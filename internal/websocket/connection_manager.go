/*
 * WebSocket Connection Manager
 * High-performance connection management with load balancing
 */

package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"

	"algoveda/internal/monitoring"
	"algoveda/internal/security"
	"algoveda/pkg/logger"
)

// ConnectionManager manages WebSocket connections
type ConnectionManager struct {
	// Configuration
	config Config
	logger *logger.Logger
	
	// Connection management
	connections     sync.Map              // map[string]*Client
	subscriptions   sync.Map              // map[string]map[string]*Client (channel -> clientID -> client)
	connectionCount int64
	
	// Message routing
	messageRouter   *MessageRouter
	loadBalancer   *LoadBalancer
	
	// Monitoring and security
	metrics         *monitoring.Metrics
	securityManager *security.Manager
	
	// Channels
	register     chan *Client
	unregister   chan *Client
	broadcast    chan *BroadcastMessage
	
	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	
	// Performance monitoring
	messagesSent     int64
	messagesReceived int64
	bytesTransferred int64
}

// Client represents a WebSocket client connection
type Client struct {
	ID              string                 `json:"id"`
	Connection      *websocket.Conn        `json:"-"`
	RemoteAddr      string                 `json:"remote_addr"`
	UserAgent       string                 `json:"user_agent"`
	Headers         map[string][]string    `json:"headers"`
	ConnectedAt     time.Time              `json:"connected_at"`
	LastActivity    time.Time              `json:"last_activity"`
	Subscriptions   map[string]bool        `json:"subscriptions"`
	Authenticated   bool                   `json:"authenticated"`
	UserID          string                 `json:"user_id,omitempty"`
	
	// Connection state
	send         chan []byte
	closed       int32
	pingTicker   *time.Ticker
	pongWait     time.Duration
	writeWait    time.Duration
	maxMessageSize int64
	
	// Statistics
	MessagesSent     int64 `json:"messages_sent"`
	MessagesReceived int64 `json:"messages_received"`
	BytesTransferred int64 `json:"bytes_transferred"`
}

// BroadcastMessage represents a message to broadcast
type BroadcastMessage struct {
	Channel string
	Data    []byte
	Exclude []string // Client IDs to exclude
}

// Config holds connection manager configuration
type Config struct {
	MaxConnections     int           `yaml:"max_connections"`
	ReadBufferSize     int           `yaml:"read_buffer_size"`
	WriteBufferSize    int           `yaml:"write_buffer_size"`
	WriteTimeout       time.Duration `yaml:"write_timeout"`
	ReadTimeout        time.Duration `yaml:"read_timeout"`
	PingInterval       time.Duration `yaml:"ping_interval"`
	MaxMessageSize     int64         `yaml:"max_message_size"`
	CompressionEnabled bool          `yaml:"compression_enabled"`
	
	Logger             *logger.Logger
	Monitor            *monitoring.Metrics
	SecurityManager    *security.Manager
}

// NewConnectionManager creates a new connection manager
func NewConnectionManager(config Config) *ConnectionManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	cm := &ConnectionManager{
		config:          config,
		logger:          config.Logger,
		metrics:         config.Monitor,
		securityManager: config.SecurityManager,
		
		register:   make(chan *Client, 1000),
		unregister: make(chan *Client, 1000),
		broadcast:  make(chan *BroadcastMessage, 10000),
		
		ctx:    ctx,
		cancel: cancel,
	}
	
	// Initialize message router
	cm.messageRouter = NewMessageRouter(MessageRouterConfig{
		Logger:  config.Logger,
		Monitor: config.Monitor,
	})
	
	// Initialize load balancer
	cm.loadBalancer = NewLoadBalancer(LoadBalancerConfig{
		Logger: config.Logger,
	})
	
	return cm
}

// Start starts the connection manager
func (cm *ConnectionManager) Start(ctx context.Context) error {
	cm.logger.Info("Starting WebSocket connection manager")
	
	// Start message router
	if err := cm.messageRouter.Start(ctx); err != nil {
		return fmt.Errorf("failed to start message router: %w", err)
	}
	
	// Start load balancer
	if err := cm.loadBalancer.Start(ctx); err != nil {
		return fmt.Errorf("failed to start load balancer: %w", err)
	}
	
	// Start connection manager loop
	cm.wg.Add(1)
	go cm.run()
	
	return nil
}

// RegisterClient registers a new WebSocket client
func (cm *ConnectionManager) RegisterClient(client *Client) error {
	// Check connection limits
	if atomic.LoadInt64(&cm.connectionCount) >= int64(cm.config.MaxConnections) {
		return fmt.Errorf("maximum connection limit reached")
	}
	
	// Initialize client
	client.send = make(chan []byte, 256)
	client.Subscriptions = make(map[string]bool)
	client.pongWait = cm.config.ReadTimeout
	client.writeWait = cm.config.WriteTimeout
	client.maxMessageSize = cm.config.MaxMessageSize
	
	// Configure WebSocket connection
	client.Connection.SetReadLimit(cm.config.MaxMessageSize)
	client.Connection.SetReadDeadline(time.Now().Add(client.pongWait))
	client.Connection.SetPongHandler(func(string) error {
		client.Connection.SetReadDeadline(time.Now().Add(client.pongWait))
		client.LastActivity = time.Now()
		return nil
	})
	
	// Register client
	select {
	case cm.register <- client:
		return nil
	default:
		return fmt.Errorf("registration queue full")
	}
}

// UnregisterClient unregisters a WebSocket client
func (cm *ConnectionManager) UnregisterClient(clientID string) {
	if client, ok := cm.connections.Load(clientID); ok {
		cm.unregister <- client.(*Client)
	}
}

// Broadcast sends a message to all clients on a channel
func (cm *ConnectionManager) Broadcast(channel string, data []byte, exclude ...string) int {
	msg := &BroadcastMessage{
		Channel: channel,
		Data:    data,
		Exclude: exclude,
	}
	
	select {
	case cm.broadcast <- msg:
		return cm.getChannelSubscriberCount(channel)
	default:
		cm.logger.Warn("Broadcast queue full, dropping message")
		return 0
	}
}

// Subscribe subscribes a client to a channel
func (cm *ConnectionManager) Subscribe(clientID, channel string) error {
	clientInterface, ok := cm.connections.Load(clientID)
	if !ok {
		return fmt.Errorf("client not found")
	}
	
	client := clientInterface.(*Client)
	
	// Add to client subscriptions
	client.Subscriptions[channel] = true
	
	// Add to channel subscriptions
	channelSubs, _ := cm.subscriptions.LoadOrStore(channel, &sync.Map{})
	channelSubs.(*sync.Map).Store(clientID, client)
	
	cm.logger.Debug("Client subscribed to channel",
		zap.String("client_id", clientID),
		zap.String("channel", channel))
	
	// Update metrics
	cm.metrics.IncrementCounter("websocket_subscriptions_total")
	
	return nil
}

// Unsubscribe unsubscribes a client from a channel
func (cm *ConnectionManager) Unsubscribe(clientID, channel string) error {
	clientInterface, ok := cm.connections.Load(clientID)
	if !ok {
		return fmt.Errorf("client not found")
	}
	
	client := clientInterface.(*Client)
	
	// Remove from client subscriptions
	delete(client.Subscriptions, channel)
	
	// Remove from channel subscriptions
	if channelSubs, ok := cm.subscriptions.Load(channel); ok {
		channelSubs.(*sync.Map).Delete(clientID)
	}
	
	cm.logger.Debug("Client unsubscribed from channel",
		zap.String("client_id", clientID),
		zap.String("channel", channel))
	
	// Update metrics
	cm.metrics.IncrementCounter("websocket_unsubscriptions_total")
	
	return nil
}

// SendToClient sends a message to a specific client
func (cm *ConnectionManager) SendToClient(clientID string, data []byte) error {
	clientInterface, ok := cm.connections.Load(clientID)
	if !ok {
		return fmt.Errorf("client not found")
	}
	
	client := clientInterface.(*Client)
	
	select {
	case client.send <- data:
		return nil
	default:
		// Client send buffer is full, disconnect
		cm.logger.Warn("Client send buffer full, disconnecting",
			zap.String("client_id", clientID))
		go cm.UnregisterClient(clientID)
		return fmt.Errorf("client send buffer full")
	}
}

// GetConnections returns all active connections
func (cm *ConnectionManager) GetConnections() []Client {
	var connections []Client
	
	cm.connections.Range(func(key, value interface{}) bool {
		client := value.(*Client)
		connections = append(connections, *client)
		return true
	})
	
	return connections
}

// GetConnectionCount returns the current connection count
func (cm *ConnectionManager) GetConnectionCount() int {
	return int(atomic.LoadInt64(&cm.connectionCount))
}

// GetSubscriptions returns subscription information
func (cm *ConnectionManager) GetSubscriptions() map[string][]string {
	subscriptions := make(map[string][]string)
	
	cm.subscriptions.Range(func(key, value interface{}) bool {
		channel := key.(string)
		channelSubs := value.(*sync.Map)
		
		var subscribers []string
		channelSubs.Range(func(clientKey, clientValue interface{}) bool {
			subscribers = append(subscribers, clientKey.(string))
			return true
		})
		
		subscriptions[channel] = subscribers
		return true
	})
	
	return subscriptions
}

// GetStats returns connection manager statistics
func (cm *ConnectionManager) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"active_connections":  atomic.LoadInt64(&cm.connectionCount),
		"messages_sent":       atomic.LoadInt64(&cm.messagesSent),
		"messages_received":   atomic.LoadInt64(&cm.messagesReceived),
		"bytes_transferred":   atomic.LoadInt64(&cm.bytesTransferred),
		"subscription_count":  cm.getSubscriptionCount(),
	}
}

// IsHealthy returns the health status
func (cm *ConnectionManager) IsHealthy() bool {
	return cm.ctx.Err() == nil
}

// IsReady returns the readiness status
func (cm *ConnectionManager) IsReady() bool {
	return cm.IsHealthy()
}

// Shutdown gracefully shuts down the connection manager
func (cm *ConnectionManager) Shutdown(ctx context.Context) {
	cm.logger.Info("Shutting down WebSocket connection manager")
	
	// Cancel context
	cm.cancel()
	
	// Close all client connections
	cm.connections.Range(func(key, value interface{}) bool {
		client := value.(*Client)
		cm.closeClient(client)
		return true
	})
	
	// Wait for goroutines to finish
	cm.wg.Wait()
	
	cm.logger.Info("WebSocket connection manager shutdown complete")
}

// Main event loop
func (cm *ConnectionManager) run() {
	defer cm.wg.Done()
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-cm.ctx.Done():
			return
			
		case client := <-cm.register:
			cm.handleClientRegistration(client)
			
		case client := <-cm.unregister:
			cm.handleClientUnregistration(client)
			
		case message := <-cm.broadcast:
			cm.handleBroadcast(message)
			
		case <-ticker.C:
			cm.cleanup()
		}
	}
}

// Handle client registration
func (cm *ConnectionManager) handleClientRegistration(client *Client) {
	// Store client connection
	cm.connections.Store(client.ID, client)
	atomic.AddInt64(&cm.connectionCount, 1)
	
	// Start client goroutines
	cm.wg.Add(2)
	go cm.clientReader(client)
	go cm.clientWriter(client)
	
	cm.logger.Info("Client registered",
		zap.String("client_id", client.ID),
		zap.String("remote_addr", client.RemoteAddr),
		zap.Int("total_connections", int(atomic.LoadInt64(&cm.connectionCount))))
	
	// Update metrics
	cm.metrics.IncrementCounter("websocket_connections_total")
	cm.metrics.SetGauge("websocket_active_connections", 
		float64(atomic.LoadInt64(&cm.connectionCount)))
}

// Handle client unregistration
func (cm *ConnectionManager) handleClientUnregistration(client *Client) {
	// Remove from connections
	if _, ok := cm.connections.LoadAndDelete(client.ID); ok {
		atomic.AddInt64(&cm.connectionCount, -1)
	}
	
	// Remove from all subscriptions
	cm.subscriptions.Range(func(key, value interface{}) bool {
		channelSubs := value.(*sync.Map)
		channelSubs.Delete(client.ID)
		return true
	})
	
	// Close client connection
	cm.closeClient(client)
	
	cm.logger.Info("Client unregistered",
		zap.String("client_id", client.ID),
		zap.Int("total_connections", int(atomic.LoadInt64(&cm.connectionCount))))
	
	// Update metrics
	cm.metrics.IncrementCounter("websocket_disconnections_total")
	cm.metrics.SetGauge("websocket_active_connections", 
		float64(atomic.LoadInt64(&cm.connectionCount)))
}

// Handle broadcast message
func (cm *ConnectionManager) handleBroadcast(message *BroadcastMessage) {
	if channelSubs, ok := cm.subscriptions.Load(message.Channel); ok {
		excludeMap := make(map[string]bool)
		for _, clientID := range message.Exclude {
			excludeMap[clientID] = true
		}
		
		sent := 0
		channelSubs.(*sync.Map).Range(func(key, value interface{}) bool {
			clientID := key.(string)
			if !excludeMap[clientID] {
				client := value.(*Client)
				select {
				case client.send <- message.Data:
					sent++
				default:
					// Client buffer full, skip
				}
			}
			return true
		})
		
		atomic.AddInt64(&cm.messagesSent, int64(sent))
		atomic.AddInt64(&cm.bytesTransferred, int64(len(message.Data)*sent))
	}
}

// Client reader goroutine
func (cm *ConnectionManager) clientReader(client *Client) {
	defer func() {
		cm.unregister <- client
		cm.wg.Done()
	}()
	
	client.Connection.SetReadLimit(client.maxMessageSize)
	client.Connection.SetReadDeadline(time.Now().Add(client.pongWait))
	
	for {
		_, message, err := client.Connection.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				cm.logger.Warn("WebSocket read error",
					zap.String("client_id", client.ID),
					zap.Error(err))
			}
			break
		}
		
		client.LastActivity = time.Now()
		atomic.AddInt64(&client.MessagesReceived, 1)
		atomic.AddInt64(&client.BytesTransferred, int64(len(message)))
		atomic.AddInt64(&cm.messagesReceived, 1)
		
		// Process message through router
		if err := cm.messageRouter.ProcessMessage(client.ID, message); err != nil {
			cm.logger.Error("Message processing error",
				zap.String("client_id", client.ID),
				zap.Error(err))
		}
	}
}

// Client writer goroutine
func (cm *ConnectionManager) clientWriter(client *Client) {
	ticker := time.NewTicker(cm.config.PingInterval)
	defer func() {
		ticker.Stop()
		client.Connection.Close()
		cm.wg.Done()
	}()
	
	for {
		select {
		case message, ok := <-client.send:
			client.Connection.SetWriteDeadline(time.Now().Add(client.writeWait))
			if !ok {
				client.Connection.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			
			if err := client.Connection.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}
			
			atomic.AddInt64(&client.MessagesSent, 1)
			atomic.AddInt64(&client.BytesTransferred, int64(len(message)))
			
		case <-ticker.C:
			client.Connection.SetWriteDeadline(time.Now().Add(client.writeWait))
			if err := client.Connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// Helper functions
func (cm *ConnectionManager) closeClient(client *Client) {
	if atomic.CompareAndSwapInt32(&client.closed, 0, 1) {
		close(client.send)
		if client.pingTicker != nil {
			client.pingTicker.Stop()
		}
	}
}

func (cm *ConnectionManager) getChannelSubscriberCount(channel string) int {
	if channelSubs, ok := cm.subscriptions.Load(channel); ok {
		count := 0
		channelSubs.(*sync.Map).Range(func(key, value interface{}) bool {
			count++
			return true
		})
		return count
	}
	return 0
}

func (cm *ConnectionManager) getSubscriptionCount() int {
	total := 0
	cm.subscriptions.Range(func(key, value interface{}) bool {
		channelSubs := value.(*sync.Map)
		channelSubs.Range(func(clientKey, clientValue interface{}) bool {
			total++
			return true
		})
		return true
	})
	return total
}

func (cm *ConnectionManager) cleanup() {
	now := time.Now()
	
	// Clean up stale connections
	cm.connections.Range(func(key, value interface{}) bool {
		client := value.(*Client)
		if now.Sub(client.LastActivity) > 5*time.Minute {
			cm.logger.Info("Removing stale connection",
				zap.String("client_id", client.ID))
			cm.unregister <- client
		}
		return true
	})
	
	// Clean up empty channel subscriptions
	cm.subscriptions.Range(func(key, value interface{}) bool {
		channel := key.(string)
		channelSubs := value.(*sync.Map)
		
		empty := true
		channelSubs.Range(func(clientKey, clientValue interface{}) bool {
			empty = false
			return false // Break on first item
		})
		
		if empty {
			cm.subscriptions.Delete(channel)
		}
		return true
	})
}
