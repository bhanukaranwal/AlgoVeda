package websocket

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "sync/atomic"
    "time"

    "github.com/gorilla/websocket"
    "go.uber.org/zap"

    "github.com/algoveda/websocket-gateway/internal/config"
    "github.com/algoveda/websocket-gateway/internal/monitoring"
    "github.com/algoveda/websocket-gateway/internal/security"
)

type ConnectionManager struct {
    config          *config.WebSocketConfig
    logger          *zap.Logger
    monitor         *monitoring.Monitor
    security        *security.Manager
    upgrader        *websocket.Upgrader
    connections     sync.Map // map[string]*Connection
    subscriptions   sync.Map // map[string]map[string]*Connection
    messageHandler  MessageHandler
    metrics         *Metrics
    shutdownCh      chan struct{}
    wg              sync.WaitGroup
}

type Connection struct {
    ID              string
    UserID          string
    SessionID       string
    RemoteAddr      string
    UserAgent       string
    ConnectedAt     time.Time
    LastActivity    time.Time
    conn            *websocket.Conn
    writeMutex      sync.Mutex
    readMutex       sync.Mutex
    subscriptions   map[string]bool
    rateLimiter     *RateLimiter
    messageQueue    chan *Message
    closeOnce       sync.Once
    closeCh         chan struct{}
    manager         *ConnectionManager
}

type Message struct {
    Type      string          `json:"type"`
    Topic     string          `json:"topic,omitempty"`
    Data      json.RawMessage `json:"data"`
    Timestamp int64           `json:"timestamp"`
    RequestID string          `json:"request_id,omitempty"`
}

type MessageHandler func(conn *Connection, msg *Message) error

type Metrics struct {
    ActiveConnections   int64
    TotalConnections    int64
    MessagesReceived    int64
    MessagesSent        int64
    BytesReceived       int64
    BytesSent          int64
    ConnectionErrors    int64
    MessageErrors      int64
}

type RateLimiter struct {
    maxRequests int
    window      time.Duration
    requests    []time.Time
    mutex       sync.Mutex
}

func NewConnectionManager(
    config *config.WebSocketConfig,
    logger *zap.Logger,
    monitor *monitoring.Monitor,
    security *security.Manager,
) (*ConnectionManager, error) {
    upgrader := &websocket.Upgrader{
        ReadBufferSize:  config.ReadBufferSize,
        WriteBufferSize: config.WriteBufferSize,
        CheckOrigin: func(r *http.Request) bool {
            // Implement origin checking logic
            return true // For now, allow all origins
        },
        EnableCompression: config.EnableCompression,
    }

    cm := &ConnectionManager{
        config:     config,
        logger:     logger,
        monitor:    monitor,
        security:   security,
        upgrader:   upgrader,
        metrics:    &Metrics{},
        shutdownCh: make(chan struct{}),
    }

    return cm, nil
}

func (cm *ConnectionManager) Start(ctx context.Context) error {
    cm.logger.Info("Starting WebSocket connection manager")

    // Start metrics reporting goroutine
    cm.wg.Add(1)
    go cm.metricsReporter(ctx)

    // Start connection cleanup goroutine
    cm.wg.Add(1)
    go cm.connectionCleaner(ctx)

    return nil
}

func (cm *ConnectionManager) Shutdown(ctx context.Context) error {
    cm.logger.Info("Shutting down WebSocket connection manager")
    
    close(cm.shutdownCh)

    // Close all connections
    cm.connections.Range(func(key, value interface{}) bool {
        if conn, ok := value.(*Connection); ok {
            conn.Close()
        }
        return true
    })

    // Wait for goroutines to finish
    done := make(chan struct{})
    go func() {
        cm.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        cm.logger.Info("All goroutines stopped")
    case <-ctx.Done():
        cm.logger.Warn("Shutdown timeout reached, forcing exit")
    }

    return nil
}

func (cm *ConnectionManager) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    // Security checks
    if !cm.security.ValidateRequest(r) {
        http.Error(w, "Unauthorized", http.StatusUnauthorized)
        return
    }

    // Rate limiting
    clientIP := cm.getClientIP(r)
    if !cm.security.CheckRateLimit(clientIP) {
        http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
        return
    }

    // Upgrade connection
    conn, err := cm.upgrader.Upgrade(w, r, nil)
    if err != nil {
        cm.logger.Error("Failed to upgrade connection", zap.Error(err))
        atomic.AddInt64(&cm.metrics.ConnectionErrors, 1)
        return
    }

    // Create connection object
    connection := cm.createConnection(conn, r)
    
    // Store connection
    cm.connections.Store(connection.ID, connection)
    atomic.AddInt64(&cm.metrics.ActiveConnections, 1)
    atomic.AddInt64(&cm.metrics.TotalConnections, 1)

    cm.logger.Info("New WebSocket connection established",
        zap.String("connection_id", connection.ID),
        zap.String("remote_addr", connection.RemoteAddr),
        zap.String("user_id", connection.UserID))

    // Start connection handlers
    go connection.readPump()
    go connection.writePump()
}

func (cm *ConnectionManager) createConnection(conn *websocket.Conn, r *http.Request) *Connection {
    connectionID := generateConnectionID()
    userID := cm.getUserID(r)
    sessionID := cm.getSessionID(r)

    return &Connection{
        ID:            connectionID,
        UserID:        userID,
        SessionID:     sessionID,
        RemoteAddr:    cm.getClientIP(r),
        UserAgent:     r.UserAgent(),
        ConnectedAt:   time.Now(),
        LastActivity:  time.Now(),
        conn:          conn,
        subscriptions: make(map[string]bool),
        rateLimiter:   NewRateLimiter(cm.config.RateLimit.MaxRequests, cm.config.RateLimit.Window),
        messageQueue:  make(chan *Message, cm.config.MessageQueueSize),
        closeCh:       make(chan struct{}),
        manager:       cm,
    }
}

func (c *Connection) readPump() {
    defer func() {
        c.manager.removeConnection(c)
        c.conn.Close()
    }()

    c.conn.SetReadLimit(int64(c.manager.config.MaxMessageSize))
    c.conn.SetReadDeadline(time.Now().Add(c.manager.config.ReadTimeout))
    c.conn.SetPongHandler(func(string) error {
        c.conn.SetReadDeadline(time.Now().Add(c.manager.config.ReadTimeout))
        return nil
    })

    for {
        select {
        case <-c.closeCh:
            return
        default:
        }

        _, messageData, err := c.conn.ReadMessage()
        if err != nil {
            if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
                c.manager.logger.Error("WebSocket read error", 
                    zap.String("connection_id", c.ID),
                    zap.Error(err))
            }
            return
        }

        atomic.AddInt64(&c.manager.metrics.MessagesReceived, 1)
        atomic.AddInt64(&c.manager.metrics.BytesReceived, int64(len(messageData)))
        c.LastActivity = time.Now()

        // Rate limiting
        if !c.rateLimiter.Allow() {
            c.manager.logger.Warn("Rate limit exceeded for connection",
                zap.String("connection_id", c.ID))
            continue
        }

        // Parse message
        var msg Message
        if err := json.Unmarshal(messageData, &msg); err != nil {
            c.manager.logger.Error("Failed to parse message",
                zap.String("connection_id", c.ID),
                zap.Error(err))
            atomic.AddInt64(&c.manager.metrics.MessageErrors, 1)
            continue
        }

        msg.Timestamp = time.Now().UnixNano()

        // Handle message
        if c.manager.messageHandler != nil {
            if err := c.manager.messageHandler(c, &msg); err != nil {
                c.manager.logger.Error("Message handler error",
                    zap.String("connection_id", c.ID),
                    zap.Error(err))
                atomic.AddInt64(&c.manager.metrics.MessageErrors, 1)
            }
        }
    }
}

func (c *Connection) writePump() {
    ticker := time.NewTicker(c.manager.config.PingPeriod)
    defer func() {
        ticker.Stop()
        c.conn.Close()
    }()

    for {
        select {
        case message, ok := <-c.messageQueue:
            c.conn.SetWriteDeadline(time.Now().Add(c.manager.config.WriteTimeout))
            if !ok {
                c.conn.WriteMessage(websocket.CloseMessage, []byte{})
                return
            }

            if err := c.writeMessage(message); err != nil {
                c.manager.logger.Error("Write message error",
                    zap.String("connection_id", c.ID),
                    zap.Error(err))
                return
            }

        case <-ticker.C:
            c.conn.SetWriteDeadline(time.Now().Add(c.manager.config.WriteTimeout))
            if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
                return
            }

        case <-c.closeCh:
            return
        }
    }
}

func (c *Connection) writeMessage(msg *Message) error {
    c.writeMutex.Lock()
    defer c.writeMutex.Unlock()

    data, err := json.Marshal(msg)
    if err != nil {
        return err
    }

    if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
        return err
    }

    atomic.AddInt64(&c.manager.metrics.MessagesSent, 1)
    atomic.AddInt64(&c.manager.metrics.BytesSent, int64(len(data)))
    return nil
}

func (c *Connection) SendMessage(msg *Message) error {
    select {
    case c.messageQueue <- msg:
        return nil
    case <-c.closeCh:
        return fmt.Errorf("connection closed")
    default:
        return fmt.Errorf("message queue full")
    }
}

func (c *Connection) Subscribe(topic string) error {
    c.subscriptions[topic] = true
    
    // Add to global subscriptions map
    subs, _ := c.manager.subscriptions.LoadOrStore(topic, make(map[string]*Connection))
    if subMap, ok := subs.(map[string]*Connection); ok {
        subMap[c.ID] = c
    }

    c.manager.logger.Debug("Connection subscribed to topic",
        zap.String("connection_id", c.ID),
        zap.String("topic", topic))
    
    return nil
}

func (c *Connection) Unsubscribe(topic string) error {
    delete(c.subscriptions, topic)
    
    // Remove from global subscriptions map
    if subs, ok := c.manager.subscriptions.Load(topic); ok {
        if subMap, ok := subs.(map[string]*Connection); ok {
            delete(subMap, c.ID)
        }
    }

    c.manager.logger.Debug("Connection unsubscribed from topic",
        zap.String("connection_id", c.ID),
        zap.String("topic", topic))
    
    return nil
}

func (c *Connection) Close() {
    c.closeOnce.Do(func() {
        close(c.closeCh)
        close(c.messageQueue)
        
        // Unsubscribe from all topics
        for topic := range c.subscriptions {
            c.Unsubscribe(topic)
        }
    })
}

func (cm *ConnectionManager) SetMessageHandler(handler MessageHandler) {
    cm.messageHandler = handler
}

func (cm *ConnectionManager) Broadcast(topic string, msg *Message) error {
    if subs, ok := cm.subscriptions.Load(topic); ok {
        if subMap, ok := subs.(map[string]*Connection); ok {
            for _, conn := range subMap {
                if err := conn.SendMessage(msg); err != nil {
                    cm.logger.Warn("Failed to send broadcast message",
                        zap.String("connection_id", conn.ID),
                        zap.String("topic", topic),
                        zap.Error(err))
                }
            }
        }
    }
    return nil
}

func (cm *ConnectionManager) GetConnection(connectionID string) (*Connection, bool) {
    if conn, ok := cm.connections.Load(connectionID); ok {
        return conn.(*Connection), true
    }
    return nil, false
}

func (cm *ConnectionManager) GetConnectionsByUser(userID string) []*Connection {
    var connections []*Connection
    cm.connections.Range(func(key, value interface{}) bool {
        if conn, ok := value.(*Connection); ok && conn.UserID == userID {
            connections = append(connections, conn)
        }
        return true
    })
    return connections
}

func (cm *ConnectionManager) removeConnection(conn *Connection) {
    cm.connections.Delete(conn.ID)
    atomic.AddInt64(&cm.metrics.ActiveConnections, -1)
    
    cm.logger.Info("WebSocket connection closed",
        zap.String("connection_id", conn.ID),
        zap.String("user_id", conn.UserID),
        zap.Duration("duration", time.Since(conn.ConnectedAt)))
}

func (cm *ConnectionManager) metricsReporter(ctx context.Context) {
    defer cm.wg.Done()
    
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            cm.reportMetrics()
        case <-ctx.Done():
            return
        case <-cm.shutdownCh:
            return
        }
    }
}

func (cm *ConnectionManager) connectionCleaner(ctx context.Context) {
    defer cm.wg.Done()
    
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            cm.cleanupStaleConnections()
        case <-ctx.Done():
            return
        case <-cm.shutdownCh:
            return
        }
    }
}

func (cm *ConnectionManager) reportMetrics() {
    cm.logger.Info("WebSocket metrics",
        zap.Int64("active_connections", atomic.LoadInt64(&cm.metrics.ActiveConnections)),
        zap.Int64("total_connections", atomic.LoadInt64(&cm.metrics.TotalConnections)),
        zap.Int64("messages_received", atomic.LoadInt64(&cm.metrics.MessagesReceived)),
        zap.Int64("messages_sent", atomic.LoadInt64(&cm.metrics.MessagesSent)),
        zap.Int64("bytes_received", atomic.LoadInt64(&cm.metrics.BytesReceived)),
        zap.Int64("bytes_sent", atomic.LoadInt64(&cm.metrics.BytesSent)),
        zap.Int64("connection_errors", atomic.LoadInt64(&cm.metrics.ConnectionErrors)),
        zap.Int64("message_errors", atomic.LoadInt64(&cm.metrics.MessageErrors)))
}

func (cm *ConnectionManager) cleanupStaleConnections() {
    now := time.Now()
    var staleConnections []string

    cm.connections.Range(func(key, value interface{}) bool {
        if conn, ok := value.(*Connection); ok {
            if now.Sub(conn.LastActivity) > cm.config.IdleTimeout {
                staleConnections = append(staleConnections, conn.ID)
            }
        }
        return true
    })

    for _, connID := range staleConnections {
        if conn, ok := cm.GetConnection(connID); ok {
            cm.logger.Info("Closing stale connection",
                zap.String("connection_id", connID),
                zap.Duration("idle_time", now.Sub(conn.LastActivity)))
            conn.Close()
        }
    }
}

// Helper functions
func (cm *ConnectionManager) getClientIP(r *http.Request) string {
    // Check X-Forwarded-For header first
    if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
        return xff
    }
    // Check X-Real-IP header
    if xri := r.Header.Get("X-Real-IP"); xri != "" {
        return xri
    }
    // Fall back to remote address
    return r.RemoteAddr
}

func (cm *ConnectionManager) getUserID(r *http.Request) string {
    // Extract user ID from JWT token or session
    // This would be implemented based on your authentication system
    return "anonymous" // Placeholder
}

func (cm *ConnectionManager) getSessionID(r *http.Request) string {
    // Extract session ID from cookie or header
    // This would be implemented based on your session management
    return generateSessionID() // Placeholder
}

func generateConnectionID() string {
    // Generate unique connection ID
    return fmt.Sprintf("conn_%d_%d", time.Now().UnixNano(), rand.Int63())
}

func generateSessionID() string {
    // Generate unique session ID
    return fmt.Sprintf("sess_%d_%d", time.Now().UnixNano(), rand.Int63())
}

// Rate limiter implementation
func NewRateLimiter(maxRequests int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        maxRequests: maxRequests,
        window:      window,
        requests:    make([]time.Time, 0),
    }
}

func (rl *RateLimiter) Allow() bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()

    now := time.Now()
    cutoff := now.Add(-rl.window)

    // Remove old requests
    var validRequests []time.Time
    for _, reqTime := range rl.requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    rl.requests = validRequests

    // Check if we can allow this request
    if len(rl.requests) < rl.maxRequests {
        rl.requests = append(rl.requests, now)
        return true
    }

    return false
}

func (cm *ConnectionManager) HandleAdminConnections(w http.ResponseWriter, r *http.Request) {
    // Admin endpoint to view connection statistics
    var connections []map[string]interface{}
    
    cm.connections.Range(func(key, value interface{}) bool {
        if conn, ok := value.(*Connection); ok {
            connections = append(connections, map[string]interface{}{
                "id":              conn.ID,
                "user_id":         conn.UserID,
                "remote_addr":     conn.RemoteAddr,
                "connected_at":    conn.ConnectedAt,
                "last_activity":   conn.LastActivity,
                "subscriptions":   len(conn.subscriptions),
            })
        }
        return true
    })

    response := map[string]interface{}{
        "total_connections": len(connections),
        "connections":      connections,
        "metrics":          cm.metrics,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
