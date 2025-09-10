/*
 * WebSocket Message Router
 * High-performance message routing and dispatch system
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
	"go.uber.org/zap"

	"algoveda/internal/monitoring"
	"algoveda/pkg/logger"
)

// MessageType represents different types of WebSocket messages
type MessageType int

const (
	MessageTypeSubscribe MessageType = iota + 1
	MessageTypeUnsubscribe
	MessageTypeBroadcast
	MessageTypePrivate
	MessageTypeHeartbeat
	MessageTypeMarketData
	MessageTypeOrderUpdate
	MessageTypeTradeUpdate
	MessageTypeRiskAlert
	MessageTypeSystemMessage
)

// Message represents a WebSocket message
type Message struct {
	ID        string                 `json:"id,omitempty"`
	Type      MessageType            `json:"type"`
	Channel   string                 `json:"channel,omitempty"`
	Data      interface{}            `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
	ClientID  string                 `json:"client_id,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// MessageHandler defines the interface for message handlers
type MessageHandler interface {
	HandleMessage(clientID string, message *Message) error
	GetSupportedTypes() []MessageType
}

// MessageRouter routes messages to appropriate handlers
type MessageRouter struct {
	logger   *logger.Logger
	metrics  *monitoring.Metrics
	
	// Message handling
	handlers map[MessageType]MessageHandler
	
	// Message queues
	incomingQueue  chan *IncomingMessage
	outgoingQueue  chan *OutgoingMessage
	
	// Routing statistics
	messagesRouted    int64
	messagesDropped   int64
	routingErrors     int64
	avgProcessingTime int64
	
	// Configuration
	queueSize        int
	maxProcessingTime time.Duration
	enableMetrics    bool
	
	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// IncomingMessage represents a message received from a client
type IncomingMessage struct {
	ClientID  string
	Message   *Message
	RawData   []byte
	Timestamp time.Time
	Callback  func(error)
}

// OutgoingMessage represents a message to send to a client
type OutgoingMessage struct {
	ClientID  string
	Message   *Message
	Timestamp time.Time
	Priority  int
	Callback  func(error)
}

// MessageRouterConfig holds configuration for the message router
type MessageRouterConfig struct {
	Logger            *logger.Logger
	Monitor           *monitoring.Metrics
	QueueSize         int
	MaxProcessingTime time.Duration
	EnableMetrics     bool
}

// NewMessageRouter creates a new message router
func NewMessageRouter(config MessageRouterConfig) *MessageRouter {
	ctx, cancel := context.WithCancel(context.Background())
	
	if config.QueueSize == 0 {
		config.QueueSize = 10000
	}
	
	if config.MaxProcessingTime == 0 {
		config.MaxProcessingTime = 100 * time.Millisecond
	}
	
	router := &MessageRouter{
		logger:            config.Logger,
		metrics:           config.Monitor,
		handlers:          make(map[MessageType]MessageHandler),
		incomingQueue:     make(chan *IncomingMessage, config.QueueSize),
		outgoingQueue:     make(chan *OutgoingMessage, config.QueueSize),
		queueSize:         config.QueueSize,
		maxProcessingTime: config.MaxProcessingTime,
		enableMetrics:     config.EnableMetrics,
		ctx:               ctx,
		cancel:            cancel,
	}
	
	// Register default handlers
	router.registerDefaultHandlers()
	
	return router
}

// Start starts the message router
func (mr *MessageRouter) Start(ctx context.Context) error {
	mr.logger.Info("Starting message router")
	
	// Start processing goroutines
	for i := 0; i < 4; i++ { // 4 processing workers
		mr.wg.Add(1)
		go mr.processIncomingMessages()
	}
	
	for i := 0; i < 2; i++ { // 2 outgoing workers
		mr.wg.Add(1)
		go mr.processOutgoingMessages()
	}
	
	// Start metrics collection if enabled
	if mr.enableMetrics {
		mr.wg.Add(1)
		go mr.metricsCollector()
	}
	
	return nil
}

// RegisterHandler registers a message handler for specific message types
func (mr *MessageRouter) RegisterHandler(handler MessageHandler) {
	supportedTypes := handler.GetSupportedTypes()
	
	for _, msgType := range supportedTypes {
		mr.handlers[msgType] = handler
		mr.logger.Debug("Registered handler for message type",
			zap.Int("message_type", int(msgType)))
	}
}

// ProcessMessage processes an incoming message from a client
func (mr *MessageRouter) ProcessMessage(clientID string, rawData []byte) error {
	startTime := time.Now()
	
	// Parse message
	var message Message
	if err := json.Unmarshal(rawData, &message); err != nil {
		mr.logger.Warn("Failed to parse message",
			zap.String("client_id", clientID),
			zap.Error(err))
		
		atomic.AddInt64(&mr.routingErrors, 1)
		return fmt.Errorf("invalid message format: %w", err)
	}
	
	// Set timestamp if not provided
	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}
	
	// Set client ID
	message.ClientID = clientID
	
	// Create incoming message
	incomingMsg := &IncomingMessage{
		ClientID:  clientID,
		Message:   &message,
		RawData:   rawData,
		Timestamp: time.Now(),
	}
	
	// Queue message for processing
	select {
	case mr.incomingQueue <- incomingMsg:
		atomic.AddInt64(&mr.messagesRouted, 1)
		
		// Update processing time metric
		if mr.enableMetrics {
			processingTime := time.Since(startTime)
			atomic.StoreInt64(&mr.avgProcessingTime, processingTime.Nanoseconds())
		}
		
		return nil
		
	default:
		// Queue is full, drop message
		atomic.AddInt64(&mr.messagesDropped, 1)
		mr.logger.Warn("Incoming message queue full, dropping message",
			zap.String("client_id", clientID),
			zap.Int("message_type", int(message.Type)))
		
		return fmt.Errorf("message queue full")
	}
}

// SendMessage sends a message to a specific client
func (mr *MessageRouter) SendMessage(clientID string, message *Message) error {
	return mr.SendMessageWithPriority(clientID, message, 0)
}

// SendMessageWithPriority sends a message with specified priority
func (mr *MessageRouter) SendMessageWithPriority(clientID string, message *Message, priority int) error {
	outgoingMsg := &OutgoingMessage{
		ClientID:  clientID,
		Message:   message,
		Timestamp: time.Now(),
		Priority:  priority,
	}
	
	select {
	case mr.outgoingQueue <- outgoingMsg:
		return nil
		
	default:
		atomic.AddInt64(&mr.messagesDropped, 1)
		mr.logger.Warn("Outgoing message queue full",
			zap.String("client_id", clientID),
			zap.Int("message_type", int(message.Type)))
		
		return fmt.Errorf("outgoing message queue full")
	}
}

// GetStats returns routing statistics
func (mr *MessageRouter) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"messages_routed":       atomic.LoadInt64(&mr.messagesRouted),
		"messages_dropped":      atomic.LoadInt64(&mr.messagesDropped),
		"routing_errors":        atomic.LoadInt64(&mr.routingErrors),
		"avg_processing_time_ns": atomic.LoadInt64(&mr.avgProcessingTime),
		"incoming_queue_size":   len(mr.incomingQueue),
		"outgoing_queue_size":   len(mr.outgoingQueue),
		"handlers_registered":   len(mr.handlers),
	}
}

// Shutdown gracefully shuts down the message router
func (mr *MessageRouter) Shutdown(ctx context.Context) {
	mr.logger.Info("Shutting down message router")
	
	mr.cancel()
	
	// Close queues
	close(mr.incomingQueue)
	close(mr.outgoingQueue)
	
	// Wait for workers to finish
	mr.wg.Wait()
	
	mr.logger.Info("Message router shutdown complete")
}

// Process incoming messages
func (mr *MessageRouter) processIncomingMessages() {
	defer mr.wg.Done()
	
	for {
		select {
		case <-mr.ctx.Done():
			return
			
		case incomingMsg, ok := <-mr.incomingQueue:
			if !ok {
				return
			}
			
			mr.handleIncomingMessage(incomingMsg)
		}
	}
}

// Handle a single incoming message
func (mr *MessageRouter) handleIncomingMessage(incomingMsg *IncomingMessage) {
	startTime := time.Now()
	defer func() {
		if mr.enableMetrics {
			processingTime := time.Since(startTime)
			mr.metrics.RecordHistogram("message_processing_time_ms", 
				float64(processingTime.Nanoseconds())/1e6)
		}
	}()
	
	// Apply processing timeout
	ctx, cancel := context.WithTimeout(mr.ctx, mr.maxProcessingTime)
	defer cancel()
	
	done := make(chan error, 1)
	
	go func() {
		handler, exists := mr.handlers[incomingMsg.Message.Type]
		if !exists {
			done <- fmt.Errorf("no handler for message type %d", incomingMsg.Message.Type)
			return
		}
		
		err := handler.HandleMessage(incomingMsg.ClientID, incomingMsg.Message)
		done <- err
	}()
	
	select {
	case <-ctx.Done():
		atomic.AddInt64(&mr.routingErrors, 1)
		mr.logger.Warn("Message processing timeout",
			zap.String("client_id", incomingMsg.ClientID),
			zap.Int("message_type", int(incomingMsg.Message.Type)),
			zap.Duration("timeout", mr.maxProcessingTime))
		
		if incomingMsg.Callback != nil {
			incomingMsg.Callback(fmt.Errorf("processing timeout"))
		}
		
	case err := <-done:
		if err != nil {
			atomic.AddInt64(&mr.routingErrors, 1)
			mr.logger.Error("Message processing error",
				zap.String("client_id", incomingMsg.ClientID),
				zap.Int("message_type", int(incomingMsg.Message.Type)),
				zap.Error(err))
		}
		
		if incomingMsg.Callback != nil {
			incomingMsg.Callback(err)
		}
	}
}

// Process outgoing messages
func (mr *MessageRouter) processOutgoingMessages() {
	defer mr.wg.Done()
	
	for {
		select {
		case <-mr.ctx.Done():
			return
			
		case outgoingMsg, ok := <-mr.outgoingQueue:
			if !ok {
				return
			}
			
			mr.handleOutgoingMessage(outgoingMsg)
		}
	}
}

// Handle a single outgoing message
func (mr *MessageRouter) handleOutgoingMessage(outgoingMsg *OutgoingMessage) {
	// This would integrate with the connection manager to send the message
	// For now, we'll log it
	
	mr.logger.Debug("Processing outgoing message",
		zap.String("client_id", outgoingMsg.ClientID),
		zap.Int("message_type", int(outgoingMsg.Message.Type)),
		zap.Int("priority", outgoingMsg.Priority))
	
	// Simulate sending message
	// In real implementation, this would call
