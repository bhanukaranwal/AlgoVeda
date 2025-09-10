/*
 * WebSocket Protocol Handler
 * Advanced protocol handling with message validation and routing
 */

package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"

	"algoveda/internal/monitoring"
	"algoveda/internal/security"
	"algoveda/pkg/logger"
)

// Protocol version constants
const (
	ProtocolVersion1_0 = "1.0"
	ProtocolVersion2_0 = "2.0"
	DefaultProtocol    = ProtocolVersion2_0
)

// Message types for AlgoVeda protocol
type AlgoVedaMessageType string

const (
	// Authentication messages
	MsgTypeAuth         AlgoVedaMessageType = "auth"
	MsgTypeAuthResponse AlgoVedaMessageType = "auth_response"
	
	// Subscription messages
	MsgTypeSubscribe    AlgoVedaMessageType = "subscribe"
	MsgTypeUnsubscribe  AlgoVedaMessageType = "unsubscribe"
	MsgTypeSubResponse  AlgoVedaMessageType = "sub_response"
	
	// Market data messages
	MsgTypeMarketData   AlgoVedaMessageType = "market_data"
	MsgTypeTick         AlgoVedaMessageType = "tick"
	MsgTypeQuote        AlgoVedaMessageType = "quote"
	MsgTypeTrade        AlgoVedaMessageType = "trade"
	MsgTypeLevel2       AlgoVedaMessageType = "level2"
	
	// Trading messages
	MsgTypeOrder        AlgoVedaMessageType = "order"
	MsgTypeOrderUpdate  AlgoVedaMessageType = "order_update"
	MsgTypeExecution    AlgoVedaMessageType = "execution"
	MsgTypePosition     AlgoVedaMessageType = "position"
	MsgTypePortfolio    AlgoVedaMessageType = "portfolio"
	
	// Risk messages
	MsgTypeRiskAlert    AlgoVedaMessageType = "risk_alert"
	MsgTypeRiskUpdate   AlgoVedaMessageType = "risk_update"
	
	// System messages
	MsgTypeHeartbeat    AlgoVedaMessageType = "heartbeat"
	MsgTypeError        AlgoVedaMessageType = "error"
	MsgTypeStatus       AlgoVedaMessageType = "status"
)

// AlgoVedaMessage represents the base message structure
type AlgoVedaMessage struct {
	ID          string              `json:"id,omitempty"`
	Type        AlgoVedaMessageType `json:"type"`
	Version     string              `json:"version,omitempty"`
	Timestamp   int64               `json:"timestamp"`
	ClientID    string              `json:"client_id,omitempty"`
	Channel     string              `json:"channel,omitempty"`
	Data        json.RawMessage     `json:"data,omitempty"`
	Error       *ErrorData          `json:"error,omitempty"`
	Metadata    map[string]string   `json:"metadata,omitempty"`
}

// ErrorData represents error information
type ErrorData struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// Authentication data structures
type AuthRequest struct {
	Token     string            `json:"token"`
	UserID    string            `json:"user_id,omitempty"`
	ClientApp string            `json:"client_app,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type AuthResponse struct {
	Success     bool              `json:"success"`
	SessionID   string            `json:"session_id,omitempty"`
	Permissions []string          `json:"permissions,omitempty"`
	ExpiresAt   int64             `json:"expires_at,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Subscription data structures
type SubscribeRequest struct {
	Channels   []string          `json:"channels"`
	Symbols    []string          `json:"symbols,omitempty"`
	Parameters map[string]string `json:"parameters,omitempty"`
}

type SubscribeResponse struct {
	Success     bool              `json:"success"`
	Channel     string            `json:"channel"`
	Symbols     []string          `json:"symbols,omitempty"`
	Status      string            `json:"status"`
	Message     string            `json:"message,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Market data structures
type TickData struct {
	Symbol    string  `json:"symbol"`
	Price     float64 `json:"price"`
	Size      float64 `json:"size"`
	Side      string  `json:"side"`
	Timestamp int64   `json:"timestamp"`
	Exchange  string  `json:"exchange,omitempty"`
}

type QuoteData struct {
	Symbol    string  `json:"symbol"`
	BidPrice  float64 `json:"bid_price"`
	AskPrice  float64 `json:"ask_price"`
	BidSize   float64 `json:"bid_size"`
	AskSize   float64 `json:"ask_size"`
	Timestamp int64   `json:"timestamp"`
	Exchange  string  `json:"exchange,omitempty"`
}

type TradeData struct {
	Symbol     string  `json:"symbol"`
	Price      float64 `json:"price"`
	Size       float64 `json:"size"`
	Side       string  `json:"side"`
	TradeID    string  `json:"trade_id"`
	Timestamp  int64   `json:"timestamp"`
	Exchange   string  `json:"exchange,omitempty"`
	Conditions []string `json:"conditions,omitempty"`
}

// Order data structures
type OrderRequest struct {
	OrderID   string            `json:"order_id"`
	Symbol    string            `json:"symbol"`
	Side      string            `json:"side"`
	Quantity  float64           `json:"quantity"`
	Price     float64           `json:"price,omitempty"`
	OrderType string            `json:"order_type"`
	TimeInForce string          `json:"time_in_force,omitempty"`
	Parameters map[string]string `json:"parameters,omitempty"`
}

type OrderUpdate struct {
	OrderID       string  `json:"order_id"`
	Status        string  `json:"status"`
	FilledQty     float64 `json:"filled_qty"`
	RemainingQty  float64 `json:"remaining_qty"`
	AvgPrice      float64 `json:"avg_price,omitempty"`
	LastPrice     float64 `json:"last_price,omitempty"`
	LastQty       float64 `json:"last_qty,omitempty"`
	UpdateTime    int64   `json:"update_time"`
	RejectReason  string  `json:"reject_reason,omitempty"`
}

// ProtocolHandler handles WebSocket protocol operations
type ProtocolHandler struct {
	logger          *logger.Logger
	metrics         *monitoring.Metrics
	securityManager *security.Manager
	
	// Protocol configuration
	supportedVersions map[string]bool
	defaultVersion    string
	maxMessageSize    int64
	compressionLevel  int
	
	// Message handlers
	handlers map[AlgoVedaMessageType]MessageHandler
	
	// Validation rules
	validators map[AlgoVedaMessageType]MessageValidator
	
	// Rate limiting
	rateLimiter *RateLimiter
	
	// State
	activeConnections sync.Map  // clientID -> *ConnectionState
	
	// Statistics
	messagesProcessed prometheus.Counter
	messageErrors     prometheus.Counter
	protocolVersions  prometheus.CounterVec
}

// ConnectionState tracks per-connection protocol state
type ConnectionState struct {
	ClientID         string
	ProtocolVersion  string
	Authenticated    bool
	Permissions      []string
	Subscriptions    map[string]bool
	LastActivity     time.Time
	MessageCount     int64
	ErrorCount       int64
	RateLimitTokens  int
	Metadata         map[string]string
	mu               sync.RWMutex
}

// MessageValidator validates message content
type MessageValidator interface {
	Validate(message *AlgoVedaMessage) error
}

// RateLimiter implements token bucket rate limiting
type RateLimiter struct {
	capacity     int
	refillRate   int
	refillPeriod time.Duration
	tokens       map[string]*TokenBucket
	mu           sync.RWMutex
}

type TokenBucket struct {
	tokens     int
	lastRefill time.Time
	mu         sync.Mutex
}

// ProtocolHandlerConfig holds configuration for the protocol handler
type ProtocolHandlerConfig struct {
	Logger              *logger.Logger
	Metrics             *monitoring.Metrics
	SecurityManager     *security.Manager
	SupportedVersions   []string
	DefaultVersion      string
	MaxMessageSize      int64
	CompressionLevel    int
	RateLimitCapacity   int
	RateLimitRefillRate int
}

// NewProtocolHandler creates a new protocol handler
func NewProtocolHandler(config ProtocolHandlerConfig) *ProtocolHandler {
	if len(config.SupportedVersions) == 0 {
		config.SupportedVersions = []string{ProtocolVersion1_0, ProtocolVersion2_0}
	}
	
	if config.DefaultVersion == "" {
		config.DefaultVersion = DefaultProtocol
	}
	
	if config.MaxMessageSize == 0 {
		config.MaxMessageSize = 1024 * 1024 // 1MB
	}
	
	supportedVersions := make(map[string]bool)
	for _, version := range config.SupportedVersions {
		supportedVersions[version] = true
	}
	
	ph := &ProtocolHandler{
		logger:            config.Logger,
		metrics:           config.Metrics,
		securityManager:   config.SecurityManager,
		supportedVersions: supportedVersions,
		defaultVersion:    config.DefaultVersion,
		maxMessageSize:    config.MaxMessageSize,
		compressionLevel:  config.CompressionLevel,
		handlers:          make(map[AlgoVedaMessageType]MessageHandler),
		validators:        make(map[AlgoVedaMessageType]MessageValidator),
		rateLimiter: &RateLimiter{
			capacity:     config.RateLimitCapacity,
			refillRate:   config.RateLimitRefillRate,
			refillPeriod: time.Second,
			tokens:       make(map[string]*TokenBucket),
		},
		messagesProcessed: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "websocket_messages_processed_total",
			Help: "Total number of WebSocket messages processed",
		}),
		messageErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "websocket_message_errors_total",
			Help: "Total number of WebSocket message errors",
		}),
		protocolVersions: *prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "websocket_protocol_versions_total",
			Help: "Total number of connections by protocol version",
		}, []string{"version"}),
	}
	
	// Register default handlers
	ph.registerDefaultHandlers()
	ph.registerDefaultValidators()
	
	return ph
}

// RegisterHandler registers a message handler for a specific message type
func (ph *ProtocolHandler) RegisterHandler(msgType AlgoVedaMessageType, handler MessageHandler) {
	ph.handlers[msgType] = handler
}

// RegisterValidator registers a message validator for a specific message type
func (ph *ProtocolHandler) RegisterValidator(msgType AlgoVedaMessageType, validator MessageValidator) {
	ph.validators[msgType] = validator
}

// HandleMessage processes an incoming WebSocket message
func (ph *ProtocolHandler) HandleMessage(clientID string, rawMessage []byte) error {
	// Check rate limiting first
	if !ph.rateLimiter.Allow(clientID) {
		ph.messageErrors.Inc()
		return fmt.Errorf("rate limit exceeded for client %s", clientID)
	}
	
	// Parse base message
	var message AlgoVedaMessage
	if err := json.Unmarshal(rawMessage, &message); err != nil {
		ph.messageErrors.Inc()
		ph.logger.Warn("Failed to parse message",
			zap.String("client_id", clientID),
			zap.Error(err))
		return ph.sendError(clientID, 400, "Invalid JSON format", err.Error())
	}
	
	// Set client ID if not present
	if message.ClientID == "" {
		message.ClientID = clientID
	}
	
	// Set timestamp if not present
	if message.Timestamp == 0 {
		message.Timestamp = time.Now().UnixNano() / 1000000 // milliseconds
	}
	
	// Set version if not present
	if message.Version == "" {
		message.Version = ph.defaultVersion
	}
	
	// Validate protocol version
	if !ph.supportedVersions[message.Version] {
		ph.messageErrors.Inc()
		return ph.sendError(clientID, 400, "Unsupported protocol version", 
			fmt.Sprintf("Version %s not supported", message.Version))
	}
	
	// Update protocol version metrics
	ph.protocolVersions.WithLabelValues(message.Version).Inc()
	
	// Get connection state
	connState := ph.getConnectionState(clientID)
	connState.mu.Lock()
	connState.ProtocolVersion = message.Version
	connState.LastActivity = time.Now()
	connState.MessageCount++
	connState.mu.Unlock()
	
	// Validate message size
	if int64(len(rawMessage)) > ph.maxMessageSize {
		ph.messageErrors.Inc()
		return ph.sendError(clientID, 413, "Message too large", 
			fmt.Sprintf("Message size %d exceeds limit %d", len(rawMessage), ph.maxMessageSize))
	}
	
	// Validate message content
	if validator, exists := ph.validators[message.Type]; exists {
		if err := validator.Validate(&message); err != nil {
			ph.messageErrors.Inc()
			return ph.sendError(clientID, 400, "Message validation failed", err.Error())
		}
	}
	
	// Check authentication for protected message types
	if ph.requiresAuthentication(message.Type) && !connState.Authenticated {
		ph.messageErrors.Inc()
		return ph.sendError(clientID, 401, "Authentication required", 
			fmt.Sprintf("Message type %s requires authentication", message.Type))
	}
	
	// Check permissions
	if ph.requiresPermission(message.Type) {
		required := ph.getRequiredPermission(message.Type)
		if !ph.hasPermission(connState, required) {
			ph.messageErrors.Inc()
			return ph.sendError(clientID, 403, "Insufficient permissions", 
				fmt.Sprintf("Permission %s required for %s", required, message.Type))
		}
	}
	
	// Route to handler
	handler, exists := ph.handlers[message.Type]
	if !exists {
		ph.messageErrors.Inc()
		return ph.sendError(clientID, 404, "Unknown message type", 
			fmt.Sprintf("No handler for message type %s", message.Type))
	}
	
	// Process message
	if err := handler.HandleMessage(clientID, &Message{
		ID:        message.ID,
		Type:      MessageType(message.Type),
		Channel:   message.Channel,
		Data:      message.Data,
		Timestamp: time.Unix(0, message.Timestamp*1000000),
		ClientID:  clientID,
		Metadata:  message.Metadata,
	}); err != nil {
		ph.messageErrors.Inc()
		ph.logger.Error("Message handler error",
			zap.String("client_id", clientID),
			zap.String("message_type", string(message.Type)),
			zap.Error(err))
		return ph.sendError(clientID, 500, "Message processing failed", err.Error())
	}
	
	ph.messagesProcessed.Inc()
	return nil
}

// SendMessage sends a message to a client
func (ph *ProtocolHandler) SendMessage(clientID string, msgType AlgoVedaMessageType, data interface{}) error {
	return ph.SendMessageWithMetadata(clientID, msgType, data, nil)
}

// SendMessageWithMetadata sends a message with metadata to a client
func (ph *ProtocolHandler) SendMessageWithMetadata(clientID string, msgType AlgoVedaMessageType, 
	data interface{}, metadata map[string]string) error {
	
	connState := ph.getConnectionState(clientID)
	
	// Serialize data
	var jsonData json.RawMessage
	if data != nil {
		dataBytes, err := json.Marshal(data)
		if err != nil {
			return fmt.Errorf("failed to marshal message data: %w", err)
		}
		jsonData = dataBytes
	}
	
	// Create message
	message := AlgoVedaMessage{
		ID:        generateMessageID(),
		Type:      msgType,
		Version:   connState.ProtocolVersion,
		Timestamp: time.Now().UnixNano() / 1000000,
		ClientID:  clientID,
		Data:      jsonData,
		Metadata:  metadata,
	}
	
	// Serialize message
	messageBytes, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}
	
	// Send through connection manager (would need reference to connection manager)
	// For now, we'll log it
	ph.logger.Debug("Sending message to client",
		zap.String("client_id", clientID),
		zap.String("message_type", string(msgType)),
		zap.Int("message_size", len(messageBytes)))
	
	return nil
}

// Helper functions
func (ph *ProtocolHandler) getConnectionState(clientID string) *ConnectionState {
	if state, ok := ph.activeConnections.Load(clientID); ok {
		return state.(*ConnectionState)
	}
	
	state := &ConnectionState{
		ClientID:        clientID,
		ProtocolVersion: ph.defaultVersion,
		Subscriptions:   make(map[string]bool),
		LastActivity:    time.Now(),
		RateLimitTokens: ph.rateLimiter.capacity,
		Metadata:        make(map[string]string),
	}
	
	ph.activeConnections.Store(clientID, state)
	return state
}

func (ph *ProtocolHandler) sendError(clientID string, code int, message, details string) error {
	errorData := &ErrorData{
		Code:    code,
		Message: message,
		Details: details,
	}
	
	return ph.SendMessage(clientID, MsgTypeError, errorData)
}

func (ph *ProtocolHandler) requiresAuthentication(msgType AlgoVedaMessageType) bool {
	protectedTypes := map[AlgoVedaMessageType]bool{
		MsgTypeOrder:       true,
		MsgTypePosition:    true,
		MsgTypePortfolio:   true,
		MsgTypeRiskAlert:   true,
		MsgTypeRiskUpdate:  true,
	}
	
	return protectedTypes[msgType]
}

func (ph *ProtocolHandler) requiresPermission(msgType AlgoVedaMessageType) bool {
	return ph.getRequiredPermission(msgType) != ""
}

func (ph *ProtocolHandler) getRequiredPermission(msgType AlgoVedaMessageType) string {
	permissions := map[AlgoVedaMessageType]string{
		MsgTypeOrder:      "trading",
		MsgTypePosition:   "portfolio_read",
		MsgTypePortfolio:  "portfolio_read",
		MsgTypeRiskAlert:  "risk_read",
		MsgTypeRiskUpdate: "risk_read",
	}
	
	return permissions[msgType]
}

func (ph *ProtocolHandler) hasPermission(state *ConnectionState, permission string) bool {
	state.mu.RLock()
	defer state.mu.RUnlock()
	
	for _, perm := range state.Permissions {
		if perm == permission || perm == "admin" {
			return true
		}
	}
	
	return false
}

func (ph *ProtocolHandler) registerDefaultHandlers() {
	// Authentication handler
	ph.RegisterHandler(MsgTypeAuth, &AuthHandler{
		protocolHandler: ph,
		securityManager: ph.securityManager,
		logger:          ph.logger,
	})
	
	// Subscription handler
	ph.RegisterHandler(MsgTypeSubscribe, &SubscriptionHandler{
		protocolHandler: ph,
		logger:          ph.logger,
	})
	
	// Heartbeat handler
	ph.RegisterHandler(MsgTypeHeartbeat, &HeartbeatHandler{
		protocolHandler: ph,
		logger:          ph.logger,
	})
}

func (ph *ProtocolHandler) registerDefaultValidators() {
	// Basic message structure validator
	ph.RegisterValidator(MsgTypeAuth, &AuthValidator{})
	ph.RegisterValidator(MsgTypeSubscribe, &SubscriptionValidator{})
	ph.RegisterValidator(MsgTypeOrder, &OrderValidator{})
}

// Rate limiter implementation
func (rl *RateLimiter) Allow(clientID string) bool {
	rl.mu.Lock()
	bucket, exists := rl.tokens[clientID]
	if !exists {
		bucket = &TokenBucket{
			tokens:     rl.capacity,
			lastRefill: time.Now(),
		}
		rl.tokens[clientID] = bucket
	}
	rl.mu.Unlock()
	
	bucket.mu.Lock()
	defer bucket.mu.Unlock()
	
	// Refill tokens
	now := time.Now()
	elapsed := now.Sub(bucket.lastRefill)
	if elapsed >= rl.refillPeriod {
		periods := int(elapsed / rl.refillPeriod)
		tokensToAdd := periods * rl.refillRate
		bucket.tokens = min(bucket.tokens + tokensToAdd, rl.capacity)
		bucket.lastRefill = bucket.lastRefill.Add(time.Duration(periods) * rl.refillPeriod)
	}
	
	// Check if request can be allowed
	if bucket.tokens > 0 {
		bucket.tokens--
		return true
	}
	
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func generateMessageID() string {
	return fmt.Sprintf("msg_%d_%d", time.Now().UnixNano(), 
		rand.Intn(1000000))
}

// Message handlers implementation
type AuthHandler struct {
	protocolHandler *ProtocolHandler
	securityManager *security.Manager
	logger          *logger.Logger
}

func (h *AuthHandler) HandleMessage(clientID string, message *Message) error {
	var authReq AuthRequest
	if err := json.Unmarshal(message.Data, &authReq); err != nil {
		return fmt.Errorf("invalid auth request format: %w", err)
	}
	
	// Validate token with security manager
	userInfo, err := h.securityManager.ValidateToken(authReq.Token)
	if err != nil {
		response := AuthResponse{
			Success: false,
		}
		return h.protocolHandler.SendMessage(clientID, MsgTypeAuthResponse, response)
	}
	
	// Update connection state
	state := h.protocolHandler.getConnectionState(clientID)
	state.mu.Lock()
	state.Authenticated = true
	state.Permissions = userInfo.Permissions
	state.Metadata = authReq.Metadata
	state.mu.Unlock()
	
	// Send success response
	response := AuthResponse{
		Success:     true,
		SessionID:   generateSessionID(),
		Permissions: userInfo.Permissions,
		ExpiresAt:   time.Now().Add(24 * time.Hour).Unix(),
	}
	
	return h.protocolHandler.SendMessage(clientID, MsgTypeAuthResponse, response)
}

func (h *AuthHandler) GetSupportedTypes() []MessageType {
	return []MessageType{MessageType(MsgTypeAuth)}
}

func generateSessionID() string {
	return fmt.Sprintf("sess_%d_%d", time.Now().UnixNano(), 
		rand.Intn(1000000))
}

// Subscription handler and other handlers would be implemented similarly...

type SubscriptionHandler struct {
	protocolHandler *ProtocolHandler
	logger          *logger.Logger
}

func (h *SubscriptionHandler) HandleMessage(clientID string, message *Message) error {
	var subReq SubscribeRequest
	if err := json.Unmarshal(message.Data, &subReq); err != nil {
		return fmt.Errorf("invalid subscription request: %w", err)
	}
	
	state := h.protocolHandler.getConnectionState(clientID)
	state.mu.Lock()
	for _, channel := range subReq.Channels {
		state.Subscriptions[channel] = true
	}
	state.mu.Unlock()
	
	// Send response for each channel
	for _, channel := range subReq.Channels {
		response := SubscribeResponse{
			Success: true,
			Channel: channel,
			Symbols: subReq.Symbols,
			Status:  "subscribed",
		}
		
		if err := h.protocolHandler.SendMessage(clientID, MsgTypeSubResponse, response); err != nil {
			h.logger.Error("Failed to send subscription response",
				zap.String("client_id", clientID),
				zap.String("channel", channel),
				zap.Error(err))
		}
	}
	
	return nil
}

func (h *SubscriptionHandler) GetSupportedTypes() []MessageType {
	return []MessageType{MessageType(MsgTypeSubscribe), MessageType(MsgTypeUnsubscribe)}
}

// Heartbeat handler
type HeartbeatHandler struct {
	protocolHandler *ProtocolHandler
	logger          *logger.Logger
}

func (h *HeartbeatHandler) HandleMessage(clientID string, message *Message) error {
	// Simply echo back a heartbeat response
	return h.protocolHandler.SendMessage(clientID, MsgTypeHeartbeat, map[string]interface{}{
		"server_time": time.Now().UnixNano() / 1000000,
		"client_time": message.Timestamp.UnixNano() / 1000000,
	})
}

func (h *HeartbeatHandler) GetSupportedTypes() []MessageType {
	return []MessageType{MessageType(MsgTypeHeartbeat)}
}

// Message validators
type AuthValidator struct{}

func (v *AuthValidator) Validate(message *AlgoVedaMessage) error {
	var authReq AuthRequest
	if err := json.Unmarshal(message.Data, &authReq); err != nil {
		return fmt.Errorf("invalid auth request format: %w", err)
	}
	
	if authReq.Token == "" {
		return fmt.Errorf("token is required")
	}
	
	return nil
}

type SubscriptionValidator struct{}

func (v *SubscriptionValidator) Validate(message *AlgoVedaMessage) error {
	var subReq SubscribeRequest
	if err := json.Unmarshal(message.Data, &subReq); err != nil {
		return fmt.Errorf("invalid subscription request format: %w", err)
	}
	
	if len(subReq.Channels) == 0 {
		return fmt.Errorf("at least one channel is required")
	}
	
	return nil
}

type OrderValidator struct{}

func (v *OrderValidator) Validate(message *AlgoVedaMessage) error {
	var orderReq OrderRequest
	if err := json.Unmarshal(message.Data, &orderReq); err != nil {
		return fmt.Errorf("invalid order request format: %w", err)
	}
	
	if orderReq.Symbol == "" {
		return fmt.Errorf("symbol is required")
	}
	
	if orderReq.Quantity <= 0 {
		return fmt.Errorf("quantity must be positive")
	}
	
	validSides := map[string]bool{"BUY": true, "SELL": true}
	if !validSides[strings.ToUpper(orderReq.Side)] {
		return fmt.Errorf("invalid side: %s", orderReq.Side)
	}
	
	return nil
}
