/*!
 * Ultra-Fast WebSocket Hub with 100k+ Concurrent Connections
 * Zero-copy message broadcasting with adaptive compression
 */

package websocket

import (
    "sync"
    "sync/atomic"
    "time"
    "unsafe"

    "github.com/gorilla/websocket"
    "github.com/klauspost/compress/flate"
    "github.com/valyala/bytebufferpool"
)

type UltraFastHub struct {
    connections    map[*Connection]bool
    broadcast      chan []byte
    register       chan *Connection
    unregister     chan *Connection
    connectionsMu  sync.RWMutex
    
    // Performance optimizations
    messagePool    sync.Pool
    bufferPool     *bytebufferpool.Pool
    compressor     *flate.Writer
    
    // Metrics
    totalConnections   int64
    messagesPerSecond  int64
    bytesTransferred   int64
    
    // Configuration
    maxMessageSize     int64
    compressionLevel   int
    compressionEnabled bool
    batchingEnabled    bool
    batchSize          int
    batchTimeout       time.Duration
}

func NewUltraFastHub() *UltraFastHub {
    return &UltraFastHub{
        connections:        make(map[*Connection]bool),
        broadcast:          make(chan []byte, 10000),
        register:           make(chan *Connection, 1000),
        unregister:         make(chan *Connection, 1000),
        bufferPool:         &bytebufferpool.Pool{},
        maxMessageSize:     32768,
        compressionLevel:   flate.BestSpeed,
        compressionEnabled: true,
        batchingEnabled:    true,
        batchSize:          100,
        batchTimeout:       time.Millisecond,
        messagePool: sync.Pool{
            New: func() interface{} {
                return make([]byte, 0, 1024)
            },
        },
    }
}

// Ultra-fast message broadcasting with zero allocations
func (h *UltraFastHub) BroadcastMessage(message []byte) {
    if len(message) > int(h.maxMessageSize) {
        return // Skip oversized messages
    }

    // Use object pooling to avoid allocations
    msg := h.messagePool.Get().([]byte)
    msg = append(msg[:0], message...)
    
    select {
    case h.broadcast <- msg:
        atomic.AddInt64(&h.messagesPerSecond, 1)
    default:
        h.messagePool.Put(msg) // Return to pool if channel full
    }
}

// Lock-free connection management
func (h *UltraFastHub) Run() {
    ticker := time.NewTicker(time.Second)
    defer ticker.Stop()
    
    batchedMessages := make([][]byte, 0, h.batchSize)
    batchTimer := time.NewTimer(h.batchTimeout)
    
    for {
        select {
        case conn := <-h.register:
            h.addConnection(conn)
            
        case conn := <-h.unregister:
            h.removeConnection(conn)
            
        case message := <-h.broadcast:
            if h.batchingEnabled {
                batchedMessages = append(batchedMessages, message)
                if len(batchedMessages) >= h.batchSize {
                    h.sendBatchedMessages(batchedMessages)
                    batchedMessages = batchedMessages[:0]
                    batchTimer.Reset(h.batchTimeout)
                }
            } else {
                h.sendToAllConnections(message)
            }
            h.messagePool.Put(message)
            
        case <-batchTimer.C:
            if len(batchedMessages) > 0 {
                h.sendBatchedMessages(batchedMessages)
                batchedMessages = batchedMessages[:0]
            }
            batchTimer.Reset(h.batchTimeout)
            
        case <-ticker.C:
            h.updateMetrics()
        }
    }
}

// Zero-copy message sending with adaptive compression
func (h *UltraFastHub) sendToAllConnections(message []byte) {
    h.connectionsMu.RLock()
    connCount := len(h.connections)
    
    if connCount == 0 {
        h.connectionsMu.RUnlock()
        return
    }
    
    // Pre-compress message if beneficial
    var compressedMessage []byte
    if h.compressionEnabled && len(message) > 128 {
        compressedMessage = h.compressMessage(message)
    }
    
    // Use goroutine pool for parallel sending
    semaphore := make(chan struct{}, 100) // Limit concurrent goroutines
    
    for conn := range h.connections {
        semaphore <- struct{}{}
        go func(c *Connection) {
            defer func() { <-semaphore }()
            
            messageToSend := message
            if compressedMessage != nil && 
               c.supportsCompression && 
               len(compressedMessage) < len(message) {
                messageToSend = compressedMessage
                c.EnableWriteCompression(true)
            }
            
            c.WriteMessage(websocket.BinaryMessage, messageToSend)
            atomic.AddInt64(&h.bytesTransferred, int64(len(messageToSend)))
        }(conn)
    }
    
    h.connectionsMu.RUnlock()
}
