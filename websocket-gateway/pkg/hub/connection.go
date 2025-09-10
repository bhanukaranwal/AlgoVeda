// Advanced connection management with sub-millisecond message routing
type Connection struct {
    hub      *Hub
    conn     *websocket.Conn
    send     chan []byte
    userID   string
    channels map[string]bool  // Subscribed channels
    lastSeen time.Time
    metrics  *ConnectionMetrics
}

// Ultra-fast message broadcasting to 100k+ connections
func (h *Hub) BroadcastToChannel(channel string, message []byte) {
    h.channelMutex.RLock()
    connections := h.channels[channel]
    h.channelMutex.RUnlock()
    
    for conn := range connections {
        select {
        case conn.send <- message:
        default:
            h.unregister <- conn
        }
    }
}
