package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/gorilla/mux"
    "github.com/gorilla/websocket"
    "algoveda/pkg/hub"
    "algoveda/pkg/auth"
    "algoveda/pkg/metrics"
)

// High-performance WebSocket gateway with 100k+ concurrent connections
func main() {
    // Initialize connection hub with advanced features
    connectionHub := hub.NewHub(&hub.Config{
        MaxConnections:    100000,
        ReadBufferSize:   4096,
        WriteBufferSize:  4096,
        HandshakeTimeout: 10 * time.Second,
        EnableCompression: true,
        MaxMessageSize:   32768,
    })

    // Start background services
    go connectionHub.Run()
    go metrics.StartMetricsServer(":9091")
    
    // Setup routes with middleware
    router := mux.NewRouter()
    router.HandleFunc("/ws", auth.ValidateJWT(handleWebSocket(connectionHub)))
    router.HandleFunc("/health", healthCheck)
    
    server := &http.Server{
        Addr:         ":8080",
        Handler:      router,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }

    // Graceful shutdown
    go func() {
        if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Server failed: %v", err)
        }
    }()

    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    server.Shutdown(ctx)
}
