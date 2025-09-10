package main

import (
    "context"
    "flag"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"

    "github.com/algoveda/websocket-gateway/internal/config"
    "github.com/algoveda/websocket-gateway/internal/monitoring"
    "github.com/algoveda/websocket-gateway/internal/networking"
    "github.com/algoveda/websocket-gateway/internal/routing"
    "github.com/algoveda/websocket-gateway/internal/security"
    "github.com/algoveda/websocket-gateway/internal/websocket"
    
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "go.uber.org/zap"
)

type Server struct {
    config          *config.Config
    logger          *zap.Logger
    wsManager       *websocket.ConnectionManager
    router          *routing.MessageRouter
    securityManager *security.Manager
    networkManager  *networking.Manager
    monitor         *monitoring.Monitor
    httpServer      *http.Server
    shutdownOnce    sync.Once
    shutdownCh      chan struct{}
}

func main() {
    var configPath = flag.String("config", "configs/production.yaml", "Path to configuration file")
    flag.Parse()

    // Load configuration
    cfg, err := config.Load(*configPath)
    if err != nil {
        log.Fatalf("Failed to load configuration: %v", err)
    }

    // Initialize logger
    logger, err := initLogger(cfg.Logging.Level, cfg.Logging.Format)
    if err != nil {
        log.Fatalf("Failed to initialize logger: %v", err)
    }
    defer logger.Sync()

    // Create server
    server, err := NewServer(cfg, logger)
    if err != nil {
        logger.Fatal("Failed to create server", zap.Error(err))
    }

    // Start server
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    if err := server.Start(ctx); err != nil {
        logger.Fatal("Failed to start server", zap.Error(err))
    }

    // Wait for shutdown signal
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    
    select {
    case sig := <-sigCh:
        logger.Info("Received shutdown signal", zap.String("signal", sig.String()))
    case <-server.shutdownCh:
        logger.Info("Server shutdown requested")
    }

    // Graceful shutdown
    shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer shutdownCancel()
    
    if err := server.Shutdown(shutdownCtx); err != nil {
        logger.Error("Error during shutdown", zap.Error(err))
    }

    logger.Info("Server shutdown complete")
}

func NewServer(cfg *config.Config, logger *zap.Logger) (*Server, error) {
    // Initialize monitoring
    monitor := monitoring.NewMonitor(cfg.Monitoring, logger)

    // Initialize security manager
    securityManager, err := security.NewManager(cfg.Security, logger)
    if err != nil {
        return nil, fmt.Errorf("failed to create security manager: %w", err)
    }

    // Initialize network manager
    networkManager, err := networking.NewManager(cfg.Networking, logger, monitor)
    if err != nil {
        return nil, fmt.Errorf("failed to create network manager: %w", err)
    }

    // Initialize WebSocket connection manager
    wsManager, err := websocket.NewConnectionManager(cfg.WebSocket, logger, monitor, securityManager)
    if err != nil {
        return nil, fmt.Errorf("failed to create WebSocket manager: %w", err)
    }

    // Initialize message router
    router, err := routing.NewMessageRouter(cfg.Routing, logger, monitor)
    if err != nil {
        return nil, fmt.Errorf("failed to create message router: %w", err)
    }

    // Setup HTTP server
    mux := http.NewServeMux()
    
    // WebSocket endpoint
    mux.HandleFunc("/ws", wsManager.HandleWebSocket)
    
    // Health check endpoint
    mux.HandleFunc("/health", handleHealthCheck)
    
    // Metrics endpoint
    mux.Handle("/metrics", promhttp.Handler())
    
    // Admin endpoints
    mux.HandleFunc("/admin/connections", wsManager.HandleAdminConnections)
    mux.HandleFunc("/admin/stats", monitor.HandleAdminStats)

    httpServer := &http.Server{
        Addr:         fmt.Sprintf(":%d", cfg.Server.Port),
        Handler:      mux,
        ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
        WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
        IdleTimeout:  time.Duration(cfg.Server.IdleTimeout) * time.Second,
    }

    return &Server{
        config:          cfg,
        logger:          logger,
        wsManager:       wsManager,
        router:          router,
        securityManager: securityManager,
        networkManager:  networkManager,
        monitor:         monitor,
        httpServer:      httpServer,
        shutdownCh:      make(chan struct{}),
    }, nil
}

func (s *Server) Start(ctx context.Context) error {
    s.logger.Info("Starting AlgoVeda WebSocket Gateway", 
        zap.String("version", s.config.Server.Version),
        zap.Int("port", s.config.Server.Port))

    // Start monitoring
    if err := s.monitor.Start(ctx); err != nil {
        return fmt.Errorf("failed to start monitoring: %w", err)
    }

    // Start network manager
    if err := s.networkManager.Start(ctx); err != nil {
        return fmt.Errorf("failed to start network manager: %w", err)
    }

    // Start WebSocket manager
    if err := s.wsManager.Start(ctx); err != nil {
        return fmt.Errorf("failed to start WebSocket manager: %w", err)
    }

    // Start message router
    if err := s.router.Start(ctx); err != nil {
        return fmt.Errorf("failed to start message router: %w", err)
    }

    // Connect WebSocket manager to router
    s.wsManager.SetMessageHandler(s.router.RouteMessage)
    s.router.SetBroadcastHandler(s.wsManager.Broadcast)

    // Start HTTP server
    go func() {
        if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            s.logger.Error("HTTP server error", zap.Error(err))
            close(s.shutdownCh)
        }
    }()

    s.logger.Info("Server started successfully")
    return nil
}

func (s *Server) Shutdown(ctx context.Context) error {
    s.shutdownOnce.Do(func() {
        s.logger.Info("Starting graceful shutdown")

        // Shutdown HTTP server
        if err := s.httpServer.Shutdown(ctx); err != nil {
            s.logger.Error("Error shutting down HTTP server", zap.Error(err))
        }

        // Shutdown WebSocket manager
        if err := s.wsManager.Shutdown(ctx); err != nil {
            s.logger.Error("Error shutting down WebSocket manager", zap.Error(err))
        }

        // Shutdown message router
        if err := s.router.Shutdown(ctx); err != nil {
            s.logger.Error("Error shutting down message router", zap.Error(err))
        }

        // Shutdown network manager
        if err := s.networkManager.Shutdown(ctx); err != nil {
            s.logger.Error("Error shutting down network manager", zap.Error(err))
        }

        // Shutdown monitoring
        if err := s.monitor.Shutdown(ctx); err != nil {
            s.logger.Error("Error shutting down monitoring", zap.Error(err))
        }
    })

    return nil
}

func initLogger(level, format string) (*zap.Logger, error) {
    var config zap.Config

    if format == "json" {
        config = zap.NewProductionConfig()
    } else {
        config = zap.NewDevelopmentConfig()
    }

    // Set log level
    switch level {
    case "debug":
        config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
    case "info":
        config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
    case "warn":
        config.Level = zap.NewAtomicLevelAt(zap.WarnLevel)
    case "error":
        config.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
    default:
        config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
    }

    return config.Build()
}

func handleHealthCheck(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    w.Write([]byte(`{"status":"healthy","timestamp":"` + time.Now().UTC().Format(time.RFC3339) + `"}`))
}

// Additional helper functions and middleware would be implemented here...
