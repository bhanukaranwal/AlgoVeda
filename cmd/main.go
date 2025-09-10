/*
 * WebSocket Gateway Main Application
 * High-performance WebSocket server for real-time trading data
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"algoveda/internal/config"
	"algoveda/internal/websocket"
	"algoveda/internal/monitoring"
	"algoveda/internal/security"
	"algoveda/internal/loadbalancer"
	"algoveda/pkg/logger"
)

var (
	configFile = flag.String("config", "config/production.yaml", "Configuration file path")
	logLevel   = flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	cpuProfile = flag.String("cpuprofile", "", "Enable CPU profiling")
	memProfile = flag.String("memprofile", "", "Enable memory profiling")
)

// Application holds the main application components
type Application struct {
	config          *config.Config
	logger          *logger.Logger
	wsManager       *websocket.Manager
	httpServer      *http.Server
	metricsServer   *http.Server
	loadBalancer    *loadbalancer.LoadBalancer
	securityManager *security.Manager
	monitoring      *monitoring.Monitor
}

func main() {
	flag.Parse()

	// Initialize application
	app, err := NewApplication()
	if err != nil {
		log.Fatalf("Failed to initialize application: %v", err)
	}

	// Start application
	if err := app.Start(); err != nil {
		log.Fatalf("Failed to start application: %v", err)
	}

	// Wait for shutdown signal
	app.WaitForShutdown()
}

// NewApplication creates a new application instance
func NewApplication() (*Application, error) {
	// Load configuration
	cfg, err := config.Load(*configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Initialize logger
	log := logger.New(logger.Config{
		Level:  *logLevel,
		Format: cfg.Logging.Format,
		Output: cfg.Logging.Output,
	})

	log.Info("Initializing AlgoVeda WebSocket Gateway",
		"version", cfg.App.Version,
		"build", cfg.App.BuildHash,
		"config", *configFile)

	// Initialize security manager
	securityManager, err := security.NewManager(cfg.Security)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize security manager: %w", err)
	}

	// Initialize monitoring
	monitor := monitoring.NewMonitor(cfg.Monitoring)

	// Initialize WebSocket manager
	wsManager := websocket.NewManager(websocket.Config{
		MaxConnections:     cfg.WebSocket.MaxConnections,
		ReadBufferSize:     cfg.WebSocket.ReadBufferSize,
		WriteBufferSize:    cfg.WebSocket.WriteBufferSize,
		WriteTimeout:       cfg.WebSocket.WriteTimeout,
		ReadTimeout:        cfg.WebSocket.ReadTimeout,
		PingInterval:       cfg.WebSocket.PingInterval,
		MaxMessageSize:     cfg.WebSocket.MaxMessageSize,
		CompressionEnabled: cfg.WebSocket.CompressionEnabled,
		Logger:             log,
		Monitor:            monitor,
		SecurityManager:    securityManager,
	})

	// Initialize load balancer
	loadBalancer := loadbalancer.New(loadbalancer.Config{
		Strategy:           cfg.LoadBalancer.Strategy,
		HealthCheckEnabled: cfg.LoadBalancer.HealthCheckEnabled,
		HealthCheckInterval: cfg.LoadBalancer.HealthCheckInterval,
		Backends:           cfg.LoadBalancer.Backends,
		Logger:             log,
	})

	return &Application{
		config:          cfg,
		logger:          log,
		wsManager:       wsManager,
		loadBalancer:    loadBalancer,
		securityManager: securityManager,
		monitoring:      monitor,
	}, nil
}

// Start starts all application components
func (app *Application) Start() error {
	ctx := context.Background()

	app.logger.Info("Starting AlgoVeda WebSocket Gateway")

	// Start monitoring first
	if err := app.monitoring.Start(ctx); err != nil {
		return fmt.Errorf("failed to start monitoring: %w", err)
	}

	// Start security manager
	if err := app.securityManager.Start(ctx); err != nil {
		return fmt.Errorf("failed to start security manager: %w", err)
	}

	// Start load balancer
	if err := app.loadBalancer.Start(ctx); err != nil {
		return fmt.Errorf("failed to start load balancer: %w", err)
	}

	// Start WebSocket manager
	if err := app.wsManager.Start(ctx); err != nil {
		return fmt.Errorf("failed to start WebSocket manager: %w", err)
	}

	// Setup HTTP routes
	if err := app.setupRoutes(); err != nil {
		return fmt.Errorf("failed to setup routes: %w", err)
	}

	// Start HTTP server
	if err := app.startHTTPServer(); err != nil {
		return fmt.Errorf("failed to start HTTP server: %w", err)
	}

	// Start metrics server
	if err := app.startMetricsServer(); err != nil {
		return fmt.Errorf("failed to start metrics server: %w", err)
	}

	app.logger.Info("AlgoVeda WebSocket Gateway started successfully",
		"http_port", app.config.Server.Port,
		"metrics_port", app.config.Monitoring.MetricsPort,
		"max_connections", app.config.WebSocket.MaxConnections)

	return nil
}

// setupRoutes configures HTTP routes
func (app *Application) setupRoutes() error {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(app.loggingMiddleware())
	router.Use(app.securityMiddleware())
	router.Use(app.rateLimitMiddleware())
	router.Use(app.corsMiddleware())

	// Health check endpoints
	router.GET("/health", app.healthCheckHandler)
	router.GET("/ready", app.readinessHandler)
	router.GET("/live", app.livenessHandler)

	// WebSocket endpoint
	router.GET("/ws", app.websocketHandler)

	// API endpoints
	api := router.Group("/api/v1")
	{
		api.GET("/stats", app.statsHandler)
		api.GET("/connections", app.connectionsHandler)
		api.POST("/broadcast", app.broadcastHandler)
		api.GET("/subscriptions", app.subscriptionsHandler)
		api.POST("/subscribe", app.subscribeHandler)
		api.POST("/unsubscribe", app.unsubscribeHandler)
	}

	// Admin endpoints (protected)
	admin := router.Group("/admin")
	admin.Use(app.adminAuthMiddleware())
	{
		admin.GET("/metrics", gin.WrapH(promhttp.Handler()))
		admin.GET("/config", app.configHandler)
		admin.POST("/reload", app.reloadConfigHandler)
		admin.POST("/shutdown", app.shutdownHandler)
	}

	app.httpServer = &http.Server{
		Addr:           fmt.Sprintf(":%d", app.config.Server.Port),
		Handler:        router,
		ReadTimeout:    app.config.Server.ReadTimeout,
		WriteTimeout:   app.config.Server.WriteTimeout,
		IdleTimeout:    app.config.Server.IdleTimeout,
		MaxHeaderBytes: app.config.Server.MaxHeaderBytes,
	}

	return nil
}

// startHTTPServer starts the main HTTP server
func (app *Application) startHTTPServer() error {
	go func() {
		app.logger.Info("Starting HTTP server", "port", app.config.Server.Port)
		
		if app.config.Server.TLS.Enabled {
			if err := app.httpServer.ListenAndServeTLS(
				app.config.Server.TLS.CertFile,
				app.config.Server.TLS.KeyFile,
			); err != nil && err != http.ErrServerClosed {
				app.logger.Error("HTTP server error", "error", err)
			}
		} else {
			if err := app.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				app.logger.Error("HTTP server error", "error", err)
			}
		}
	}()

	return nil
}

// startMetricsServer starts the Prometheus metrics server
func (app *Application) startMetricsServer() error {
	metricsRouter := gin.New()
	metricsRouter.Use(gin.Recovery())
	metricsRouter.GET("/metrics", gin.WrapH(promhttp.Handler()))
	metricsRouter.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	app.metricsServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", app.config.Monitoring.MetricsPort),
		Handler: metricsRouter,
	}

	go func() {
		app.logger.Info("Starting metrics server", "port", app.config.Monitoring.MetricsPort)
		if err := app.metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			app.logger.Error("Metrics server error", "error", err)
		}
	}()

	return nil
}

// WebSocket handler
func (app *Application) websocketHandler(c *gin.Context) {
	// Upgrade HTTP connection to WebSocket
	upgrader := websocket.Upgrader{
		ReadBufferSize:  app.config.WebSocket.ReadBufferSize,
		WriteBufferSize: app.config.WebSocket.WriteBufferSize,
		CheckOrigin: func(r *http.Request) bool {
			return app.securityManager.CheckOrigin(r)
		},
		EnableCompression: app.config.WebSocket.CompressionEnabled,
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		app.logger.Error("WebSocket upgrade failed", "error", err, "remote_addr", c.ClientIP())
		return
	}

	// Create client context
	clientID := app.generateClientID()
	client := &websocket.Client{
		ID:         clientID,
		Connection: conn,
		RemoteAddr: c.ClientIP(),
		UserAgent:  c.Request.UserAgent(),
		Headers:    c.Request.Header,
		ConnectedAt: time.Now(),
	}

	// Authenticate client
	if err := app.securityManager.AuthenticateClient(client, c.Request); err != nil {
		app.logger.Warn("Client authentication failed", 
			"client_id", clientID, 
			"remote_addr", c.ClientIP(), 
			"error", err)
		conn.Close()
		return
	}

	// Register client with WebSocket manager
	if err := app.wsManager.RegisterClient(client); err != nil {
		app.logger.Error("Failed to register client", 
			"client_id", clientID, 
			"error", err)
		conn.Close()
		return
	}

	app.logger.Info("WebSocket client connected", 
		"client_id", clientID, 
		"remote_addr", c.ClientIP())

	// Update metrics
	app.monitoring.IncrementCounter("websocket_connections_total")
	app.monitoring.SetGauge("websocket_active_connections", 
		float64(app.wsManager.GetConnectionCount()))
}

// Middleware functions
func (app *Application) loggingMiddleware() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
			param.ClientIP,
			param.TimeStamp.Format(time.RFC3339),
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.Request.UserAgent(),
			param.ErrorMessage,
		)
	})
}

func (app *Application) securityMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Security headers
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Referrer-Policy", "strict-origin-when-cross-origin")

		// Rate limiting check
		if !app.securityManager.CheckRateLimit(c.ClientIP()) {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "Rate limit exceeded",
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

func (app *Application) rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !app.securityManager.CheckRateLimit(c.ClientIP()) {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "Rate limit exceeded",
			})
			c.Abort()
			return
		}
		c.Next()
	}
}

func (app *Application) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		if app.securityManager.IsAllowedOrigin(origin) {
			c.Header("Access-Control-Allow-Origin", origin)
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			c.Header("Access-Control-Allow-Headers", "Accept, Authorization, Content-Type, X-CSRF-Token")
			c.Header("Access-Control-Allow-Credentials", "true")
		}

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func (app *Application) adminAuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := c.GetHeader("Authorization")
		if !app.securityManager.ValidateAdminToken(token) {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "Unauthorized",
			})
			c.Abort()
			return
		}
		c.Next()
	}
}

// HTTP handlers
func (app *Application) healthCheckHandler(c *gin.Context) {
	status := app.getHealthStatus()
	httpStatus := http.StatusOK
	if status["status"] != "healthy" {
		httpStatus = http.StatusServiceUnavailable
	}
	c.JSON(httpStatus, status)
}

func (app *Application) readinessHandler(c *gin.Context) {
	ready := app.wsManager.IsReady() && 
			app.loadBalancer.IsReady() && 
			app.securityManager.IsReady()
	
	if ready {
		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	} else {
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "not ready"})
	}
}

func (app *Application) livenessHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "alive"})
}

func (app *Application) statsHandler(c *gin.Context) {
	stats := app.wsManager.GetStats()
	c.JSON(http.StatusOK, stats)
}

func (app *Application) connectionsHandler(c *gin.Context) {
	connections := app.wsManager.GetConnections()
	c.JSON(http.StatusOK, gin.H{
		"connections": connections,
		"count":       len(connections),
	})
}

func (app *Application) broadcastHandler(c *gin.Context) {
	var request struct {
		Message string `json:"message" binding:"required"`
		Channel string `json:"channel"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	count := app.wsManager.Broadcast(request.Channel, []byte(request.Message))
	c.JSON(http.StatusOK, gin.H{
		"message": "Broadcast sent",
		"recipients": count,
	})
}

func (app *Application) subscriptionsHandler(c *gin.Context) {
	subscriptions := app.wsManager.GetSubscriptions()
	c.JSON(http.StatusOK, subscriptions)
}

func (app *Application) subscribeHandler(c *gin.Context) {
	var request struct {
		ClientID string `json:"client_id" binding:"required"`
		Channel  string `json:"channel" binding:"required"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := app.wsManager.Subscribe(request.ClientID, request.Channel); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Subscribed successfully"})
}

func (app *Application) unsubscribeHandler(c *gin.Context) {
	var request struct {
		ClientID string `json:"client_id" binding:"required"`
		Channel  string `json:"channel" binding:"required"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := app.wsManager.Unsubscribe(request.ClientID, request.Channel); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Unsubscribed successfully"})
}

func (app *Application) configHandler(c *gin.Context) {
	c.JSON(http.StatusOK, app.config)
}

func (app *Application) reloadConfigHandler(c *gin.Context) {
	newConfig, err := config.Load(*configFile)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to reload config: " + err.Error(),
		})
		return
	}

	app.config = newConfig
	c.JSON(http.StatusOK, gin.H{"message": "Configuration reloaded"})
}

func (app *Application) shutdownHandler(c *gin.Context) {
	go func() {
		time.Sleep(100 * time.Millisecond)
		app.Shutdown()
	}()
	c.JSON(http.StatusOK, gin.H{"message": "Shutdown initiated"})
}

// Helper functions
func (app *Application) generateClientID() string {
	return fmt.Sprintf("client_%d_%d", time.Now().UnixNano(), 
		app.wsManager.GetConnectionCount()+1)
}

func (app *Application) getHealthStatus() map[string]interface{} {
	status := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC(),
		"version":   app.config.App.Version,
		"uptime":    time.Since(app.monitoring.GetStartTime()).String(),
	}

	// Check component health
	checks := map[string]bool{
		"websocket_manager": app.wsManager.IsHealthy(),
		"load_balancer":     app.loadBalancer.IsHealthy(),
		"security_manager":  app.securityManager.IsHealthy(),
		"monitoring":        app.monitoring.IsHealthy(),
	}

	allHealthy := true
	for component, healthy := range checks {
		status[component] = map[string]interface{}{
			"healthy": healthy,
		}
		if !healthy {
			allHealthy = false
		}
	}

	if !allHealthy {
		status["status"] = "unhealthy"
	}

	return status
}

// WaitForShutdown waits for shutdown signals
func (app *Application) WaitForShutdown() {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	sig := <-quit
	app.logger.Info("Received shutdown signal", "signal", sig.String())

	app.Shutdown()
}

// Shutdown gracefully shuts down the application
func (app *Application) Shutdown() {
	app.logger.Info("Shutting down AlgoVeda WebSocket Gateway")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Shutdown HTTP servers
	if app.httpServer != nil {
		if err := app.httpServer.Shutdown(ctx); err != nil {
			app.logger.Error("HTTP server shutdown error", "error", err)
		}
	}

	if app.metricsServer != nil {
		if err := app.metricsServer.Shutdown(ctx); err != nil {
			app.logger.Error("Metrics server shutdown error", "error", err)
		}
	}

	// Shutdown components
	if app.wsManager != nil {
		app.wsManager.Shutdown(ctx)
	}

	if app.loadBalancer != nil {
		app.loadBalancer.Shutdown(ctx)
	}

	if app.securityManager != nil {
		app.securityManager.Shutdown(ctx)
	}

	if app.monitoring != nil {
		app.monitoring.Shutdown(ctx)
	}

	app.logger.Info("AlgoVeda WebSocket Gateway shutdown complete")
}
