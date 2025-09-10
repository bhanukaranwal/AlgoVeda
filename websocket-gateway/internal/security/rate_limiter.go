/*!
 * Ultra-High Performance Rate Limiter for AlgoVeda WebSocket Gateway
 * Token bucket implementation with distributed coordination
 */

package security

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-redis/redis/v8"
	"golang.org/x/time/rate"
)

// RateLimiter provides high-performance rate limiting with multiple algorithms
type RateLimiter struct {
	// Token bucket limiter for burst handling
	tokenBucket *rate.Limiter
	
	// Sliding window counters for precise limiting
	slidingWindow *SlidingWindowLimiter
	
	// Distributed coordination via Redis
	redisClient *redis.Client
	
	// Configuration
	config *RateLimiterConfig
	
	// Metrics
	metrics *RateLimiterMetrics
	
	// Client-specific limiters
	clientLimiters sync.Map // map[string]*ClientLimiter
}

type RateLimiterConfig struct {
	GlobalRateLimit    int           `yaml:"global_rate_limit"`    // Requests per second globally
	ClientRateLimit    int           `yaml:"client_rate_limit"`    // Requests per second per client
	BurstSize          int           `yaml:"burst_size"`           // Maximum burst size
	WindowSize         time.Duration `yaml:"window_size"`          // Sliding window size
	CleanupInterval    time.Duration `yaml:"cleanup_interval"`     // Cleanup interval for old entries
	RedisKeyPrefix     string        `yaml:"redis_key_prefix"`     // Redis key prefix
	DistributedMode    bool          `yaml:"distributed_mode"`     // Enable distributed coordination
	BackoffMultiplier  float64       `yaml:"backoff_multiplier"`   // Exponential backoff multiplier
	MaxBackoffDuration time.Duration `yaml:"max_backoff_duration"` // Maximum backoff duration
}

type RateLimiterMetrics struct {
	requestsAllowed    int64
	requestsRejected   int64
	clientsTracked     int64
	averageLatency     int64 // microseconds
	lastCleanupTime    int64
}

type ClientLimiter struct {
	limiter           *rate.Limiter
	requestCount      int64
	lastRequestTime   int64
	backoffUntil      int64
	isBlocked         int32 // atomic bool
	windowStart       int64
	windowRequests    int64
}

type SlidingWindowLimiter struct {
	windows    sync.Map // map[string]*Window
	windowSize time.Duration
	maxRequests int64
}

type Window struct {
	requests []int64 // timestamps
	mutex    sync.RWMutex
}

// NewRateLimiter creates a new high-performance rate limiter[6][14]
func NewRateLimiter(config *RateLimiterConfig, redisClient *redis.Client) *RateLimiter {
	rl := &RateLimiter{
		tokenBucket: rate.NewLimiter(rate.Limit(config.GlobalRateLimit), config.BurstSize),
		slidingWindow: &SlidingWindowLimiter{
			windowSize:  config.WindowSize,
			maxRequests: int64(config.ClientRateLimit),
		},
		redisClient: redisClient,
		config:      config,
		metrics: &RateLimiterMetrics{},
	}

	// Start background cleanup
	go rl.cleanup()
	
	// Start metrics collection
	go rl.collectMetrics()

	return rl
}

// Allow checks if a request should be allowed for a given client
func (rl *RateLimiter) Allow(clientID string) (bool, error) {
	start := time.Now()
	defer func() {
		latency := time.Since(start).Microseconds()
		atomic.StoreInt64(&rl.metrics.averageLatency, latency)
	}()

	// Check global rate limit first
	if !rl.tokenBucket.Allow() {
		atomic.AddInt64(&rl.metrics.requestsRejected, 1)
		return false, nil
	}

	// Get or create client-specific limiter
	clientLimiter := rl.getOrCreateClientLimiter(clientID)
	
	// Check if client is currently blocked
	if atomic.LoadInt32(&clientLimiter.isBlocked) == 1 {
		if time.Now().UnixNano() < atomic.LoadInt64(&clientLimiter.backoffUntil) {
			atomic.AddInt64(&rl.metrics.requestsRejected, 1)
			return false, nil
		}
		// Unblock client
		atomic.StoreInt32(&clientLimiter.isBlocked, 0)
	}

	// Check client-specific rate limit
	allowed := clientLimiter.limiter.Allow()
	if !allowed {
		// Implement exponential backoff
		rl.applyBackoff(clientLimiter)
		atomic.AddInt64(&rl.metrics.requestsRejected, 1)
		return false, nil
	}

	// Check sliding window limit
	if rl.config.DistributedMode {
		allowed, err := rl.checkDistributedLimit(clientID)
		if err != nil {
			return false, err
		}
		if !allowed {
			atomic.AddInt64(&rl.metrics.requestsRejected, 1)
			return false, nil
		}
	} else {
		if !rl.slidingWindow.Allow(clientID) {
			atomic.AddInt64(&rl.metrics.requestsRejected, 1)
			return false, nil
		}
	}

	// Update client metrics
	atomic.StoreInt64(&clientLimiter.lastRequestTime, time.Now().UnixNano())
	atomic.AddInt64(&clientLimiter.requestCount, 1)
	atomic.AddInt64(&rl.metrics.requestsAllowed, 1)

	return true, nil
}

// getOrCreateClientLimiter gets or creates a client-specific limiter
func (rl *RateLimiter) getOrCreateClientLimiter(clientID string) *ClientLimiter {
	if limiter, ok := rl.clientLimiters.Load(clientID); ok {
		return limiter.(*ClientLimiter)
	}

	// Create new client limiter
	clientLimiter := &ClientLimiter{
		limiter: rate.NewLimiter(
			rate.Limit(rl.config.ClientRateLimit),
			rl.config.BurstSize,
		),
		lastRequestTime: time.Now().UnixNano(),
		windowStart:     time.Now().UnixNano(),
	}

	// Store in map (atomic operation)
	actual, loaded := rl.clientLimiters.LoadOrStore(clientID, clientLimiter)
	if loaded {
		return actual.(*ClientLimiter)
	}

	atomic.AddInt64(&rl.metrics.clientsTracked, 1)
	return clientLimiter
}

// applyBackoff applies exponential backoff to a client
func (rl *RateLimiter) applyBackoff(clientLimiter *ClientLimiter) {
	backoffDuration := time.Duration(float64(time.Second) * rl.config.BackoffMultiplier)
	if backoffDuration > rl.config.MaxBackoffDuration {
		backoffDuration = rl.config.MaxBackoffDuration
	}

	backoffUntil := time.Now().Add(backoffDuration).UnixNano()
	atomic.StoreInt64(&clientLimiter.backoffUntil, backoffUntil)
	atomic.StoreInt32(&clientLimiter.isBlocked, 1)
}

// checkDistributedLimit checks rate limit using Redis for distributed coordination
func (rl *RateLimiter) checkDistributedLimit(clientID string) (bool, error) {
	ctx := context.Background()
	key := fmt.Sprintf("%s:client:%s", rl.config.RedisKeyPrefix, clientID)
	
	// Use Redis sliding window counter
	pipe := rl.redisClient.Pipeline()
	
	now := time.Now().UnixNano()
	windowStart := now - rl.config.WindowSize.Nanoseconds()
	
	// Remove old entries
	pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
	
	// Add current request
	pipe.ZAdd(ctx, key, &redis.Z{
		Score:  float64(now),
		Member: now,
	})
	
	// Count requests in window
	pipe.ZCard(ctx, key)
	
	// Set expiration
	pipe.Expire(ctx, key, rl.config.WindowSize*2)
	
	results, err := pipe.Exec(ctx)
	if err != nil {
		return false, err
	}
	
	// Get count result
	countResult := results[22].(*redis.IntCmd)
	count, err := countResult.Result()
	if err != nil {
		return false, err
	}
	
	return count <= int64(rl.config.ClientRateLimit), nil
}

// Allow method for SlidingWindowLimiter
func (sw *SlidingWindowLimiter) Allow(clientID string) bool {
	windowInterface, _ := sw.windows.LoadOrStore(clientID, &Window{
		requests: make([]int64, 0),
	})
	window := windowInterface.(*Window)
	
	window.mutex.Lock()
	defer window.mutex.Unlock()
	
	now := time.Now().UnixNano()
	windowStart := now - sw.windowSize.Nanoseconds()
	
	// Remove old requests
	i := 0
	for i < len(window.requests) && window.requests[i] < windowStart {
		i++
	}
	window.requests = window.requests[i:]
	
	// Check if we can add new request
	if int64(len(window.requests)) >= sw.maxRequests {
		return false
	}
	
	// Add current request
	window.requests = append(window.requests, now)
	return true
}

// cleanup removes old client limiters periodically
func (rl *RateLimiter) cleanup() {
	ticker := time.NewTicker(rl.config.CleanupInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		cutoff := time.Now().Add(-rl.config.CleanupInterval * 2).UnixNano()
		
		rl.clientLimiters.Range(func(key, value interface{}) bool {
			clientLimiter := value.(*ClientLimiter)
			lastRequest := atomic.LoadInt64(&clientLimiter.lastRequestTime)
			
			if lastRequest < cutoff {
				rl.clientLimiters.Delete(key)
				atomic.AddInt64(&rl.metrics.clientsTracked, -1)
			}
			return true
		})
		
		atomic.StoreInt64(&rl.metrics.lastCleanupTime, time.Now().UnixNano())
	}
}

// collectMetrics collects and reports metrics
func (rl *RateLimiter) collectMetrics() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		allowed := atomic.LoadInt64(&rl.metrics.requestsAllowed)
		rejected := atomic.LoadInt64(&rl.metrics.requestsRejected)
		clients := atomic.LoadInt64(&rl.metrics.clientsTracked)
		latency := atomic.LoadInt64(&rl.metrics.averageLatency)
		
		// Reset counters
		atomic.StoreInt64(&rl.metrics.requestsAllowed, 0)
		atomic.StoreInt64(&rl.metrics.requestsRejected, 0)
		
		// Log metrics (in production, send to monitoring system)
		fmt.Printf("RateLimiter Metrics: Allowed=%d, Rejected=%d, Clients=%d, AvgLatency=%dμs\n",
			allowed, rejected, clients, latency)
	}
}

// GetMetrics returns current rate limiter metrics
func (rl *RateLimiter) GetMetrics() RateLimiterMetrics {
	return RateLimiterMetrics{
		requestsAllowed:    atomic.LoadInt64(&rl.metrics.requestsAllowed),
		requestsRejected:   atomic.LoadInt64(&rl.metrics.requestsRejected),
		clientsTracked:     atomic.LoadInt64(&rl.metrics.clientsTracked),
		averageLatency:     atomic.LoadInt64(&rl.metrics.averageLatency),
		lastCleanupTime:    atomic.LoadInt64(&rl.metrics.lastCleanupTime),
	}
}

// Reset resets all rate limiting state
func (rl *RateLimiter) Reset() {
	rl.clientLimiters.Range(func(key, value interface{}) bool {
		rl.clientLimiters.Delete(key)
		return true
	})
	
	rl.slidingWindow.windows.Range(func(key, value interface{}) bool {
		rl.slidingWindow.windows.Delete(key)
		return true
	})
	
	atomic.StoreInt64(&rl.metrics.clientsTracked, 0)
}
