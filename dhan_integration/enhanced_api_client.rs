/*!
 * DhanHQ Enhanced API Client Integration for AlgoVeda
 *
 * Provides robust, high-performance integration with DhanHQ APIs,
 * supporting advanced rate limiting, failover, and streaming.
 */

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use reqwest::Client;
use tracing::{info, warn, error, instrument};
use chrono::{DateTime, Utc};

use crate::config::DhanConfig;
use crate::trading::{Order, TradingResult, TradingError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanOrder {
    pub dhan_client_id: String,
    pub correlation_id: String,
    pub order_id: Option<String>,
    pub exchange_segment: String,
    pub product_type: String,
    pub order_type: String,
    pub validity: String,
    pub trading_symbol: String,
    pub security_id: String,
    pub quantity: u32,
    pub disclosed_quantity: u32,
    pub price: f64,
    pub trigger_price: f64,
    pub after_market_order: bool,
    pub amo_time: String,
    pub bo_profit_value: f64,
    pub bo_stop_loss_value: f64,
}

#[derive(Debug, Clone)]
pub struct DhanResponse<T> {
    pub status: String,
    pub data: Option<T>,
    pub error: Option<String>,
    pub remarks: HashMap<String, String>,
}

#[derive(Debug)]
pub struct EnhancedApiClient {
    config: Arc<DhanConfig>,
    http_client: Client,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    retry_handler: RetryHandler,
    connection_pool: ConnectionPool,
    is_connected: Arc<RwLock<bool>>,
    last_request_time: Arc<RwLock<Instant>>,
}

#[derive(Debug)]
struct RateLimiter {
    requests_per_second: u32,
    tokens: u32,
    last_refill: Instant,
}

#[derive(Debug)]
struct CircuitBreaker {
    failure_count: u32,
    success_count: u32,
    failure_threshold: u32,
    recovery_timeout: Duration,
    state: CircuitBreakerState,
    last_failure_time: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug)]
struct RetryHandler {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
}

#[derive(Debug)]
struct ConnectionPool {
    connections: u32,
    max_connections: u32,
}

impl EnhancedApiClient {
    pub async fn new(config: &DhanConfig) -> TradingResult<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| TradingError::ExecutionFailed(format!("HTTP client creation failed: {}", e)))?;

        let rate_limiter = RateLimiter {
            requests_per_second: config.rate_limit,
            tokens: config.rate_limit,
            last_refill: Instant::now(),
        };

        let circuit_breaker = CircuitBreaker {
            failure_count: 0,
            success_count: 0,
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            state: CircuitBreakerState::Closed,
            last_failure_time: None,
        };

        let retry_handler = RetryHandler {
            max_retries: config.retry_attempts,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
        };

        let connection_pool = ConnectionPool {
            connections: 0,
            max_connections: 10,
        };

        Ok(Self {
            config: Arc::new(config.clone()),
            http_client,
            rate_limiter: Arc::new(RwLock::new(rate_limiter)),
            circuit_breaker: Arc::new(RwLock::new(circuit_breaker)),
            retry_handler,
            connection_pool,
            is_connected: Arc::new(RwLock::new(false)),
            last_request_time: Arc::new(RwLock::new(Instant::now())),
        })
    }

    #[instrument(skip(self))]
    pub async fn place_order(&self, order: &DhanOrder) -> TradingResult<DhanResponse<String>> {
        self.check_circuit_breaker().await?;
        self.apply_rate_limiting().await?;

        let url = format!("{}/orders", self.config.base_url);
        let mut attempt = 0;

        loop {
            match self.execute_request(&url, order).await {
                Ok(response) => {
                    self.record_success().await;
                    return Ok(response);
                }
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.retry_handler.max_retries {
                        self.record_failure().await;
                        return Err(e);
                    }
                    
                    let delay = self.calculate_retry_delay(attempt);
                    tokio::time::sleep(delay).await;
                    warn!("Retrying order placement, attempt {}/{}", attempt + 1, self.retry_handler.max_retries);
                }
            }
        }
    }

    async fn execute_request<T: Serialize>(
        &self,
        url: &str,
        payload: &T,
    ) -> TradingResult<DhanResponse<String>> {
        let response = self
            .http_client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await
            .map_err(|e| TradingError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        if response.status().is_success() {
            let dhan_response: DhanResponse<String> = response
                .json()
                .await
                .map_err(|e| TradingError::ExecutionFailed(format!("Response parsing failed: {}", e)))?;
            
            Ok(dhan_response)
        } else {
            let error_text = response.text().await.unwrap_or_default();
            Err(TradingError::ExecutionFailed(format!(
                "DhanHQ API error: {} - {}", 
                response.status(), 
                error_text
            )))
        }
    }

    async fn check_circuit_breaker(&self) -> TradingResult<()> {
        let mut breaker = self.circuit_breaker.write();
        
        match breaker.state {
            CircuitBreakerState::Open => {
                if let Some(last_failure) = breaker.last_failure_time {
                    if last_failure.elapsed() > breaker.recovery_timeout {
                        breaker.state = CircuitBreakerState::HalfOpen;
                        info!("Circuit breaker moved to half-open state");
                    } else {
                        return Err(TradingError::ExecutionFailed(
                            "Circuit breaker is open - requests blocked".to_string()
                        ));
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }

    async fn apply_rate_limiting(&self) -> TradingResult<()> {
        let mut limiter = self.rate_limiter.write();
        let now = Instant::now();
        
        // Refill tokens based on time elapsed
        let elapsed = now.duration_since(limiter.last_refill).as_secs();
        if elapsed >= 1 {
            limiter.tokens = limiter.requests_per_second;
            limiter.last_refill = now;
        }
        
        if limiter.tokens > 0 {
            limiter.tokens -= 1;
            Ok(())
        } else {
            let wait_time = Duration::from_millis(1000 / limiter.requests_per_second as u64);
            tokio::time::sleep(wait_time).await;
            self.apply_rate_limiting().await
        }
    }

    async fn record_success(&self) {
        let mut breaker = self.circuit_breaker.write();
        breaker.success_count += 1;
        breaker.failure_count = 0;
        
        if breaker.state == CircuitBreakerState::HalfOpen {
            breaker.state = CircuitBreakerState::Closed;
            info!("Circuit breaker closed after successful request");
        }
    }

    async fn record_failure(&self) {
        let mut breaker = self.circuit_breaker.write();
        breaker.failure_count += 1;
        breaker.success_count = 0;
        
        if breaker.failure_count >= breaker.failure_threshold {
            breaker.state = CircuitBreakerState::Open;
            breaker.last_failure_time = Some(Instant::now());
            error!("Circuit breaker opened due to excessive failures");
        }
    }

    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let delay = self.retry_handler.base_delay * 2_u32.pow(attempt - 1);
        delay.min(self.retry_handler.max_delay)
    }

    pub async fn get_positions(&self) -> TradingResult<DhanResponse<Vec<serde_json::Value>>> {
        self.check_circuit_breaker().await?;
        self.apply_rate_limiting().await?;

        let url = format!("{}/positions", self.config.base_url);
        
        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .send()
            .await
            .map_err(|e| TradingError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        if response.status().is_success() {
            let positions: DhanResponse<Vec<serde_json::Value>> = response
                .json()
                .await
                .map_err(|e| TradingError::ExecutionFailed(format!("Response parsing failed: {}", e)))?;
            
            self.record_success().await;
            Ok(positions)
        } else {
            self.record_failure().await;
            Err(TradingError::ExecutionFailed(format!(
                "Failed to fetch positions: {}", 
                response.status()
            )))
        }
    }

    pub async fn get_orders(&self) -> TradingResult<DhanResponse<Vec<serde_json::Value>>> {
        self.check_circuit_breaker().await?;
        self.apply_rate_limiting().await?;

        let url = format!("{}/orders", self.config.base_url);
        
        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .send()
            .await
            .map_err(|e| TradingError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        if response.status().is_success() {
            let orders: DhanResponse<Vec<serde_json::Value>> = response
                .json()
                .await
                .map_err(|e| TradingError::ExecutionFailed(format!("Response parsing failed: {}", e)))?;
            
            self.record_success().await;
            Ok(orders)
        } else {
            self.record_failure().await;
            Err(TradingError::ExecutionFailed(format!(
                "Failed to fetch orders: {}", 
                response.status()
            )))
        }
    }

    pub async fn cancel_order(&self, order_id: &str) -> TradingResult<DhanResponse<String>> {
        self.check_circuit_breaker().await?;
        self.apply_rate_limiting().await?;

        let url = format!("{}/orders/{}", self.config.base_url, order_id);
        
        let response = self
            .http_client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .send()
            .await
            .map_err(|e| TradingError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        if response.status().is_success() {
            let result: DhanResponse<String> = response
                .json()
                .await
                .map_err(|e| TradingError::ExecutionFailed(format!("Response parsing failed: {}", e)))?;
            
            self.record_success().await;
            Ok(result)
        } else {
            self.record_failure().await;
            Err(TradingError::ExecutionFailed(format!(
                "Failed to cancel order: {}", 
                response.status()
            )))
        }
    }

    pub async fn start_connection(&self) -> TradingResult<()> {
        info!("Starting DhanHQ API connection");
        
        // Test connectivity
        let health_check = self.health_check().await;
        if health_check {
            let mut connected = self.is_connected.write();
            *connected = true;
            info!("DhanHQ connection established successfully");
        } else {
            return Err(TradingError::ExecutionFailed(
                "Failed to establish DhanHQ connection".to_string()
            ));
        }
        
        Ok(())
    }

    pub async fn close_connection(&self) -> TradingResult<()> {
        info!("Closing DhanHQ API connection");
        let mut connected = self.is_connected.write();
        *connected = false;
        Ok(())
    }

    pub async fn health_check(&self) -> bool {
        // Simple connectivity test
        match self.get_positions().await {
            Ok(_) => true,
            Err(e) => {
                warn!("DhanHQ health check failed: {}", e);
                false
            }
        }
    }

    pub async fn get_connection_stats(&self) -> ConnectionStats {
        let breaker = self.circuit_breaker.read();
        let limiter = self.rate_limiter.read();
        
        ConnectionStats {
            is_connected: *self.is_connected.read(),
            circuit_breaker_state: breaker.state,
            failure_count: breaker.failure_count,
            success_count: breaker.success_count,
            remaining_tokens: limiter.tokens,
            last_request_time: *self.last_request_time.read(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub is_connected: bool,
    pub circuit_breaker_state: CircuitBreakerState,
    pub failure_count: u32,
    pub success_count: u32,
    pub remaining_tokens: u32,
    pub last_request_time: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_client_creation() {
        let config = DhanConfig {
            client_id: "test_client".to_string(),
            access_token: "test_token".to_string(),
            api_key: "test_key".to_string(),
            base_url: "https://api.dhan.co".to_string(),
            websocket_url: "wss://api-feed.dhan.co".to_string(),
            rate_limit: 100,
            timeout: 30,
            retry_attempts: 3,
        };
        
        let client = EnhancedApiClient::new(&config).await;
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = DhanConfig {
            rate_limit: 1, // 1 request per second
            ..DhanConfig::default()
        };
        
        let client = EnhancedApiClient::new(&config).await.unwrap();
        
        // First request should go through
        let start = Instant::now();
        client.apply_rate_limiting().await.unwrap();
        let first_duration = start.elapsed();
        
        // Second request should be delayed
        let start = Instant::now();
        client.apply_rate_limiting().await.unwrap();
        let second_duration = start.elapsed();
        
        assert!(second_duration > first_duration);
    }
}
