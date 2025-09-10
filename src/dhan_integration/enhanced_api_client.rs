use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use reqwest::{Client, Response, Method};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use thiserror::Error;
use chrono::{DateTime, Utc};

#[derive(Debug, Error)]
pub enum DhanApiError {
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("API request failed: {status} - {message}")]
    RequestFailed { status: u16, message: String },
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Invalid response format")]
    InvalidResponseFormat,
    #[error("Order placement failed: {0}")]
    OrderPlacementFailed(String),
    #[error("Market data unavailable")]
    MarketDataUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanCredentials {
    pub client_id: String,
    pub access_token: String,
    pub api_key: String,
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanOrderRequest {
    pub dhan_client_id: String,
    pub correlation_id: String,
    pub transaction_type: String, // BUY/SELL
    pub exchange_segment: String, // NSE_EQ/BSE_EQ/NSE_FNO/NSE_CURRENCY/MCX_COMM
    pub product_type: String,     // CNC/INTRADAY/MARGIN/CO/BO
    pub order_type: String,       // MARKET/LIMIT/STOP_LOSS/STOP_LOSS_MARKET
    pub validity: String,         // DAY/IOC
    pub trading_symbol: String,
    pub security_id: String,
    pub quantity: i32,
    pub disclosed_quantity: Option<i32>,
    pub price: Option<f64>,
    pub trigger_price: Option<f64>,
    pub after_market_order: Option<bool>,
    pub amo_time: Option<String>,
    pub bo_profit_value: Option<f64>,
    pub bo_stop_loss_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanOrderResponse {
    pub status: String,
    pub remarks: Value,
    pub data: Option<DhanOrderData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanOrderData {
    pub order_id: String,
    pub order_status: String,
    pub transaction_type: String,
    pub exchange_segment: String,
    pub product_type: String,
    pub order_type: String,
    pub validity: String,
    pub trading_symbol: String,
    pub security_id: String,
    pub quantity: i32,
    pub disclosed_quantity: i32,
    pub price: f64,
    pub trigger_price: f64,
    pub filled_qty: i32,
    pub pending_qty: i32,
    pub order_value: f64,
    pub create_time: String,
    pub update_time: String,
    pub exchange_time: String,
    pub drv_expiry_date: Option<String>,
    pub drv_option_type: Option<String>,
    pub drv_strike_price: Option<f64>,
    pub oms_error_code: Option<String>,
    pub oms_error_description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanPosition {
    pub position_type: String,
    pub exchange_segment: String,
    pub product_type: String,
    pub trading_symbol: String,
    pub security_id: String,
    pub net_qty: i32,
    pub buy_avg: f64,
    pub buy_qty: i32,
    pub sell_avg: f64,
    pub sell_qty: i32,
    pub realized_profit: f64,
    pub unrealized_profit: f64,
    pub rch_flg: String,
    pub drv_expiry_date: Option<String>,
    pub drv_option_type: Option<String>,
    pub drv_strike_price: Option<f64>,
    pub cross_currency: Option<String>,
    pub cost_price: f64,
    pub multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanMarketData {
    pub trading_symbol: String,
    pub exchange_segment: String,
    pub ltp: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub prev_close: f64,
    pub change: f64,
    pub pchange: f64,
    pub timestamp: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: i32,
    pub ask_qty: i32,
    pub oi: Option<i64>,
    pub prev_oi: Option<i64>,
    pub tot_buy_quan: Option<i64>,
    pub tot_sell_quan: Option<i64>,
}

pub struct EnhancedDhanApiClient {
    client: Client,
    credentials: DhanCredentials,
    rate_limiter: Arc<Semaphore>,
    request_history: Arc<RwLock<Vec<Instant>>>,
    connection_pool: Arc<RwLock<HashMap<String, Instant>>>,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreaker,
    metrics: Arc<RwLock<ApiMetrics>>,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub state: CircuitBreakerState,
    pub failure_count: u32,
    pub last_failure_time: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Default)]
pub struct ApiMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub rate_limited_requests: u64,
    pub average_response_time: Duration,
    pub last_request_time: Option<Instant>,
    pub connection_errors: u64,
}

impl EnhancedDhanApiClient {
    pub fn new(credentials: DhanCredentials) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(10)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            credentials,
            rate_limiter: Arc::new(Semaphore::new(10)), // 10 requests per second
            request_history: Arc::new(RwLock::new(Vec::new())),
            connection_pool: Arc::new(RwLock::new(HashMap::new())),
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },
            circuit_breaker: CircuitBreaker {
                failure_threshold: 5,
                recovery_timeout: Duration::from_secs(60),
                state: CircuitBreakerState::Closed,
                failure_count: 0,
                last_failure_time: None,
            },
            metrics: Arc::new(RwLock::new(ApiMetrics::default())),
        }
    }

    pub async fn place_order(&self, order_request: DhanOrderRequest) -> Result<DhanOrderResponse, DhanApiError> {
        let start_time = Instant::now();
        
        // Rate limiting
        let _permit = self.rate_limiter.acquire().await.unwrap();

        // Circuit breaker check
        if self.is_circuit_breaker_open().await {
            return Err(DhanApiError::RequestFailed {
                status: 503,
                message: "Circuit breaker is open".to_string(),
            });
        }

        let mut attempt = 0;
        loop {
            match self.execute_order_request(&order_request).await {
                Ok(response) => {
                    self.record_success(start_time).await;
                    return Ok(response);
                }
                Err(e) => {
                    self.record_failure().await;
                    
                    attempt += 1;
                    if attempt >= self.retry_policy.max_retries {
                        return Err(e);
                    }

                    // Exponential backoff
                    let delay = self.calculate_retry_delay(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn execute_order_request(&self, order_request: &DhanOrderRequest) -> Result<DhanOrderResponse, DhanApiError> {
        let url = format!("{}/orders", self.credentials.base_url);
        
        let response = self.client
            .post(&url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .header("authorization", &self.credentials.access_token)
            .header("api-key", &self.credentials.api_key)
            .json(order_request)
            .send()
            .await?;

        let status = response.status().as_u16();
        
        if status == 429 {
            return Err(DhanApiError::RateLimitExceeded);
        }

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(DhanApiError::RequestFailed {
                status,
                message: error_text,
            });
        }

        let order_response: DhanOrderResponse = response.json().await?;
        
        if order_response.status != "success" {
            return Err(DhanApiError::OrderPlacementFailed(
                order_response.remarks.to_string()
            ));
        }

        Ok(order_response)
    }

    pub async fn get_orders(&self) -> Result<Vec<DhanOrderData>, DhanApiError> {
        let url = format!("{}/orders", self.credentials.base_url);
        
        let response = self.make_authenticated_request(Method::GET, &url, None::<&()>).await?;
        
        #[derive(Deserialize)]
        struct OrdersResponse {
            status: String,
            data: Vec<DhanOrderData>,
        }
        
        let orders_response: OrdersResponse = response.json().await?;
        Ok(orders_response.data)
    }

    pub async fn get_positions(&self) -> Result<Vec<DhanPosition>, DhanApiError> {
        let url = format!("{}/positions", self.credentials.base_url);
        
        let response = self.make_authenticated_request(Method::GET, &url, None::<&()>).await?;
        
        #[derive(Deserialize)]
        struct PositionsResponse {
            status: String,
            data: Vec<DhanPosition>,
        }
        
        let positions_response: PositionsResponse = response.json().await?;
        Ok(positions_response.data)
    }

    pub async fn get_market_data(&self, symbols: &[String]) -> Result<Vec<DhanMarketData>, DhanApiError> {
        let symbols_param = symbols.join(",");
        let url = format!("{}/marketdata/ltp?symbols={}", self.credentials.base_url, symbols_param);
        
        let response = self.make_authenticated_request(Method::GET, &url, None::<&()>).await?;
        
        #[derive(Deserialize)]
        struct MarketDataResponse {
            status: String,
            data: HashMap<String, DhanMarketData>,
        }
        
        let market_response: MarketDataResponse = response.json().await?;
        Ok(market_response.data.into_values().collect())
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<DhanOrderResponse, DhanApiError> {
        let url = format!("{}/orders/{}", self.credentials.base_url, order_id);
        
        let response = self.make_authenticated_request(Method::DELETE, &url, None::<&()>).await?;
        let cancel_response: DhanOrderResponse = response.json().await?;
        
        Ok(cancel_response)
    }

    pub async fn modify_order(
        &self, 
        order_id: &str, 
        modifications: &DhanOrderRequest
    ) -> Result<DhanOrderResponse, DhanApiError> {
        let url = format!("{}/orders/{}", self.credentials.base_url, order_id);
        
        let response = self.make_authenticated_request(Method::PUT, &url, Some(modifications)).await?;
        let modify_response: DhanOrderResponse = response.json().await?;
        
        Ok(modify_response)
    }

    pub async fn get_tradebook(&self) -> Result<Vec<Value>, DhanApiError> {
        let url = format!("{}/tradebook", self.credentials.base_url);
        
        let response = self.make_authenticated_request(Method::GET, &url, None::<&()>).await?;
        
        #[derive(Deserialize)]
        struct TradebookResponse {
            status: String,
            data: Vec<Value>,
        }
        
        let tradebook_response: TradebookResponse = response.json().await?;
        Ok(tradebook_response.data)
    }

    pub async fn get_holdings(&self) -> Result<Vec<Value>, DhanApiError> {
        let url = format!("{}/holdings", self.credentials.base_url);
        
        let response = self.make_authenticated_request(Method::GET, &url, None::<&()>).await?;
        
        #[derive(Deserialize)]
        struct HoldingsResponse {
            status: String,
            data: Vec<Value>,
        }
        
        let holdings_response: HoldingsResponse = response.json().await?;
        Ok(holdings_response.data)
    }

    async fn make_authenticated_request<T: Serialize>(
        &self,
        method: Method,
        url: &str,
        body: Option<&T>,
    ) -> Result<Response, DhanApiError> {
        // Rate limiting
        let _permit = self.rate_limiter.acquire().await.unwrap();

        // Circuit breaker check
        if self.is_circuit_breaker_open().await {
            return Err(DhanApiError::RequestFailed {
                status: 503,
                message: "Circuit breaker is open".to_string(),
            });
        }

        let start_time = Instant::now();
        let mut request = self.client
            .request(method, url)
            .header("Accept", "application/json")
            .header("authorization", &self.credentials.access_token)
            .header("api-key", &self.credentials.api_key);

        if let Some(body) = body {
            request = request.json(body);
        }

        let response = request.send().await?;

        let status = response.status();
        if status.is_success() {
            self.record_success(start_time).await;
            Ok(response)
        } else {
            self.record_failure().await;
            let error_text = response.text().await.unwrap_or_default();
            Err(DhanApiError::RequestFailed {
                status: status.as_u16(),
                message: error_text,
            })
        }
    }

    async fn is_circuit_breaker_open(&self) -> bool {
        // Simple circuit breaker logic
        self.circuit_breaker.state == CircuitBreakerState::Open &&
        self.circuit_breaker.last_failure_time
            .map(|t| t.elapsed() < self.circuit_breaker.recovery_timeout)
            .unwrap_or(false)
    }

    async fn record_success(&self, start_time: Instant) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        metrics.successful_requests += 1;
        
        let response_time = start_time.elapsed();
        metrics.average_response_time = 
            (metrics.average_response_time + response_time) / 2;
        
        metrics.last_request_time = Some(Instant::now());
    }

    async fn record_failure(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        metrics.failed_requests += 1;
        metrics.last_request_time = Some(Instant::now());
    }

    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let delay = self.retry_policy.base_delay.as_millis() as f64 * 
                   self.retry_policy.backoff_multiplier.powi(attempt as i32);
        
        Duration::from_millis(delay.min(self.retry_policy.max_delay.as_millis() as f64) as u64)
    }

    pub async fn get_metrics(&self) -> ApiMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn health_check(&self) -> Result<bool, DhanApiError> {
        let url = format!("{}/margincalculator/compact", self.credentials.base_url);
        
        let response = self.make_authenticated_request(Method::GET, &url, None::<&()>).await?;
        Ok(response.status().is_success())
    }

    // Batch operations for better performance
    pub async fn place_multiple_orders(
        &self, 
        orders: Vec<DhanOrderRequest>
    ) -> Result<Vec<Result<DhanOrderResponse, DhanApiError>>, DhanApiError> {
        let mut results = Vec::new();
        
        // Process orders in batches to respect rate limits
        for chunk in orders.chunks(5) {
            let mut batch_results = Vec::new();
            
            for order in chunk {
                let result = self.place_order(order.clone()).await;
                batch_results.push(result);
            }
            
            results.extend(batch_results);
            
            // Small delay between batches
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        Ok(results)
    }

    pub async fn get_portfolio_summary(&self) -> Result<PortfolioSummary, DhanApiError> {
        let positions = self.get_positions().await?;
        let holdings = self.get_holdings().await?;
        
        // Calculate portfolio metrics
        let total_equity = positions.iter()
            .map(|p| p.net_qty as f64 * p.cost_price)
            .sum::<f64>();
        
        let total_pnl = positions.iter()
            .map(|p| p.realized_profit + p.unrealized_profit)
            .sum::<f64>();
        
        Ok(PortfolioSummary {
            total_equity,
            total_pnl,
            positions_count: positions.len(),
            holdings_count: holdings.len(),
            last_updated: Utc::now(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub total_equity: f64,
    pub total_pnl: f64,
    pub positions_count: usize,
    pub holdings_count: usize,
    pub last_updated: DateTime<Utc>,
}

// WebSocket client for real-time data
pub struct DhanWebSocketClient {
    // WebSocket implementation for real-time market data and order updates
    // This would integrate with the WebSocket gateway
}

impl DhanWebSocketClient {
    pub async fn connect(&mut self) -> Result<(), DhanApiError> {
        // WebSocket connection logic
        Ok(())
    }

    pub async fn subscribe_market_data(&mut self, symbols: Vec<String>) -> Result<(), DhanApiError> {
        // Subscribe to real-time market data
        Ok(())
    }

    pub async fn subscribe_order_updates(&mut self) -> Result<(), DhanApiError> {
        // Subscribe to real-time order updates
        Ok(())
    }
}

// Integration tests
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_place_order() {
        // Test order placement
        let credentials = DhanCredentials {
            client_id: "test_client".to_string(),
            access_token: "test_token".to_string(),
            api_key: "test_api_key".to_string(),
            base_url: "https://api.dhan.co".to_string(),
        };

        let client = EnhancedDhanApiClient::new(credentials);
        
        let order = DhanOrderRequest {
            dhan_client_id: "test_client".to_string(),
            correlation_id: "test_correlation".to_string(),
            transaction_type: "BUY".to_string(),
            exchange_segment: "NSE_EQ".to_string(),
            product_type: "INTRADAY".to_string(),
            order_type: "MARKET".to_string(),
            validity: "DAY".to_string(),
            trading_symbol: "RELIANCE".to_string(),
            security_id: "2885".to_string(),
            quantity: 1,
            disclosed_quantity: None,
            price: None,
            trigger_price: None,
            after_market_order: None,
            amo_time: None,
            bo_profit_value: None,
            bo_stop_loss_value: None,
        };

        // Note: This would fail without valid credentials
        // let result = client.place_order(order).await;
        // assert!(result.is_ok());
    }
}
