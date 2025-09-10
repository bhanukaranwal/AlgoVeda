/*!
 * Execution Engine
 * Multi-venue order execution with smart routing and fill optimization
 */

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout},
};
use uuid::Uuid;
use rust_decimal::Decimal;
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};

use crate::{
    config::TradingConfig,
    trading::{Order, OrderStatus, OrderSide, Fill, Trade},
    networking::VenueConnector,
    monitoring::MetricsCollector,
    error::{Result, AlgoVedaError},
    utils::latency::LatencyTracker,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub order_id: Uuid,
    pub parent_order_id: Option<Uuid>,
    pub strategy_id: String,
    pub venue: String,
    pub execution_algo: ExecutionAlgorithm,
    pub urgency: ExecutionUrgency,
    pub constraints: ExecutionConstraints,
    pub created_at: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    Market,
    Limit,
    TWAP { duration_ms: u64, slice_size: Decimal },
    VWAP { participation_rate: f64 },
    ImplementationShortfall { risk_aversion: f64 },
    Iceberg { visible_size: Decimal, refresh_rate: f64 },
    Sniper { max_levels: u32, timeout_ms: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionUrgency {
    Low,
    Normal,
    High,
    Urgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    pub max_execution_time_ms: u64,
    pub max_slippage_bps: u32,
    pub min_fill_ratio: f64,
    pub preferred_venues: Vec<String>,
    pub avoid_venues: Vec<String>,
    pub time_constraints: Option<TimeConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    pub start_time: Option<Instant>,
    pub end_time: Option<Instant>,
    pub no_cross_session: bool,
}

pub struct ExecutionEngine {
    config: Arc<TradingConfig>,
    venue_connectors: Arc<RwLock<HashMap<String, Arc<VenueConnector>>>>,
    active_executions: Arc<RwLock<HashMap<Uuid, ExecutionContext>>>,
    execution_queue: Arc<Mutex<mpsc::UnboundedReceiver<ExecutionRequest>>>,
    execution_sender: mpsc::UnboundedSender<ExecutionRequest>,
    fill_sender: broadcast::Sender<Fill>,
    metrics: Arc<MetricsCollector>,
    latency_tracker: Arc<LatencyTracker>,
}

#[derive(Debug, Clone)]
pub struct ExecutionRequest {
    pub order: Order,
    pub context: ExecutionContext,
    pub callback: Option<mpsc::UnboundedSender<ExecutionResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub order_id: Uuid,
    pub status: ExecutionStatus,
    pub fills: Vec<Fill>,
    pub remaining_quantity: Decimal,
    pub average_price: Option<Decimal>,
    pub total_commission: Decimal,
    pub execution_time_ms: u64,
    pub slippage_bps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Failed(String),
}

impl ExecutionEngine {
    pub async fn new(
        config: Arc<TradingConfig>,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        let (execution_sender, execution_receiver) = mpsc::unbounded_channel();
        let (fill_sender, _) = broadcast::channel(10000);

        Ok(Self {
            config,
            venue_connectors: Arc::new(RwLock::new(HashMap::new())),
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            execution_queue: Arc::new(Mutex::new(execution_receiver)),
            execution_sender,
            fill_sender,
            metrics,
            latency_tracker: Arc::new(LatencyTracker::new()),
        })
    }

    #[instrument(skip(self))]
    pub async fn submit_order(&self, order: Order, context: ExecutionContext) -> Result<Uuid> {
        let execution_id = Uuid::new_v4();
        
        debug!("Submitting order for execution: {}", order.id);
        
        // Validate execution request
        self.validate_execution_request(&order, &context)?;
        
        // Store execution context
        {
            let mut executions = self.active_executions.write().unwrap();
            executions.insert(execution_id, context.clone());
        }
        
        // Create execution request
        let request = ExecutionRequest {
            order,
            context,
            callback: None,
        };
        
        // Submit to execution queue
        self.execution_sender.send(request)
            .map_err(|e| AlgoVedaError::Trading(format!("Failed to submit execution: {}", e)))?;
        
        // Update metrics
        self.metrics.increment_counter("execution_requests_submitted", &[]);
        
        Ok(execution_id)
    }

    #[instrument(skip(self))]
    pub async fn run(&self) -> Result<()> {
        info!("Starting execution engine");
        
        let mut execution_queue = self.execution_queue.lock().await;
        let mut heartbeat_interval = interval(Duration::from_millis(1000));
        
        loop {
            tokio::select! {
                // Process execution requests
                Some(request) = execution_queue.recv() => {
                    self.process_execution_request(request).await;
                }
                
                // Heartbeat and maintenance
                _ = heartbeat_interval.tick() => {
                    self.perform_maintenance().await;
                }
            }
        }
    }

    async fn process_execution_request(&self, request: ExecutionRequest) {
        let start_time = Instant::now();
        let order_id = request.order.id;
        
        debug!("Processing execution request for order: {}", order_id);
        
        let result = match request.context.execution_algo {
            ExecutionAlgorithm::Market => {
                self.execute_market_order(request).await
            }
            ExecutionAlgorithm::Limit => {
                self.execute_limit_order(request).await
            }
            ExecutionAlgorithm::TWAP { duration_ms, slice_size } => {
                self.execute_twap_order(request, duration_ms, slice_size).await
            }
            ExecutionAlgorithm::VWAP { participation_rate } => {
                self.execute_vwap_order(request, participation_rate).await
            }
            ExecutionAlgorithm::ImplementationShortfall { risk_aversion } => {
                self.execute_implementation_shortfall(request, risk_aversion).await
            }
            ExecutionAlgorithm::Iceberg { visible_size, refresh_rate } => {
                self.execute_iceberg_order(request, visible_size, refresh_rate).await
            }
            ExecutionAlgorithm::Sniper { max_levels, timeout_ms } => {
                self.execute_sniper_order(request, max_levels, timeout_ms).await
            }
        };

        let execution_time = start_time.elapsed();
        
        // Update latency tracking
        self.latency_tracker.record_latency("order_execution", execution_time);
        
        // Update metrics based on result
        match &result {
            Ok(exec_result) => {
                self.metrics.record_histogram(
                    "execution_time_ms",
                    execution_time.as_millis() as f64,
                    &[("status", "success")],
                );
                
                match exec_result.status {
                    ExecutionStatus::Filled => {
                        self.metrics.increment_counter("orders_filled", &[]);
                    }
                    ExecutionStatus::PartiallyFilled => {
                        self.metrics.increment_counter("orders_partially_filled", &[]);
                    }
                    _ => {}
                }
            }
            Err(e) => {
                error!("Execution failed for order {}: {}", order_id, e);
                self.metrics.increment_counter("execution_failures", &[]);
            }
        }
    }

    async fn execute_market_order(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        let order = &request.order;
        debug!("Executing market order: {}", order.id);
        
        // Select best venue for market order
        let venue = self.select_venue_for_market_order(order).await?;
        
        // Get venue connector
        let connector = self.get_venue_connector(&venue)?;
        
        // Submit order to venue
        let venue_order_id = connector.submit_market_order(order).await?;
        
        // Wait for fills with timeout
        let timeout_duration = Duration::from_millis(
            request.context.constraints.max_execution_time_ms
        );
        
        match timeout(timeout_duration, self.wait_for_fills(venue_order_id)).await {
            Ok(fills) => {
                let (status, avg_price, total_commission) = self.calculate_execution_stats(&fills);
                
                Ok(ExecutionResult {
                    order_id: order.id,
                    status,
                    fills,
                    remaining_quantity: Decimal::ZERO, // Market orders fill completely or fail
                    average_price: avg_price,
                    total_commission,
                    execution_time_ms: request.context.created_at.elapsed().as_millis() as u64,
                    slippage_bps: None, // TODO: Calculate slippage
                })
            }
            Err(_) => {
                // Timeout - cancel order
                let _ = connector.cancel_order(venue_order_id).await;
                
                Err(AlgoVedaError::Trading(
                    "Market order execution timeout".to_string()
                ))
            }
        }
    }

    async fn execute_limit_order(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        let order = &request.order;
        debug!("Executing limit order: {} at {}", order.id, order.price.unwrap());
        
        // Select venue for limit order
        let venue = self.select_venue_for_limit_order(order).await?;
        
        // Get venue connector
        let connector = self.get_venue_connector(&venue)?;
        
        // Submit limit order
        let venue_order_id = connector.submit_limit_order(order).await?;
        
        // Monitor order status and fills
        let timeout_duration = Duration::from_millis(
            request.context.constraints.max_execution_time_ms
        );
        
        match timeout(timeout_duration, self.monitor_limit_order(venue_order_id)).await {
            Ok(fills) => {
                let (status, avg_price, total_commission) = self.calculate_execution_stats(&fills);
                let remaining_quantity = order.quantity - fills.iter()
                    .map(|f| f.quantity)
                    .sum::<Decimal>();
                
                Ok(ExecutionResult {
                    order_id: order.id,
                    status,
                    fills,
                    remaining_quantity,
                    average_price: avg_price,
                    total_commission,
                    execution_time_ms: request.context.created_at.elapsed().as_millis() as u64,
                    slippage_bps: None,
                })
            }
            Err(_) => {
                // Timeout - cancel remaining
                let _ = connector.cancel_order(venue_order_id).await;
                
                Err(AlgoVedaError::Trading(
                    "Limit order execution timeout".to_string()
                ))
            }
        }
    }

    async fn execute_twap_order(
        &self,
        request: ExecutionRequest,
        duration_ms: u64,
        slice_size: Decimal,
    ) -> Result<ExecutionResult> {
        let order = &request.order;
        debug!("Executing TWAP order: {} over {}ms with slice size {}", 
               order.id, duration_ms, slice_size);
        
        let total_quantity = order.quantity;
        let num_slices = (total_quantity / slice_size).ceil().to_u64().unwrap() as usize;
        let slice_interval = Duration::from_millis(duration_ms / num_slices as u64);
        
        let mut all_fills = Vec::new();
        let mut remaining_quantity = total_quantity;
        
        // Execute slices over time
        for slice_num in 0..num_slices {
            if remaining_quantity <= Decimal::ZERO {
                break;
            }
            
            let current_slice_size = slice_size.min(remaining_quantity);
            
            // Create child order for this slice
            let mut child_order = order.clone();
            child_order.id = Uuid::new_v4();
            child_order.quantity = current_slice_size;
            
            // Execute slice as market order
            let venue = self.select_venue_for_market_order(&child_order).await?;
            let connector = self.get_venue_connector(&venue)?;
            
            match connector.submit_market_order(&child_order).await {
                Ok(venue_order_id) => {
                    // Wait for fills from this slice
                    if let Ok(slice_fills) = timeout(
                        Duration::from_millis(10000), // 10s timeout per slice
                        self.wait_for_fills(venue_order_id)
                    ).await {
                        let filled_quantity: Decimal = slice_fills.iter()
                            .map(|f| f.quantity)
                            .sum();
                        
                        remaining_quantity -= filled_quantity;
                        all_fills.extend(slice_fills);
                        
                        self.metrics.increment_counter("twap_slices_executed", &[]);
                    }
                }
                Err(e) => {
                    warn!("TWAP slice {} failed: {}", slice_num, e);
                }
            }
            
            // Wait for next slice time (except for last slice)
            if slice_num < num_slices - 1 {
                tokio::time::sleep(slice_interval).await;
            }
        }
        
        let (status, avg_price, total_commission) = self.calculate_execution_stats(&all_fills);
        
        Ok(ExecutionResult {
            order_id: order.id,
            status,
            fills: all_fills,
            remaining_quantity,
            average_price: avg_price,
            total_commission,
            execution_time_ms: request.context.created_at.elapsed().as_millis() as u64,
            slippage_bps: None,
        })
    }

    async fn execute_vwap_order(
        &self,
        _request: ExecutionRequest,
        _participation_rate: f64,
    ) -> Result<ExecutionResult> {
        // TODO: Implement VWAP execution algorithm
        Err(AlgoVedaError::Trading("VWAP execution not implemented".to_string()))
    }

    async fn execute_implementation_shortfall(
        &self,
        _request: ExecutionRequest,
        _risk_aversion: f64,
    ) -> Result<ExecutionResult> {
        // TODO: Implement Implementation Shortfall algorithm
        Err(AlgoVedaError::Trading("Implementation Shortfall not implemented".to_string()))
    }

    async fn execute_iceberg_order(
        &self,
        _request: ExecutionRequest,
        _visible_size: Decimal,
        _refresh_rate: f64,
    ) -> Result<ExecutionResult> {
        // TODO: Implement Iceberg execution algorithm
        Err(AlgoVedaError::Trading("Iceberg execution not implemented".to_string()))
    }

    async fn execute_sniper_order(
        &self,
        _request: ExecutionRequest,
        _max_levels: u32,
        _timeout_ms: u64,
    ) -> Result<ExecutionResult> {
        // TODO: Implement Sniper execution algorithm
        Err(AlgoVedaError::Trading("Sniper execution not implemented".to_string()))
    }

    async fn select_venue_for_market_order(&self, order: &Order) -> Result<String> {
        // Simple venue selection - in practice this would be much more sophisticated
        let venues = self.config.get_enabled_venues();
        
        if venues.is_empty() {
            return Err(AlgoVedaError::Trading("No venues available".to_string()));
        }
        
        // Select highest priority venue
        Ok(venues[0].name.clone())
    }

    async fn select_venue_for_limit_order(&self, order: &Order) -> Result<String> {
        // For limit orders, consider liquidity and spreads
        self.select_venue_for_market_order(order).await
    }

    fn get_venue_connector(&self, venue: &str) -> Result<Arc<VenueConnector>> {
        let connectors = self.venue_connectors.read().unwrap();
        connectors.get(venue)
            .cloned()
            .ok_or_else(|| AlgoVedaError::Trading(format!("Venue connector not found: {}", venue)))
    }

    async fn wait_for_fills(&self, venue_order_id: Uuid) -> Vec<Fill> {
        // TODO: Implement fill monitoring
        // This would listen to the venue connector for fill updates
        Vec::new()
    }

    async fn monitor_limit_order(&self, venue_order_id: Uuid) -> Vec<Fill> {
        // TODO: Implement limit order monitoring
        // This would track partial fills and order status updates
        Vec::new()
    }

    fn calculate_execution_stats(&self, fills: &[Fill]) -> (ExecutionStatus, Option<Decimal>, Decimal) {
        if fills.is_empty() {
            return (ExecutionStatus::Pending, None, Decimal::ZERO);
        }
        
        let total_quantity: Decimal = fills.iter().map(|f| f.quantity).sum();
        let total_value: Decimal = fills.iter().map(|f| f.quantity * f.price).sum();
        let total_commission: Decimal = fills.iter().map(|f| f.commission).sum();
        
        let average_price = if total_quantity > Decimal::ZERO {
            Some(total_value / total_quantity)
        } else {
            None
        };
        
        // Determine status based on fills
        let status = if total_quantity > Decimal::ZERO {
            ExecutionStatus::PartiallyFilled // Simplified - would need original order quantity
        } else {
            ExecutionStatus::Pending
        };
        
        (status, average_price, total_commission)
    }

    fn validate_execution_request(&self, order: &Order, context: &ExecutionContext) -> Result<()> {
        // Validate order
        if order.quantity <= Decimal::ZERO {
            return Err(AlgoVedaError::Trading("Order quantity must be positive".to_string()));
        }
        
        // Validate execution constraints
        if context.constraints.max_execution_time_ms == 0 {
            return Err(AlgoVedaError::Trading("Max execution time must be positive".to_string()));
        }
        
        // Validate venues
        if !context.constraints.preferred_venues.is_empty() {
            for venue in &context.constraints.preferred_venues {
                if self.config.get_venue(venue).is_none() {
                    return Err(AlgoVedaError::Trading(
                        format!("Unknown preferred venue: {}", venue)
                    ));
                }
            }
        }
        
        Ok(())
    }

    async fn perform_maintenance(&self) {
        // Clean up completed executions
        let mut executions = self.active_executions.write().unwrap();
        let now = Instant::now();
        
        executions.retain(|_, context| {
            now.duration_since(context.created_at) < Duration::from_secs(3600) // Keep for 1 hour
        });
        
        // Update metrics
        self.metrics.set_gauge("active_executions", executions.len() as f64, &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TradingConfig;

    #[tokio::test]
    async fn test_execution_engine_creation() {
        let config = Arc::new(TradingConfig::default());
        let metrics = Arc::new(MetricsCollector::new_test());
        
        let engine = ExecutionEngine::new(config, metrics).await;
        assert!(engine.is_ok());
    }

    #[test]
    fn test_execution_request_validation() {
        let config = Arc::new(TradingConfig::default());
        let metrics = Arc::new(MetricsCollector::new_test());
        let engine = ExecutionEngine::new(config, metrics).await.unwrap();
        
        let order = Order {
            id: Uuid::new_v4(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::new(100, 0),
            price: Some(Decimal::new(150, 0)),
            order_type: crate::trading::OrderType::Limit,
            status: OrderStatus::New,
            created_at: Instant::now(),
            updated_at: Instant::now(),
        };
        
        let context = ExecutionContext {
            order_id: order.id,
            parent_order_id: None,
            strategy_id: "test".to_string(),
            venue: "test_venue".to_string(),
            execution_algo: ExecutionAlgorithm::Market,
            urgency: ExecutionUrgency::Normal,
            constraints: ExecutionConstraints {
                max_execution_time_ms: 30000,
                max_slippage_bps: 50,
                min_fill_ratio: 0.8,
                preferred_venues: vec![],
                avoid_venues: vec![],
                time_constraints: None,
            },
            created_at: Instant::now(),
        };
        
        assert!(engine.validate_execution_request(&order, &context).is_ok());
    }
}
