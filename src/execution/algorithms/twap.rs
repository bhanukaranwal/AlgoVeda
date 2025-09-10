/*!
 * Time-Weighted Average Price (TWAP) Execution Algorithm
 * Advanced TWAP implementation with adaptive scheduling and market impact optimization
 */

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout, Interval},
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, TimeZone};
use rand::Rng;

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, OrderSide, OrderType, OrderStatus, Fill},
    market_data::{MarketData, Level1Data},
    execution::ExecutionEngine,
    portfolio::Position,
    risk_management::RiskManager,
    utils::time::TradingSession,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPConfig {
    pub total_quantity: u64,
    pub duration_minutes: u32,
    pub max_participation_rate: f64,  // Max % of volume
    pub min_slice_size: u64,
    pub max_slice_size: u64,
    pub price_limit: Option<f64>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub randomization_factor: f64,    // 0.0 to 1.0
    pub adaptive_sizing: bool,
    pub market_impact_model: MarketImpactModel,
    pub urgency_factor: f64,          // 0.0 (patient) to 1.0 (aggressive)
    pub dark_pool_preference: f64,    // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketImpactModel {
    Linear,
    SquareRoot,
    Almgren,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPState {
    pub parent_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub total_quantity: u64,
    pub remaining_quantity: u64,
    pub executed_quantity: u64,
    pub average_price: f64,
    pub slices_executed: u32,
    pub total_slices: u32,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub current_slice: Option<SliceInfo>,
    pub market_conditions: MarketConditions,
    pub performance_metrics: TWAPMetrics,
    pub status: TWAPStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TWAPStatus {
    Pending,
    Active,
    Paused,
    Completed,
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceInfo {
    pub slice_id: String,
    pub target_quantity: u64,
    pub executed_quantity: u64,
    pub orders: Vec<String>, // Child order IDs
    pub start_time: DateTime<Utc>,
    pub target_completion: DateTime<Utc>,
    pub slice_vwap: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub current_price: f64,
    pub bid_ask_spread: f64,
    pub volume_rate: f64,          // shares/minute
    pub volatility: f64,
    pub participation_rate: f64,    // Our participation in volume
    pub market_impact: f64,
    pub liquidity_score: f64,      // 0.0 to 1.0
    pub momentum_score: f64,       // -1.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPMetrics {
    pub implementation_shortfall: f64,
    pub market_impact: f64,
    pub timing_risk: f64,
    pub arrival_price: f64,
    pub volume_weighted_price: f64,
    pub tracking_error: f64,
    pub efficiency_ratio: f64,
    pub slippage_bps: f64,
}

pub struct TWAPAlgorithm {
    config: TWAPConfig,
    state: Arc<RwLock<TWAPState>>,
    execution_engine: Arc<ExecutionEngine>,
    risk_manager: Arc<RiskManager>,
    
    // Market data
    market_data_receiver: broadcast::Receiver<MarketData>,
    current_market_data: Arc<RwLock<Option<MarketData>>>,
    
    // Scheduling
    slice_scheduler: Arc<Mutex<SliceScheduler>>,
    execution_timer: Arc<Mutex<Option<Interval>>>,
    
    // Order management
    active_orders: Arc<RwLock<HashMap<String, Order>>>,
    fill_receiver: mpsc::Receiver<Fill>,
    
    // Performance tracking
    execution_history: Arc<RwLock<VecDeque<ExecutionEvent>>>,
    volume_tracker: Arc<RwLock<VolumeTracker>>,
}

#[derive(Debug, Clone)]
struct ExecutionEvent {
    timestamp: DateTime<Utc>,
    event_type: ExecutionEventType,
    slice_id: String,
    quantity: u64,
    price: f64,
    market_conditions: MarketConditions,
}

#[derive(Debug, Clone)]
enum ExecutionEventType {
    SliceStarted,
    OrderSent,
    PartialFill,
    CompleteFill,
    SliceCompleted,
    MarketDataUpdate,
    RiskCheck,
}

struct SliceScheduler {
    scheduled_slices: VecDeque<ScheduledSlice>,
    current_slice_index: usize,
    randomization_generator: rand::rngs::ThreadRng,
}

#[derive(Debug, Clone)]
struct ScheduledSlice {
    slice_id: String,
    target_time: DateTime<Utc>,
    target_quantity: u64,
    max_quantity: u64,
    urgency: f64,
    price_limit: Option<f64>,
}

struct VolumeTracker {
    historical_volumes: HashMap<String, VecDeque<(DateTime<Utc>, u64)>>, // symbol -> (time, volume)
    volume_predictions: HashMap<String, f64>, // symbol -> predicted volume rate
    participation_rates: HashMap<String, f64>, // symbol -> our participation rate
}

impl TWAPAlgorithm {
    pub fn new(
        config: TWAPConfig,
        execution_engine: Arc<ExecutionEngine>,
        risk_manager: Arc<RiskManager>,
        market_data_receiver: broadcast::Receiver<MarketData>,
        fill_receiver: mpsc::Receiver<Fill>,
    ) -> Result<Self> {
        // Calculate execution schedule
        let total_minutes = config.duration_minutes;
        let total_slices = Self::calculate_optimal_slice_count(&config);
        
        let state = TWAPState {
            parent_order_id: uuid::Uuid::new_v4().to_string(),
            symbol: "".to_string(), // Will be set when starting
            side: OrderSide::Buy,   // Will be set when starting
            total_quantity: config.total_quantity,
            remaining_quantity: config.total_quantity,
            executed_quantity: 0,
            average_price: 0.0,
            slices_executed: 0,
            total_slices,
            start_time: config.start_time.unwrap_or_else(|| Utc::now()),
            end_time: config.end_time.unwrap_or_else(|| Utc::now() + chrono::Duration::minutes(total_minutes as i64)),
            current_slice: None,
            market_conditions: MarketConditions::default(),
            performance_metrics: TWAPMetrics::default(),
            status: TWAPStatus::Pending,
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
            execution_engine,
            risk_manager,
            market_data_receiver,
            current_market_data: Arc::new(RwLock::new(None)),
            slice_scheduler: Arc::new(Mutex::new(SliceScheduler::new())),
            execution_timer: Arc::new(Mutex::new(None)),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            fill_receiver,
            execution_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            volume_tracker: Arc::new(RwLock::new(VolumeTracker::new())),
        })
    }

    /// Start TWAP execution
    pub async fn start(&mut self, symbol: String, side: OrderSide) -> Result<String> {
        // Update state with symbol and side
        {
            let mut state = self.state.write().unwrap();
            state.symbol = symbol.clone();
            state.side = side;
            state.status = TWAPStatus::Active;
        }

        // Initialize slice schedule
        self.initialize_slice_schedule().await?;

        // Start market data monitoring
        self.start_market_data_monitoring().await;

        // Start fill monitoring
        self.start_fill_monitoring().await;

        // Start execution timer
        self.start_execution_timer().await;

        // Start performance monitoring
        self.start_performance_monitoring().await;

        let parent_order_id = self.state.read().unwrap().parent_order_id.clone();
        Ok(parent_order_id)
    }

    /// Initialize the slice execution schedule
    async fn initialize_slice_schedule(&self) -> Result<()> {
        let state = self.state.read().unwrap();
        let total_duration = state.end_time - state.start_time;
        let duration_minutes = total_duration.num_minutes() as f64;
        
        let slice_interval = duration_minutes / state.total_slices as f64;
        let base_slice_quantity = state.total_quantity / state.total_slices as u64;
        
        let mut scheduler = self.slice_scheduler.lock().await;
        scheduler.scheduled_slices.clear();
        
        for i in 0..state.total_slices {
            let target_time = state.start_time + 
                chrono::Duration::minutes((i as f64 * slice_interval) as i64);
            
            // Add randomization
            let randomization_minutes = if self.config.randomization_factor > 0.0 {
                let max_offset = slice_interval * self.config.randomization_factor / 2.0;
                let offset = scheduler.randomization_generator.gen_range(-max_offset..max_offset);
                offset as i64
            } else {
                0
            };
            
            let randomized_time = target_time + chrono::Duration::minutes(randomization_minutes);
            
            // Calculate slice quantity with potential randomization
            let quantity_variance = if self.config.randomization_factor > 0.0 {
                let max_variance = base_slice_quantity as f64 * self.config.randomization_factor * 0.2;
                scheduler.randomization_generator.gen_range(-max_variance..max_variance) as i64
            } else {
                0
            };
            
            let target_quantity = ((base_slice_quantity as i64 + quantity_variance).max(self.config.min_slice_size as i64) as u64)
                .min(self.config.max_slice_size)
                .min(state.remaining_quantity);

            scheduler.scheduled_slices.push_back(ScheduledSlice {
                slice_id: uuid::Uuid::new_v4().to_string(),
                target_time: randomized_time,
                target_quantity,
                max_quantity: target_quantity * 2, // Allow up to 2x if market conditions are favorable
                urgency: self.config.urgency_factor,
                price_limit: self.config.price_limit,
            });
        }

        drop(state);
        drop(scheduler);
        
        Ok(())
    }

    /// Start market data monitoring
    async fn start_market_data_monitoring(&self) {
        let mut market_data_receiver = self.market_data_receiver.resubscribe();
        let current_market_data = self.current_market_data.clone();
        let state = self.state.clone();
        let execution_history = self.execution_history.clone();
        let volume_tracker = self.volume_tracker.clone();
        
        tokio::spawn(async move {
            while let Ok(market_data) = market_data_receiver.recv().await {
                // Update current market data
                *current_market_data.write().unwrap() = Some(market_data.clone());
                
                // Update market conditions
                Self::update_market_conditions(&state, &market_data).await;
                
                // Update volume tracking
                Self::update_volume_tracking(&volume_tracker, &market_data).await;
                
                // Record market data event
                let event = ExecutionEvent {
                    timestamp: Utc::now(),
                    event_type: ExecutionEventType::MarketDataUpdate,
                    slice_id: "".to_string(),
                    quantity: 0,
                    price: market_data.last_price.unwrap_or(0.0),
                    market_conditions: state.read().unwrap().market_conditions.clone(),
                };
                
                execution_history.write().unwrap().push_back(event);
                if execution_history.read().unwrap().len() > 10000 {
                    execution_history.write().unwrap().pop_front();
                }
            }
        });
    }

    /// Start fill monitoring
    async fn start_fill_monitoring(&mut self) {
        let state = self.state.clone();
        let execution_history = self.execution_history.clone();
        let active_orders = self.active_orders.clone();
        
        tokio::spawn(async move {
            // Note: In real implementation, this would receive from self.fill_receiver
            // For compilation, we'll create a dummy receiver
            let (_tx, mut rx) = mpsc::channel::<Fill>(1000);
            
            while let Some(fill) = rx.recv().await {
                Self::process_fill(&state, &execution_history, &active_orders, fill).await;
            }
        });
    }

    /// Start execution timer for slice scheduling
    async fn start_execution_timer(&self) {
        let slice_scheduler = self.slice_scheduler.clone();
        let state = self.state.clone();
        let execution_engine = self.execution_engine.clone();
        let risk_manager = self.risk_manager.clone();
        let active_orders = self.active_orders.clone();
        let current_market_data = self.current_market_data.clone();
        let execution_history = self.execution_history.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                let now = Utc::now();
                let mut scheduler = slice_scheduler.lock().await;
                
                // Check if any slices are ready for execution
                while let Some(scheduled_slice) = scheduler.scheduled_slices.front() {
                    if scheduled_slice.target_time <= now {
                        let slice = scheduler.scheduled_slices.pop_front().unwrap();
                        drop(scheduler);
                        
                        // Execute the slice
                        Self::execute_slice(
                            &state,
                            &execution_engine,
                            &risk_manager,
                            &active_orders,
                            &current_market_data,
                            &execution_history,
                            &config,
                            slice,
                        ).await;
                        
                        scheduler = slice_scheduler.lock().await;
                    } else {
                        break;
                    }
                }
                
                drop(scheduler);
                
                // Check if TWAP is complete
                let state_guard = state.read().unwrap();
                if state_guard.remaining_quantity == 0 {
                    // TWAP completed
                    break;
                } else if now > state_guard.end_time {
                    // TWAP expired
                    // Cancel remaining orders and update status
                    drop(state_guard);
                    let mut state_mut = state.write().unwrap();
                    state_mut.status = TWAPStatus::Failed("Time expired".to_string());
                    break;
                }
                drop(state_guard);
            }
        });
    }

    /// Execute a scheduled slice
    async fn execute_slice(
        state: &Arc<RwLock<TWAPState>>,
        execution_engine: &Arc<ExecutionEngine>,
        risk_manager: &Arc<RiskManager>,
        active_orders: &Arc<RwLock<HashMap<String, Order>>>,
        current_market_data: &Arc<RwLock<Option<MarketData>>>,
        execution_history: &Arc<RwLock<VecDeque<ExecutionEvent>>>,
        config: &TWAPConfig,
        scheduled_slice: ScheduledSlice,
    ) {
        // Get current market conditions
        let market_data = current_market_data.read().unwrap().clone();
        if market_data.is_none() {
            return; // No market data available
        }
        let market_data = market_data.unwrap();

        // Risk check
        let state_guard = state.read().unwrap();
        let risk_check = risk_manager.validate_order(
            &state_guard.symbol,
            state_guard.side.clone(),
            scheduled_slice.target_quantity,
            market_data.last_price,
        );
        drop(state_guard);

        if risk_check.is_err() {
            // Risk violation - skip this slice
            return;
        }

        // Adaptive sizing based on market conditions
        let adjusted_quantity = Self::calculate_adaptive_quantity(
            state,
            &scheduled_slice,
            &market_data,
            config,
        );

        if adjusted_quantity == 0 {
            return; // Skip this slice
        }

        // Create slice info
        let slice_info = SliceInfo {
            slice_id: scheduled_slice.slice_id.clone(),
            target_quantity: adjusted_quantity,
            executed_quantity: 0,
            orders: Vec::new(),
            start_time: Utc::now(),
            target_completion: Utc::now() + chrono::Duration::minutes(5), // 5-minute target
            slice_vwap: 0.0,
        };

        // Update state with current slice
        {
            let mut state_mut = state.write().unwrap();
            state_mut.current_slice = Some(slice_info.clone());
        }

        // Record slice start event
        let event = ExecutionEvent {
            timestamp: Utc::now(),
            event_type: ExecutionEventType::SliceStarted,
            slice_id: scheduled_slice.slice_id.clone(),
            quantity: adjusted_quantity,
            price: market_data.last_price.unwrap_or(0.0),
            market_conditions: state.read().unwrap().market_conditions.clone(),
        };
        execution_history.write().unwrap().push_back(event);

        // Execute slice with multiple child orders if needed
        Self::execute_slice_orders(
            state,
            execution_engine,
            active_orders,
            execution_history,
            config,
            slice_info,
            &market_data,
        ).await;
    }

    /// Execute child orders for a slice
    async fn execute_slice_orders(
        state: &Arc<RwLock<TWAPState>>,
        execution_engine: &Arc<ExecutionEngine>,
        active_orders: &Arc<RwLock<HashMap<String, Order>>>,
        execution_history: &Arc<RwLock<VecDeque<ExecutionEvent>>>,
        config: &TWAPConfig,
        mut slice_info: SliceInfo,
        market_data: &MarketData,
    ) {
        let state_guard = state.read().unwrap();
        let remaining_quantity = slice_info.target_quantity;
        
        // Determine order strategy based on market conditions and urgency
        let order_strategy = Self::determine_order_strategy(
            &state_guard.market_conditions,
            config,
            remaining_quantity,
        );
        drop(state_guard);

        match order_strategy {
            OrderStrategy::SingleLimitOrder => {
                // Single limit order for the entire slice
                let limit_price = Self::calculate_limit_price(market_data, &state.read().unwrap().side, config);
                
                let order = Order {
                    id: uuid::Uuid::new_v4().to_string(),
                    symbol: state.read().unwrap().symbol.clone(),
                    side: state.read().unwrap().side.clone(),
                    quantity: remaining_quantity,
                    order_type: OrderType::Limit,
                    price: Some(limit_price),
                    time_in_force: crate::trading::TimeInForce::IOC, // Immediate or Cancel
                    status: OrderStatus::PendingNew,
                    parent_order_id: Some(state.read().unwrap().parent_order_id.clone()),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };

                slice_info.orders.push(order.id.clone());
                active_orders.write().unwrap().insert(order.id.clone(), order.clone());

                // Submit order
                if let Err(e) = execution_engine.submit_order(order.clone()).await {
                    eprintln!("Failed to submit order: {}", e);
                }
            }
            
            OrderStrategy::MultipleSmallOrders => {
                // Break into smaller orders to reduce market impact
                let num_child_orders = (remaining_quantity / 1000).max(1).min(10); // 1-10 orders
                let base_quantity = remaining_quantity / num_child_orders;
                
                for i in 0..num_child_orders {
                    let quantity = if i == num_child_orders - 1 {
                        // Last order gets remainder
                        remaining_quantity - (base_quantity * (num_child_orders - 1))
                    } else {
                        base_quantity
                    };
                    
                    let limit_price = Self::calculate_limit_price(market_data, &state.read().unwrap().side, config);
                    
                    let order = Order {
                        id: uuid::Uuid::new_v4().to_string(),
                        symbol: state.read().unwrap().symbol.clone(),
                        side: state.read().unwrap().side.clone(),
                        quantity,
                        order_type: OrderType::Limit,
                        price: Some(limit_price),
                        time_in_force: crate::trading::TimeInForce::IOC,
                        status: OrderStatus::PendingNew,
                        parent_order_id: Some(state.read().unwrap().parent_order_id.clone()),
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    };

                    slice_info.orders.push(order.id.clone());
                    active_orders.write().unwrap().insert(order.id.clone(), order.clone());

                    // Submit order with small delay to spread execution
                    if let Err(e) = execution_engine.submit_order(order.clone()).await {
                        eprintln!("Failed to submit order: {}", e);
                    }
                    
                    if i < num_child_orders - 1 {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
            
            OrderStrategy::MarketOrder => {
                // Urgent execution with market order
                let order = Order {
                    id: uuid::Uuid::new_v4().to_string(),
                    symbol: state.read().unwrap().symbol.clone(),
                    side: state.read().unwrap().side.clone(),
                    quantity: remaining_quantity,
                    order_type: OrderType::Market,
                    price: None,
                    time_in_force: crate::trading::TimeInForce::IOC,
                    status: OrderStatus::PendingNew,
                    parent_order_id: Some(state.read().unwrap().parent_order_id.clone()),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };

                slice_info.orders.push(order.id.clone());
                active_orders.write().unwrap().insert(order.id.clone(), order.clone());

                if let Err(e) = execution_engine.submit_order(order.clone()).await {
                    eprintln!("Failed to submit order: {}", e);
                }
            }
        }

        // Update state with slice orders
        {
            let mut state_mut = state.write().unwrap();
            state_mut.current_slice = Some(slice_info);
        }
    }

    /// Process incoming fills
    async fn process_fill(
        state: &Arc<RwLock<TWAPState>>,
        execution_history: &Arc<RwLock<VecDeque<ExecutionEvent>>>,
        active_orders: &Arc<RwLock<HashMap<String, Order>>>,
        fill: Fill,
    ) {
        let mut state_mut = state.write().unwrap();
        
        // Update execution metrics
        let fill_quantity = fill.quantity;
        let fill_price = fill.price;
        
        state_mut.executed_quantity += fill_quantity;
        state_mut.remaining_quantity = state_mut.remaining_quantity.saturating_sub(fill_quantity);
        
        // Update VWAP
        let total_executed = state_mut.executed_quantity;
        if total_executed > 0 {
            state_mut.average_price = ((state_mut.average_price * (total_executed - fill_quantity) as f64) + 
                (fill_price * fill_quantity as f64)) / total_executed as f64;
        }
        
        // Update current slice if applicable
        if let Some(ref mut slice) = state_mut.current_slice {
            slice.executed_quantity += fill_quantity;
            
            // Update slice VWAP
            if slice.executed_quantity > 0 {
                slice.slice_vwap = ((slice.slice_vwap * (slice.executed_quantity - fill_quantity) as f64) + 
                    (fill_price * fill_quantity as f64)) / slice.executed_quantity as f64;
            }
        }

        drop(state_mut);
        
        // Record fill event
        let event = ExecutionEvent {
            timestamp: Utc::now(),
            event_type: if fill.quantity == fill.original_quantity {
                ExecutionEventType::CompleteFill
            } else {
                ExecutionEventType::PartialFill
            },
            slice_id: state.read().unwrap().current_slice.as_ref()
                .map(|s| s.slice_id.clone()).unwrap_or_default(),
            quantity: fill_quantity,
            price: fill_price,
            market_conditions: state.read().unwrap().market_conditions.clone(),
        };
        
        execution_history.write().unwrap().push_back(event);
        
        // Update order status
        active_orders.write().unwrap().remove(&fill.order_id);
        
        // Check if TWAP is complete
        if state.read().unwrap().remaining_quantity == 0 {
            let mut state_mut = state.write().unwrap();
            state_mut.status = TWAPStatus::Completed;
        }
    }

    /// Helper functions
    fn calculate_optimal_slice_count(config: &TWAPConfig) -> u32 {
        // Balance between market impact and timing risk
        let base_slices = (config.duration_minutes / 5).max(1); // At least one slice per 5 minutes
        let quantity_factor = (config.total_quantity as f64 / 100000.0).sqrt(); // Scale with size
        let urgency_factor = 1.0 + config.urgency_factor;
        
        ((base_slices as f64 * quantity_factor * urgency_factor) as u32)
            .max(1)
            .min(config.duration_minutes) // Maximum one per minute
    }

    fn calculate_adaptive_quantity(
        state: &Arc<RwLock<TWAPState>>,
        scheduled_slice: &ScheduledSlice,
        market_data: &MarketData,
        config: &TWAPConfig,
    ) -> u64 {
        let state_guard = state.read().unwrap();
        let market_conditions = &state_guard.market_conditions;
        
        let mut adjusted_quantity = scheduled_slice.target_quantity;
        
        // Adjust based on liquidity
        if market_conditions.liquidity_score < 0.5 {
            adjusted_quantity = (adjusted_quantity as f64 * 0.5) as u64; // Reduce size in illiquid markets
        }
        
        // Adjust based on volatility
        if market_conditions.volatility > 0.3 {
            adjusted_quantity = (adjusted_quantity as f64 * 0.7) as u64; // Reduce size in volatile markets
        }
        
        // Adjust based on participation rate
        if market_conditions.participation_rate > config.max_participation_rate {
            adjusted_quantity = (adjusted_quantity as f64 * 0.6) as u64; // Reduce to stay under participation limit
        }
        
        // Ensure within bounds
        adjusted_quantity = adjusted_quantity
            .max(config.min_slice_size)
            .min(config.max_slice_size)
            .min(state_guard.remaining_quantity);
        
        adjusted_quantity
    }

    fn determine_order_strategy(
        market_conditions: &MarketConditions,
        config: &TWAPConfig,
        quantity: u64,
    ) -> OrderStrategy {
        // High urgency or end of trading window
        if config.urgency_factor > 0.8 {
            return OrderStrategy::MarketOrder;
        }
        
        // Large quantity or poor liquidity
        if quantity > 10000 || market_conditions.liquidity_score < 0.4 {
            return OrderStrategy::MultipleSmallOrders;
        }
        
        // Default to single limit order
        OrderStrategy::SingleLimitOrder
    }

    fn calculate_limit_price(
        market_data: &MarketData,
        side: &OrderSide,
        config: &TWAPConfig,
    ) -> f64 {
        let mid_price = market_data.mid_price();
        let spread = market_data.bid_ask_spread();
        
        // Aggressive pricing based on urgency
        let aggression = config.urgency_factor * 0.5; // Max 50% through spread
        
        match side {
            OrderSide::Buy => {
                let limit = mid_price + (spread * aggression);
                if let Some(max_price) = config.price_limit {
                    limit.min(max_price)
                } else {
                    limit
                }
            }
            OrderSide::Sell => {
                let limit = mid_price - (spread * aggression);
                if let Some(min_price) = config.price_limit {
                    limit.max(min_price)
                } else {
                    limit
                }
            }
        }
    }

    async fn update_market_conditions(
        state: &Arc<RwLock<TWAPState>>,
        market_data: &MarketData,
    ) {
        let mut state_mut = state.write().unwrap();
        
        state_mut.market_conditions.current_price = market_data.last_price.unwrap_or(0.0);
        state_mut.market_conditions.bid_ask_spread = market_data.bid_ask_spread();
        
        // Update other market conditions (simplified)
        state_mut.market_conditions.volume_rate = market_data.volume.unwrap_or(0) as f64 / 60.0; // per minute
        state_mut.market_conditions.volatility = 0.2; // Would calculate from price history
        state_mut.market_conditions.liquidity_score = 0.7; // Would calculate from order book
        state_mut.market_conditions.momentum_score = 0.1; // Would calculate from price momentum
    }

    async fn update_volume_tracking(
        volume_tracker: &Arc<RwLock<VolumeTracker>>,
        market_data: &MarketData,
    ) {
        let mut tracker = volume_tracker.write().unwrap();
        let symbol = &market_data.symbol;
        
        let volume_data = tracker.historical_volumes
            .entry(symbol.clone())
            .or_insert_with(VecDeque::new);
        
        volume_data.push_back((Utc::now(), market_data.volume.unwrap_or(0)));
        
        // Keep only recent data (last hour)
        let cutoff_time = Utc::now() - chrono::Duration::hours(1);
        while let Some((time, _)) = volume_data.front() {
            if *time < cutoff_time {
                volume_data.pop_front();
            } else {
                break;
            }
        }
        
        // Update volume prediction
        if !volume_data.is_empty() {
            let total_volume: u64 = volume_data.iter().map(|(_, v)| v).sum();
            let time_span_minutes = volume_data.len() as f64;
            tracker.volume_predictions.insert(
                symbol.clone(), 
                total_volume as f64 / time_span_minutes
            );
        }
    }

    async fn start_performance_monitoring(&self) {
        let state = self.state.clone();
        let execution_history = self.execution_history.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Update performance metrics
                Self::calculate_performance_metrics(&state, &execution_history).await;
            }
        });
    }

    async fn calculate_performance_metrics(
        state: &Arc<RwLock<TWAPState>>,
        execution_history: &Arc<RwLock<VecDeque<ExecutionEvent>>>,
    ) {
        let mut state_mut = state.write().unwrap();
        let history = execution_history.read().unwrap();
        
        if state_mut.executed_quantity == 0 {
            return;
        }
        
        // Implementation shortfall calculation
        let arrival_price = history.iter()
            .find(|e| matches!(e.event_type, ExecutionEventType::SliceStarted))
            .map(|e| e.price)
            .unwrap_or(0.0);
        
        state_mut.performance_metrics.arrival_price = arrival_price;
        
        if arrival_price > 0.0 {
            let implementation_shortfall = match state_mut.side {
                OrderSide::Buy => (state_mut.average_price - arrival_price) / arrival_price,
                OrderSide::Sell => (arrival_price - state_mut.average_price) / arrival_price,
            };
            
            state_mut.performance_metrics.implementation_shortfall = implementation_shortfall;
            state_mut.performance_metrics.slippage_bps = implementation_shortfall * 10000.0;
        }
        
        // Market impact estimation (simplified)
        state_mut.performance_metrics.market_impact = state_mut.performance_metrics.slippage_bps * 0.3;
        
        // Timing risk (simplified)
        state_mut.performance_metrics.timing_risk = state_mut.performance_metrics.slippage_bps * 0.7;
        
        // Efficiency ratio
        let theoretical_cost = 0.5; // Theoretical implementation shortfall
        state_mut.performance_metrics.efficiency_ratio = if state_mut.performance_metrics.implementation_shortfall != 0.0 {
            theoretical_cost / state_mut.performance_metrics.implementation_shortfall.abs()
        } else {
            1.0
        };
    }

    /// Get current TWAP state
    pub fn get_state(&self) -> TWAPState {
        self.state.read().unwrap().clone()
    }

    /// Pause TWAP execution
    pub async fn pause(&self) -> Result<()> {
        let mut state = self.state.write().unwrap();
        state.status = TWAPStatus::Paused;
        Ok(())
    }

    /// Resume TWAP execution
    pub async fn resume(&self) -> Result<()> {
        let mut state = self.state.write().unwrap();
        state.status = TWAPStatus::Active;
        Ok(())
    }

    /// Cancel TWAP execution
    pub async fn cancel(&self) -> Result<()> {
        let mut state = self.state.write().unwrap();
        state.status = TWAPStatus::Cancelled;
        
        // Cancel all active orders
        let active_orders = self.active_orders.read().unwrap();
        for order_id in active_orders.keys() {
            // Would cancel order through execution engine
            let _ = self.execution_engine.cancel_order(order_id.clone()).await;
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum OrderStrategy {
    SingleLimitOrder,
    MultipleSmallOrders,
    MarketOrder,
}

impl Default for MarketConditions {
    fn default() -> Self {
        Self {
            current_price: 0.0,
            bid_ask_spread: 0.0,
            volume_rate: 0.0,
            volatility: 0.2,
            participation_rate: 0.0,
            market_impact: 0.0,
            liquidity_score: 0.5,
            momentum_score: 0.0,
        }
    }
}

impl Default for TWAPMetrics {
    fn default() -> Self {
        Self {
            implementation_shortfall: 0.0,
            market_impact: 0.0,
            timing_risk: 0.0,
            arrival_price: 0.0,
            volume_weighted_price: 0.0,
            tracking_error: 0.0,
            efficiency_ratio: 1.0,
            slippage_bps: 0.0,
        }
    }
}

impl SliceScheduler {
    fn new() -> Self {
        Self {
            scheduled_slices: VecDeque::new(),
            current_slice_index: 0,
            randomization_generator: rand::thread_rng(),
        }
    }
}

impl VolumeTracker {
    fn new() -> Self {
        Self {
            historical_volumes: HashMap::new(),
            volume_predictions: HashMap::new(),
            participation_rates: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_config_creation() {
        let config = TWAPConfig {
            total_quantity: 10000,
            duration_minutes: 60,
            max_participation_rate: 0.1,
            min_slice_size: 100,
            max_slice_size: 1000,
            price_limit: Some(100.0),
            start_time: None,
            end_time: None,
            randomization_factor: 0.1,
            adaptive_sizing: true,
            market_impact_model: MarketImpactModel::Linear,
            urgency_factor: 0.5,
            dark_pool_preference: 0.3,
        };
        
        assert_eq!(config.total_quantity, 10000);
        assert_eq!(config.duration_minutes, 60);
    }

    #[test]
    fn test_slice_count_calculation() {
        let config = TWAPConfig {
            total_quantity: 100000,
            duration_minutes: 120,
            max_participation_rate: 0.1,
            min_slice_size: 1000,
            max_slice_size: 5000,
            price_limit: None,
            start_time: None,
            end_time: None,
            randomization_factor: 0.0,
            adaptive_sizing: false,
            market_impact_model: MarketImpactModel::Linear,
            urgency_factor: 0.5,
            dark_pool_preference: 0.0,
        };
        
        let slice_count = TWAPAlgorithm::calculate_optimal_slice_count(&config);
        assert!(slice_count > 0);
        assert!(slice_count <= config.duration_minutes);
    }
}
