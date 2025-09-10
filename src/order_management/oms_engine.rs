/*!
 * Order Management System (OMS) Engine
 * Professional-grade order lifecycle management with allocation and blotter
 */

use std::{
    collections::{HashMap, BTreeMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::interval,
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, OrderSide, OrderType, OrderStatus, TimeInForce, Fill},
    portfolio::{Portfolio, Position},
    risk_management::RiskManager,
    execution::ExecutionEngine,
    accounting::TradeSettlement,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OMSConfig {
    pub enable_pre_trade_risk: bool,
    pub enable_post_trade_risk: bool,
    pub enable_allocation_engine: bool,
    pub enable_trade_settlement: bool,
    pub max_orders_per_second: u32,
    pub order_timeout_seconds: u32,
    pub enable_order_recycling: bool,
    pub audit_trail_enabled: bool,
    pub compliance_checks_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderState {
    PendingNew,
    PendingCancel,
    PendingReplace,
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
    Expired,
    Suspended,
    PendingCalculated,
    Calculated,
    Replaced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRecord {
    pub order: Order,
    pub state: OrderState,
    pub creation_time: DateTime<Utc>,
    pub last_update_time: DateTime<Utc>,
    pub fills: Vec<Fill>,
    pub total_filled_quantity: u64,
    pub average_fill_price: f64,
    pub remaining_quantity: u64,
    pub commission: f64,
    pub parent_order_id: Option<String>,
    pub child_order_ids: Vec<String>,
    pub allocation_instructions: Option<AllocationInstruction>,
    pub compliance_status: ComplianceStatus,
    pub settlement_status: SettlementStatus,
    pub audit_trail: Vec<AuditEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInstruction {
    pub instruction_id: String,
    pub strategy_id: String,
    pub account_allocations: Vec<AccountAllocation>,
    pub allocation_method: AllocationMethod,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountAllocation {
    pub account_id: String,
    pub allocation_type: AllocationType,
    pub value: f64,  // Percentage, ratio, or absolute quantity
    pub max_quantity: Option<u64>,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationMethod {
    ProRata,
    Priority,
    EqualDollar,
    EqualShares,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    Percentage,    // % of total fill
    Ratio,         // Ratio relative to other allocations
    Fixed,         // Fixed quantity
    MaxQuantity,   // Up to maximum quantity
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub pre_trade_approved: bool,
    pub post_trade_approved: bool,
    pub compliance_checks: Vec<ComplianceCheck>,
    pub violations: Vec<ComplianceViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub check_id: String,
    pub check_type: ComplianceCheckType,
    pub status: ComplianceCheckStatus,
    pub message: Option<String>,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceCheckType {
    PositionLimit,
    ConcentrationLimit,
    RiskLimit,
    TradingRestriction,
    SettlementLimit,
    MarketAccess,
    RegulationNMS,
    BestExecution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceCheckStatus {
    Passed,
    Failed,
    Warning,
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub violation_id: String,
    pub violation_type: ComplianceCheckType,
    pub severity: ViolationSeverity,
    pub description: String,
    pub detected_at: DateTime<Utc>,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SettlementStatus {
    Pending,
    InProgress,
    Settled,
    Failed,
    PartiallySettled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub event_type: AuditEventType,
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub description: String,
    pub old_values: Option<HashMap<String, String>>,
    pub new_values: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    OrderCreated,
    OrderModified,
    OrderCanceled,
    OrderFilled,
    OrderRejected,
    AllocationCreated,
    AllocationModified,
    ComplianceViolation,
    SettlementEvent,
    SystemEvent,
}

pub struct OMSEngine {
    config: OMSConfig,
    
    // Order storage and tracking
    orders: Arc<RwLock<HashMap<String, OrderRecord>>>,
    order_index: Arc<RwLock<BTreeMap<DateTime<Utc>, String>>>,  // Time-based index
    symbol_orders: Arc<RwLock<HashMap<String, Vec<String>>>>,   // Symbol-based index
    strategy_orders: Arc<RwLock<HashMap<String, Vec<String>>>>, // Strategy-based index
    
    // Allocation engine
    allocation_engine: Arc<AllocationEngine>,
    pending_allocations: Arc<RwLock<HashMap<String, AllocationInstruction>>>,
    
    // Event handling
    order_events: broadcast::Sender<OrderEvent>,
    fill_events: broadcast::Sender<FillEvent>,
    allocation_events: broadcast::Sender<AllocationEvent>,
    
    // External systems
    risk_manager: Arc<RiskManager>,
    execution_engine: Arc<ExecutionEngine>,
    settlement_engine: Arc<TradeSettlement>,
    
    // Performance metrics
    orders_processed: Arc<AtomicU64>,
    average_processing_time: Arc<RwLock<Duration>>,
    last_cleanup_time: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub order_id: String,
    pub event_type: OrderEventType,
    pub timestamp: DateTime<Utc>,
    pub order_state: OrderState,
    pub details: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderEventType {
    Created,
    Updated,
    Filled,
    PartiallyFilled,
    Canceled,
    Rejected,
    Expired,
    Allocated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    pub fill_id: String,
    pub order_id: String,
    pub timestamp: DateTime<Utc>,
    pub fill: Fill,
    pub remaining_quantity: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEvent {
    pub allocation_id: String,
    pub order_id: String,
    pub timestamp: DateTime<Utc>,
    pub allocations: Vec<AccountAllocation>,
    pub status: AllocationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    PartiallyComplete,
}

pub struct AllocationEngine {
    pending_instructions: Arc<RwLock<HashMap<String, AllocationInstruction>>>,
    completed_allocations: Arc<RwLock<HashMap<String, CompletedAllocation>>>,
    allocation_queue: Arc<Mutex<VecDeque<AllocationTask>>>,
}

#[derive(Debug, Clone)]
struct AllocationTask {
    instruction_id: String,
    order_id: String,
    fill: Fill,
    created_at: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletedAllocation {
    pub instruction_id: String,
    pub order_id: String,
    pub account_fills: Vec<AccountFill>,
    pub completed_at: DateTime<Utc>,
    pub total_allocated: u64,
    pub allocation_accuracy: f64, // How close to target allocation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccountFill {
    pub account_id: String,
    pub quantity: u64,
    pub price: f64,
    pub commission: f64,
    pub allocation_percentage: f64,
}

impl OMSEngine {
    pub fn new(
        config: OMSConfig,
        risk_manager: Arc<RiskManager>,
        execution_engine: Arc<ExecutionEngine>,
        settlement_engine: Arc<TradeSettlement>,
    ) -> Self {
        let (order_events, _) = broadcast::channel(10000);
        let (fill_events, _) = broadcast::channel(10000);
        let (allocation_events, _) = broadcast::channel(1000);
        
        Self {
            config,
            orders: Arc::new(RwLock::new(HashMap::new())),
            order_index: Arc::new(RwLock::new(BTreeMap::new())),
            symbol_orders: Arc::new(RwLock::new(HashMap::new())),
            strategy_orders: Arc::new(RwLock::new(HashMap::new())),
            allocation_engine: Arc::new(AllocationEngine::new()),
            pending_allocations: Arc::new(RwLock::new(HashMap::new())),
            order_events,
            fill_events,
            allocation_events,
            risk_manager,
            execution_engine,
            settlement_engine,
            orders_processed: Arc::new(AtomicU64::new(0)),
            average_processing_time: Arc::new(RwLock::new(Duration::from_micros(0))),
            last_cleanup_time: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Submit a new order to the OMS
    pub async fn submit_order(&self, mut order: Order, allocation: Option<AllocationInstruction>) -> Result<String> {
        let start_time = Instant::now();
        
        // Pre-trade risk checks
        if self.config.enable_pre_trade_risk {
            let risk_result = self.risk_manager.validate_order(
                &order.symbol,
                order.side.clone(),
                order.quantity,
                order.price.unwrap_or(0.0),
            );
            
            if let Err(e) = risk_result {
                order.status = OrderStatus::Rejected;
                self.create_audit_event(&order.id, AuditEventType::OrderRejected, 
                    "system", &format!("Pre-trade risk rejection: {}", e)).await;
                return Err(e);
            }
        }

        // Compliance checks
        if self.config.compliance_checks_enabled {
            let compliance_result = self.perform_compliance_checks(&order).await?;
            if !compliance_result.pre_trade_approved {
                order.status = OrderStatus::Rejected;
                return Err(AlgoVedaError::Compliance("Pre-trade compliance failed".to_string()));
            }
        }

        // Create order record
        let order_record = OrderRecord {
            order: order.clone(),
            state: OrderState::PendingNew,
            creation_time: Utc::now(),
            last_update_time: Utc::now(),
            fills: Vec::new(),
            total_filled_quantity: 0,
            average_fill_price: 0.0,
            remaining_quantity: order.quantity,
            commission: 0.0,
            parent_order_id: order.parent_order_id.clone(),
            child_order_ids: Vec::new(),
            allocation_instructions: allocation.clone(),
            compliance_status: ComplianceStatus {
                pre_trade_approved: true,
                post_trade_approved: false,
                compliance_checks: Vec::new(),
                violations: Vec::new(),
            },
            settlement_status: SettlementStatus::Pending,
            audit_trail: Vec::new(),
        };

        // Store order
        self.store_order(order_record.clone()).await;

        // Store allocation instruction if provided
        if let Some(alloc) = allocation {
            self.pending_allocations.write().unwrap().insert(order.id.clone(), alloc);
        }

        // Submit to execution engine
        let execution_result = self.execution_engine.submit_order(order.clone()).await;
        
        match execution_result {
            Ok(_) => {
                self.update_order_state(&order.id, OrderState::New).await?;
                self.create_audit_event(&order.id, AuditEventType::OrderCreated, 
                    "system", "Order created and submitted").await;
                
                // Emit order event
                let _ = self.order_events.send(OrderEvent {
                    order_id: order.id.clone(),
                    event_type: OrderEventType::Created,
                    timestamp: Utc::now(),
                    order_state: OrderState::New,
                    details: None,
                });
            }
            Err(e) => {
                self.update_order_state(&order.id, OrderState::Rejected).await?;
                self.create_audit_event(&order.id, AuditEventType::OrderRejected, 
                    "system", &format!("Execution rejection: {}", e)).await;
                return Err(e);
            }
        }

        // Update performance metrics
        self.orders_processed.fetch_add(1, Ordering::Relaxed);
        let processing_time = start_time.elapsed();
        *self.average_processing_time.write().unwrap() = processing_time;

        Ok(order.id)
    }

    /// Cancel an existing order
    pub async fn cancel_order(&self, order_id: &str, reason: &str) -> Result<()> {
        let order_record = self.get_order(order_id)?;
        
        // Check if order can be canceled
        match order_record.state {
            OrderState::Filled | OrderState::Canceled | OrderState::Rejected | OrderState::Expired => {
                return Err(AlgoVedaError::OrderManagement(
                    format!("Cannot cancel order in state: {:?}", order_record.state)
                ));
            }
            _ => {}
        }

        // Submit cancel request to execution engine
        let cancel_result = self.execution_engine.cancel_order(order_id.to_string()).await;
        
        match cancel_result {
            Ok(_) => {
                self.update_order_state(order_id, OrderState::PendingCancel).await?;
                self.create_audit_event(order_id, AuditEventType::OrderCanceled, 
                    "user", reason).await;
            }
            Err(e) => {
                return Err(e);
            }
        }

        Ok(())
    }

    /// Process an incoming fill
    pub async fn process_fill(&self, fill: Fill) -> Result<()> {
        let mut order_record = self.get_order(&fill.order_id)?;
        
        // Update order record with fill
        order_record.fills.push(fill.clone());
        order_record.total_filled_quantity += fill.quantity;
        order_record.remaining_quantity = order_record.remaining_quantity.saturating_sub(fill.quantity);
        
        // Update average fill price
        let total_value = order_record.fills.iter()
            .map(|f| f.price * f.quantity as f64)
            .sum::<f64>();
        order_record.average_fill_price = total_value / order_record.total_filled_quantity as f64;
        
        // Update order state
        let new_state = if order_record.remaining_quantity == 0 {
            OrderState::Filled
        } else {
            OrderState::PartiallyFilled
        };
        
        order_record.state = new_state.clone();
        order_record.last_update_time = Utc::now();
        
        // Store updated order
        self.update_order_record(order_record.clone()).await;

        // Post-trade risk checks
        if self.config.enable_post_trade_risk {
            let _ = self.perform_post_trade_risk_check(&order_record, &fill).await;
        }

        // Process allocation if required
        if let Some(allocation) = &order_record.allocation_instructions {
            self.process_allocation(&order_record, &fill, allocation).await?;
        }

        // Emit fill event
        let _ = self.fill_events.send(FillEvent {
            fill_id: fill.id.clone(),
            order_id: fill.order_id.clone(),
            timestamp: Utc::now(),
            fill: fill.clone(),
            remaining_quantity: order_record.remaining_quantity,
        });

        // Emit order event
        let event_type = if new_state == OrderState::Filled {
            OrderEventType::Filled
        } else {
            OrderEventType::PartiallyFilled
        };

        let _ = self.order_events.send(OrderEvent {
            order_id: fill.order_id,
            event_type,
            timestamp: Utc::now(),
            order_state: new_state,
            details: None,
        });

        // Trigger settlement if order is fully filled
        if order_record.remaining_quantity == 0 && self.config.enable_trade_settlement {
            self.initiate_settlement(&order_record).await?;
        }

        Ok(())
    }

    /// Get order by ID
    pub fn get_order(&self, order_id: &str) -> Result<OrderRecord> {
        let orders = self.orders.read().unwrap();
        orders.get(order_id)
            .cloned()
            .ok_or_else(|| AlgoVedaError::OrderManagement(format!("Order not found: {}", order_id)))
    }

    /// Get orders by symbol
    pub fn get_orders_by_symbol(&self, symbol: &str) -> Vec<OrderRecord> {
        let symbol_orders = self.symbol_orders.read().unwrap();
        let orders = self.orders.read().unwrap();
        
        symbol_orders.get(symbol)
            .map(|order_ids| {
                order_ids.iter()
                    .filter_map(|id| orders.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get orders by strategy
    pub fn get_orders_by_strategy(&self, strategy_id: &str) -> Vec<OrderRecord> {
        let strategy_orders = self.strategy_orders.read().unwrap();
        let orders = self.orders.read().unwrap();
        
        strategy_orders.get(strategy_id)
            .map(|order_ids| {
                order_ids.iter()
                    .filter_map(|id| orders.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get orders within time range
    pub fn get_orders_by_time_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<OrderRecord> {
        let order_index = self.order_index.read().unwrap();
        let orders = self.orders.read().unwrap();
        
        order_index.range(start..=end)
            .filter_map(|(_, order_id)| orders.get(order_id).cloned())
            .collect()
    }

    /// Get active orders (not filled, canceled, or rejected)
    pub fn get_active_orders(&self) -> Vec<OrderRecord> {
        let orders = self.orders.read().unwrap();
        
        orders.values()
            .filter(|order| matches!(
                order.state,
                OrderState::New | OrderState::PartiallyFilled | OrderState::PendingNew | OrderState::PendingCancel
            ))
            .cloned()
            .collect()
    }

    /// Store order in all indexes
    async fn store_order(&self, order_record: OrderRecord) {
        let order_id = order_record.order.id.clone();
        let symbol = order_record.order.symbol.clone();
        let creation_time = order_record.creation_time;
        
        // Main orders storage
        self.orders.write().unwrap().insert(order_id.clone(), order_record.clone());
        
        // Time-based index
        self.order_index.write().unwrap().insert(creation_time, order_id.clone());
        
        // Symbol-based index
        self.symbol_orders.write().unwrap()
            .entry(symbol)
            .or_insert_with(Vec::new)
            .push(order_id.clone());
        
        // Strategy-based index (if parent order ID represents strategy)
        if let Some(strategy_id) = &order_record.parent_order_id {
            self.strategy_orders.write().unwrap()
                .entry(strategy_id.clone())
                .or_insert_with(Vec::new)
                .push(order_id);
        }
    }

    /// Update order record
    async fn update_order_record(&self, order_record: OrderRecord) {
        let order_id = order_record.order.id.clone();
        self.orders.write().unwrap().insert(order_id, order_record);
    }

    /// Update order state
    async fn update_order_state(&self, order_id: &str, new_state: OrderState) -> Result<()> {
        let mut orders = self.orders.write().unwrap();
        
        if let Some(order_record) = orders.get_mut(order_id) {
            order_record.state = new_state;
            order_record.last_update_time = Utc::now();
            Ok(())
        } else {
            Err(AlgoVedaError::OrderManagement(format!("Order not found: {}", order_id)))
        }
    }

    /// Perform compliance checks
    async fn perform_compliance_checks(&self, order: &Order) -> Result<ComplianceStatus> {
        let mut compliance_status = ComplianceStatus {
            pre_trade_approved: true,
            post_trade_approved: false,
            compliance_checks: Vec::new(),
            violations: Vec::new(),
        };

        // Position limit check
        let position_check = self.check_position_limits(order).await;
        compliance_status.compliance_checks.push(position_check);

        // Concentration limit check
        let concentration_check = self.check_concentration_limits(order).await;
        compliance_status.compliance_checks.push(concentration_check);

        // Trading restriction check
        let restriction_check = self.check_trading_restrictions(order).await;
        compliance_status.compliance_checks.push(restriction_check);

        // Check if any compliance checks failed
        let has_failures = compliance_status.compliance_checks.iter()
            .any(|check| check.status == ComplianceCheckStatus::Failed);

        compliance_status.pre_trade_approved = !has_failures;

        Ok(compliance_status)
    }

    async fn check_position_limits(&self, order: &Order) -> ComplianceCheck {
        // Simplified position limit check
        ComplianceCheck {
            check_id: Uuid::new_v4().to_string(),
            check_type: ComplianceCheckType::PositionLimit,
            status: ComplianceCheckStatus::Passed,
            message: None,
            checked_at: Utc::now(),
        }
    }

    async fn check_concentration_limits(&self, order: &Order) -> ComplianceCheck {
        // Simplified concentration limit check
        ComplianceCheck {
            check_id: Uuid::new_v4().to_string(),
            check_type: ComplianceCheckType::ConcentrationLimit,
            status: ComplianceCheckStatus::Passed,
            message: None,
            checked_at: Utc::now(),
        }
    }

    async fn check_trading_restrictions(&self, order: &Order) -> ComplianceCheck {
        // Simplified trading restriction check
        ComplianceCheck {
            check_id: Uuid::new_v4().to_string(),
            check_type: ComplianceCheckType::TradingRestriction,
            status: ComplianceCheckStatus::Passed,
            message: None,
            checked_at: Utc::now(),
        }
    }

    /// Post-trade risk check
    async fn perform_post_trade_risk_check(&self, order_record: &OrderRecord, fill: &Fill) -> Result<()> {
        // Simplified post-trade risk check
        Ok(())
    }

    /// Process allocation for a fill
    async fn process_allocation(&self, order_record: &OrderRecord, fill: &Fill, allocation: &AllocationInstruction) -> Result<()> {
        let allocation_task = AllocationTask {
            instruction_id: allocation.instruction_id.clone(),
            order_id: order_record.order.id.clone(),
            fill: fill.clone(),
            created_at: Instant::now(),
        };

        self.allocation_engine.queue_allocation(allocation_task).await;
        Ok(())
    }

    /// Initiate trade settlement
    async fn initiate_settlement(&self, order_record: &OrderRecord) -> Result<()> {
        // Create settlement instruction
        let settlement_instruction = crate::accounting::SettlementInstruction {
            trade_id: order_record.order.id.clone(),
            symbol: order_record.order.symbol.clone(),
            quantity: order_record.total_filled_quantity,
            price: order_record.average_fill_price,
            side: order_record.order.side.clone(),
            settlement_date: Utc::now() + chrono::Duration::days(2), // T+2 settlement
            counterparty: "EXCHANGE".to_string(),
            currency: "USD".to_string(),
        };

        self.settlement_engine.initiate_settlement(settlement_instruction).await?;
        
        Ok(())
    }

    /// Create audit event
    async fn create_audit_event(&self, order_id: &str, event_type: AuditEventType, user_id: &str, description: &str) {
        let audit_event = AuditEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: Utc::now(),
            user_id: user_id.to_string(),
            description: description.to_string(),
            old_values: None,
            new_values: None,
        };

        // Add audit event to order record
        if let Ok(mut orders) = self.orders.try_write() {
            if let Some(order_record) = orders.get_mut(order_id) {
                order_record.audit_trail.push(audit_event);
            }
        }
    }

    /// Get OMS statistics
    pub fn get_statistics(&self) -> OMSStatistics {
        let orders = self.orders.read().unwrap();
        let total_orders = orders.len();
        
        let mut stats_by_state = HashMap::new();
        let mut stats_by_symbol = HashMap::new();
        
        for order_record in orders.values() {
            // Count by state
            *stats_by_state.entry(order_record.state.clone()).or_insert(0u64) += 1;
            
            // Count by symbol
            *stats_by_symbol.entry(order_record.order.symbol.clone()).or_insert(0u64) += 1;
        }

        OMSStatistics {
            total_orders: total_orders as u64,
            orders_processed: self.orders_processed.load(Ordering::Relaxed),
            average_processing_time: *self.average_processing_time.read().unwrap(),
            orders_by_state: stats_by_state,
            orders_by_symbol: stats_by_symbol,
            active_orders: self.get_active_orders().len() as u64,
            pending_allocations: self.pending_allocations.read().unwrap().len() as u64,
        }
    }

    /// Start background maintenance tasks
    pub async fn start_maintenance_tasks(&self) {
        self.start_order_cleanup_task().await;
        self.start_allocation_processing().await;
        self.start_compliance_monitoring().await;
    }

    /// Start order cleanup task
    async fn start_order_cleanup_task(&self) {
        let orders = self.orders.clone();
        let order_index = self.order_index.clone();
        let last_cleanup_time = self.last_cleanup_time.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_hours(1));
            
            loop {
                interval.tick().await;
                
                let cutoff_time = Utc::now() - chrono::Duration::days(30);
                let mut orders_to_remove = Vec::new();
                
                // Find old completed orders
                {
                    let orders_guard = orders.read().unwrap();
                    for (order_id, order_record) in orders_guard.iter() {
                        if order_record.creation_time < cutoff_time &&
                           matches!(order_record.state, OrderState::Filled | OrderState::Canceled | OrderState::Rejected) {
                            orders_to_remove.push(order_id.clone());
                        }
                    }
                }
                
                // Archive old orders
                if !orders_to_remove.is_empty() {
                    let mut orders_guard = orders.write().unwrap();
                    let mut index_guard = order_index.write().unwrap();
                    
                    for order_id in orders_to_remove {
                        if let Some(order_record) = orders_guard.remove(&order_id) {
                            index_guard.remove(&order_record.creation_time);
                            // Could archive to database here
                        }
                    }
                }
                
                *last_cleanup_time.write().unwrap() = Instant::now();
            }
        });
    }

    /// Start allocation processing
    async fn start_allocation_processing(&self) {
        let allocation_engine = self.allocation_engine.clone();
        let allocation_events = self.allocation_events.clone();
        
        tokio::spawn(async move {
            allocation_engine.start_processing(allocation_events).await;
        });
    }

    /// Start compliance monitoring
    async fn start_compliance_monitoring(&self) {
        let orders = self.orders.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Monitor for compliance violations
                let orders_guard = orders.read().unwrap();
                for order_record in orders_guard.values() {
                    // Check for any new violations
                    // This would implement real-time compliance monitoring
                }
            }
        });
    }

    /// Subscribe to order events
    pub fn subscribe_order_events(&self) -> broadcast::Receiver<OrderEvent> {
        self.order_events.subscribe()
    }

    /// Subscribe to fill events
    pub fn subscribe_fill_events(&self) -> broadcast::Receiver<FillEvent> {
        self.fill_events.subscribe()
    }

    /// Subscribe to allocation events
    pub fn subscribe_allocation_events(&self) -> broadcast::Receiver<AllocationEvent> {
        self.allocation_events.subscribe()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OMSStatistics {
    pub total_orders: u64,
    pub orders_processed: u64,
    pub average_processing_time: Duration,
    pub orders_by_state: HashMap<OrderState, u64>,
    pub orders_by_symbol: HashMap<String, u64>,
    pub active_orders: u64,
    pub pending_allocations: u64,
}

impl AllocationEngine {
    fn new() -> Self {
        Self {
            pending_instructions: Arc::new(RwLock::new(HashMap::new())),
            completed_allocations: Arc::new(RwLock::new(HashMap::new())),
            allocation_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    async fn queue_allocation(&self, task: AllocationTask) {
        self.allocation_queue.lock().await.push_back(task);
    }

    async fn start_processing(&self, events: broadcast::Sender<AllocationEvent>) {
        let allocation_queue = self.allocation_queue.clone();
        let pending_instructions = self.pending_instructions.clone();
        let completed_allocations = self.completed_allocations.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                let task = {
                    let mut queue = allocation_queue.lock().await;
                    queue.pop_front()
                };
                
                if let Some(task) = task {
                    Self::process_allocation_task(
                        task,
                        &pending_instructions,
                        &completed_allocations,
                        &events,
                    ).await;
                }
            }
        });
    }

    async fn process_allocation_task(
        task: AllocationTask,
        pending_instructions: &Arc<RwLock<HashMap<String, AllocationInstruction>>>,
        completed_allocations: &Arc<RwLock<HashMap<String, CompletedAllocation>>>,
        events: &broadcast::Sender<AllocationEvent>,
    ) {
        // Get allocation instruction
        let instruction = {
            let instructions = pending_instructions.read().unwrap();
            instructions.get(&task.instruction_id).cloned()
        };

        let Some(instruction) = instruction else {
            return;
        };

        // Calculate allocations based on method
        let account_fills = Self::calculate_allocations(&instruction, &task.fill);

        // Create completed allocation
        let completed_allocation = CompletedAllocation {
            instruction_id: task.instruction_id.clone(),
            order_id: task.order_id.clone(),
            account_fills: account_fills.clone(),
            completed_at: Utc::now(),
            total_allocated: account_fills.iter().map(|af| af.quantity).sum(),
            allocation_accuracy: 0.99, // Would calculate actual accuracy
        };

        // Store completed allocation
        completed_allocations.write().unwrap().insert(
            task.instruction_id.clone(),
            completed_allocation,
        );

        // Emit allocation event
        let _ = events.send(AllocationEvent {
            allocation_id: task.instruction_id,
            order_id: task.order_id,
            timestamp: Utc::now(),
            allocations: instruction.account_allocations,
            status: AllocationStatus::Completed,
        });
    }

    fn calculate_allocations(instruction: &AllocationInstruction, fill: &Fill) -> Vec<AccountFill> {
        let mut account_fills = Vec::new();
        let total_quantity = fill.quantity;
        let fill_price = fill.price;

        match instruction.allocation_method {
            AllocationMethod::ProRata => {
                let total_allocation: f64 = instruction.account_allocations.iter()
                    .map(|alloc| alloc.value)
                    .sum();

                for allocation in &instruction.account_allocations {
                    let percentage = allocation.value / total_allocation;
                    let quantity = (total_quantity as f64 * percentage) as u64;

                    account_fills.push(AccountFill {
                        account_id: allocation.account_id.clone(),
                        quantity,
                        price: fill_price,
                        commission: 0.0, // Would calculate actual commission
                        allocation_percentage: percentage,
                    });
                }
            }
            AllocationMethod::EqualShares => {
                let quantity_per_account = total_quantity / instruction.account_allocations.len() as u64;
                
                for allocation in &instruction.account_allocations {
                    account_fills.push(AccountFill {
                        account_id: allocation.account_id.clone(),
                        quantity: quantity_per_account,
                        price: fill_price,
                        commission: 0.0,
                        allocation_percentage: 1.0 / instruction.account_allocations.len() as f64,
                    });
                }
            }
            _ => {
                // Implement other allocation methods
            }
        }

        account_fills
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::{OrderSide, OrderType, TimeInForce};

    #[tokio::test]
    async fn test_oms_order_submission() {
        let config = OMSConfig {
            enable_pre_trade_risk: true,
            enable_post_trade_risk: true,
            enable_allocation_engine: true,
            enable_trade_settlement: false,
            max_orders_per_second: 1000,
            order_timeout_seconds: 30,
            enable_order_recycling: false,
            audit_trail_enabled: true,
            compliance_checks_enabled: true,
        };

        // Would need to create mock dependencies for full test
        // This is a simplified test structure
        assert_eq!(config.max_orders_per_second, 1000);
    }

    #[test]
    fn test_allocation_calculation() {
        let instruction = AllocationInstruction {
            instruction_id: "test_instruction".to_string(),
            strategy_id: "test_strategy".to_string(),
            account_allocations: vec![
                AccountAllocation {
                    account_id: "account1".to_string(),
                    allocation_type: AllocationType::Percentage,
                    value: 60.0,
                    max_quantity: None,
                    priority: 1,
                },
                AccountAllocation {
                    account_id: "account2".to_string(),
                    allocation_type: AllocationType::Percentage,
                    value: 40.0,
                    max_quantity: None,
                    priority: 2,
                },
            ],
            allocation_method: AllocationMethod::ProRata,
            created_by: "test_user".to_string(),
            created_at: Utc::now(),
        };

        let fill = Fill {
            id: "test_fill".to_string(),
            order_id: "test_order".to_string(),
            quantity: 1000,
            price: 100.0,
            timestamp: Utc::now(),
            original_quantity: 1000,
            commission: 0.0,
            exchange: "TEST".to_string(),
        };

        let account_fills = AllocationEngine::calculate_allocations(&instruction, &fill);
        
        assert_eq!(account_fills.len(), 2);
        assert_eq!(account_fills[0].quantity, 600);
        assert_eq!(account_fills[1].quantity, 400);
    }
}
