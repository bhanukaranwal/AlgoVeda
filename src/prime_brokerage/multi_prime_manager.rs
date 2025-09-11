/*!
 * Multi-Prime Brokerage Manager
 * Advanced prime brokerage connectivity with cross-margining and risk aggregation
 */

use std::{
    collections::{HashMap, BTreeMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout},
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration, NaiveTime};
use uuid::Uuid;

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, Fill, OrderSide},
    portfolio::{Portfolio, Position},
    risk_management::RiskManager,
    execution::ExecutionEngine,
    fix::fix_gateway::FIXEngine,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeBrokerageConfig {
    pub enabled_primes: Vec<PrimeBroker>,
    pub default_allocation_method: AllocationMethod,
    pub enable_cross_margining: bool,
    pub enable_netting: bool,
    pub enable_auto_rebalancing: bool,
    pub margin_call_threshold: f64,
    pub liquidation_threshold: f64,
    pub max_leverage_ratio: f64,
    pub settlement_cycles: HashMap<String, u32>, // Asset class -> days
    pub funding_preferences: FundingPreferences,
    pub reporting_requirements: ReportingRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeBroker {
    pub id: String,
    pub name: String,
    pub tier: PrimeTier,
    pub asset_classes: Vec<AssetClass>,
    pub supported_currencies: Vec<String>,
    pub geographic_coverage: Vec<String>,
    pub connectivity: ConnectivityOptions,
    pub margin_rates: MarginRates,
    pub financing_rates: FinancingRates,
    pub commission_schedule: CommissionSchedule,
    pub collateral_requirements: CollateralRequirements,
    pub risk_limits: PrimeRiskLimits,
    pub sla_requirements: SLARequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimeTier {
    Bulgebracket,    // Top-tier investment banks
    Regional,        // Regional prime brokers
    Electronic,      // Electronic/retail prime brokers
    Specialty,       // Specialized prime services
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetClass {
    Equities,
    FixedIncome,
    ForeignExchange,
    Commodities,
    Derivatives,
    Cryptocurrencies,
    PrivateMarkets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityOptions {
    pub fix_enabled: bool,
    pub api_enabled: bool,
    pub oms_integration: bool,
    pub stp_enabled: bool,      // Straight-through processing
    pub real_time_reporting: bool,
    pub fix_version: String,
    pub api_endpoints: Vec<String>,
    pub backup_connectivity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginRates {
    pub initial_margin_rate: f64,
    pub maintenance_margin_rate: f64,
    pub variation_margin_rate: f64,
    pub excess_margin_rate: f64,
    pub haircut_rates: HashMap<String, f64>, // Security type -> haircut %
    pub cross_margin_benefit: f64,           // Cross-margin offset %
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancingRates {
    pub base_rate: f64,
    pub spread: f64,
    pub borrow_rate: f64,
    pub repo_rates: HashMap<String, f64>, // Currency -> repo rate
    pub funding_costs: HashMap<AssetClass, f64>,
    pub dividend_pass_through: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommissionSchedule {
    pub equity_commission: f64,        // Per share or %
    pub fixed_income_commission: f64,  // Basis points
    pub fx_commission: f64,           // Basis points
    pub derivatives_commission: f64,   // Per contract
    pub minimum_commission: f64,
    pub volume_discounts: Vec<VolumeDiscount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDiscount {
    pub volume_threshold: f64,
    pub discount_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralRequirements {
    pub eligible_collateral: Vec<CollateralType>,
    pub concentration_limits: HashMap<String, f64>,
    pub valuation_frequency: ValuationFrequency,
    pub margin_call_timeline: u32, // Hours to post additional margin
    pub segregation_requirements: SegregationLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollateralType {
    Cash,
    GovernmentBonds,
    CorporateBonds,
    Equities,
    ETFs,
    CommodityCollateral,
    CryptocurrencyCollateral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValuationFrequency {
    RealTime,
    Hourly,
    Daily,
    Weekly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SegregationLevel {
    Omnibus,      // Commingled with other clients
    Segregated,   // Separately identified
    Individual,   // Individual custody account
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeRiskLimits {
    pub gross_exposure_limit: f64,
    pub net_exposure_limit: f64,
    pub concentration_limits: HashMap<String, f64>,
    pub sector_limits: HashMap<String, f64>,
    pub var_limit: f64,
    pub stress_test_limits: HashMap<String, f64>,
    pub leverage_limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLARequirements {
    pub trade_execution_sla: u64,     // Milliseconds
    pub settlement_sla: u32,          // Business days
    pub margin_call_response: u32,    // Hours
    pub reporting_sla: u32,           // Hours
    pub uptime_requirement: f64,      // Percentage
    pub disaster_recovery_rto: u32,   // Hours (Recovery Time Objective)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationMethod {
    ProRata,
    BestExecution,
    CostOptimized,
    RiskOptimized,
    LiquidityOptimized,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingPreferences {
    pub preferred_currencies: Vec<String>,
    pub funding_sources: Vec<FundingSource>,
    pub auto_funding: bool,
    pub funding_thresholds: HashMap<String, f64>,
    pub overnight_funding_limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingSource {
    pub source_type: FundingSourceType,
    pub currency: String,
    pub available_amount: f64,
    pub cost_basis_points: f64,
    pub availability_hours: Vec<(NaiveTime, NaiveTime)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FundingSourceType {
    CreditLine,
    RepurchaseAgreement,
    SecuritiesLending,
    CashDeposit,
    CrossCurrencySwap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingRequirements {
    pub daily_reports: Vec<ReportType>,
    pub weekly_reports: Vec<ReportType>,
    pub monthly_reports: Vec<ReportType>,
    pub real_time_notifications: Vec<NotificationType>,
    pub regulatory_reporting: bool,
    pub custom_reports: Vec<CustomReportConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    PositionReport,
    MarginReport,
    RiskReport,
    PnLReport,
    TradeReport,
    SettlementReport,
    CollateralReport,
    ComplianceReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    MarginCall,
    LimitBreach,
    TradeConfirmation,
    SettlementFail,
    RiskAlert,
    SystemAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomReportConfig {
    pub report_id: String,
    pub name: String,
    pub frequency: String,
    pub format: String,
    pub delivery_method: String,
    pub recipients: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeAllocation {
    pub allocation_id: String,
    pub parent_order_id: String,
    pub prime_allocations: Vec<PrimeOrderAllocation>,
    pub allocation_method: AllocationMethod,
    pub total_quantity: u64,
    pub allocated_quantity: u64,
    pub allocation_status: AllocationStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeOrderAllocation {
    pub prime_broker_id: String,
    pub order_id: String,
    pub allocated_quantity: u64,
    pub allocation_percentage: f64,
    pub expected_execution_time: u64,
    pub expected_commission: f64,
    pub risk_score: f64,
    pub liquidity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    PartiallyCompleted,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossMarginAccount {
    pub account_id: String,
    pub prime_brokers: Vec<String>,
    pub base_currency: String,
    pub total_equity: f64,
    pub total_margin_used: f64,
    pub available_margin: f64,
    pub excess_margin: f64,
    pub cross_margin_benefit: f64,
    pub leverage_ratio: f64,
    pub positions: Vec<CrossMarginPosition>,
    pub margin_requirements: HashMap<String, f64>, // Prime -> margin required
    pub netting_benefit: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossMarginPosition {
    pub symbol: String,
    pub asset_class: AssetClass,
    pub prime_broker_id: String,
    pub position_size: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub margin_requirement: f64,
    pub haircut_applied: f64,
    pub risk_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NettingCalculation {
    pub calculation_id: String,
    pub timestamp: DateTime<Utc>,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub netting_benefit: f64,
    pub cross_margining_benefit: f64,
    pub total_margin_saved: f64,
    pub risk_reduction: f64,
    pub positions_netted: Vec<NetPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetPosition {
    pub symbol: String,
    pub gross_long: f64,
    pub gross_short: f64,
    pub net_position: f64,
    pub margin_before_netting: f64,
    pub margin_after_netting: f64,
    pub margin_saved: f64,
}

pub struct MultiPrimeManager {
    config: PrimeBrokerageConfig,
    
    // Prime broker connections
    prime_brokers: Arc<RwLock<HashMap<String, PrimeBroker>>>,
    prime_connections: Arc<RwLock<HashMap<String, Arc<dyn PrimeConnection + Send + Sync>>>>,
    
    // Cross-margining
    cross_margin_account: Arc<RwLock<CrossMarginAccount>>,
    netting_calculator: Arc<NettingCalculator>,
    
    // Order allocation
    allocation_engine: Arc<AllocationEngine>,
    pending_allocations: Arc<RwLock<HashMap<String, PrimeAllocation>>>,
    
    // Risk management
    risk_manager: Arc<RiskManager>,
    margin_monitor: Arc<MarginMonitor>,
    
    // Event handling
    prime_events: broadcast::Sender<PrimeEvent>,
    
    // Performance tracking
    allocations_processed: Arc<AtomicU64>,
    total_margin_saved: Arc<RwLock<f64>>,
    last_netting_calculation: Arc<RwLock<Option<DateTime<Utc>>>>,
    
    // Control
    is_running: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeEvent {
    pub event_id: String,
    pub event_type: PrimeEventType,
    pub timestamp: DateTime<Utc>,
    pub prime_broker_id: Option<String>,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrimeEventType {
    AllocationCompleted,
    MarginCallIssued,
    SettlementCompleted,
    RiskLimitBreached,
    ConnectionStatusChanged,
    NetPositionUpdated,
    CollateralPosted,
    FundingExecuted,
}

// Trait for prime broker connections
pub trait PrimeConnection {
    async fn submit_order(&self, order: Order) -> Result<String>;
    async fn cancel_order(&self, order_id: &str) -> Result<()>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
    async fn get_margin_info(&self) -> Result<MarginInfo>;
    async fn post_collateral(&self, collateral: CollateralPosting) -> Result<String>;
    async fn get_funding_rates(&self) -> Result<HashMap<String, f64>>;
    fn get_connection_status(&self) -> ConnectionStatus;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginInfo {
    pub total_equity: f64,
    pub margin_used: f64,
    pub available_margin: f64,
    pub margin_call_amount: f64,
    pub liquidation_threshold: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralPosting {
    pub posting_id: String,
    pub collateral_type: CollateralType,
    pub amount: f64,
    pub currency: String,
    pub purpose: CollateralPurpose,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollateralPurpose {
    InitialMargin,
    VariationMargin,
    ExcessCollateral,
    RegulatoryCapital,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Degraded,
    Maintenance,
}

// Supporting engines
pub struct AllocationEngine {
    allocation_algorithms: HashMap<AllocationMethod, Box<dyn AllocationAlgorithm + Send + Sync>>,
    execution_cost_models: HashMap<String, ExecutionCostModel>,
    liquidity_models: HashMap<String, LiquidityModel>,
}

pub trait AllocationAlgorithm {
    fn allocate(&self, order: &Order, primes: &[PrimeBroker]) -> Result<Vec<PrimeOrderAllocation>>;
}

#[derive(Debug, Clone)]
struct ExecutionCostModel {
    base_cost: f64,
    size_impact: f64,
    urgency_premium: f64,
    market_impact_model: String,
}

#[derive(Debug, Clone)]
struct LiquidityModel {
    base_liquidity_score: f64,
    time_of_day_factors: Vec<f64>,
    market_conditions_adjustment: f64,
}

pub struct NettingCalculator {
    netting_algorithms: Vec<NettingAlgorithm>,
    cross_asset_correlation: HashMap<(String, String), f64>,
    position_cache: Arc<RwLock<HashMap<String, Vec<Position>>>>,
}

#[derive(Debug, Clone)]
enum NettingAlgorithm {
    SimpleNetting,
    RiskAdjustedNetting,
    CorrelationBasedNetting,
    OptimalNetting,
}

pub struct MarginMonitor {
    margin_thresholds: HashMap<String, f64>,
    alert_manager: Arc<AlertManager>,
    auto_margin_posting: bool,
}

struct AlertManager;

impl MultiPrimeManager {
    pub fn new(
        config: PrimeBrokerageConfig,
        risk_manager: Arc<RiskManager>,
    ) -> Self {
        let (prime_events, _) = broadcast::channel(1000);
        
        // Initialize cross-margin account
        let cross_margin_account = CrossMarginAccount {
            account_id: "MASTER_ACCOUNT".to_string(),
            prime_brokers: config.enabled_primes.iter().map(|p| p.id.clone()).collect(),
            base_currency: "USD".to_string(),
            total_equity: 0.0,
            total_margin_used: 0.0,
            available_margin: 0.0,
            excess_margin: 0.0,
            cross_margin_benefit: 0.0,
            leverage_ratio: 1.0,
            positions: Vec::new(),
            margin_requirements: HashMap::new(),
            netting_benefit: 0.0,
            last_updated: Utc::now(),
        };
        
        Self {
            config: config.clone(),
            prime_brokers: Arc::new(RwLock::new(
                config.enabled_primes.into_iter().map(|p| (p.id.clone(), p)).collect()
            )),
            prime_connections: Arc::new(RwLock::new(HashMap::new())),
            cross_margin_account: Arc::new(RwLock::new(cross_margin_account)),
            netting_calculator: Arc::new(NettingCalculator::new()),
            allocation_engine: Arc::new(AllocationEngine::new()),
            pending_allocations: Arc::new(RwLock::new(HashMap::new())),
            risk_manager,
            margin_monitor: Arc::new(MarginMonitor::new()),
            prime_events,
            allocations_processed: Arc::new(AtomicU64::new(0)),
            total_margin_saved: Arc::new(RwLock::new(0.0)),
            last_netting_calculation: Arc::new(RwLock::new(None)),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the multi-prime manager
    pub async fn start(&self) -> Result<()> {
        self.is_running.store(true, Ordering::Relaxed);
        
        // Initialize prime broker connections
        self.initialize_prime_connections().await?;
        
        // Start monitoring tasks
        self.start_margin_monitoring().await;
        self.start_netting_calculations().await;
        self.start_position_reconciliation().await;
        
        Ok(())
    }

    /// Stop the multi-prime manager
    pub async fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Allocate order across prime brokers
    pub async fn allocate_order(&self, order: Order) -> Result<String> {
        let allocation_id = Uuid::new_v4().to_string();
        
        // Get available prime brokers
        let primes: Vec<PrimeBroker> = self.prime_brokers.read().unwrap().values().cloned().collect();
        
        // Calculate allocation
        let allocations = self.allocation_engine.allocate_order(&order, &primes, &self.config.default_allocation_method).await?;
        
        // Create allocation record
        let prime_allocation = PrimeAllocation {
            allocation_id: allocation_id.clone(),
            parent_order_id: order.id.clone(),
            prime_allocations: allocations.clone(),
            allocation_method: self.config.default_allocation_method.clone(),
            total_quantity: order.quantity,
            allocated_quantity: allocations.iter().map(|a| a.allocated_quantity).sum(),
            allocation_status: AllocationStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        // Store allocation
        self.pending_allocations.write().unwrap().insert(allocation_id.clone(), prime_allocation);
        
        // Execute allocations
        for allocation in allocations {
            self.execute_allocation(&allocation_id, allocation).await?;
        }
        
        self.allocations_processed.fetch_add(1, Ordering::Relaxed);
        
        Ok(allocation_id)
    }

    /// Execute individual allocation with a prime broker
    async fn execute_allocation(&self, allocation_id: &str, allocation: PrimeOrderAllocation) -> Result<()> {
        let prime_connections = self.prime_connections.read().unwrap();
        
        if let Some(connection) = prime_connections.get(&allocation.prime_broker_id) {
            // Create child order
            let child_order = Order {
                id: allocation.order_id.clone(),
                symbol: "TEMP".to_string(), // Would get from parent order
                side: OrderSide::Buy, // Would get from parent order
                quantity: allocation.allocated_quantity,
                order_type: crate::trading::OrderType::Market, // Would get from parent
                price: None,
                time_in_force: crate::trading::TimeInForce::Day,
                status: crate::trading::OrderStatus::PendingNew,
                parent_order_id: Some(allocation_id.to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            // Submit to prime broker
            let execution_id = connection.submit_order(child_order).await?;
            
            // Update allocation status
            // Would update the allocation record with execution ID
        }
        
        Ok(())
    }

    /// Calculate cross-margining benefits
    pub async fn calculate_cross_margining(&self) -> Result<NettingCalculation> {
        let calculation_id = Uuid::new_v4().to_string();
        let positions = self.get_all_positions().await?;
        
        let netting_result = self.netting_calculator.calculate_netting(&positions).await?;
        
        // Update cross-margin account
        {
            let mut account = self.cross_margin_account.write().unwrap();
            account.netting_benefit = netting_result.netting_benefit;
            account.cross_margin_benefit = netting_result.cross_margining_benefit;
            account.last_updated = Utc::now();
        }
        
        *self.total_margin_saved.write().unwrap() += netting_result.total_margin_saved;
        *self.last_netting_calculation.write().unwrap() = Some(Utc::now());
        
        // Emit netting event
        let _ = self.prime_events.send(PrimeEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: PrimeEventType::NetPositionUpdated,
            timestamp: Utc::now(),
            prime_broker_id: None,
            data: serde_json::to_value(&netting_result).unwrap_or(serde_json::Value::Null),
        });
        
        Ok(netting_result)
    }

    /// Get aggregate margin requirements
    pub async fn get_margin_requirements(&self) -> Result<HashMap<String, MarginInfo>> {
        let mut margin_info = HashMap::new();
        let prime_connections = self.prime_connections.read().unwrap();
        
        for (prime_id, connection) in prime_connections.iter() {
            match connection.get_margin_info().await {
                Ok(info) => {
                    margin_info.insert(prime_id.clone(), info);
                }
                Err(e) => {
                    eprintln!("Failed to get margin info from {}: {}", prime_id, e);
                }
            }
        }
        
        Ok(margin_info)
    }

    /// Post collateral to prime broker
    pub async fn post_collateral(
        &self, 
        prime_broker_id: &str, 
        collateral: CollateralPosting
    ) -> Result<String> {
        let prime_connections = self.prime_connections.read().unwrap();
        
        if let Some(connection) = prime_connections.get(prime_broker_id) {
            let posting_id = connection.post_collateral(collateral.clone()).await?;
            
            // Emit collateral event
            let _ = self.prime_events.send(PrimeEvent {
                event_id: Uuid::new_v4().to_string(),
                event_type: PrimeEventType::CollateralPosted,
                timestamp: Utc::now(),
                prime_broker_id: Some(prime_broker_id.to_string()),
                data: serde_json::to_value(&collateral).unwrap_or(serde_json::Value::Null),
            });
            
            Ok(posting_id)
        } else {
            Err(AlgoVedaError::PrimeBrokerage(format!("Prime broker not found: {}", prime_broker_id)))
        }
    }

    /// Get funding costs across all primes
    pub async fn get_funding_costs(&self) -> Result<HashMap<String, HashMap<String, f64>>> {
        let mut all_funding_costs = HashMap::new();
        let prime_connections = self.prime_connections.read().unwrap();
        
        for (prime_id, connection) in prime_connections.iter() {
            match connection.get_funding_rates().await {
                Ok(rates) => {
                    all_funding_costs.insert(prime_id.clone(), rates);
                }
                Err(e) => {
                    eprintln!("Failed to get funding rates from {}: {}", prime_id, e);
                }
            }
        }
        
        Ok(all_funding_costs)
    }

    /// Optimize funding across primes
    pub async fn optimize_funding(&self, required_funding: f64, currency: &str) -> Result<Vec<FundingAllocation>> {
        let funding_costs = self.get_funding_costs().await?;
        let mut funding_options = Vec::new();
        
        // Collect funding options from all primes
        for (prime_id, rates) in funding_costs {
            if let Some(&rate) = rates.get(currency) {
                funding_options.push(FundingOption {
                    prime_broker_id: prime_id,
                    currency: currency.to_string(),
                    available_amount: 10000000.0, // Would get actual available amount
                    cost_bps: rate * 10000.0,
                    tenor_days: 1, // Overnight funding
                });
            }
        }
        
        // Sort by cost (cheapest first)
        funding_options.sort_by(|a, b| a.cost_bps.partial_cmp(&b.cost_bps).unwrap());
        
        // Allocate funding optimally
        let mut allocations = Vec::new();
        let mut remaining_funding = required_funding;
        
        for option in funding_options {
            if remaining_funding <= 0.0 {
                break;
            }
            
            let allocation_amount = remaining_funding.min(option.available_amount);
            allocations.push(FundingAllocation {
                prime_broker_id: option.prime_broker_id,
                currency: option.currency,
                amount: allocation_amount,
                cost_bps: option.cost_bps,
                tenor_days: option.tenor_days,
            });
            
            remaining_funding -= allocation_amount;
        }
        
        Ok(allocations)
    }

    /// Helper methods
    async fn initialize_prime_connections(&self) -> Result<()> {
        let primes = self.prime_brokers.read().unwrap().clone();
        let mut connections = self.prime_connections.write().unwrap();
        
        for (prime_id, prime) in primes {
            // Create appropriate connection type based on connectivity options
            if prime.connectivity.fix_enabled {
                // Would create FIX connection
                let connection = Arc::new(FIXPrimeConnection::new(prime.clone())?);
                connections.insert(prime_id, connection);
            } else if prime.connectivity.api_enabled {
                // Would create API connection  
                let connection = Arc::new(APIPrimeConnection::new(prime.clone())?);
                connections.insert(prime_id, connection);
            }
        }
        
        Ok(())
    }

    async fn get_all_positions(&self) -> Result<Vec<Position>> {
        let mut all_positions = Vec::new();
        let prime_connections = self.prime_connections.read().unwrap();
        
        for connection in prime_connections.values() {
            match connection.get_positions().await {
                Ok(positions) => {
                    all_positions.extend(positions);
                }
                Err(e) => {
                    eprintln!("Failed to get positions: {}", e);
                }
            }
        }
        
        Ok(all_positions)
    }

    async fn start_margin_monitoring(&self) {
        let margin_monitor = self.margin_monitor.clone();
        let cross_margin_account = self.cross_margin_account.clone();
        let prime_events = self.prime_events.clone();
        let is_running = self.is_running.clone();
        let threshold = self.config.margin_call_threshold;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let account = cross_margin_account.read().unwrap();
                let margin_ratio = account.total_margin_used / account.total_equity.max(1.0);
                
                if margin_ratio > threshold {
                    // Emit margin call event
                    let _ = prime_events.send(PrimeEvent {
                        event_id: Uuid::new_v4().to_string(),
                        event_type: PrimeEventType::MarginCallIssued,
                        timestamp: Utc::now(),
                        prime_broker_id: None,
                        data: serde_json::json!({
                            "margin_ratio": margin_ratio,
                            "threshold": threshold,
                            "call_amount": account.total_margin_used - (account.total_equity * threshold)
                        }),
                    });
                }
            }
        });
    }

    async fn start_netting_calculations(&self) {
        let netting_calculator = self.netting_calculator.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // Every hour
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Would perform netting calculations
                // self.calculate_cross_margining().await;
            }
        });
    }

    async fn start_position_reconciliation(&self) {
        let prime_connections = self.prime_connections.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1800)); // Every 30 minutes
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Reconcile positions across all prime brokers
                // Would implement position reconciliation logic
            }
        });
    }

    /// Get statistics
    pub fn get_statistics(&self) -> MultiPrimeStatistics {
        let prime_brokers = self.prime_brokers.read().unwrap();
        let cross_margin_account = self.cross_margin_account.read().unwrap();
        
        MultiPrimeStatistics {
            active_prime_brokers: prime_brokers.len() as u64,
            allocations_processed: self.allocations_processed.load(Ordering::Relaxed),
            total_equity: cross_margin_account.total_equity,
            total_margin_saved: *self.total_margin_saved.read().unwrap(),
            cross_margin_benefit: cross_margin_account.cross_margin_benefit,
            leverage_ratio: cross_margin_account.leverage_ratio,
            netting_benefit: cross_margin_account.netting_benefit,
            last_netting_calculation: *self.last_netting_calculation.read().unwrap(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingOption {
    pub prime_broker_id: String,
    pub currency: String,
    pub available_amount: f64,
    pub cost_bps: f64,
    pub tenor_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingAllocation {
    pub prime_broker_id: String,
    pub currency: String,
    pub amount: f64,
    pub cost_bps: f64,
    pub tenor_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPrimeStatistics {
    pub active_prime_brokers: u64,
    pub allocations_processed: u64,
    pub total_equity: f64,
    pub total_margin_saved: f64,
    pub cross_margin_benefit: f64,
    pub leverage_ratio: f64,
    pub netting_benefit: f64,
    pub last_netting_calculation: Option<DateTime<Utc>>,
}

// Mock implementation of prime connections
struct FIXPrimeConnection {
    prime: PrimeBroker,
    fix_engine: Arc<FIXEngine>,
}

impl FIXPrimeConnection {
    fn new(prime: PrimeBroker) -> Result<Self> {
        // Would initialize FIX connection
        Ok(Self {
            prime,
            fix_engine: Arc::new(FIXEngine::new(Default::default())?),
        })
    }
}

impl PrimeConnection for FIXPrimeConnection {
    async fn submit_order(&self, order: Order) -> Result<String> {
        // Would submit via FIX
        Ok(Uuid::new_v4().to_string())
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        Ok(())
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        Ok(Vec::new())
    }

    async fn get_margin_info(&self) -> Result<MarginInfo> {
        Ok(MarginInfo {
            total_equity: 1000000.0,
            margin_used: 200000.0,
            available_margin: 800000.0,
            margin_call_amount: 0.0,
            liquidation_threshold: 150000.0,
            timestamp: Utc::now(),
        })
    }

    async fn post_collateral(&self, collateral: CollateralPosting) -> Result<String> {
        Ok(collateral.posting_id)
    }

    async fn get_funding_rates(&self) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }

    fn get_connection_status(&self) -> ConnectionStatus {
        ConnectionStatus::Connected
    }
}

struct APIPrimeConnection {
    prime: PrimeBroker,
}

impl APIPrimeConnection {
    fn new(prime: PrimeBroker) -> Result<Self> {
        Ok(Self { prime })
    }
}

impl PrimeConnection for APIPrimeConnection {
    async fn submit_order(&self, order: Order) -> Result<String> {
        Ok(Uuid::new_v4().to_string())
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        Ok(())
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        Ok(Vec::new())
    }

    async fn get_margin_info(&self) -> Result<MarginInfo> {
        Ok(MarginInfo {
            total_equity: 1000000.0,
            margin_used: 200000.0,
            available_margin: 800000.0,
            margin_call_amount: 0.0,
            liquidation_threshold: 150000.0,
            timestamp: Utc::now(),
        })
    }

    async fn post_collateral(&self, collateral: CollateralPosting) -> Result<String> {
        Ok(collateral.posting_id)
    }

    async fn get_funding_rates(&self) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }

    fn get_connection_status(&self) -> ConnectionStatus {
        ConnectionStatus::Connected
    }
}

// Implementation of supporting engines
impl AllocationEngine {
    fn new() -> Self {
        let mut allocation_algorithms: HashMap<AllocationMethod, Box<dyn AllocationAlgorithm + Send + Sync>> = HashMap::new();
        allocation_algorithms.insert(AllocationMethod::ProRata, Box::new(ProRataAllocator));
        allocation_algorithms.insert(AllocationMethod::BestExecution, Box::new(BestExecutionAllocator));
        
        Self {
            allocation_algorithms,
            execution_cost_models: HashMap::new(),
            liquidity_models: HashMap::new(),
        }
    }

    async fn allocate_order(
        &self, 
        order: &Order, 
        primes: &[PrimeBroker], 
        method: &AllocationMethod
    ) -> Result<Vec<PrimeOrderAllocation>> {
        if let Some(algorithm) = self.allocation_algorithms.get(method) {
            algorithm.allocate(order, primes)
        } else {
            Err(AlgoVedaError::PrimeBrokerage("Allocation method not found".to_string()))
        }
    }
}

struct ProRataAllocator;

impl AllocationAlgorithm for ProRataAllocator {
    fn allocate(&self, order: &Order, primes: &[PrimeBroker]) -> Result<Vec<PrimeOrderAllocation>> {
        let mut allocations = Vec::new();
        let allocation_per_prime = order.quantity / primes.len() as u64;
        
        for prime in primes {
            allocations.push(PrimeOrderAllocation {
                prime_broker_id: prime.id.clone(),
                order_id: Uuid::new_v4().to_string(),
                allocated_quantity: allocation_per_prime,
                allocation_percentage: 1.0 / primes.len() as f64,
                expected_execution_time: 1000, // 1 second
                expected_commission: 0.002,
                risk_score: 0.5,
                liquidity_score: 0.8,
            });
        }
        
        Ok(allocations)
    }
}

struct BestExecutionAllocator;

impl AllocationAlgorithm for BestExecutionAllocator {
    fn allocate(&self, order: &Order, primes: &[PrimeBroker]) -> Result<Vec<PrimeOrderAllocation>> {
        // Would implement sophisticated best execution logic
        // For now, just allocate to single best prime
        if let Some(best_prime) = primes.first() {
            Ok(vec![PrimeOrderAllocation {
                prime_broker_id: best_prime.id.clone(),
                order_id: Uuid::new_v4().to_string(),
                allocated_quantity: order.quantity,
                allocation_percentage: 1.0,
                expected_execution_time: 800,
                expected_commission: 0.0015,
                risk_score: 0.3,
                liquidity_score: 0.9,
            }])
        } else {
            Err(AlgoVedaError::PrimeBrokerage("No prime brokers available".to_string()))
        }
    }
}

impl NettingCalculator {
    fn new() -> Self {
        Self {
            netting_algorithms: vec![NettingAlgorithm::SimpleNetting],
            cross_asset_correlation: HashMap::new(),
            position_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn calculate_netting(&self, positions: &[Position]) -> Result<NettingCalculation> {
        let calculation_id = Uuid::new_v4().to_string();
        
        // Group positions by symbol
        let mut symbol_positions: HashMap<String, Vec<&Position>> = HashMap::new();
        for position in positions {
            symbol_positions.entry(position.symbol.clone()).or_insert(Vec::new()).push(position);
        }
        
        let mut net_positions = Vec::new();
        let mut total_margin_saved = 0.0;
        let mut gross_exposure = 0.0;
        let mut net_exposure = 0.0;
        
        for (symbol, positions) in symbol_positions {
            let gross_long = positions.iter()
                .filter(|p| p.quantity > 0)
                .map(|p| p.quantity as f64 * p.average_price)
                .sum::<f64>();
            
            let gross_short = positions.iter()
                .filter(|p| p.quantity < 0)
                .map(|p| (-p.quantity) as f64 * p.average_price)
                .sum::<f64>();
            
            let net_position = gross_long - gross_short;
            
            // Simplified margin calculation
            let margin_before_netting = (gross_long + gross_short) * 0.25; // 25% margin
            let margin_after_netting = net_position.abs() * 0.25;
            let margin_saved = margin_before_netting - margin_after_netting;
            
            gross_exposure += gross_long + gross_short;
            net_exposure += net_position.abs();
            total_margin_saved += margin_saved;
            
            net_positions.push(NetPosition {
                symbol,
                gross_long,
                gross_short,
                net_position,
                margin_before_netting,
                margin_after_netting,
                margin_saved,
            });
        }
        
        let netting_benefit = (gross_exposure - net_exposure) / gross_exposure.max(1.0);
        
        Ok(NettingCalculation {
            calculation_id,
            timestamp: Utc::now(),
            gross_exposure,
            net_exposure,
            netting_benefit,
            cross_margining_benefit: netting_benefit * 0.8, // 80% of netting benefit
            total_margin_saved,
            risk_reduction: netting_benefit * 0.6,
            positions_netted: net_positions,
        })
    }
}

impl MarginMonitor {
    fn new() -> Self {
        Self {
            margin_thresholds: HashMap::new(),
            alert_manager: Arc::new(AlertManager),
            auto_margin_posting: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_broker_creation() {
        let prime = PrimeBroker {
            id: "GOLDMAN_SACHS".to_string(),
            name: "Goldman Sachs Prime Services".to_string(),
            tier: PrimeTier::Bulgebracket,
            asset_classes: vec![AssetClass::Equities, AssetClass::FixedIncome],
            supported_currencies: vec!["USD".to_string(), "EUR".to_string()],
            geographic_coverage: vec!["US".to_string(), "EU".to_string()],
            connectivity: ConnectivityOptions {
                fix_enabled: true,
                api_enabled: true,
                oms_integration: true,
                stp_enabled: true,
                real_time_reporting: true,
                fix_version: "4.4".to_string(),
                api_endpoints: vec!["https://api.gs.com/prime".to_string()],
                backup_connectivity: true,
            },
            margin_rates: MarginRates {
                initial_margin_rate: 0.25,
                maintenance_margin_rate: 0.15,
                variation_margin_rate: 0.05,
                excess_margin_rate: 0.02,
                haircut_rates: HashMap::new(),
                cross_margin_benefit: 0.2,
            },
            financing_rates: FinancingRates {
                base_rate: 0.05,
                spread: 0.0075,
                borrow_rate: 0.06,
                repo_rates: HashMap::new(),
                funding_costs: HashMap::new(),
                dividend_pass_through: 0.85,
            },
            commission_schedule: CommissionSchedule {
                equity_commission: 0.002,
                fixed_income_commission: 0.25,
                fx_commission: 0.5,
                derivatives_commission: 2.0,
                minimum_commission: 25.0,
                volume_discounts: Vec::new(),
            },
            collateral_requirements: CollateralRequirements {
                eligible_collateral: vec![CollateralType::Cash, CollateralType::GovernmentBonds],
                concentration_limits: HashMap::new(),
                valuation_frequency: ValuationFrequency::Daily,
                margin_call_timeline: 2,
                segregation_requirements: SegregationLevel::Segregated,
            },
            risk_limits: PrimeRiskLimits {
                gross_exposure_limit: 1000000000.0,
                net_exposure_limit: 500000000.0,
                concentration_limits: HashMap::new(),
                sector_limits: HashMap::new(),
                var_limit: 10000000.0,
                stress_test_limits: HashMap::new(),
                leverage_limit: 4.0,
            },
            sla_requirements: SLARequirements {
                trade_execution_sla: 100,
                settlement_sla: 1,
                margin_call_response: 2,
                reporting_sla: 1,
                uptime_requirement: 99.9,
                disaster_recovery_rto: 4,
            },
        };

        assert_eq!(prime.id, "GOLDMAN_SACHS");
        assert_eq!(prime.tier, PrimeTier::BulgeAccount);
    }

    #[tokio::test]
    async fn test_allocation_engine() {
        let engine = AllocationEngine::new();
        
        let order = Order {
            id: "TEST_ORDER".to_string(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: 10000,
            order_type: crate::trading::OrderType::Market,
            price: None,
            time_in_force: crate::trading::TimeInForce::Day,
            status: crate::trading::OrderStatus::PendingNew,
            parent_order_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let primes = vec![
            PrimeBroker {
                id: "PRIME1".to_string(),
                name: "Prime 1".to_string(),
                tier: PrimeTier::BulgeAccount,
                asset_classes: vec![AssetClass::Equities],
                supported_currencies: vec!["USD".to_string()],
                geographic_coverage: vec!["US".to_string()],
                connectivity: ConnectivityOptions {
                    fix_enabled: true,
                    api_enabled: false,
                    oms_integration: false,
                    stp_enabled: false,
                    real_time_reporting: false,
                    fix_version: "4.4".to_string(),
                    api_endpoints: vec![],
                    backup_connectivity: false,
                },
                margin_rates: MarginRates {
                    initial_margin_rate: 0.25,
                    maintenance_margin_rate: 0.15,
                    variation_margin_rate: 0.05,
                    excess_margin_rate: 0.02,
                    haircut_rates: HashMap::new(),
                    cross_margin_benefit: 0.2,
                },
                financing_rates: FinancingRates {
                    base_rate: 0.05,
                    spread: 0.01,
                    borrow_rate: 0.06,
                    repo_rates: HashMap::new(),
                    funding_costs: HashMap::new(),
                    dividend_pass_through: 0.8,
                },
                commission_schedule: CommissionSchedule {
                    equity_commission: 0.002,
                    fixed_income_commission: 0.0,
                    fx_commission: 0.0,
                    derivatives_commission: 0.0,
                    minimum_commission: 10.0,
                    volume_discounts: vec![],
                },
                collateral_requirements: CollateralRequirements {
                    eligible_collateral: vec![CollateralType::Cash],
                    concentration_limits: HashMap::new(),
                    valuation_frequency: ValuationFrequency::Daily,
                    margin_call_timeline: 1,
                    segregation_requirements: SegregationLevel::Omnibus,
                },
                risk_limits: PrimeRiskLimits {
                    gross_exposure_limit: 100000000.0,
                    net_exposure_limit: 50000000.0,
                    concentration_limits: HashMap::new(),
                    sector_limits: HashMap::new(),
                    var_limit: 1000000.0,
                    stress_test_limits: HashMap::new(),
                    leverage_limit: 2.0,
                },
                sla_requirements: SLARequirements {
                    trade_execution_sla: 200,
                    settlement_sla: 2,
                    margin_call_response: 4,
                    reporting_sla: 6,
                    uptime_requirement: 99.5,
                    disaster_recovery_rto: 8,
                },
            },
        ];

        let allocations = engine.allocate_order(&order, &primes, &AllocationMethod::ProRata).await.unwrap();
        
        assert_eq!(allocations.len(), 1);
        assert_eq!(allocations[0].allocated_quantity, 10000);
    }
}
