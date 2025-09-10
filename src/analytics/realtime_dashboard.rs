/*!
 * Real-time Analytics Dashboard
 * Live portfolio monitoring, P&L tracking, and risk analytics
 */

use std::{
    collections::{HashMap, BTreeMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout},
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use uuid::Uuid;

use crate::{
    error::{Result, AlgoVedaError},
    portfolio::{Portfolio, Position},
    trading::{Fill, Order},
    market_data::MarketData,
    risk_engine::var_calculator::VarResult,
    order_management::OrderRecord,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub update_frequency_ms: u64,
    pub historical_data_retention_hours: u32,
    pub enable_real_time_pnl: bool,
    pub enable_risk_monitoring: bool,
    pub enable_order_flow_analysis: bool,
    pub enable_performance_analytics: bool,
    pub max_symbols_tracked: usize,
    pub websocket_port: u16,
    pub authentication_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub timestamp: DateTime<Utc>,
    pub portfolio_summary: PortfolioSummary,
    pub pnl_summary: PnLSummary,
    pub risk_metrics: RiskMetrics,
    pub order_flow_metrics: OrderFlowMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub positions: Vec<PositionSnapshot>,
    pub active_orders: Vec<OrderSnapshot>,
    pub recent_fills: Vec<FillSnapshot>,
    pub alerts: Vec<Alert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub total_market_value: f64,
    pub total_cash: f64,
    pub total_exposure: f64,
    pub net_exposure: f64,
    pub long_exposure: f64,
    pub short_exposure: f64,
    pub leverage: f64,
    pub number_of_positions: u32,
    pub number_of_symbols: u32,
    pub largest_position_pct: f64,
    pub concentration_ratio: f64,
    pub asset_allocation: HashMap<String, f64>,
    pub sector_allocation: HashMap<String, f64>,
    pub geographic_allocation: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLSummary {
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub total_pnl: f64,
    pub daily_pnl: f64,
    pub weekly_pnl: f64,
    pub monthly_pnl: f64,
    pub ytd_pnl: f64,
    pub pnl_by_symbol: HashMap<String, f64>,
    pub pnl_by_strategy: HashMap<String, f64>,
    pub pnl_by_asset_class: HashMap<String, f64>,
    pub pnl_attribution: PnLAttribution,
    pub high_water_mark: f64,
    pub drawdown_current: f64,
    pub drawdown_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLAttribution {
    pub market_pnl: f64,        // Due to market movements
    pub alpha_pnl: f64,         // Due to stock selection
    pub interaction_pnl: f64,   // Interaction effects
    pub trading_costs: f64,     // Transaction costs
    pub funding_costs: f64,     // Borrowing/lending costs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub portfolio_var_1d: f64,
    pub portfolio_var_10d: f64,
    pub expected_shortfall: f64,
    pub portfolio_volatility: f64,
    pub portfolio_beta: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub information_ratio: f64,
    pub tracking_error: f64,
    pub risk_by_symbol: HashMap<String, SymbolRisk>,
    pub concentration_risk: f64,
    pub liquidity_risk: f64,
    pub correlation_risk: f64,
    pub tail_risk_measures: TailRiskMeasures,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolRisk {
    pub symbol: String,
    pub var_contribution: f64,
    pub volatility: f64,
    pub beta: f64,
    pub correlation_to_portfolio: f64,
    pub liquidity_score: f64,
    pub concentration_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailRiskMeasures {
    pub skewness: f64,
    pub kurtosis: f64,
    pub var_99: f64,
    pub expected_shortfall_99: f64,
    pub maximum_loss_1d: f64,
    pub maximum_loss_1w: f64,
    pub stress_test_results: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowMetrics {
    pub orders_today: u64,
    pub fills_today: u64,
    pub volume_today: f64,
    pub notional_today: f64,
    pub average_fill_time_ms: f64,
    pub fill_rate: f64,
    pub cancel_rate: f64,
    pub reject_rate: f64,
    pub order_flow_by_venue: HashMap<String, VenueMetrics>,
    pub order_flow_by_strategy: HashMap<String, StrategyMetrics>,
    pub market_impact_bps: f64,
    pub implementation_shortfall_bps: f64,
    pub slippage_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueMetrics {
    pub venue_name: String,
    pub orders: u64,
    pub fills: u64,
    pub volume: f64,
    pub fill_rate: f64,
    pub average_fill_time_ms: f64,
    pub market_impact_bps: f64,
    pub rebates_earned: f64,
    pub fees_paid: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy_name: String,
    pub orders: u64,
    pub fills: u64,
    pub pnl: f64,
    pub volume: f64,
    pub success_rate: f64,
    pub average_holding_period_hours: f64,
    pub sharpe_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub benchmark_return: f64,
    pub active_return: f64,
    pub volatility: f64,
    pub benchmark_volatility: f64,
    pub tracking_error: f64,
    pub information_ratio: f64,
    pub sharpe_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub consecutive_wins: u32,
    pub consecutive_losses: u32,
    pub performance_by_period: HashMap<String, PeriodPerformance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodPerformance {
    pub period: String,
    pub return_pct: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub trades: u64,
    pub win_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    pub symbol: String,
    pub quantity: i64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub total_pnl: f64,
    pub average_cost: f64,
    pub current_price: f64,
    pub daily_change: f64,
    pub daily_change_pct: f64,
    pub weight_pct: f64,
    pub sector: String,
    pub asset_class: String,
    pub currency: String,
    pub beta: f64,
    pub volatility: f64,
    pub var_contribution: f64,
    pub liquidity_score: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSnapshot {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub quantity: u64,
    pub filled_quantity: u64,
    pub remaining_quantity: u64,
    pub price: Option<f64>,
    pub average_fill_price: f64,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub time_in_force: String,
    pub strategy: Option<String>,
    pub venue: String,
    pub estimated_fill_time_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillSnapshot {
    pub fill_id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: u64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub commission: f64,
    pub market_impact_bps: f64,
    pub implementation_shortfall_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub symbol: Option<String>,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
    pub timestamp: DateTime<Utc>,
    pub acknowledged: bool,
    pub auto_resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    RiskLimit,
    PositionLimit,
    PnLLimit,
    VaRBreach,
    DrawdownLimit,
    ConcentrationLimit,
    LiquidityAlert,
    MarketDataStale,
    OrderTimeout,
    FillAbnormal,
    SystemError,
    ComplianceViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

pub struct RealTimeDashboard {
    config: DashboardConfig,
    
    // Data storage
    current_snapshot: Arc<RwLock<Option<DashboardSnapshot>>>,
    historical_snapshots: Arc<RwLock<VecDeque<DashboardSnapshot>>>,
    
    // Live data streams
    portfolio_data: Arc<RwLock<Option<Portfolio>>>,
    market_data_cache: Arc<RwLock<HashMap<String, MarketDataPoint>>>,
    order_cache: Arc<RwLock<HashMap<String, OrderRecord>>>,
    fill_history: Arc<RwLock<VecDeque<Fill>>>,
    
    // Risk data
    var_results: Arc<RwLock<Option<VarResult>>>,
    
    // Event streams
    dashboard_events: broadcast::Sender<DashboardEvent>,
    alert_queue: Arc<Mutex<VecDeque<Alert>>>,
    
    // Performance tracking
    update_count: Arc<AtomicU64>,
    last_update_time: Arc<RwLock<Option<Instant>>>,
    processing_time_ms: Arc<RwLock<VecDeque<u64>>>,
    
    // Control
    is_running: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct MarketDataPoint {
    symbol: String,
    price: f64,
    bid: f64,
    ask: f64,
    volume: u64,
    timestamp: DateTime<Utc>,
    daily_change: f64,
    daily_change_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardEvent {
    pub event_id: String,
    pub event_type: DashboardEventType,
    pub timestamp: DateTime<Utc>,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardEventType {
    SnapshotUpdated,
    AlertGenerated,
    PositionChanged,
    OrderFilled,
    RiskLimitBreached,
    PerformanceUpdate,
    SystemStatus,
}

impl RealTimeDashboard {
    pub fn new(config: DashboardConfig) -> Self {
        let (dashboard_events, _) = broadcast::channel(1000);
        
        Self {
            config,
            current_snapshot: Arc::new(RwLock::new(None)),
            historical_snapshots: Arc::new(RwLock::new(VecDeque::new())),
            portfolio_data: Arc::new(RwLock::new(None)),
            market_data_cache: Arc::new(RwLock::new(HashMap::new())),
            order_cache: Arc::new(RwLock::new(HashMap::new())),
            fill_history: Arc::new(RwLock::new(VecDeque::new())),
            var_results: Arc::new(RwLock::new(None)),
            dashboard_events,
            alert_queue: Arc::new(Mutex::new(VecDeque::new())),
            update_count: Arc::new(AtomicU64::new(0)),
            last_update_time: Arc::new(RwLock::new(None)),
            processing_time_ms: Arc::new(RwLock::new(VecDeque::new())),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the real-time dashboard
    pub async fn start(&self) -> Result<()> {
        self.is_running.store(true, Ordering::Relaxed);
        
        // Start update loop
        self.start_update_loop().await;
        
        // Start alert processing
        self.start_alert_processor().await;
        
        // Start data cleanup
        self.start_data_cleanup().await;
        
        // Start websocket server (would be implemented)
        if self.config.websocket_port > 0 {
            // self.start_websocket_server().await;
        }
        
        Ok(())
    }

    /// Stop the dashboard
    pub async fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Update portfolio data
    pub async fn update_portfolio(&self, portfolio: Portfolio) {
        *self.portfolio_data.write().unwrap() = Some(portfolio);
    }

    /// Update market data
    pub async fn update_market_data(&self, symbol: String, market_data: MarketData) {
        let market_point = MarketDataPoint {
            symbol: symbol.clone(),
            price: market_data.last_price.unwrap_or(0.0),
            bid: market_data.bid.unwrap_or(0.0),
            ask: market_data.ask.unwrap_or(0.0),
            volume: market_data.volume.unwrap_or(0),
            timestamp: Utc::now(),
            daily_change: 0.0, // Would calculate from previous close
            daily_change_pct: 0.0,
        };
        
        self.market_data_cache.write().unwrap().insert(symbol, market_point);
    }

    /// Update order data
    pub async fn update_order(&self, order_record: OrderRecord) {
        self.order_cache.write().unwrap().insert(order_record.order.id.clone(), order_record);
    }

    /// Add fill data
    pub async fn add_fill(&self, fill: Fill) {
        let mut fill_history = self.fill_history.write().unwrap();
        fill_history.push_back(fill);
        
        // Keep only recent fills
        while fill_history.len() > 10000 {
            fill_history.pop_front();
        }
    }

    /// Update VaR results
    pub async fn update_var_results(&self, var_result: VarResult) {
        *self.var_results.write().unwrap() = Some(var_result);
    }

    /// Get current dashboard snapshot
    pub fn get_current_snapshot(&self) -> Option<DashboardSnapshot> {
        self.current_snapshot.read().unwrap().clone()
    }

    /// Get historical snapshots
    pub fn get_historical_snapshots(&self, limit: usize) -> Vec<DashboardSnapshot> {
        let snapshots = self.historical_snapshots.read().unwrap();
        snapshots.iter().rev().take(limit).cloned().collect()
    }

    /// Start the main update loop
    async fn start_update_loop(&self) {
        let current_snapshot = self.current_snapshot.clone();
        let historical_snapshots = self.historical_snapshots.clone();
        let portfolio_data = self.portfolio_data.clone();
        let market_data_cache = self.market_data_cache.clone();
        let order_cache = self.order_cache.clone();
        let fill_history = self.fill_history.clone();
        let var_results = self.var_results.clone();
        let dashboard_events = self.dashboard_events.clone();
        let alert_queue = self.alert_queue.clone();
        let update_count = self.update_count.clone();
        let last_update_time = self.last_update_time.clone();
        let processing_time_ms = self.processing_time_ms.clone();
        let is_running = self.is_running.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.update_frequency_ms));
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let start_time = Instant::now();
                
                // Generate new snapshot
                let snapshot = Self::generate_snapshot(
                    &portfolio_data,
                    &market_data_cache,
                    &order_cache,
                    &fill_history,
                    &var_results,
                ).await;
                
                if let Some(snapshot) = snapshot {
                    // Check for alerts
                    let alerts = Self::check_for_alerts(&snapshot, &alert_queue).await;
                    
                    // Update current snapshot
                    *current_snapshot.write().unwrap() = Some(snapshot.clone());
                    
                    // Add to historical snapshots
                    let mut historical = historical_snapshots.write().unwrap();
                    historical.push_back(snapshot.clone());
                    
                    // Keep only recent history
                    let max_history = config.historical_data_retention_hours as usize * 60 * 60 / (config.update_frequency_ms as usize / 1000);
                    while historical.len() > max_history {
                        historical.pop_front();
                    }
                    
                    // Emit dashboard event
                    let _ = dashboard_events.send(DashboardEvent {
                        event_id: Uuid::new_v4().to_string(),
                        event_type: DashboardEventType::SnapshotUpdated,
                        timestamp: Utc::now(),
                        data: serde_json::to_value(&snapshot).unwrap_or(serde_json::Value::Null),
                    });
                }
                
                // Update performance metrics
                let processing_time = start_time.elapsed().as_millis() as u64;
                {
                    let mut times = processing_time_ms.write().unwrap();
                    times.push_back(processing_time);
                    if times.len() > 100 {
                        times.pop_front();
                    }
                }
                
                update_count.fetch_add(1, Ordering::Relaxed);
                *last_update_time.write().unwrap() = Some(start_time);
            }
        });
    }

    /// Generate a complete dashboard snapshot
    async fn generate_snapshot(
        portfolio_data: &Arc<RwLock<Option<Portfolio>>>,
        market_data_cache: &Arc<RwLock<HashMap<String, MarketDataPoint>>>,
        order_cache: &Arc<RwLock<HashMap<String, OrderRecord>>>,
        fill_history: &Arc<RwLock<VecDeque<Fill>>>,
        var_results: &Arc<RwLock<Option<VarResult>>>,
    ) -> Option<DashboardSnapshot> {
        let portfolio = portfolio_data.read().unwrap().clone()?;
        let market_data = market_data_cache.read().unwrap().clone();
        let orders = order_cache.read().unwrap().clone();
        let fills = fill_history.read().unw
        let fills = fill_history.read().unwrap().clone();
        let var_result = var_results.read().unwrap().clone();

        // Calculate portfolio summary
        let portfolio_summary = Self::calculate_portfolio_summary(&portfolio, &market_data);
        
        // Calculate P&L summary
        let pnl_summary = Self::calculate_pnl_summary(&portfolio, &market_data, &fills);
        
        // Calculate risk metrics
        let risk_metrics = Self::calculate_risk_metrics(&portfolio, &market_data, &var_result);
        
        // Calculate order flow metrics
        let order_flow_metrics = Self::calculate_order_flow_metrics(&orders, &fills);
        
        // Calculate performance metrics
        let performance_metrics = Self::calculate_performance_metrics(&portfolio, &fills);
        
        // Generate position snapshots
        let positions = Self::generate_position_snapshots(&portfolio, &market_data);
        
        // Generate order snapshots
        let active_orders = Self::generate_order_snapshots(&orders);
        
        // Generate recent fill snapshots
        let recent_fills = Self::generate_fill_snapshots(&fills, 100);
        
        Some(DashboardSnapshot {
            timestamp: Utc::now(),
            portfolio_summary,
            pnl_summary,
            risk_metrics,
            order_flow_metrics,
            performance_metrics,
            positions,
            active_orders,
            recent_fills,
            alerts: Vec::new(), // Will be populated by alert checker
        })
    }

    fn calculate_portfolio_summary(
        portfolio: &Portfolio,
        market_data: &HashMap<String, MarketDataPoint>,
    ) -> PortfolioSummary {
        let positions = portfolio.get_positions();
        let mut total_market_value = 0.0;
        let mut long_exposure = 0.0;
        let mut short_exposure = 0.0;
        let mut asset_allocation = HashMap::new();
        let mut sector_allocation = HashMap::new();

        for position in positions {
            if let Some(market_point) = market_data.get(&position.symbol) {
                let market_value = position.quantity as f64 * market_point.price;
                total_market_value += market_value.abs();
                
                if position.quantity > 0 {
                    long_exposure += market_value;
                } else {
                    short_exposure += market_value.abs();
                }
                
                // Asset allocation (simplified)
                *asset_allocation.entry("EQUITY".to_string()).or_insert(0.0) += market_value.abs();
                *sector_allocation.entry("TECHNOLOGY".to_string()).or_insert(0.0) += market_value.abs();
            }
        }

        let net_exposure = long_exposure - short_exposure;
        let total_exposure = long_exposure + short_exposure;
        let leverage = if total_market_value > 0.0 { total_exposure / total_market_value } else { 0.0 };

        // Calculate concentration ratio (top 5 positions / total)
        let mut position_values: Vec<f64> = positions.iter()
            .filter_map(|p| market_data.get(&p.symbol).map(|md| (p.quantity as f64 * md.price).abs()))
            .collect();
        position_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let top_5_value: f64 = position_values.iter().take(5).sum();
        let concentration_ratio = if total_market_value > 0.0 { top_5_value / total_market_value } else { 0.0 };
        let largest_position_pct = position_values.get(0).unwrap_or(&0.0) / total_market_value.max(1.0) * 100.0;

        PortfolioSummary {
            total_market_value,
            total_cash: portfolio.get_cash_balance(),
            total_exposure,
            net_exposure,
            long_exposure,
            short_exposure,
            leverage,
            number_of_positions: positions.len() as u32,
            number_of_symbols: positions.len() as u32,
            largest_position_pct,
            concentration_ratio,
            asset_allocation,
            sector_allocation,
            geographic_allocation: HashMap::new(),
        }
    }

    fn calculate_pnl_summary(
        portfolio: &Portfolio,
        market_data: &HashMap<String, MarketDataPoint>,
        fills: &VecDeque<Fill>,
    ) -> PnLSummary {
        let positions = portfolio.get_positions();
        let mut unrealized_pnl = 0.0;
        let mut realized_pnl = 0.0;
        let mut pnl_by_symbol = HashMap::new();

        // Calculate unrealized P&L
        for position in positions {
            if let Some(market_point) = market_data.get(&position.symbol) {
                let position_pnl = (market_point.price - position.average_price) * position.quantity as f64;
                unrealized_pnl += position_pnl;
                pnl_by_symbol.insert(position.symbol.clone(), position_pnl);
            }
        }

        // Calculate realized P&L from fills (simplified)
        for fill in fills.iter().rev().take(1000) {
            // This would need more sophisticated P&L calculation
            realized_pnl += fill.quantity as f64 * 0.01; // Placeholder
        }

        let total_pnl = unrealized_pnl + realized_pnl;

        // Calculate period P&L (would need historical data)
        let daily_pnl = total_pnl * 0.1; // Placeholder
        let weekly_pnl = total_pnl * 0.3;
        let monthly_pnl = total_pnl * 0.8;
        let ytd_pnl = total_pnl;

        PnLSummary {
            unrealized_pnl,
            realized_pnl,
            total_pnl,
            daily_pnl,
            weekly_pnl,
            monthly_pnl,
            ytd_pnl,
            pnl_by_symbol,
            pnl_by_strategy: HashMap::new(),
            pnl_by_asset_class: HashMap::new(),
            pnl_attribution: PnLAttribution {
                market_pnl: total_pnl * 0.7,
                alpha_pnl: total_pnl * 0.2,
                interaction_pnl: total_pnl * 0.05,
                trading_costs: total_pnl * -0.03,
                funding_costs: total_pnl * -0.02,
            },
            high_water_mark: total_pnl * 1.1,
            drawdown_current: total_pnl * -0.05,
            drawdown_max: total_pnl * -0.15,
        }
    }

    fn calculate_risk_metrics(
        portfolio: &Portfolio,
        market_data: &HashMap<String, MarketDataPoint>,
        var_result: &Option<VarResult>,
    ) -> RiskMetrics {
        let var_1d = var_result.as_ref().map(|v| v.var_absolute).unwrap_or(0.0);
        let expected_shortfall = var_result.as_ref().map(|v| v.expected_shortfall).unwrap_or(0.0);

        let mut risk_by_symbol = HashMap::new();
        for position in portfolio.get_positions() {
            risk_by_symbol.insert(position.symbol.clone(), SymbolRisk {
                symbol: position.symbol.clone(),
                var_contribution: var_1d * 0.1, // Simplified
                volatility: 0.25, // Would calculate from market data
                beta: 1.0,
                correlation_to_portfolio: 0.8,
                liquidity_score: 0.9,
                concentration_pct: 5.0,
            });
        }

        RiskMetrics {
            portfolio_var_1d: var_1d,
            portfolio_var_10d: var_1d * 3.16, // Sqrt(10) scaling
            expected_shortfall,
            portfolio_volatility: 0.15,
            portfolio_beta: 1.0,
            max_drawdown: 0.08,
            sharpe_ratio: 1.2,
            sortino_ratio: 1.5,
            information_ratio: 0.8,
            tracking_error: 0.05,
            risk_by_symbol,
            concentration_risk: 0.15,
            liquidity_risk: 0.05,
            correlation_risk: 0.25,
            tail_risk_measures: TailRiskMeasures {
                skewness: -0.5,
                kurtosis: 3.2,
                var_99: var_1d * 1.5,
                expected_shortfall_99: expected_shortfall * 1.3,
                maximum_loss_1d: var_1d * 2.0,
                maximum_loss_1w: var_1d * 4.0,
                stress_test_results: HashMap::new(),
            },
        }
    }

    fn calculate_order_flow_metrics(
        orders: &HashMap<String, OrderRecord>,
        fills: &VecDeque<Fill>,
    ) -> OrderFlowMetrics {
        let today = Utc::now().date_naive();
        
        let orders_today = orders.values()
            .filter(|o| o.creation_time.date_naive() == today)
            .count() as u64;

        let fills_today: Vec<&Fill> = fills.iter()
            .filter(|f| f.timestamp.date_naive() == today)
            .collect();

        let volume_today = fills_today.iter().map(|f| f.quantity as f64).sum();
        let notional_today = fills_today.iter().map(|f| f.quantity as f64 * f.price).sum();

        let filled_orders = orders.values()
            .filter(|o| o.total_filled_quantity > 0)
            .count();
        let fill_rate = if orders_today > 0 { filled_orders as f64 / orders_today as f64 } else { 0.0 };

        OrderFlowMetrics {
            orders_today,
            fills_today: fills_today.len() as u64,
            volume_today,
            notional_today,
            average_fill_time_ms: 150.0, // Would calculate from actual data
            fill_rate,
            cancel_rate: 0.05,
            reject_rate: 0.01,
            order_flow_by_venue: HashMap::new(),
            order_flow_by_strategy: HashMap::new(),
            market_impact_bps: 2.5,
            implementation_shortfall_bps: 5.2,
            slippage_bps: 3.1,
        }
    }

    fn calculate_performance_metrics(
        portfolio: &Portfolio,
        fills: &VecDeque<Fill>,
    ) -> PerformanceMetrics {
        // This would require historical portfolio values
        // Simplified calculation for demonstration
        
        PerformanceMetrics {
            total_return: 0.125, // 12.5%
            benchmark_return: 0.08, // 8%
            active_return: 0.045, // 4.5%
            volatility: 0.15,
            benchmark_volatility: 0.12,
            tracking_error: 0.05,
            information_ratio: 0.9,
            sharpe_ratio: 1.2,
            calmar_ratio: 1.8,
            max_drawdown: 0.08,
            win_rate: 0.65,
            profit_factor: 1.8,
            average_win: 0.02,
            average_loss: -0.011,
            largest_win: 0.085,
            largest_loss: -0.045,
            consecutive_wins: 5,
            consecutive_losses: 2,
            performance_by_period: HashMap::new(),
        }
    }

    fn generate_position_snapshots(
        portfolio: &Portfolio,
        market_data: &HashMap<String, MarketDataPoint>,
    ) -> Vec<PositionSnapshot> {
        portfolio.get_positions().iter().filter_map(|position| {
            market_data.get(&position.symbol).map(|market_point| {
                let market_value = position.quantity as f64 * market_point.price;
                let unrealized_pnl = (market_point.price - position.average_price) * position.quantity as f64;
                
                PositionSnapshot {
                    symbol: position.symbol.clone(),
                    quantity: position.quantity,
                    market_value,
                    unrealized_pnl,
                    realized_pnl: 0.0, // Would track from trade history
                    total_pnl: unrealized_pnl,
                    average_cost: position.average_price,
                    current_price: market_point.price,
                    daily_change: market_point.daily_change,
                    daily_change_pct: market_point.daily_change_pct,
                    weight_pct: market_value / 1000000.0 * 100.0, // Assume $1M portfolio
                    sector: "TECHNOLOGY".to_string(), // Would lookup
                    asset_class: "EQUITY".to_string(),
                    currency: "USD".to_string(),
                    beta: 1.0, // Would calculate
                    volatility: 0.25,
                    var_contribution: 0.05,
                    liquidity_score: 0.9,
                    last_updated: market_point.timestamp,
                }
            })
        }).collect()
    }

    fn generate_order_snapshots(orders: &HashMap<String, OrderRecord>) -> Vec<OrderSnapshot> {
        orders.values().filter_map(|order_record| {
            // Only include active orders
            if matches!(order_record.state, crate::order_management::OrderState::New | 
                       crate::order_management::OrderState::PartiallyFilled) {
                Some(OrderSnapshot {
                    order_id: order_record.order.id.clone(),
                    symbol: order_record.order.symbol.clone(),
                    side: format!("{:?}", order_record.order.side),
                    order_type: format!("{:?}", order_record.order.order_type),
                    quantity: order_record.order.quantity,
                    filled_quantity: order_record.total_filled_quantity,
                    remaining_quantity: order_record.remaining_quantity,
                    price: order_record.order.price,
                    average_fill_price: order_record.average_fill_price,
                    status: format!("{:?}", order_record.state),
                    created_at: order_record.creation_time,
                    updated_at: order_record.last_update_time,
                    time_in_force: format!("{:?}", order_record.order.time_in_force),
                    strategy: order_record.parent_order_id.clone(),
                    venue: "NASDAQ".to_string(), // Would track actual venue
                    estimated_fill_time_sec: 30.0,
                })
            } else {
                None
            }
        }).collect()
    }

    fn generate_fill_snapshots(fills: &VecDeque<Fill>, limit: usize) -> Vec<FillSnapshot> {
        fills.iter().rev().take(limit).map(|fill| {
            FillSnapshot {
                fill_id: fill.id.clone(),
                order_id: fill.order_id.clone(),
                symbol: "AAPL".to_string(), // Would lookup from order
                side: format!("{:?}", fill.side.unwrap_or(crate::trading::OrderSide::Buy)),
                quantity: fill.quantity,
                price: fill.price,
                timestamp: fill.timestamp,
                venue: fill.exchange.clone(),
                commission: fill.commission,
                market_impact_bps: 2.5, // Would calculate
                implementation_shortfall_bps: 5.0,
            }
        }).collect()
    }

    async fn check_for_alerts(
        snapshot: &DashboardSnapshot,
        alert_queue: &Arc<Mutex<VecDeque<Alert>>>,
    ) -> Vec<Alert> {
        let mut alerts = Vec::new();

        // Check risk limits
        if snapshot.risk_metrics.portfolio_var_1d > 100000.0 {
            alerts.push(Alert {
                id: Uuid::new_v4().to_string(),
                alert_type: AlertType::VaRBreach,
                severity: AlertSeverity::Warning,
                message: "Portfolio VaR exceeds $100k limit".to_string(),
                symbol: None,
                value: Some(snapshot.risk_metrics.portfolio_var_1d),
                threshold: Some(100000.0),
                timestamp: Utc::now(),
                acknowledged: false,
                auto_resolved: false,
            });
        }

        // Check position limits
        if snapshot.portfolio_summary.largest_position_pct > 10.0 {
            alerts.push(Alert {
                id: Uuid::new_v4().to_string(),
                alert_type: AlertType::ConcentrationLimit,
                severity: AlertSeverity::Warning,
                message: format!("Largest position exceeds 10% limit: {:.1}%", snapshot.portfolio_summary.largest_position_pct),
                symbol: None,
                value: Some(snapshot.portfolio_summary.largest_position_pct),
                threshold: Some(10.0),
                timestamp: Utc::now(),
                acknowledged: false,
                auto_resolved: false,
            });
        }

        // Check drawdown limits
        if snapshot.pnl_summary.drawdown_current < -0.05 {
            alerts.push(Alert {
                id: Uuid::new_v4().to_string(),
                alert_type: AlertType::DrawdownLimit,
                severity: AlertSeverity::Critical,
                message: format!("Current drawdown exceeds 5% limit: {:.1}%", snapshot.pnl_summary.drawdown_current * 100.0),
                symbol: None,
                value: Some(snapshot.pnl_summary.drawdown_current),
                threshold: Some(-0.05),
                timestamp: Utc::now(),
                acknowledged: false,
                auto_resolved: false,
            });
        }

        // Add alerts to queue
        let mut queue = alert_queue.lock().await;
        for alert in &alerts {
            queue.push_back(alert.clone());
        }

        alerts
    }

    async fn start_alert_processor(&self) {
        let alert_queue = self.alert_queue.clone();
        let dashboard_events = self.dashboard_events.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let alert = {
                    alert_queue.lock().await.pop_front()
                };
                
                if let Some(alert) = alert {
                    // Process alert (send notifications, etc.)
                    
                    // Emit alert event
                    let _ = dashboard_events.send(DashboardEvent {
                        event_id: Uuid::new_v4().to_string(),
                        event_type: DashboardEventType::AlertGenerated,
                        timestamp: Utc::now(),
                        data: serde_json::to_value(&alert).unwrap_or(serde_json::Value::Null),
                    });
                }
            }
        });
    }

    async fn start_data_cleanup(&self) {
        let historical_snapshots = self.historical_snapshots.clone();
        let fill_history = self.fill_history.clone();
        let processing_time_ms = self.processing_time_ms.clone();
        let is_running = self.is_running.clone();
        let retention_hours = self.config.historical_data_retention_hours;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let cutoff_time = Utc::now() - ChronoDuration::hours(retention_hours as i64);
                
                // Clean historical snapshots
                {
                    let mut snapshots = historical_snapshots.write().unwrap();
                    snapshots.retain(|snapshot| snapshot.timestamp > cutoff_time);
                }
                
                // Clean fill history
                {
                    let mut fills = fill_history.write().unwrap();
                    fills.retain(|fill| fill.timestamp > cutoff_time);
                }
                
                // Keep processing times bounded
                {
                    let mut times = processing_time_ms.write().unwrap();
                    while times.len() > 1000 {
                        times.pop_front();
                    }
                }
            }
        });
    }

    /// Get dashboard statistics
    pub fn get_statistics(&self) -> DashboardStatistics {
        let processing_times = self.processing_time_ms.read().unwrap();
        let avg_processing_time = if !processing_times.is_empty() {
            processing_times.iter().sum::<u64>() as f64 / processing_times.len() as f64
        } else {
            0.0
        };

        DashboardStatistics {
            updates_processed: self.update_count.load(Ordering::Relaxed),
            average_processing_time_ms: avg_processing_time,
            last_update: self.last_update_time.read().unwrap().clone(),
            snapshots_stored: self.historical_snapshots.read().unwrap().len() as u64,
            symbols_tracked: self.market_data_cache.read().unwrap().len() as u64,
            active_orders: self.order_cache.read().unwrap().values()
                .filter(|o| matches!(o.state, crate::order_management::OrderState::New | 
                           crate::order_management::OrderState::PartiallyFilled))
                .count() as u64,
            fills_tracked: self.fill_history.read().unwrap().len() as u64,
            pending_alerts: self.alert_queue.try_lock().map(|q| q.len() as u64).unwrap_or(0),
        }
    }

    /// Subscribe to dashboard events
    pub fn subscribe_events(&self) -> broadcast::Receiver<DashboardEvent> {
        self.dashboard_events.subscribe()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatistics {
    pub updates_processed: u64,
    pub average_processing_time_ms: f64,
    pub last_update: Option<Instant>,
    pub snapshots_stored: u64,
    pub symbols_tracked: u64,
    pub active_orders: u64,
    pub fills_tracked: u64,
    pub pending_alerts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_config_creation() {
        let config = DashboardConfig {
            update_frequency_ms: 1000,
            historical_data_retention_hours: 24,
            enable_real_time_pnl: true,
            enable_risk_monitoring: true,
            enable_order_flow_analysis: true,
            enable_performance_analytics: true,
            max_symbols_tracked: 10000,
            websocket_port: 8080,
            authentication_required: false,
        };
        
        assert_eq!(config.update_frequency_ms, 1000);
        assert!(config.enable_real_time_pnl);
    }

    #[tokio::test]
    async fn test_dashboard_creation() {
        let config = DashboardConfig {
            update_frequency_ms: 100,
            historical_data_retention_hours: 1,
            enable_real_time_pnl: true,
            enable_risk_monitoring: true,
            enable_order_flow_analysis: true,
            enable_performance_analytics: true,
            max_symbols_tracked: 100,
            websocket_port: 0,
            authentication_required: false,
        };
        
        let dashboard = RealTimeDashboard::new(config);
        assert!(!dashboard.is_running.load(Ordering::Relaxed));
    }
}
