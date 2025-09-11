/*!
 * Commodities Trading Engine
 * Comprehensive commodity futures trading with energy, metals, and agricultural markets
 */

use std::{
    collections::{HashMap, BTreeMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::interval,
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration, NaiveDate};
use uuid::Uuid;

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, Fill, OrderSide},
    market_data::MarketData,
    risk_management::RiskManager,
    execution::ExecutionEngine,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityConfig {
    pub supported_sectors: Vec<CommoditySector>,
    pub supported_exchanges: Vec<CommodityExchange>,
    pub enable_physical_delivery: bool,
    pub enable_curve_trading: bool,
    pub enable_spread_trading: bool,
    pub enable_crack_spreads: bool,
    pub max_position_limits: HashMap<String, f64>,
    pub storage_costs: HashMap<String, f64>,
    pub transportation_costs: HashMap<String, f64>,
    pub quality_specifications: HashMap<String, QualitySpec>,
    pub seasonal_adjustments: bool,
    pub weather_integration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommoditySector {
    Energy,         // Oil, Gas, Power, Coal
    PreciousMetals, // Gold, Silver, Platinum, Palladium
    BaseMetals,     // Copper, Aluminum, Zinc, Nickel
    Agriculture,    // Grains, Livestock, Softs
    Livestock,      // Cattle, Hogs, Feeder Cattle
    Softs,         // Coffee, Sugar, Cocoa, Cotton
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommodityExchange {
    NYMEX,    // New York Mercantile Exchange
    COMEX,    // Commodity Exchange
    CBOT,     // Chicago Board of Trade
    CME,      // Chicago Mercantile Exchange
    ICE,      // Intercontinental Exchange
    LME,      // London Metal Exchange
    SHFE,     // Shanghai Futures Exchange
    DCE,      // Dalian Commodity Exchange
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityInstrument {
    pub symbol: String,
    pub name: String,
    pub sector: CommoditySector,
    pub exchange: CommodityExchange,
    pub contract_size: f64,
    pub tick_size: f64,
    pub tick_value: f64,
    pub currency: String,
    pub delivery_months: Vec<String>,
    pub first_notice_day: Option<NaiveDate>,
    pub last_trading_day: NaiveDate,
    pub settlement_method: SettlementMethod,
    pub delivery_locations: Vec<DeliveryLocation>,
    pub quality_specs: QualitySpec,
    pub storage_rate: f64,        // Per unit per day
    pub insurance_rate: f64,      // Percentage of value
    pub margin_requirements: MarginRequirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SettlementMethod {
    PhysicalDelivery,
    CashSettlement,
    FinancialSettlement,
    ElectiveDelivery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryLocation {
    pub location_id: String,
    pub name: String,
    pub country: String,
    pub coordinates: (f64, f64), // lat, lng
    pub storage_capacity: f64,
    pub handling_costs: f64,
    pub premium_discount: f64,   // Basis adjustment
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySpec {
    pub grade: String,
    pub purity: Option<f64>,        // Percentage for metals
    pub sulfur_content: Option<f64>, // For energy products
    pub moisture_content: Option<f64>, // For agriculture
    pub protein_content: Option<f64>, // For grains
    pub specific_gravity: Option<f64>, // For liquids
    pub deliverable_grades: Vec<DeliverableGrade>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliverableGrade {
    pub grade_name: String,
    pub premium_discount: f64,   // Price adjustment
    pub conversion_factor: f64,
    pub quality_tolerances: HashMap<String, (f64, f64)>, // min, max
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginRequirement {
    pub initial_margin: f64,
    pub maintenance_margin: f64,
    pub delivery_margin: f64,
    pub inter_month_margin: f64,  // For calendar spreads
    pub inter_commodity_margin: f64, // For related commodities
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityCurve {
    pub curve_id: String,
    pub commodity: String,
    pub curve_date: NaiveDate,
    pub curve_points: BTreeMap<NaiveDate, f64>, // maturity -> price
    pub curve_type: CurveType,
    pub interpolation_method: InterpolationMethod,
    pub seasonality_factors: HashMap<u32, f64>, // month -> factor
    pub storage_costs: f64,
    pub convenience_yield: f64,
    pub forward_curve_model: ForwardCurveModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurveType {
    Spot,
    Forward,
    Strip,      // Average price over period
    Swing,      // Option-like structure
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Linear,
    CubicSpline,
    StepFunction,
    SeasonalAdjusted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForwardCurveModel {
    CostOfCarry,
    ConvenienceYield,
    SeasonalModel,
    MeanReversion,
    JumpDiffusion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadTrade {
    pub spread_id: String,
    pub spread_type: SpreadType,
    pub legs: Vec<SpreadLeg>,
    pub net_price: f64,
    pub total_quantity: u64,
    pub margin_requirement: f64,
    pub risk_metrics: SpreadRiskMetrics,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpreadType {
    CalendarSpread,   // Same commodity, different months
    InterCommodity,   // Related commodities
    CrackSpread,      // Crude oil vs refined products
    SparkSpread,      // Natural gas vs electricity
    CrushSpread,      // Soybeans vs meal and oil
    LocationSpread,   // Same commodity, different locations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadLeg {
    pub instrument: String,
    pub side: OrderSide,
    pub quantity: u64,
    pub weight: f64,             // Ratio in spread
    pub delivery_month: String,
    pub price: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadRiskMetrics {
    pub delta: f64,              // Price sensitivity
    pub gamma: f64,              // Convexity
    pub theta: f64,              // Time decay
    pub vega: f64,               // Volatility sensitivity
    pub correlation_risk: f64,   // Correlation between legs
    pub basis_risk: f64,         // Basis convergence risk
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalDelivery {
    pub delivery_id: String,
    pub contract_id: String,
    pub commodity: String,
    pub quantity: f64,
    pub quality_grade: String,
    pub delivery_location: DeliveryLocation,
    pub delivery_date: NaiveDate,
    pub warehouse_receipt: Option<String>,
    pub inspection_certificate: Option<String>,
    pub shipping_documents: Vec<String>,
    pub delivery_status: DeliveryStatus,
    pub storage_costs_accrued: f64,
    pub handling_fees: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryStatus {
    NoticeIssued,
    Pending,
    InTransit,
    Delivered,
    Accepted,
    Rejected,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub location: String,
    pub temperature: f64,
    pub precipitation: f64,
    pub humidity: f64,
    pub wind_speed: f64,
    pub pressure: f64,
    pub forecast_confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub impact_assessment: WeatherImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherImpact {
    pub crop_yield_impact: f64,      // Percentage impact on yield
    pub demand_impact: f64,          // Impact on energy demand
    pub transportation_impact: f64,   // Impact on logistics
    pub storage_impact: f64,         // Impact on storage conditions
}

pub struct CommodityTradingEngine {
    config: CommodityConfig,
    
    // Instrument universe
    instruments: Arc<RwLock<HashMap<String, CommodityInstrument>>>,
    
    // Market data and curves
    commodity_prices: Arc<RwLock<HashMap<String, f64>>>,
    forward_curves: Arc<RwLock<HashMap<String, CommodityCurve>>>,
    
    // Trading and positions
    positions: Arc<RwLock<HashMap<String, CommodityPosition>>>,
    spread_trades: Arc<RwLock<HashMap<String, SpreadTrade>>>,
    
    // Physical delivery
    delivery_tracker: Arc<RwLock<HashMap<String, PhysicalDelivery>>>,
    warehouse_inventory: Arc<RwLock<HashMap<String, f64>>>,
    
    // Weather and fundamentals
    weather_data: Arc<RwLock<HashMap<String, WeatherData>>>,
    fundamental_data: Arc<RwLock<HashMap<String, FundamentalData>>>,
    
    // Analytics engines
    curve_builder: Arc<CurveBuilder>,
    spread_analyzer: Arc<SpreadAnalyzer>,
    seasonality_analyzer: Arc<SeasonalityAnalyzer>,
    weather_analyzer: Arc<WeatherAnalyzer>,
    
    // External systems
    execution_engine: Arc<ExecutionEngine>,
    risk_manager: Arc<RiskManager>,
    
    // Event handling
    commodity_events: broadcast::Sender<CommodityEvent>,
    
    // Performance tracking
    trades_executed: Arc<AtomicU64>,
    physical_deliveries: Arc<AtomicU64>,
    spread_trades_executed: Arc<AtomicU64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityPosition {
    pub instrument: String,
    pub position_size: f64,          // Number of contracts
    pub average_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub margin_used: f64,
    pub delivery_exposure: f64,      // Contracts subject to delivery
    pub storage_costs: f64,          // Accumulated storage costs
    pub quality_adjustments: f64,    // Price adjustments for grade
    pub seasonal_adjustment: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundamentalData {
    pub commodity: String,
    pub supply_data: SupplyData,
    pub demand_data: DemandData,
    pub inventory_data: InventoryData,
    pub economic_indicators: EconomicIndicators,
    pub seasonal_patterns: HashMap<u32, f64>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyData {
    pub production: f64,
    pub capacity_utilization: f64,
    pub planned_capacity: f64,
    pub disruptions: Vec<SupplyDisruption>,
    pub seasonal_factors: HashMap<u32, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyDisruption {
    pub event_type: String,
    pub severity: f64,           // 0-1 scale
    pub duration_days: u32,
    pub affected_capacity: f64,
    pub recovery_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandData {
    pub consumption: f64,
    pub growth_rate: f64,
    pub elasticity: f64,
    pub substitution_effects: HashMap<String, f64>,
    pub seasonal_demand: HashMap<u32, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryData {
    pub total_stocks: f64,
    pub commercial_stocks: f64,
    pub strategic_reserves: f64,
    pub days_of_supply: f64,
    pub inventory_turnover: f64,
    pub storage_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicIndicators {
    pub gdp_growth: f64,
    pub industrial_production: f64,
    pub exchange_rates: HashMap<String, f64>,
    pub interest_rates: HashMap<String, f64>,
    pub inflation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityEvent {
    pub event_id: String,
    pub event_type: CommodityEventType,
    pub timestamp: DateTime<Utc>,
    pub commodity: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommodityEventType {
    PriceUpdate,
    CurveUpdate,
    SpreadTradeExecuted,
    DeliveryNotice,
    WeatherAlert,
    SupplyDisruption,
    StorageUpdate,
    SeasonalAdjustment,
}

// Supporting analytics engines
pub struct CurveBuilder {
    interpolation_methods: HashMap<String, InterpolationMethod>,
    seasonality_models: HashMap<String, SeasonalityModel>,
}

#[derive(Debug, Clone)]
struct SeasonalityModel {
    seasonal_factors: HashMap<u32, f64>, // month -> factor
    trend_component: f64,
    volatility_seasonality: HashMap<u32, f64>,
}

pub struct SpreadAnalyzer {
    correlation_calculator: CorrelationCalculator,
    spread_models: HashMap<SpreadType, SpreadModel>,
}

#[derive(Debug, Clone)]
struct SpreadModel {
    mean_reversion_rate: f64,
    long_term_mean: f64,
    volatility: f64,
    jump_intensity: f64,
}

#[derive(Debug, Clone)]
struct CorrelationCalculator {
    correlation_matrix: HashMap<(String, String), f64>,
    lookback_periods: Vec<u32>,
}

pub struct SeasonalityAnalyzer {
    seasonal_patterns: HashMap<String, SeasonalPattern>,
    weather_correlations: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
struct SeasonalPattern {
    commodity: String,
    monthly_factors: [f64; 12],
    volatility_factors: [f64; 12],
    demand_drivers: Vec<String>,
    supply_drivers: Vec<String>,
}

pub struct WeatherAnalyzer {
    weather_models: HashMap<String, WeatherModel>,
    impact_calculators: HashMap<String, ImpactCalculator>,
}

#[derive(Debug, Clone)]
struct WeatherModel {
    temperature_sensitivity: f64,
    precipitation_sensitivity: f64,
    seasonal_adjustments: HashMap<u32, f64>,
}

#[derive(Debug, Clone)]
struct ImpactCalculator {
    yield_elasticity: f64,
    demand_elasticity: f64,
    price_impact_model: String,
}

impl CommodityTradingEngine {
    pub fn new(
        config: CommodityConfig,
        execution_engine: Arc<ExecutionEngine>,
        risk_manager: Arc<RiskManager>,
    ) -> Self {
        let (commodity_events, _) = broadcast::channel(1000);
        
        Self {
            config,
            instruments: Arc::new(RwLock::new(HashMap::new())),
            commodity_prices: Arc::new(RwLock::new(HashMap::new())),
            forward_curves: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            spread_trades: Arc::new(RwLock::new(HashMap::new())),
            delivery_tracker: Arc::new(RwLock::new(HashMap::new())),
            warehouse_inventory: Arc::new(RwLock::new(HashMap::new())),
            weather_data: Arc::new(RwLock::new(HashMap::new())),
            fundamental_data: Arc::new(RwLock::new(HashMap::new())),
            curve_builder: Arc::new(CurveBuilder::new()),
            spread_analyzer: Arc::new(SpreadAnalyzer::new()),
            seasonality_analyzer: Arc::new(SeasonalityAnalyzer::new()),
            weather_analyzer: Arc::new(WeatherAnalyzer::new()),
            execution_engine,
            risk_manager,
            commodity_events,
            trades_executed: Arc::new(AtomicU64::new(0)),
            physical_deliveries: Arc::new(AtomicU64::new(0)),
            spread_trades_executed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Add commodity instrument
    pub async fn add_instrument(&self, instrument: CommodityInstrument) -> Result<()> {
        let symbol = instrument.symbol.clone();
        self.instruments.write().unwrap().insert(symbol, instrument);
        Ok(())
    }

    /// Update commodity price
    pub async fn update_price(&self, symbol: String, price: f64) -> Result<()> {
        self.commodity_prices.write().unwrap().insert(symbol.clone(), price);
        
        // Update position P&L
        self.update_position_pnl(&symbol).await?;
        
        // Update forward curve if applicable
        self.update_forward_curve(&symbol, price).await?;
        
        // Emit price update event
        let _ = self.commodity_events.send(CommodityEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CommodityEventType::PriceUpdate,
            timestamp: Utc::now(),
            commodity: symbol,
            data: serde_json::json!({ "price": price }),
        });
        
        Ok(())
    }

    /// Execute commodity trade
    pub async fn execute_trade(
        &self,
        symbol: String,
        side: OrderSide,
        quantity: u64,
        price: Option<f64>,
        delivery_month: Option<String>,
    ) -> Result<String> {
        // Validate instrument exists
        let instrument = self.instruments.read().unwrap()
            .get(&symbol)
            .cloned()
            .ok_or_else(|| AlgoVedaError::Commodity(format!("Instrument not found: {}", symbol)))?;

        // Risk checks
        self.risk_manager.validate_commodity_order(&instrument, side.clone(), quantity, price.unwrap_or(0.0))?;

        // Check position limits
        let position_limit = self.config.max_position_limits.get(&symbol).copied().unwrap_or(1000.0);
        let current_position = self.get_position_size(&symbol).await;
        let new_position = match side {
            OrderSide::Buy => current_position + quantity as f64,
            OrderSide::Sell => current_position - quantity as f64,
        };

        if new_position.abs() > position_limit {
            return Err(AlgoVedaError::Commodity("Position limit exceeded".to_string()));
        }

        // Create order
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.clone(),
            side,
            quantity,
            order_type: if price.is_some() { 
                crate::trading::OrderType::Limit 
            } else { 
                crate::trading::OrderType::Market 
            },
            price,
            time_in_force: crate::trading::TimeInForce::Day,
            status: crate::trading::OrderStatus::PendingNew,
            parent_order_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Submit to execution engine
        let execution_id = self.execution_engine.submit_order(order).await?;
        
        self.trades_executed.fetch_add(1, Ordering::Relaxed);
        
        Ok(execution_id)
    }

    /// Execute spread trade
    pub async fn execute_spread_trade(&self, spread: SpreadTrade) -> Result<String> {
        let spread_id = spread.spread_id.clone();
        
        // Validate all legs
        for leg in &spread.legs {
            if !self.instruments.read().unwrap().contains_key(&leg.instrument) {
                return Err(AlgoVedaError::Commodity(format!("Instrument not found: {}", leg.instrument)));
            }
        }

        // Calculate spread margin requirement
        let margin_required = self.calculate_spread_margin(&spread).await?;
        
        // Execute each leg
        for leg in &spread.legs {
            let leg_order = Order {
                id: Uuid::new_v4().to_string(),
                symbol: leg.instrument.clone(),
                side: leg.side.clone(),
                quantity: leg.quantity,
                order_type: crate::trading::OrderType::Limit,
                price: leg.price,
                time_in_force: crate::trading::TimeInForce::Day,
                status: crate::trading::OrderStatus::PendingNew,
                parent_order_id: Some(spread_id.clone()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            self.execution_engine.submit_order(leg_order).await?;
        }

        // Store spread trade
        self.spread_trades.write().unwrap().insert(spread_id.clone(), spread);
        
        self.spread_trades_executed.fetch_add(1, Ordering::Relaxed);
        
        // Emit spread trade event
        let _ = self.commodity_events.send(CommodityEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CommodityEventType::SpreadTradeExecuted,
            timestamp: Utc::now(),
            commodity: "SPREAD".to_string(),
            data: serde_json::json!({ "spread_id": spread_id }),
        });
        
        Ok(spread_id)
    }

    /// Build forward curve
    pub async fn build_forward_curve(
        &self,
        commodity: String,
        curve_type: CurveType,
        market_prices: Vec<(NaiveDate, f64)>,
    ) -> Result<CommodityCurve> {
        let curve = self.curve_builder.build_curve(
            commodity.clone(),
            curve_type,
            market_prices,
        ).await?;
        
        // Store curve
        self.forward_curves.write().unwrap().insert(commodity.clone(), curve.clone());
        
        // Emit curve update event
        let _ = self.commodity_events.send(CommodityEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CommodityEventType::CurveUpdate,
            timestamp: Utc::now(),
            commodity,
            data: serde_json::to_value(&curve).unwrap_or(serde_json::Value::Null),
        });
        
        Ok(curve)
    }

    /// Process delivery notice
    pub async fn process_delivery_notice(
        &self,
        contract_symbol: String,
        quantity: f64,
        delivery_location: DeliveryLocation,
        delivery_date: NaiveDate,
    ) -> Result<String> {
        let delivery_id = Uuid::new_v4().to_string();
        
        let delivery = PhysicalDelivery {
            delivery_id: delivery_id.clone(),
            contract_id: contract_symbol.clone(),
            commodity: contract_symbol.clone(),
            quantity,
            quality_grade: "Standard".to_string(), // Would determine from contract
            delivery_location,
            delivery_date,
            warehouse_receipt: None,
            inspection_certificate: None,
            shipping_documents: Vec::new(),
            delivery_status: DeliveryStatus::NoticeIssued,
            storage_costs_accrued: 0.0,
            handling_fees: 0.0,
        };
        
        // Store delivery
        self.delivery_tracker.write().unwrap().insert(delivery_id.clone(), delivery);
        
        // Update inventory
        let location_id = delivery_location.location_id;
        *self.warehouse_inventory.write().unwrap().entry(location_id).or_insert(0.0) += quantity;
        
        self.physical_deliveries.fetch_add(1, Ordering::Relaxed);
        
        // Emit delivery event
        let _ = self.commodity_events.send(CommodityEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: CommodityEventType::DeliveryNotice,
            timestamp: Utc::now(),
            commodity: contract_symbol,
            data: serde_json::json!({ "delivery_id": delivery_id, "quantity": quantity }),
        });
        
        Ok(delivery_id)
    }

    /// Update weather data
    pub async fn update_weather_data(&self, weather: WeatherData) -> Result<()> {
        let location = weather.location.clone();
        
        // Calculate weather impact
        let impact = self.weather_analyzer.calculate_impact(&weather).await?;
        
        let mut updated_weather = weather;
        updated_weather.impact_assessment = impact;
        
        self.weather_data.write().unwrap().insert(location.clone(), updated_weather);
        
        // Check for significant weather events
        if impact.crop_yield_impact.abs() > 0.1 || impact.demand_impact.abs() > 0.05 {
            let _ = self.commodity_events.send(CommodityEvent {
                event_id: Uuid::new_v4().to_string(),
                event_type: CommodityEventType::WeatherAlert,
                timestamp: Utc::now(),
                commodity: "WEATHER".to_string(),
                data: serde_json::to_value(&impact).unwrap_or(serde_json::Value::Null),
            });
        }
        
        Ok(())
    }

    /// Calculate seasonal adjustment
    pub async fn calculate_seasonal_adjustment(&self, commodity: &str, month: u32) -> f64 {
        self.seasonality_analyzer.get_seasonal_factor(commodity, month).await
    }

    /// Helper methods
    async fn update_position_pnl(&self, symbol: &str) -> Result<()> {
        if let Some(current_price) = self.commodity_prices.read().unwrap().get(symbol).copied() {
            let mut positions = self.positions.write().unwrap();
            
            if let Some(position) = positions.get_mut(symbol) {
                let price_change = current_price - position.average_price;
                position.unrealized_pnl = position.position_size * price_change;
                position.last_updated = Utc::now();
            }
        }
        
        Ok(())
    }

    async fn update_forward_curve(&self, symbol: &str, spot_price: f64) -> Result<()> {
        // Update forward curve based on new spot price
        // This would implement curve bootstrapping logic
        Ok(())
    }

    async fn get_position_size(&self, symbol: &str) -> f64 {
        self.positions.read().unwrap()
            .get(symbol)
            .map(|p| p.position_size)
            .unwrap_or(0.0)
    }

    async fn calculate_spread_margin(&self, spread: &SpreadTrade) -> Result<f64> {
        // Calculate margin requirement for spread trade
        let base_margin: f64 = spread.legs.iter()
            .map(|leg| {
                // Get instrument margin requirement
                self.instruments.read().unwrap()
                    .get(&leg.instrument)
                    .map(|inst| inst.margin_requirements.initial_margin * leg.quantity as f64)
                    .unwrap_or(0.0)
            })
            .sum();
        
        // Apply spread margin offset (typically lower than sum of individual margins)
        let spread_offset = match spread.spread_type {
            SpreadType::CalendarSpread => 0.2,     // 20% of individual margins
            SpreadType::InterCommodity => 0.5,     // 50% of individual margins
            SpreadType::CrackSpread => 0.3,        // 30% of individual margins
            _ => 0.6,                              // 60% default
        };
        
        Ok(base_margin * spread_offset)
    }

    /// Get statistics
    pub fn get_statistics(&self) -> CommodityStatistics {
        let instruments = self.instruments.read().unwrap();
        let positions = self.positions.read().unwrap();
        let spread_trades = self.spread_trades.read().unwrap();
        
        CommodityStatistics {
            instruments_tracked: instruments.len() as u64,
            active_positions: positions.len() as u64,
            trades_executed: self.trades_executed.load(Ordering::Relaxed),
            spread_trades_executed: self.spread_trades_executed.load(Ordering::Relaxed),
            physical_deliveries: self.physical_deliveries.load(Ordering::Relaxed),
            total_unrealized_pnl: positions.values().map(|p| p.unrealized_pnl).sum(),
            total_realized_pnl: positions.values().map(|p| p.realized_pnl).sum(),
            active_spread_trades: spread_trades.len() as u64,
            warehouse_locations: self.warehouse_inventory.read().unwrap().len() as u64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommodityStatistics {
    pub instruments_tracked: u64,
    pub active_positions: u64,
    pub trades_executed: u64,
    pub spread_trades_executed: u64,
    pub physical_deliveries: u64,
    pub total_unrealized_pnl: f64,
    pub total_realized_pnl: f64,
    pub active_spread_trades: u64,
    pub warehouse_locations: u64,
}

// Implementation of supporting engines
impl CurveBuilder {
    fn new() -> Self {
        Self {
            interpolation_methods: HashMap::new(),
            seasonality_models: HashMap::new(),
        }
    }

    async fn build_curve(
        &self,
        commodity: String,
        curve_type: CurveType,
        market_prices: Vec<(NaiveDate, f64)>,
    ) -> Result<CommodityCurve> {
        let mut curve_points = BTreeMap::new();
        
        // Simple interpolation for demonstration
        for (date, price) in market_prices {
            curve_points.insert(date, price);
        }
        
        Ok(CommodityCurve {
            curve_id: format!("{}_{:?}", commodity, curve_type),
            commodity,
            curve_date: Utc::now().date_naive(),
            curve_points,
            curve_type,
            interpolation_method: InterpolationMethod::Linear,
            seasonality_factors: HashMap::new(),
            storage_costs: 0.0,
            convenience_yield: 0.0,
            forward_curve_model: ForwardCurveModel::CostOfCarry,
        })
    }
}

impl SpreadAnalyzer {
    fn new() -> Self {
        Self {
            correlation_calculator: CorrelationCalculator {
                correlation_matrix: HashMap::new(),
                lookback_periods: vec![20, 60, 252],
            },
            spread_models: HashMap::new(),
        }
    }
}

impl SeasonalityAnalyzer {
    fn new() -> Self {
        Self {
            seasonal_patterns: HashMap::new(),
            weather_correlations: HashMap::new(),
        }
    }

    async fn get_seasonal_factor(&self, commodity: &str, month: u32) -> f64 {
        self.seasonal_patterns.get(commodity)
            .and_then(|pattern| pattern.monthly_factors.get((month - 1) as usize))
            .copied()
            .unwrap_or(1.0)
    }
}

impl WeatherAnalyzer {
    fn new() -> Self {
        Self {
            weather_models: HashMap::new(),
            impact_calculators: HashMap::new(),
        }
    }

    async fn calculate_impact(&self, weather: &WeatherData) -> Result<WeatherImpact> {
        // Simplified weather impact calculation
        Ok(WeatherImpact {
            crop_yield_impact: if weather.temperature > 35.0 { -0.1 } else { 0.0 },
            demand_impact: if weather.temperature < 0.0 { 0.15 } else { 0.0 },
            transportation_impact: if weather.precipitation > 50.0 { -0.05 } else { 0.0 },
            storage_impact: if weather.humidity > 80.0 { -0.02 } else { 0.0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commodity_instrument_creation() {
        let instrument = CommodityInstrument {
            symbol: "CLZ3".to_string(),
            name: "Crude Oil December 2023".to_string(),
            sector: CommoditySector::Energy,
            exchange: CommodityExchange::NYMEX,
            contract_size: 1000.0,
            tick_size: 0.01,
            tick_value: 10.0,
            currency: "USD".to_string(),
            delivery_months: vec!["DEC".to_string()],
            first_notice_day: Some(NaiveDate::from_ymd_opt(2023, 11, 20).unwrap()),
            last_trading_day: NaiveDate::from_ymd_opt(2023, 11, 17).unwrap(),
            settlement_method: SettlementMethod::PhysicalDelivery,
            delivery_locations: vec![],
            quality_specs: QualitySpec {
                grade: "WTI".to_string(),
                purity: None,
                sulfur_content: Some(0.42),
                moisture_content: None,
                protein_content: None,
                specific_gravity: Some(0.827),
                deliverable_grades: vec![],
            },
            storage_rate: 0.05,
            insurance_rate: 0.001,
            margin_requirements: MarginRequirement {
                initial_margin: 5000.0,
                maintenance_margin: 3500.0,
                delivery_margin: 7500.0,
                inter_month_margin: 1000.0,
                inter_commodity_margin: 2500.0,
            },
        };

        assert_eq!(instrument.symbol, "CLZ3");
        assert_eq!(instrument.sector, CommoditySector::Energy);
    }

    #[test]
    fn test_spread_trade_creation() {
        let spread = SpreadTrade {
            spread_id: "CL_CALENDAR_SPREAD".to_string(),
            spread_type: SpreadType::CalendarSpread,
            legs: vec![
                SpreadLeg {
                    instrument: "CLZ3".to_string(),
                    side: OrderSide::Buy,
                    quantity: 10,
                    weight: 1.0,
                    delivery_month: "DEC23".to_string(),
                    price: Some(75.50),
                },
                SpreadLeg {
                    instrument: "CLF4".to_string(),
                    side: OrderSide::Sell,
                    quantity: 10,
                    weight: -1.0,
                    delivery_month: "JAN24".to_string(),
                    price: Some(74.80),
                },
            ],
            net_price: 0.70, // $0.70 premium for Dec vs Jan
            total_quantity: 10,
            margin_requirement: 1000.0,
            risk_metrics: SpreadRiskMetrics {
                delta: 0.95,
                gamma: 0.02,
                theta: -0.01,
                vega: 0.10,
                correlation_risk: 0.05,
                basis_risk: 0.03,
            },
            created_at: Utc::now(),
        };

        assert_eq!(spread.spread_type, SpreadType::CalendarSpread);
        assert_eq!(spread.legs.len(), 2);
        assert_eq!(spread.net_price, 0.70);
    }
}
