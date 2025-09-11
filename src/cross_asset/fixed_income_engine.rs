/*!
 * Fixed Income Trading Engine
 * Comprehensive bond trading, yield curve analytics, and credit risk management
 */

use std::{
    collections::{HashMap, BTreeMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout},
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration, NaiveDate};
use uuid::Uuid;
use nalgebra::{DMatrix, DVector};

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, Fill, OrderSide},
    market_data::MarketData,
    risk_management::RiskManager,
    execution::ExecutionEngine,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedIncomeConfig {
    pub supported_bond_types: Vec<BondType>,
    pub yield_curve_models: Vec<YieldCurveModel>,
    pub credit_models: Vec<CreditModel>,
    pub enable_duration_hedging: bool,
    pub enable_convexity_adjustment: bool,
    pub enable_credit_spread_analysis: bool,
    pub max_duration_exposure: f64,
    pub max_credit_exposure: f64,
    pub default_settlement_days: u32,
    pub curve_interpolation_method: InterpolationMethod,
    pub risk_free_curve_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BondType {
    Treasury,
    Corporate,
    Municipal,
    Agency,
    MortgageBacked,
    AssetBacked,
    Convertible,
    InflationLinked,
    International,
    Sukuk,  // Islamic bonds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum YieldCurveModel {
    NelsonSiegel,
    Svensson,
    CubicSpline,
    LinearInterpolation,
    Vasicek,
    CoxIngersollRoss,
    HullWhite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreditModel {
    Merton,
    ReducedForm,
    CreditMetrics,
    KMV,
    CreditRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Linear,
    CubicSpline,
    Hermite,
    Akima,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondInstrument {
    pub cusip: String,
    pub isin: String,
    pub ticker: String,
    pub issuer: String,
    pub bond_type: BondType,
    pub coupon_rate: f64,
    pub face_value: f64,
    pub issue_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub first_coupon_date: Option<NaiveDate>,
    pub coupon_frequency: CouponFrequency,
    pub day_count_convention: DayCountConvention,
    pub currency: String,
    pub credit_rating: CreditRating,
    pub callable: bool,
    pub putable: bool,
    pub convertible: bool,
    pub inflation_linked: bool,
    pub tax_exempt: bool,
    pub minimum_denomination: f64,
    pub increment: f64,
    pub settlement_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CouponFrequency {
    Monthly = 12,
    Quarterly = 4,
    SemiAnnual = 2,
    Annual = 1,
    ZeroCoupon = 0,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DayCountConvention {
    Actual360,
    Actual365,
    ActualActual,
    Thirty360,
    ThirtyE360,
    ActualActualICMA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditRating {
    pub agency: RatingAgency,
    pub rating: String,
    pub outlook: Outlook,
    pub watch: bool,
    pub numeric_score: f64,  // Internal numeric mapping
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RatingAgency {
    Moodys,
    SP,
    Fitch,
    DBRS,
    JCR,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outlook {
    Positive,
    Stable,
    Negative,
    Developing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YieldCurve {
    pub curve_id: String,
    pub currency: String,
    pub curve_type: CurveType,
    pub as_of_date: NaiveDate,
    pub points: BTreeMap<f64, f64>,  // tenor in years -> yield
    pub model: YieldCurveModel,
    pub model_parameters: Vec<f64>,
    pub goodness_of_fit: f64,
    pub confidence_intervals: HashMap<f64, (f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurveType {
    Treasury,
    Swap,
    Corporate,
    Municipal,
    Agency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondPricing {
    pub bond: BondInstrument,
    pub clean_price: f64,
    pub dirty_price: f64,
    pub accrued_interest: f64,
    pub yield_to_maturity: f64,
    pub yield_to_call: Option<f64>,
    pub yield_to_worst: f64,
    pub duration: f64,
    pub modified_duration: f64,
    pub effective_duration: f64,
    pub convexity: f64,
    pub dv01: f64,  // Dollar value of 01
    pub spread_to_benchmark: f64,
    pub z_spread: f64,
    pub option_adjusted_spread: Option<f64>,
    pub credit_spread: f64,
    pub probability_of_default: f64,
    pub recovery_rate: f64,
    pub expected_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAnalytics {
    pub total_market_value: f64,
    pub total_par_value: f64,
    pub weighted_average_maturity: f64,
    pub weighted_average_coupon: f64,
    pub portfolio_duration: f64,
    pub portfolio_convexity: f64,
    pub portfolio_yield: f64,
    pub credit_quality_breakdown: HashMap<String, f64>,
    pub maturity_buckets: HashMap<String, f64>,
    pub sector_allocation: HashMap<String, f64>,
    pub duration_buckets: HashMap<String, f64>,
    pub key_rate_durations: HashMap<f64, f64>,
    pub spread_duration: f64,
    pub credit_spread_duration: f64,
}

pub struct FixedIncomeEngine {
    config: FixedIncomeConfig,
    
    // Bond universe
    bond_universe: Arc<RwLock<HashMap<String, BondInstrument>>>,
    
    // Yield curves
    yield_curves: Arc<RwLock<HashMap<String, YieldCurve>>>,
    
    // Market data
    bond_prices: Arc<RwLock<HashMap<String, BondPricing>>>,
    credit_spreads: Arc<RwLock<HashMap<String, f64>>>,
    
    // Analytics engines
    yield_curve_builder: Arc<YieldCurveBuilder>,
    bond_pricer: Arc<BondPricer>,
    portfolio_analyzer: Arc<PortfolioAnalyzer>,
    risk_calculator: Arc<FixedIncomeRiskCalculator>,
    
    // External integrations
    execution_engine: Arc<ExecutionEngine>,
    risk_manager: Arc<RiskManager>,
    
    // Event handling
    pricing_events: broadcast::Sender<PricingEvent>,
    
    // Performance tracking
    calculations_performed: Arc<AtomicU64>,
    last_curve_update: Arc<RwLock<Option<DateTime<Utc>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingEvent {
    pub event_id: String,
    pub event_type: PricingEventType,
    pub timestamp: DateTime<Utc>,
    pub bond_id: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PricingEventType {
    PriceUpdate,
    YieldCurveUpdate,
    CreditSpreadUpdate,
    RatingChange,
    CallableExercise,
    MaturityReached,
}

pub struct YieldCurveBuilder {
    interpolation_method: InterpolationMethod,
    optimization_method: OptimizationMethod,
}

#[derive(Debug, Clone)]
enum OptimizationMethod {
    LevenbergMarquardt,
    GaussNewton,
    SimulatedAnnealing,
    GeneticAlgorithm,
}

pub struct BondPricer {
    day_count_calculator: DayCountCalculator,
    cashflow_generator: CashflowGenerator,
    discount_engine: DiscountEngine,
}

pub struct PortfolioAnalyzer {
    duration_calculator: DurationCalculator,
    convexity_calculator: ConvexityCalculator,
    spread_analyzer: SpreadAnalyzer,
}

pub struct FixedIncomeRiskCalculator {
    var_calculator: VaRCalculator,
    stress_tester: StressTester,
    scenario_generator: ScenarioGenerator,
}

// Supporting structures
struct DayCountCalculator;
struct CashflowGenerator;
struct DiscountEngine;
struct DurationCalculator;
struct ConvexityCalculator;
struct SpreadAnalyzer;
struct VaRCalculator;
struct StressTester;
struct ScenarioGenerator;

impl FixedIncomeEngine {
    pub fn new(
        config: FixedIncomeConfig,
        execution_engine: Arc<ExecutionEngine>,
        risk_manager: Arc<RiskManager>,
    ) -> Self {
        let (pricing_events, _) = broadcast::channel(1000);
        
        Self {
            config: config.clone(),
            bond_universe: Arc::new(RwLock::new(HashMap::new())),
            yield_curves: Arc::new(RwLock::new(HashMap::new())),
            bond_prices: Arc::new(RwLock::new(HashMap::new())),
            credit_spreads: Arc::new(RwLock::new(HashMap::new())),
            yield_curve_builder: Arc::new(YieldCurveBuilder::new(config.curve_interpolation_method.clone())),
            bond_pricer: Arc::new(BondPricer::new()),
            portfolio_analyzer: Arc::new(PortfolioAnalyzer::new()),
            risk_calculator: Arc::new(FixedIncomeRiskCalculator::new()),
            execution_engine,
            risk_manager,
            pricing_events,
            calculations_performed: Arc::new(AtomicU64::new(0)),
            last_curve_update: Arc::new(RwLock::new(None)),
        }
    }

    /// Add bond to universe
    pub async fn add_bond(&self, bond: BondInstrument) -> Result<()> {
        let bond_id = bond.cusip.clone();
        self.bond_universe.write().unwrap().insert(bond_id, bond);
        Ok(())
    }

    /// Update yield curve
    pub async fn update_yield_curve(&self, curve: YieldCurve) -> Result<()> {
        let curve_id = curve.curve_id.clone();
        self.yield_curves.write().unwrap().insert(curve_id, curve);
        *self.last_curve_update.write().unwrap() = Some(Utc::now());
        
        // Reprice all bonds affected by this curve
        self.reprice_bonds_for_curve(&curve_id).await?;
        
        Ok(())
    }

    /// Price a bond
    pub async fn price_bond(&self, cusip: &str, yield_to_maturity: Option<f64>) -> Result<BondPricing> {
        let bond = self.bond_universe.read().unwrap()
            .get(cusip)
            .cloned()
            .ok_or_else(|| AlgoVedaError::FixedIncome(format!("Bond not found: {}", cusip)))?;

        let pricing = self.bond_pricer.price_bond(&bond, yield_to_maturity, &self.yield_curves).await?;
        
        // Store pricing
        self.bond_prices.write().unwrap().insert(cusip.to_string(), pricing.clone());
        
        // Emit pricing event
        let _ = self.pricing_events.send(PricingEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: PricingEventType::PriceUpdate,
            timestamp: Utc::now(),
            bond_id: cusip.to_string(),
            data: serde_json::to_value(&pricing).unwrap_or(serde_json::Value::Null),
        });

        self.calculations_performed.fetch_add(1, Ordering::Relaxed);
        
        Ok(pricing)
    }

    /// Build yield curve from market data
    pub async fn build_yield_curve(
        &self, 
        curve_id: String,
        currency: String,
        curve_type: CurveType,
        market_quotes: Vec<(f64, f64)>  // (tenor, rate)
    ) -> Result<YieldCurve> {
        let curve = self.yield_curve_builder.build_curve(
            curve_id,
            currency,
            curve_type,
            market_quotes,
        ).await?;
        
        self.update_yield_curve(curve.clone()).await?;
        
        Ok(curve)
    }

    /// Analyze portfolio
    pub async fn analyze_portfolio(&self, holdings: Vec<(String, f64)>) -> Result<PortfolioAnalytics> {
        let mut bonds = Vec::new();
        let mut weights = Vec::new();

        for (cusip, amount) in holdings {
            if let Some(bond) = self.bond_universe.read().unwrap().get(&cusip) {
                if let Some(pricing) = self.bond_prices.read().unwrap().get(&cusip) {
                    bonds.push((bond.clone(), pricing.clone()));
                    weights.push(amount);
                }
            }
        }

        let analytics = self.portfolio_analyzer.analyze_portfolio(&bonds, &weights).await?;
        Ok(analytics)
    }

    /// Calculate portfolio duration
    pub async fn calculate_portfolio_duration(&self, holdings: Vec<(String, f64)>) -> Result<f64> {
        let mut total_duration = 0.0;
        let mut total_value = 0.0;

        for (cusip, amount) in holdings {
            if let Some(pricing) = self.bond_prices.read().unwrap().get(&cusip) {
                let value = amount * pricing.dirty_price;
                total_duration += pricing.duration * value;
                total_value += value;
            }
        }

        if total_value > 0.0 {
            Ok(total_duration / total_value)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate DV01 for portfolio
    pub async fn calculate_portfolio_dv01(&self, holdings: Vec<(String, f64)>) -> Result<f64> {
        let mut total_dv01 = 0.0;

        for (cusip, amount) in holdings {
            if let Some(pricing) = self.bond_prices.read().unwrap().get(&cusip) {
                total_dv01 += pricing.dv01 * amount;
            }
        }

        Ok(total_dv01)
    }

    /// Calculate key rate durations
    pub async fn calculate_key_rate_durations(
        &self, 
        holdings: Vec<(String, f64)>
    ) -> Result<HashMap<f64, f64>> {
        let mut key_rate_durations = HashMap::new();
        let key_rates = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];

        for tenor in key_rates {
            let mut duration_contribution = 0.0;

            for (cusip, amount) in &holdings {
                if let Some(bond) = self.bond_universe.read().unwrap().get(cusip) {
                    if let Some(pricing) = self.bond_prices.read().unwrap().get(cusip) {
                        // Calculate key rate duration contribution
                        let krd = self.calculate_key_rate_duration(&bond, &pricing, tenor).await?;
                        duration_contribution += krd * amount;
                    }
                }
            }

            key_rate_durations.insert(tenor, duration_contribution);
        }

        Ok(key_rate_durations)
    }

    /// Perform stress testing
    pub async fn stress_test_portfolio(
        &self,
        holdings: Vec<(String, f64)>,
        scenarios: Vec<StressScenario>,
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for scenario in scenarios {
            let mut portfolio_pnl = 0.0;

            for (cusip, amount) in &holdings {
                if let Some(bond) = self.bond_universe.read().unwrap().get(cusip) {
                    if let Some(pricing) = self.bond_prices.read().unwrap().get(cusip) {
                        let bond_pnl = self.calculate_scenario_pnl(&bond, &pricing, &scenario, *amount).await?;
                        portfolio_pnl += bond_pnl;
                    }
                }
            }

            results.insert(scenario.name, portfolio_pnl);
        }

        Ok(results)
    }

    /// Execute bond trade
    pub async fn execute_trade(
        &self,
        cusip: String,
        side: OrderSide,
        quantity: f64,
        price_type: BondPriceType,
        limit_price: Option<f64>,
    ) -> Result<String> {
        // Validate bond exists
        let bond = self.bond_universe.read().unwrap()
            .get(&cusip)
            .cloned()
            .ok_or_else(|| AlgoVedaError::FixedIncome(format!("Bond not found: {}", cusip)))?;

        // Risk checks
        self.risk_manager.validate_fixed_income_order(&bond, side.clone(), quantity, limit_price.unwrap_or(0.0))?;

        // Create order
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: cusip.clone(),
            side,
            quantity: (quantity * 1000.0) as u64, // Convert to bond units
            order_type: match price_type {
                BondPriceType::Market => crate::trading::OrderType::Market,
                BondPriceType::Limit => crate::trading::OrderType::Limit,
                BondPriceType::Stop => crate::trading::OrderType::Stop,
            },
            price: limit_price,
            time_in_force: crate::trading::TimeInForce::Day,
            status: crate::trading::OrderStatus::PendingNew,
            parent_order_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Submit to execution engine
        let order_id = self.execution_engine.submit_order(order).await?;
        
        Ok(order_id)
    }

    /// Helper methods
    async fn reprice_bonds_for_curve(&self, curve_id: &str) -> Result<()> {
        let bond_universe = self.bond_universe.read().unwrap().clone();
        
        for (cusip, bond) in bond_universe {
            // Check if bond uses this curve
            if self.bond_uses_curve(&bond, curve_id) {
                let _ = self.price_bond(&cusip, None).await;
            }
        }
        
        Ok(())
    }

    fn bond_uses_curve(&self, bond: &BondInstrument, curve_id: &str) -> bool {
        // Determine if a bond's pricing depends on a specific curve
        match &bond.bond_type {
            BondType::Treasury => curve_id.contains("treasury"),
            BondType::Corporate => curve_id.contains("corporate") || curve_id.contains("treasury"),
            BondType::Municipal => curve_id.contains("municipal"),
            _ => curve_id.contains("treasury"), // Default to treasury curve
        }
    }

    async fn calculate_key_rate_duration(
        &self,
        bond: &BondInstrument,
        pricing: &BondPricing,
        tenor: f64,
    ) -> Result<f64> {
        // Simplified key rate duration calculation
        // In practice, this would involve bumping the specific tenor point
        let time_to_maturity = self.calculate_time_to_maturity(bond)?;
        
        if (time_to_maturity - tenor).abs() < 0.5 {
            Ok(pricing.duration * 0.8) // Simplified weighting
        } else {
            Ok(pricing.duration * 0.1) // Minimal contribution
        }
    }

    async fn calculate_scenario_pnl(
        &self,
        bond: &BondInstrument,
        pricing: &BondPricing,
        scenario: &StressScenario,
        position: f64,
    ) -> Result<f64> {
        // Calculate P&L for stress scenario
        let mut pnl = 0.0;
        
        // Interest rate shock
        if let Some(rate_shock) = scenario.interest_rate_shock {
            let duration_pnl = -pricing.duration * rate_shock / 100.0 * pricing.dirty_price * position;
            let convexity_pnl = 0.5 * pricing.convexity * (rate_shock / 100.0).powi(2) * pricing.dirty_price * position;
            pnl += duration_pnl + convexity_pnl;
        }
        
        // Credit spread shock
        if let Some(spread_shock) = scenario.credit_spread_shock {
            let spread_duration = pricing.duration * 0.8; // Simplified
            let spread_pnl = -spread_duration * spread_shock / 10000.0 * pricing.dirty_price * position;
            pnl += spread_pnl;
        }
        
        Ok(pnl)
    }

    fn calculate_time_to_maturity(&self, bond: &BondInstrument) -> Result<f64> {
        let today = Utc::now().date_naive();
        let days_to_maturity = (bond.maturity_date - today).num_days();
        Ok(days_to_maturity as f64 / 365.25)
    }

    /// Get engine statistics
    pub fn get_statistics(&self) -> FixedIncomeStatistics {
        let bond_universe = self.bond_universe.read().unwrap();
        let yield_curves = self.yield_curves.read().unwrap();
        let bond_prices = self.bond_prices.read().unwrap();
        
        FixedIncomeStatistics {
            bonds_in_universe: bond_universe.len() as u64,
            yield_curves_loaded: yield_curves.len() as u64,
            bonds_priced: bond_prices.len() as u64,
            calculations_performed: self.calculations_performed.load(Ordering::Relaxed),
            last_curve_update: *self.last_curve_update.read().unwrap(),
            supported_bond_types: self.config.supported_bond_types.len() as u64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BondPriceType {
    Market,
    Limit,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    pub name: String,
    pub interest_rate_shock: Option<f64>,  // basis points
    pub credit_spread_shock: Option<f64>,  // basis points
    pub volatility_shock: Option<f64>,     // percentage
    pub correlation_shock: Option<f64>,    // percentage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedIncomeStatistics {
    pub bonds_in_universe: u64,
    pub yield_curves_loaded: u64,
    pub bonds_priced: u64,
    pub calculations_performed: u64,
    pub last_curve_update: Option<DateTime<Utc>>,
    pub supported_bond_types: u64,
}

// Implementation of helper classes
impl YieldCurveBuilder {
    fn new(interpolation_method: InterpolationMethod) -> Self {
        Self {
            interpolation_method,
            optimization_method: OptimizationMethod::LevenbergMarquardt,
        }
    }

    async fn build_curve(
        &self,
        curve_id: String,
        currency: String,
        curve_type: CurveType,
        market_quotes: Vec<(f64, f64)>,
    ) -> Result<YieldCurve> {
        // Build yield curve from market quotes
        let mut points = BTreeMap::new();
        
        // Simple interpolation for demonstration
        for (tenor, rate) in market_quotes {
            points.insert(tenor, rate);
        }
        
        // Fit model parameters (simplified)
        let model_parameters = self.fit_nelson_siegel_parameters(&points)?;
        
        // Calculate goodness of fit
        let goodness_of_fit = self.calculate_goodness_of_fit(&points, &model_parameters);
        
        Ok(YieldCurve {
            curve_id,
            currency,
            curve_type,
            as_of_date: Utc::now().date_naive(),
            points,
            model: YieldCurveModel::NelsonSiegel,
            model_parameters,
            goodness_of_fit,
            confidence_intervals: HashMap::new(),
        })
    }

    fn fit_nelson_siegel_parameters(&self, points: &BTreeMap<f64, f64>) -> Result<Vec<f64>> {
        // Simplified Nelson-Siegel fitting
        // In practice, this would use numerical optimization
        Ok(vec![0.05, -0.01, 0.02, 2.0]) // beta0, beta1, beta2, tau
    }

    fn calculate_goodness_of_fit(&self, points: &BTreeMap<f64, f64>, parameters: &[f64]) -> f64 {
        // Calculate R-squared or similar metric
        0.95 // Placeholder
    }
}

impl BondPricer {
    fn new() -> Self {
        Self {
            day_count_calculator: DayCountCalculator,
            cashflow_generator: CashflowGenerator,
            discount_engine: DiscountEngine,
        }
    }

    async fn price_bond(
        &self,
        bond: &BondInstrument,
        yield_to_maturity: Option<f64>,
        yield_curves: &Arc<RwLock<HashMap<String, YieldCurve>>>,
    ) -> Result<BondPricing> {
        // Generate cashflows
        let cashflows = self.generate_cashflows(bond)?;
        
        // Get discount rates
        let discount_rates = if let Some(ytm) = yield_to_maturity {
            self.flat_discount_curve(ytm, &cashflows)
        } else {
            self.get_discount_rates_from_curve(bond, yield_curves, &cashflows)?
        };

        // Calculate present value
        let mut pv = 0.0;
        for (cf_date, cf_amount, discount_rate) in cashflows.iter().zip(discount_rates.iter()) {
            let time_to_cf = self.calculate_time_to_cashflow(cf_date)?;
            pv += cf_amount * (-discount_rate * time_to_cf).exp();
        }

        let clean_price = pv / bond.face_value * 100.0;
        let accrued_interest = self.calculate_accrued_interest(bond)?;
        let dirty_price = clean_price + accrued_interest;

        // Calculate yield to maturity if not provided
        let ytm = yield_to_maturity.unwrap_or_else(|| self.solve_for_yield(bond, clean_price).unwrap_or(0.05));

        // Calculate risk metrics
        let duration = self.calculate_duration(bond, ytm)?;
        let modified_duration = duration / (1.0 + ytm / bond.coupon_frequency as i32 as f64);
        let convexity = self.calculate_convexity(bond, ytm)?;
        let dv01 = modified_duration * dirty_price * bond.face_value / 10000.0;

        Ok(BondPricing {
            bond: bond.clone(),
            clean_price,
            dirty_price,
            accrued_interest,
            yield_to_maturity: ytm,
            yield_to_call: None, // Would calculate if callable
            yield_to_worst: ytm, // Simplified
            duration,
            modified_duration,
            effective_duration: duration, // Simplified
            convexity,
            dv01,
            spread_to_benchmark: 0.0, // Would calculate vs treasury
            z_spread: 0.0,
            option_adjusted_spread: None,
            credit_spread: 0.0,
            probability_of_default: 0.01, // 1% default probability
            recovery_rate: 0.4, // 40% recovery
            expected_loss: 0.006, // PD * (1 - RR)
        })
    }

    fn generate_cashflows(&self, bond: &BondInstrument) -> Result<Vec<(NaiveDate, f64)>> {
        let mut cashflows = Vec::new();
        
        if bond.coupon_rate == 0.0 {
            // Zero coupon bond
            cashflows.push((bond.maturity_date, bond.face_value));
        } else {
            // Regular coupon bond
            let coupon_payment = bond.coupon_rate * bond.face_value / bond.coupon_frequency as i32 as f64;
            
            // Generate coupon dates
            let mut current_date = bond.first_coupon_date.unwrap_or(bond.issue_date);
            while current_date <= bond.maturity_date {
                if current_date <= bond.maturity_date {
                    let payment = if current_date == bond.maturity_date {
                        coupon_payment + bond.face_value // Principal + final coupon
                    } else {
                        coupon_payment
                    };
                    cashflows.push((current_date, payment));
                }
                
                // Next coupon date
                current_date = self.add_coupon_period(current_date, bond.coupon_frequency.clone());
            }
        }
        
        Ok(cashflows)
    }

    fn add_coupon_period(&self, date: NaiveDate, frequency: CouponFrequency) -> NaiveDate {
        let months_to_add = 12 / frequency as u32;
        date + chrono::Duration::days((months_to_add * 30) as i64) // Simplified
    }

    fn flat_discount_curve(&self, rate: f64, cashflows: &[(NaiveDate, f64)]) -> Vec<f64> {
        vec![rate; cashflows.len()]
    }

    fn get_discount_rates_from_curve(
        &self,
        bond: &BondInstrument,
        yield_curves: &Arc<RwLock<HashMap<String, YieldCurve>>>,
        cashflows: &[(NaiveDate, f64)],
    ) -> Result<Vec<f64>> {
        let curves = yield_curves.read().unwrap();
        let curve_key = format!("{}_{}", bond.currency, "treasury"); // Simplified curve selection
        
        if let Some(curve) = curves.get(&curve_key) {
            let mut rates = Vec::new();
            for (cf_date, _) in cashflows {
                let time_to_cf = self.calculate_time_to_cashflow(cf_date)?;
                let rate = self.interpolate_rate(curve, time_to_cf);
                rates.push(rate);
            }
            Ok(rates)
        } else {
            Ok(vec![0.05; cashflows.len()]) // Fallback rate
        }
    }

    fn interpolate_rate(&self, curve: &YieldCurve, tenor: f64) -> f64 {
        // Linear interpolation (simplified)
        let points: Vec<(f64, f64)> = curve.points.iter().map(|(&t, &r)| (t, r)).collect();
        
        for i in 0..points.len()-1 {
            if tenor >= points[i].0 && tenor <= points[i+1].0 {
                let t1 = points[i].0;
                let r1 = points[i].1;
                let t2 = points[i+1].0;
                let r2 = points[i+1].1;
                
                return r1 + (r2 - r1) * (tenor - t1) / (t2 - t1);
            }
        }
        
        // Extrapolate using last point
        points.last().map(|(_, r)| *r).unwrap_or(0.05)
    }

    fn calculate_time_to_cashflow(&self, cf_date: &NaiveDate) -> Result<f64> {
        let today = Utc::now().date_naive();
        let days = (*cf_date - today).num_days();
        Ok(days as f64 / 365.25)
    }

    fn calculate_accrued_interest(&self, bond: &BondInstrument) -> Result<f64> {
        if bond.coupon_rate == 0.0 {
            return Ok(0.0);
        }

        // Simplified accrued interest calculation
        let annual_coupon = bond.coupon_rate * bond.face_value;
        let last_coupon_date = bond.issue_date; // Simplified - would find actual last coupon date
        let today = Utc::now().date_naive();
        let days_since_coupon = (today - last_coupon_date).num_days();
        let days_in_period = 365 / bond.coupon_frequency as i32;
        
        Ok(annual_coupon * days_since_coupon as f64 / days_in_period as f64)
    }

    fn solve_for_yield(&self, bond: &BondInstrument, price: f64) -> Result<f64> {
        // Newton-Raphson method to solve for yield
        let mut yield_guess = 0.05; // 5% initial guess
        
        for _ in 0..20 { // Maximum 20 iterations
            let calculated_price = self.calculate_price_from_yield(bond, yield_guess)?;
            let price_diff = calculated_price - price;
            
            if price_diff.abs() < 0.0001 {
                return Ok(yield_guess);
            }
            
            // Calculate derivative (simplified)
            let derivative = self.calculate_price_derivative(bond, yield_guess)?;
            yield_guess -= price_diff / derivative;
        }
        
        Ok(yield_guess)
    }

    fn calculate_price_from_yield(&self, bond: &BondInstrument, yield_rate: f64) -> Result<f64> {
        let cashflows = self.generate_cashflows(bond)?;
        let mut pv = 0.0;
        
        for (cf_date, cf_amount) in cashflows {
            let time_to_cf = self.calculate_time_to_cashflow(&cf_date)?;
            pv += cf_amount * (1.0 + yield_rate).powf(-time_to_cf);
        }
        
        Ok(pv / bond.face_value * 100.0)
    }

    fn calculate_price_derivative(&self, bond: &BondInstrument, yield_rate: f64) -> Result<f64> {
        let cashflows = self.generate_cashflows(bond)?;
        let mut derivative = 0.0;
        
        for (cf_date, cf_amount) in cashflows {
            let time_to_cf = self.calculate_time_to_cashflow(&cf_date)?;
            derivative -= time_to_cf * cf_amount * (1.0 + yield_rate).powf(-time_to_cf - 1.0);
        }
        
        Ok(derivative / bond.face_value * 100.0)
    }

    fn calculate_duration(&self, bond: &BondInstrument, yield_rate: f64) -> Result<f64> {
        let cashflows = self.generate_cashflows(bond)?;
        let mut weighted_time = 0.0;
        let mut total_pv = 0.0;
        
        for (cf_date, cf_amount) in cashflows {
            let time_to_cf = self.calculate_time_to_cashflow(&cf_date)?;
            let pv = cf_amount * (1.0 + yield_rate).powf(-time_to_cf);
            weighted_time += time_to_cf * pv;
            total_pv += pv;
        }
        
        Ok(weighted_time / total_pv)
    }

    fn calculate_convexity(&self, bond: &BondInstrument, yield_rate: f64) -> Result<f64> {
        let cashflows = self.generate_cashflows(bond)?;
        let mut weighted_time_squared = 0.0;
        let mut total_pv = 0.0;
        
        for (cf_date, cf_amount) in cashflows {
            let time_to_cf = self.calculate_time_to_cashflow(&cf_date)?;
            let pv = cf_amount * (1.0 + yield_rate).powf(-time_to_cf);
            weighted_time_squared += time_to_cf * (time_to_cf + 1.0) * pv;
            total_pv += pv;
        }
        
        Ok(weighted_time_squared / (total_pv * (1.0 + yield_rate).powi(2)))
    }
}

impl PortfolioAnalyzer {
    fn new() -> Self {
        Self {
            duration_calculator: DurationCalculator,
            convexity_calculator: ConvexityCalculator,
            spread_analyzer: SpreadAnalyzer,
        }
    }

    async fn analyze_portfolio(
        &self,
        bonds: &[(BondInstrument, BondPricing)],
        weights: &[f64],
    ) -> Result<PortfolioAnalytics> {
        let mut total_market_value = 0.0;
        let mut total_par_value = 0.0;
        let mut weighted_duration = 0.0;
        let mut weighted_convexity = 0.0;
        let mut weighted_yield = 0.0;
        let mut weighted_maturity = 0.0;
        let mut weighted_coupon = 0.0;

        let mut credit_quality_breakdown = HashMap::new();
        let mut maturity_buckets = HashMap::new();
        let mut sector_allocation = HashMap::new();

        for ((bond, pricing), &weight) in bonds.iter().zip(weights.iter()) {
            let market_value = weight * pricing.dirty_price * bond.face_value / 100.0;
            total_market_value += market_value;
            total_par_value += weight * bond.face_value;

            weighted_duration += pricing.duration * market_value;
            weighted_convexity += pricing.convexity * market_value;
            weighted_yield += pricing.yield_to_maturity * market_value;
            weighted_coupon += bond.coupon_rate * market_value;

            // Time to maturity
            let time_to_maturity = self.calculate_time_to_maturity(&bond)?;
            weighted_maturity += time_to_maturity * market_value;

            // Credit quality
            let rating = &bond.credit_rating.rating;
            *credit_quality_breakdown.entry(rating.clone()).or_insert(0.0) += market_value;

            // Maturity buckets
            let bucket = self.get_maturity_bucket(time_to_maturity);
            *maturity_buckets.entry(bucket).or_insert(0.0) += market_value;

            // Sector allocation
            let sector = format!("{:?}", bond.bond_type);
            *sector_allocation.entry(sector).or_insert(0.0) += market_value;
        }

        // Convert to percentages
        for value in credit_quality_breakdown.values_mut() {
            *value = *value / total_market_value * 100.0;
        }
        for value in maturity_buckets.values_mut() {
            *value = *value / total_market_value * 100.0;
        }
        for value in sector_allocation.values_mut() {
            *value = *value / total_market_value * 100.0;
        }

        Ok(PortfolioAnalytics {
            total_market_value,
            total_par_value,
            weighted_average_maturity: weighted_maturity / total_market_value,
            weighted_average_coupon: weighted_coupon / total_market_value,
            portfolio_duration: weighted_duration / total_market_value,
            portfolio_convexity: weighted_convexity / total_market_value,
            portfolio_yield: weighted_yield / total_market_value,
            credit_quality_breakdown,
            maturity_buckets,
            sector_allocation,
            duration_buckets: HashMap::new(), // Would implement
            key_rate_durations: HashMap::new(), // Would implement
            spread_duration: 0.0, // Would implement
            credit_spread_duration: 0.0, // Would implement
        })
    }

    fn calculate_time_to_maturity(&self, bond: &BondInstrument) -> Result<f64> {
        let today = Utc::now().date_naive();
        let days_to_maturity = (bond.maturity_date - today).num_days();
        Ok(days_to_maturity as f64 / 365.25)
    }

    fn get_maturity_bucket(&self, time_to_maturity: f64) -> String {
        match time_to_maturity {
            t if t <= 1.0 => "0-1 Year".to_string(),
            t if t <= 3.0 => "1-3 Years".to_string(),
            t if t <= 5.0 => "3-5 Years".to_string(),
            t if t <= 10.0 => "5-10 Years".to_string(),
            t if t <= 20.0 => "10-20 Years".to_string(),
            _ => "20+ Years".to_string(),
        }
    }
}

impl FixedIncomeRiskCalculator {
    fn new() -> Self {
        Self {
            var_calculator: VaRCalculator,
            stress_tester: StressTester,
            scenario_generator: ScenarioGenerator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_creation() {
        let bond = BondInstrument {
            cusip: "912828XM4".to_string(),
            isin: "US912828XM49".to_string(),
            ticker: "T 2.625 02/15/2029".to_string(),
            issuer: "US Treasury".to_string(),
            bond_type: BondType::Treasury,
            coupon_rate: 0.02625,
            face_value: 1000.0,
            issue_date: NaiveDate::from_ymd_opt(2019, 2, 15).unwrap(),
            maturity_date: NaiveDate::from_ymd_opt(2029, 2, 15).unwrap(),
            first_coupon_date: Some(NaiveDate::from_ymd_opt(2019, 8, 15).unwrap()),
            coupon_frequency: CouponFrequency::SemiAnnual,
            day_count_convention: DayCountConvention::ActualActual,
            currency: "USD".to_string(),
            credit_rating: CreditRating {
                agency: RatingAgency::SP,
                rating: "AAA".to_string(),
                outlook: Outlook::Stable,
                watch: false,
                numeric_score: 100.0,
            },
            callable: false,
            putable: false,
            convertible: false,
            inflation_linked: false,
            tax_exempt: false,
            minimum_denomination: 1000.0,
            increment: 1000.0,
            settlement_days: 1,
        };

        assert_eq!(bond.cusip, "912828XM4");
        assert_eq!(bond.coupon_rate, 0.02625);
    }

    #[tokio::test]
    async fn test_yield_curve_builder() {
        let builder = YieldCurveBuilder::new(InterpolationMethod::Linear);
        
        let market_quotes = vec![
            (0.25, 0.01),   // 3M
            (0.5, 0.015),   // 6M
            (1.0, 0.02),    // 1Y
            (2.0, 0.025),   // 2Y
            (5.0, 0.03),    // 5Y
            (10.0, 0.035),  // 10Y
            (30.0, 0.04),   // 30Y
        ];

        let curve = builder.build_curve(
            "USD_TREASURY".to_string(),
            "USD".to_string(),
            CurveType::Treasury,
            market_quotes,
        ).await.unwrap();

        assert_eq!(curve.curve_id, "USD_TREASURY");
        assert_eq!(curve.points.len(), 7);
    }
}
