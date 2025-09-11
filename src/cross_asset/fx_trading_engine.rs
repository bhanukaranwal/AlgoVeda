/*!
 * Foreign Exchange (FX) Trading Engine
 * Advanced FX trading with multi-currency support, cross-rates, and carry strategies
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
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use uuid::Uuid;

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, Fill, OrderSide, OrderType, TimeInForce},
    market_data::MarketData,
    risk_management::RiskManager,
    execution::ExecutionEngine,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FXConfig {
    pub supported_currencies: Vec<Currency>,
    pub major_pairs: Vec<CurrencyPair>,
    pub cross_pairs: Vec<CurrencyPair>,
    pub exotic_pairs: Vec<CurrencyPair>,
    pub enable_carry_trading: bool,
    pub enable_triangular_arbitrage: bool,
    pub enable_forward_trading: bool,
    pub enable_options_trading: bool,
    pub max_position_per_pair: f64,
    pub max_total_exposure: f64,
    pub base_currency: Currency,
    pub settlement_days: HashMap<CurrencyPair, u32>,
    pub trading_sessions: Vec<TradingSession>,
    pub spread_thresholds: HashMap<CurrencyPair, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Currency {
    USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, SEK, NOK, DKK, PLN, CZK, HUF,
    CNY, HKD, SGD, KRW, INR, BRL, MXN, ZAR, RUB, TRY, THB, MYR, IDR, PHP,
    BTC, ETH, // Cryptocurrencies
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CurrencyPair {
    pub base: Currency,
    pub quote: Currency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSession {
    pub name: String,
    pub start_time: String,  // UTC time "08:00"
    pub end_time: String,    // UTC time "17:00"
    pub days: Vec<u8>,       // 1=Monday, 7=Sunday
    pub pairs: Vec<CurrencyPair>,
    pub liquidity_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FXRate {
    pub pair: CurrencyPair,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread: f64,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub volume: Option<f64>,
    pub high_24h: Option<f64>,
    pub low_24h: Option<f64>,
    pub change_24h: Option<f64>,
    pub volatility: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FXPosition {
    pub pair: CurrencyPair,
    pub position: f64,  // Positive = long base currency, negative = short
    pub average_rate: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub swap_points: f64,  // Rollover interest
    pub margin_used: f64,
    pub leverage: f64,
    pub opened_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarryTradeSignal {
    pub signal_id: String,
    pub long_currency: Currency,
    pub short_currency: Currency,
    pub suggested_pair: CurrencyPair,
    pub carry_yield: f64,  // Annual yield percentage
    pub risk_score: f64,   // 0-1 risk assessment
    pub confidence: f64,   // 0-1 confidence level
    pub time_horizon_days: u32,
    pub entry_rate: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub opportunity_id: String,
    pub arbitrage_type: ArbitrageType,
    pub currency_path: Vec<Currency>,
    pub expected_profit_bps: f64,
    pub required_notional: f64,
    pub execution_time_ms: u64,
    pub confidence: f64,
    pub rates_used: Vec<(CurrencyPair, f64)>,
    pub detected_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArbitrageType {
    Triangular,    // Three-currency arbitrage
    Locational,    // Same pair, different venues
    Temporal,      // Forward vs spot arbitrage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FXForward {
    pub pair: CurrencyPair,
    pub spot_rate: f64,
    pub forward_rate: f64,
    pub forward_points: f64,
    pub maturity_date: DateTime<Utc>,
    pub days_to_maturity: u32,
    pub interest_rate_differential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapRate {
    pub pair: CurrencyPair,
    pub overnight_rate: f64,
    pub tom_next_rate: f64,
    pub spot_next_rate: f64,
    pub one_week_rate: f64,
    pub one_month_rate: f64,
    pub three_month_rate: f64,
    pub six_month_rate: f64,
    pub one_year_rate: f64,
    pub timestamp: DateTime<Utc>,
}

pub struct FXTradingEngine {
    config: FXConfig,
    
    // Market data
    fx_rates: Arc<RwLock<HashMap<CurrencyPair, FXRate>>>,
    forward_rates: Arc<RwLock<HashMap<(CurrencyPair, u32), FXForward>>>,
    swap_rates: Arc<RwLock<HashMap<CurrencyPair, SwapRate>>>,
    interest_rates: Arc<RwLock<HashMap<Currency, f64>>>,
    
    // Positions and P&L
    positions: Arc<RwLock<HashMap<CurrencyPair, FXPosition>>>,
    currency_exposure: Arc<RwLock<HashMap<Currency, f64>>>,
    
    // Signal generation
    carry_signals: Arc<RwLock<VecDeque<CarryTradeSignal>>>,
    arbitrage_opportunities: Arc<RwLock<VecDeque<ArbitrageOpportunity>>>,
    
    // Analytics engines
    volatility_calculator: Arc<VolatilityCalculator>,
    correlation_calculator: Arc<CorrelationCalculator>,
    carry_analyzer: Arc<CarryAnalyzer>,
    arbitrage_detector: Arc<ArbitrageDetector>,
    
    // External systems
    execution_engine: Arc<ExecutionEngine>,
    risk_manager: Arc<RiskManager>,
    
    // Event handling
    fx_events: broadcast::Sender<FXEvent>,
    
    // Performance tracking
    trades_executed: Arc<AtomicU64>,
    pnl_realized: Arc<RwLock<f64>>,
    last_rate_update: Arc<RwLock<Option<DateTime<Utc>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FXEvent {
    pub event_id: String,
    pub event_type: FXEventType,
    pub timestamp: DateTime<Utc>,
    pub pair: Option<CurrencyPair>,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FXEventType {
    RateUpdate,
    PositionOpened,
    PositionClosed,
    CarrySignalGenerated,
    ArbitrageDetected,
    SessionChange,
    VolatilitySpike,
    MarginCall,
}

// Supporting analytics engines
pub struct VolatilityCalculator {
    historical_window: u32,
    models: Vec<VolatilityModel>,
}

#[derive(Debug, Clone)]
enum VolatilityModel {
    EWMA,
    GARCH,
    RealizedVolatility,
    ImpliedVolatility,
}

pub struct CorrelationCalculator {
    correlation_matrix: Arc<RwLock<HashMap<(CurrencyPair, CurrencyPair), f64>>>,
    lookback_periods: Vec<u32>,
}

pub struct CarryAnalyzer {
    interest_rate_forecasts: Arc<RwLock<HashMap<Currency, Vec<f64>>>>,
    carry_signals_history: VecDeque<CarryTradeSignal>,
}

pub struct ArbitrageDetector {
    rate_precision: f64,
    minimum_profit_bps: f64,
    execution_cost_bps: f64,
}

impl FXTradingEngine {
    pub fn new(
        config: FXConfig,
        execution_engine: Arc<ExecutionEngine>,
        risk_manager: Arc<RiskManager>,
    ) -> Self {
        let (fx_events, _) = broadcast::channel(1000);
        
        Self {
            config: config.clone(),
            fx_rates: Arc::new(RwLock::new(HashMap::new())),
            forward_rates: Arc::new(RwLock::new(HashMap::new())),
            swap_rates: Arc::new(RwLock::new(HashMap::new())),
            interest_rates: Arc::new(RwLock::new(HashMap::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            currency_exposure: Arc::new(RwLock::new(HashMap::new())),
            carry_signals: Arc::new(RwLock::new(VecDeque::new())),
            arbitrage_opportunities: Arc::new(RwLock::new(VecDeque::new())),
            volatility_calculator: Arc::new(VolatilityCalculator::new()),
            correlation_calculator: Arc::new(CorrelationCalculator::new()),
            carry_analyzer: Arc::new(CarryAnalyzer::new()),
            arbitrage_detector: Arc::new(ArbitrageDetector::new()),
            execution_engine,
            risk_manager,
            fx_events,
            trades_executed: Arc::new(AtomicU64::new(0)),
            pnl_realized: Arc::new(RwLock::new(0.0)),
            last_rate_update: Arc::new(RwLock::new(None)),
        }
    }

    /// Update FX rate
    pub async fn update_rate(&self, rate: FXRate) -> Result<()> {
        let pair = rate.pair.clone();
        
        // Store rate
        self.fx_rates.write().unwrap().insert(pair.clone(), rate.clone());
        *self.last_rate_update.write().unwrap() = Some(Utc::now());
        
        // Update cross rates if this is a major pair
        if self.is_major_pair(&pair) {
            self.update_cross_rates(&pair).await?;
        }
        
        // Update position P&L
        self.update_position_pnl(&pair).await?;
        
        // Check for arbitrage opportunities
        if self.config.enable_triangular_arbitrage {
            self.detect_arbitrage_opportunities().await?;
        }
        
        // Generate carry signals
        if self.config.enable_carry_trading {
            self.generate_carry_signals().await?;
        }
        
        // Emit rate update event
        let _ = self.fx_events.send(FXEvent {
            event_id: Uuid::new_v4().to_string(),
            event_type: FXEventType::RateUpdate,
            timestamp: Utc::now(),
            pair: Some(pair),
            data: serde_json::to_value(&rate).unwrap_or(serde_json::Value::Null),
        });
        
        Ok(())
    }

    /// Execute FX trade
    pub async fn execute_trade(
        &self,
        pair: CurrencyPair,
        side: OrderSide,
        amount: f64,
        order_type: OrderType,
        rate: Option<f64>,
    ) -> Result<String> {
        // Validate pair is supported
        if !self.is_supported_pair(&pair) {
            return Err(AlgoVedaError::FX(format!("Unsupported currency pair: {:?}", pair)));
        }

        // Risk checks
        self.risk_manager.validate_fx_order(&pair, side.clone(), amount, rate.unwrap_or(0.0))?;
        
        // Check position limits
        let current_exposure = self.get_currency_exposure(&pair.base).await;
        let new_exposure = match side {
            OrderSide::Buy => current_exposure + amount,
            OrderSide::Sell => current_exposure - amount,
        };
        
        if new_exposure.abs() > self.config.max_position_per_pair {
            return Err(AlgoVedaError::FX("Position limit exceeded".to_string()));
        }

        // Create FX order
        let order_id = Uuid::new_v4().to_string();
        let symbol = self.pair_to_symbol(&pair);
        
        let order = Order {
            id: order_id.clone(),
            symbol,
            side,
            quantity: (amount * 100000.0) as u64, // Convert to base units
            order_type,
            price: rate,
            time_in_force: TimeInForce::IOC, // FX typically IOC or FOK
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

    /// Process fill for FX trade
    pub async fn process_fill(&self, fill: Fill) -> Result<()> {
        let pair = self.symbol_to_pair(&fill.order_id)?; // Would need order mapping
        let amount = fill.quantity as f64 / 100000.0; // Convert from base units
        
        // Update position
        self.update_position(&pair, amount, fill.price, fill.side).await?;
        
        // Update currency exposures
        self.update_currency_exposures(&pair, amount, fill.side.unwrap_or(OrderSide::Buy)).await?;
        
        // Calculate and record P&L
        self.calculate_trade_pnl(&pair, amount, fill.price).await?;
        
        Ok(())
    }

    /// Get current rate for currency pair
    pub async fn get_rate(&self, pair: &CurrencyPair) -> Option<FXRate> {
        // Try direct rate first
        if let Some(rate) = self.fx_rates.read().unwrap().get(pair).cloned() {
            return Some(rate);
        }
        
        // Try inverse rate
        let inverse_pair = CurrencyPair { base: pair.quote.clone(), quote: pair.base.clone() };
        if let Some(inverse_rate) = self.fx_rates.read().unwrap().get(&inverse_pair).cloned() {
            return Some(FXRate {
                pair: pair.clone(),
                bid: 1.0 / inverse_rate.ask,
                ask: 1.0 / inverse_rate.bid,
                mid: 1.0 / inverse_rate.mid,
                spread: inverse_rate.spread / (inverse_rate.mid * inverse_rate.mid),
                timestamp: inverse_rate.timestamp,
                source: inverse_rate.source,
                volume: inverse_rate.volume,
                high_24h: inverse_rate.low_24h.map(|l| 1.0 / l),
                low_24h: inverse_rate.high_24h.map(|h| 1.0 / h),
                change_24h: inverse_rate.change_24h.map(|c| -c),
                volatility: inverse_rate.volatility,
            });
        }
        
        // Try cross rate calculation
        self.calculate_cross_rate(pair).await
    }

    /// Calculate cross rate through USD or EUR
    async fn calculate_cross_rate(&self, pair: &CurrencyPair) -> Option<FXRate> {
        let rates = self.fx_rates.read().unwrap();
        
        // Try through USD
        let base_usd = CurrencyPair { base: pair.base.clone(), quote: Currency::USD };
        let quote_usd = CurrencyPair { base: pair.quote.clone(), quote: Currency::USD };
        
        if let (Some(base_rate), Some(quote_rate)) = (rates.get(&base_usd), rates.get(&quote_usd)) {
            return Some(FXRate {
                pair: pair.clone(),
                bid: base_rate.bid / quote_rate.ask,
                ask: base_rate.ask / quote_rate.bid,
                mid: base_rate.mid / quote_rate.mid,
                spread: (base_rate.ask / quote_rate.bid) - (base_rate.bid / quote_rate.ask),
                timestamp: std::cmp::max(base_rate.timestamp, quote_rate.timestamp),
                source: format!("Cross: {} + {}", base_rate.source, quote_rate.source),
                volume: None,
                high_24h: None,
                low_24h: None,
                change_24h: None,
                volatility: None,
            });
        }
        
        None
    }

    /// Generate carry trade signals
    async fn generate_carry_signals(&self) -> Result<()> {
        let interest_rates = self.interest_rates.read().unwrap().clone();
        let mut signals = Vec::new();
        
        for pair in &self.config.major_pairs {
            if let (Some(&base_rate), Some(&quote_rate)) = 
                (interest_rates.get(&pair.base), interest_rates.get(&pair.quote)) {
                
                let carry_yield = base_rate - quote_rate;
                
                // Only generate signal if carry is significant
                if carry_yield.abs() > 0.005 { // 50 basis points minimum
                    let (long_currency, short_currency) = if carry_yield > 0.0 {
                        (pair.base.clone(), pair.quote.clone())
                    } else {
                        (pair.quote.clone(), pair.base.clone())
                    };
                    
                    // Calculate risk score based on volatility
                    let volatility = self.volatility_calculator.get_volatility(pair).await.unwrap_or(0.1);
                    let risk_score = (volatility * 100.0).min(1.0);
                    
                    // Get current rate for entry
                    let current_rate = self.get_rate(pair).await;
                    let entry_rate = current_rate.map(|r| r.mid).unwrap_or(1.0);
                    
                    let signal = CarryTradeSignal {
                        signal_id: Uuid::new_v4().to_string(),
                        long_currency,
                        short_currency,
                        suggested_pair: pair.clone(),
                        carry_yield: carry_yield.abs(),
                        risk_score,
                        confidence: 0.7, // Would calculate based on multiple factors
                        time_horizon_days: 30,
                        entry_rate,
                        stop_loss: Some(entry_rate * (1.0 - 0.02)), // 2% stop loss
                        take_profit: Some(entry_rate * (1.0 + 0.01)), // 1% take profit
                        created_at: Utc::now(),
                    };
                    
                    signals.push(signal);
                }
            }
        }
        
        // Store signals
        let mut carry_signals = self.carry_signals.write().unwrap();
        for signal in signals {
            carry_signals.push_back(signal.clone());
            
            // Emit signal event
            let _ = self.fx_events.send(FXEvent {
                event_id: Uuid::new_v4().to_string(),
                event_type: FXEventType::CarrySignalGenerated,
                timestamp: Utc::now(),
                pair: Some(signal.suggested_pair),
                data: serde_json::to_value(&signal).unwrap_or(serde_json::Value::Null),
            });
        }
        
        // Keep only recent signals
        while carry_signals.len() > 100 {
            carry_signals.pop_front();
        }
        
        Ok(())
    }

    /// Detect triangular arbitrage opportunities
    async fn detect_arbitrage_opportunities(&self) -> Result<()> {
        let rates = self.fx_rates.read().unwrap();
        let mut opportunities = Vec::new();
        
        // Check major currency triangles
        let major_currencies = vec![Currency::USD, Currency::EUR, Currency::GBP, Currency::JPY, Currency::CHF];
        
        for i in 0..major_currencies.len() {
            for j in (i+1)..major_currencies.len() {
                for k in (j+1)..major_currencies.len() {
                    let curr_a = &major_currencies[i];
                    let curr_b = &major_currencies[j];
                    let curr_c = &major_currencies[k];
                    
                    // Check both directions of the triangle
                    if let Some(opportunity) = self.check_triangular_arbitrage(curr_a, curr_b, curr_c, &rates).await {
                        opportunities.push(opportunity);
                    }
                    
                    if let Some(opportunity) = self.check_triangular_arbitrage(curr_a, curr_c, curr_b, &rates).await {
                        opportunities.push(opportunity);
                    }
                }
            }
        }
        
        // Store opportunities
        let mut arb_opportunities = self.arbitrage_opportunities.write().unwrap();
        for opportunity in opportunities {
            arb_opportunities.push_back(opportunity.clone());
            
            // Emit arbitrage event
            let _ = self.fx_events.send(FXEvent {
                event_id: Uuid::new_v4().to_string(),
                event_type: FXEventType::ArbitrageDetected,
                timestamp: Utc::now(),
                pair: None,
                data: serde_json::to_value(&opportunity).unwrap_or(serde_json::Value::Null),
            });
        }
        
        // Clean up expired opportunities
        let now = Utc::now();
        arb_opportunities.retain(|opp| opp.expires_at > now);
        
        Ok(())
    }

    /// Check for triangular arbitrage between three currencies
    async fn check_triangular_arbitrage(
        &self,
        curr_a: &Currency,
        curr_b: &Currency,
        curr_c: &Currency,
        rates: &HashMap<CurrencyPair, FXRate>,
    ) -> Option<ArbitrageOpportunity> {
        // Path: A -> B -> C -> A
        let pair_ab = CurrencyPair { base: curr_a.clone(), quote: curr_b.clone() };
        let pair_bc = CurrencyPair { base: curr_b.clone(), quote: curr_c.clone() };
        let pair_ca = CurrencyPair { base: curr_c.clone(), quote: curr_a.clone() };
        
        // Get rates (including inverse rates if needed)
        let rate_ab = self.get_effective_rate(&pair_ab, rates, true)?; // A->B (sell A, buy B)
        let rate_bc = self.get_effective_rate(&pair_bc, rates, true)?; // B->C (sell B, buy C)  
        let rate_ca = self.get_effective_rate(&pair_ca, rates, true)?; // C->A (sell C, buy A)
        
        // Calculate arbitrage profit
        let arbitrage_rate = rate_ab * rate_bc * rate_ca;
        let profit_bps = (arbitrage_rate - 1.0) * 10000.0;
        
        // Check if profitable after transaction costs
        let min_profit = self.arbitrage_detector.minimum_profit_bps + self.arbitrage_detector.execution_cost_bps * 3.0;
        
        if profit_bps > min_profit {
            Some(ArbitrageOpportunity {
                opportunity_id: Uuid::new_v4().to_string(),
                arbitrage_type: ArbitrageType::Triangular,
                currency_path: vec![curr_a.clone(), curr_b.clone(), curr_c.clone(), curr_a.clone()],
                expected_profit_bps: profit_bps,
                required_notional: 100000.0, // $100k base amount
                execution_time_ms: 500, // Estimated execution time
                confidence: 0.85,
                rates_used: vec![
                    (pair_ab, rate_ab),
                    (pair_bc, rate_bc),
                    (pair_ca, rate_ca),
                ],
                detected_at: Utc::now(),
                expires_at: Utc::now() + ChronoDuration::seconds(30),
            })
        } else {
            None
        }
    }

    /// Get effective rate considering bid/ask spread and direction
    fn get_effective_rate(&self, pair: &CurrencyPair, rates: &HashMap<CurrencyPair, FXRate>, is_sell_base: bool) -> Option<f64> {
        if let Some(rate) = rates.get(pair) {
            return Some(if is_sell_base { rate.bid } else { rate.ask });
        }
        
        // Try inverse pair
        let inverse_pair = CurrencyPair { base: pair.quote.clone(), quote: pair.base.clone() };
        if let Some(rate) = rates.get(&inverse_pair) {
            return Some(if is_sell_base { 1.0 / rate.ask } else { 1.0 / rate.bid });
        }
        
        None
    }

    /// Update position after trade
    async fn update_position(&self, pair: &CurrencyPair, amount: f64, rate: f64, side: Option<OrderSide>) -> Result<()> {
        let mut positions = self.positions.write().unwrap();
        let position = positions.entry(pair.clone()).or_insert_with(|| FXPosition {
            pair: pair.clone(),
            position: 0.0,
            average_rate: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            swap_points: 0.0,
            margin_used: 0.0,
            leverage: 1.0,
            opened_at: Utc::now(),
            last_updated: Utc::now(),
        });

        let trade_amount = match side.unwrap_or(OrderSide::Buy) {
            OrderSide::Buy => amount,
            OrderSide::Sell => -amount,
        };

        // Update position size and average rate
        if position.position == 0.0 {
            position.position = trade_amount;
            position.average_rate = rate;
            position.opened_at = Utc::now();
        } else if (position.position > 0.0 && trade_amount > 0.0) || (position.position < 0.0 && trade_amount < 0.0) {
            // Adding to existing position
            let total_value = position.position * position.average_rate + trade_amount * rate;
            position.position += trade_amount;
            if position.position != 0.0 {
                position.average_rate = total_value / position.position;
            }
        } else {
            // Reducing or reversing position
            let reduced_amount = trade_amount.abs().min(position.position.abs());
            let remaining_trade = trade_amount.abs() - reduced_amount;
            
            // Calculate realized P&L for reduced portion
            let realized_pnl = if position.position > 0.0 {
                (rate - position.average_rate) * reduced_amount
            } else {
                (position.average_rate - rate) * reduced_amount
            };
            
            position.realized_pnl += realized_pnl;
            position.position -= trade_amount.signum() * reduced_amount;
            
            // If there's remaining trade amount, it creates a new position
            if remaining_trade > 0.0 {
                position.position = trade_amount.signum() * remaining_trade;
                position.average_rate = rate;
            }
        }
        
        position.last_updated = Utc::now();
        
        Ok(())
    }

    /// Update currency exposures
    async fn update_currency_exposures(&self, pair: &CurrencyPair, amount: f64, side: OrderSide) -> Result<()> {
        let mut exposures = self.currency_exposure.write().unwrap();
        
        let (base_change, quote_change) = match side {
            OrderSide::Buy => (amount, -amount), // Buy base, sell quote
            OrderSide::Sell => (-amount, amount), // Sell base, buy quote
        };
        
        *exposures.entry(pair.base.clone()).or_insert(0.0) += base_change;
        *exposures.entry(pair.quote.clone()).or_insert(0.0) += quote_change;
        
        Ok(())
    }

    /// Update position P&L based on current rates
    async fn update_position_pnl(&self, pair: &CurrencyPair) -> Result<()> {
        if let Some(current_rate) = self.get_rate(pair).await {
            let mut positions = self.positions.write().unwrap();
            
            if let Some(position) = positions.get_mut(pair) {
                if position.position != 0.0 {
                    let current_value = position.position * current_rate.mid;
                    let cost_value = position.position * position.average_rate;
                    position.unrealized_pnl = current_value - cost_value;
                    position.last_updated = Utc::now();
                }
            }
        }
        
        Ok(())
    }

    /// Helper methods
    fn is_major_pair(&self, pair: &CurrencyPair) -> bool {
        self.config.major_pairs.contains(pair)
    }

    fn is_supported_pair(&self, pair: &CurrencyPair) -> bool {
        self.config.major_pairs.contains(pair) || 
        self.config.cross_pairs.contains(pair) || 
        self.config.exotic_pairs.contains(pair)
    }

    fn pair_to_symbol(&self, pair: &CurrencyPair) -> String {
        format!("{:?}{:?}", pair.base, pair.quote)
    }

    fn symbol_to_pair(&self, symbol: &str) -> Result<CurrencyPair> {
        // Simplified - would need proper symbol parsing
        if symbol.len() >= 6 {
            let base_str = &symbol[0..3];
            let quote_str = &symbol[3..6];
            
            // Would need proper currency parsing
            Ok(CurrencyPair {
                base: Currency::USD, // Placeholder
                quote: Currency::EUR, // Placeholder
            })
        } else {
            Err(AlgoVedaError::FX("Invalid symbol format".to_string()))
        }
    }

    async fn calculate_trade_pnl(&self, pair: &CurrencyPair, amount: f64, rate: f64) -> Result<()> {
        // P&L calculation would depend on base currency conversion
        // This is a simplified version
        let pnl_usd = amount * rate * 0.0001; // Placeholder calculation
        *self.pnl_realized.write().unwrap() += pnl_usd;
        Ok(())
    }

    async fn update_cross_rates(&self, updated_pair: &CurrencyPair) -> Result<()> {
        // Update cross rates that depend on this major pair
        // Implementation would calculate cross rates for all supported pairs
        Ok(())
    }

    async fn get_currency_exposure(&self, currency: &Currency) -> f64 {
        self.currency_exposure.read().unwrap().get(currency).copied().unwrap_or(0.0)
    }

    /// Get latest carry signals
    pub fn get_carry_signals(&self, limit: usize) -> Vec<CarryTradeSignal> {
        self.carry_signals.read().unwrap().iter().rev().take(limit).cloned().collect()
    }

    /// Get arbitrage opportunities
    pub fn get_arbitrage_opportunities(&self, limit: usize) -> Vec<ArbitrageOpportunity> {
        let now = Utc::now();
        self.arbitrage_opportunities.read().unwrap()
            .iter()
            .filter(|opp| opp.expires_at > now)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get current positions
    pub fn get_positions(&self) -> Vec<FXPosition> {
        self.positions.read().unwrap().values().cloned().collect()
    }

    /// Get currency exposures
    pub fn get_currency_exposures(&self) -> HashMap<Currency, f64> {
        self.currency_exposure.read().unwrap().clone()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> FXStatistics {
        let positions = self.positions.read().unwrap();
        let rates = self.fx_rates.read().unwrap();
        
        FXStatistics {
            active_positions: positions.len() as u64,
            currency_pairs_tracked: rates.len() as u64,
            trades_executed: self.trades_executed.load(Ordering::Relaxed),
            realized_pnl: *self.pnl_realized.read().unwrap(),
            unrealized_pnl: positions.values().map(|p| p.unrealized_pnl).sum(),
            total_exposure: self.currency_exposure.read().unwrap().values().map(|e| e.abs()).sum(),
            carry_signals_active: self.carry_signals.read().unwrap().len() as u64,
            arbitrage_opportunities_active: {
                let now = Utc::now();
                self.arbitrage_opportunities.read().unwrap()
                    .iter()
                    .filter(|opp| opp.expires_at > now)
                    .count() as u64
            },
            last_rate_update: *self.last_rate_update.read().unwrap(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FXStatistics {
    pub active_positions: u64,
    pub currency_pairs_tracked: u64,
    pub trades_executed: u64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_exposure: f64,
    pub carry_signals_active: u64,
    pub arbitrage_opportunities_active: u64,
    pub last_rate_update: Option<DateTime<Utc>>,
}

// Implementation of analytics engines
impl VolatilityCalculator {
    fn new() -> Self {
        Self {
            historical_window: 20,
            models: vec![VolatilityModel::EWMA, VolatilityModel::RealizedVolatility],
        }
    }

    async fn get_volatility(&self, pair: &CurrencyPair) -> Option<f64> {
        // Would calculate volatility based on historical price data
        // This is a placeholder
        Some(0.12) // 12% annualized volatility
    }
}

impl CorrelationCalculator {
    fn new() -> Self {
        Self {
            correlation_matrix: Arc::new(RwLock::new(HashMap::new())),
            lookback_periods: vec![20, 60, 252],
        }
    }
}

impl CarryAnalyzer {
    fn new() -> Self {
        Self {
            interest_rate_forecasts: Arc::new(RwLock::new(HashMap::new())),
            carry_signals_history: VecDeque::new(),
        }
    }
}

impl ArbitrageDetector {
    fn new() -> Self {
        Self {
            rate_precision: 0.00001, // 5 decimal places
            minimum_profit_bps: 1.0, // 1 basis point minimum profit
            execution_cost_bps: 0.5, // 0.5 basis points execution cost
        }
    }
}

impl CurrencyPair {
    pub fn new(base: Currency, quote: Currency) -> Self {
        Self { base, quote }
    }
    
    pub fn inverse(&self) -> Self {
        Self { base: self.quote.clone(), quote: self.base.clone() }
    }
}

impl std::fmt::Display for CurrencyPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{:?}", self.base, self.quote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_currency_pair_creation() {
        let pair = CurrencyPair::new(Currency::EUR, Currency::USD);
        assert_eq!(pair.base, Currency::EUR);
        assert_eq!(pair.quote, Currency::USD);
        
        let inverse = pair.inverse();
        assert_eq!(inverse.base, Currency::USD);
        assert_eq!(inverse.quote, Currency::EUR);
    }

    #[test]
    fn test_currency_pair_display() {
        let pair = CurrencyPair::new(Currency::GBP, Currency::JPY);
        assert_eq!(format!("{}", pair), "GBP/JPY");
    }

    #[tokio::test]
    async fn test_fx_rate_update() {
        let config = FXConfig {
            supported_currencies: vec![Currency::USD, Currency::EUR],
            major_pairs: vec![CurrencyPair::new(Currency::EUR, Currency::USD)],
            cross_pairs: vec![],
            exotic_pairs: vec![],
            enable_carry_trading: false,
            enable_triangular_arbitrage: false,
            enable_forward_trading: false,
            enable_options_trading: false,
            max_position_per_pair: 1000000.0,
            max_total_exposure: 10000000.0,
            base_currency: Currency::USD,
            settlement_days: HashMap::new(),
            trading_sessions: vec![],
            spread_thresholds: HashMap::new(),
        };

        let execution_engine = Arc::new(ExecutionEngine::new(Default::default()));
        let risk_manager = Arc::new(RiskManager::new(Default::default()));
        
        let fx_engine = FXTradingEngine::new(config, execution_engine, risk_manager);
        
        let rate = FXRate {
            pair: CurrencyPair::new(Currency::EUR, Currency::USD),
            bid: 1.0950,
            ask: 1.0952,
            mid: 1.0951,
            spread: 0.0002,
            timestamp: Utc::now(),
            source: "TEST".to_string(),
            volume: Some(1000000.0),
            high_24h: Some(1.0960),
            low_24h: Some(1.0940),
            change_24h: Some(0.0011),
            volatility: Some(0.12),
        };

        let result = fx_engine.update_rate(rate).await;
        assert!(result.is_ok());
    }
}
