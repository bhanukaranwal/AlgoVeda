/*!
 * Time-Weighted Average Price (TWAP) Execution Engine
 * 
 * Advanced TWAP implementation with adaptive scheduling,
 * market impact minimization, and real-time optimization.
 */

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::time::{Duration, Instant, interval};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, warn, instrument};
use uuid::Uuid;

use crate::trading::{Order, OrderSide, OrderType, TradingResult, MarketData};
use crate::config::TradingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapParams {
    pub total_quantity: f64,
    pub duration_minutes: u32,
    pub slice_interval_seconds: u32,
    pub min_slice_size: f64,
    pub max_slice_size: f64,
    pub participation_rate: f64, // Max % of volume
    pub price_improvement_threshold: f64, // BPS
    pub enable_market_impact_model: bool,
    pub enable_adaptive_scheduling: bool,
}

#[derive(Debug, Clone)]
pub struct TwapSlice {
    pub slice_id: Uuid,
    pub parent_order_id: Uuid,
    pub quantity: f64,
    pub target_time: DateTime<Utc>,
    pub actual_time: Option<DateTime<Utc>>,
    pub order: Option<Order>,
    pub status: SliceStatus,
    pub market_conditions: MarketConditions,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SliceStatus {
    Pending,
    Scheduled,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub volume_rate: f64,
    pub volatility: f64,
    pub spread_bps: f64,
    pub market_impact_bps: f64,
    pub liquidity_score: f64,
}

/// TWAP Engine with advanced market microstructure awareness
#[derive(Debug)]
pub struct TwapEngine {
    config: Arc<TradingConfig>,
    active_strategies: Arc<RwLock<Vec<TwapStrategy>>>,
    market_data_cache: Arc<RwLock<std::collections::HashMap<String, MarketData>>>,
    volume_tracker: Arc<RwLock<VolumeTracker>>,
    impact_model: MarketImpactModel,
}

#[derive(Debug)]
pub struct TwapStrategy {
    pub strategy_id: Uuid,
    pub parent_order: Order,
    pub params: TwapParams,
    pub slices: VecDeque<TwapSlice>,
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub is_active: bool,
    pub performance_metrics: TwapPerformance,
}

#[derive(Debug, Clone, Default)]
pub struct TwapPerformance {
    pub implementation_shortfall: f64,
    pub market_impact_bps: f64,
    pub timing_risk_bps: f64,
    pub total_cost_bps: f64,
    pub participation_rate_achieved: f64,
    pub schedule_adherence: f64,
}

#[derive(Debug)]
struct VolumeTracker {
    symbol_volumes: std::collections::HashMap<String, VolumeData>,
}

#[derive(Debug, Clone)]
struct VolumeData {
    volume_history: VecDeque<VolumePoint>,
    current_session_volume: u64,
    avg_volume_per_minute: f64,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct VolumePoint {
    timestamp: DateTime<Utc>,
    volume: u64,
    price: f64,
}

#[derive(Debug)]
struct MarketImpactModel {
    temporary_impact_coefficient: f64,
    permanent_impact_coefficient: f64,
    volatility_adjustment: f64,
}

impl TwapEngine {
    pub fn new(config: Arc<TradingConfig>) -> Self {
        Self {
            config,
            active_strategies: Arc::new(RwLock::new(Vec::new())),
            market_data_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
            volume_tracker: Arc::new(RwLock::new(VolumeTracker {
                symbol_volumes: std::collections::HashMap::new(),
            })),
            impact_model: MarketImpactModel {
                temporary_impact_coefficient: 0.5,
                permanent_impact_coefficient: 0.1,
                volatility_adjustment: 1.0,
            },
        }
    }

    #[instrument(skip(self))]
    pub fn create_twap_strategy(
        &self, 
        parent_order: Order, 
        params: TwapParams
    ) -> TradingResult<Uuid> {
        let strategy_id = Uuid::new_v4();
        let now = Utc::now();
        let end_time = now + chrono::Duration::minutes(params.duration_minutes as i64);

        // Generate initial slice schedule
        let slices = self.generate_slice_schedule(&parent_order, &params, now, end_time)?;

        let strategy = TwapStrategy {
            strategy_id,
            parent_order: parent_order.clone(),
            params,
            slices,
            executed_quantity: 0.0,
            remaining_quantity: parent_order.quantity,
            average_price: 0.0,
            start_time: now,
            end_time,
            is_active: true,
            performance_metrics: TwapPerformance::default(),
        };

        {
            let mut strategies = self.active_strategies.write();
            strategies.push(strategy);
        }

        info!("Created TWAP strategy {} for order {} with duration {}min", 
              strategy_id, parent_order.id, params.duration_minutes);

        Ok(strategy_id)
    }

    fn generate_slice_schedule(
        &self,
        parent_order: &Order,
        params: &TwapParams,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> TradingResult<VecDeque<TwapSlice>> {
        let total_duration_seconds = (end_time - start_time).num_seconds() as u32;
        let num_slices = total_duration_seconds / params.slice_interval_seconds;
        let base_slice_size = params.total_quantity / num_slices as f64;

        let mut slices = VecDeque::new();
        let mut current_time = start_time;

        for i in 0..num_slices {
            let slice_size = if params.enable_adaptive_scheduling {
                self.calculate_adaptive_slice_size(
                    base_slice_size,
                    &parent_order.symbol,
                    current_time,
                    params,
                )?
            } else {
                base_slice_size
            };

            let clamped_size = slice_size
                .max(params.min_slice_size)
                .min(params.max_slice_size);

            let slice = TwapSlice {
                slice_id: Uuid::new_v4(),
                parent_order_id: parent_order.id,
                quantity: clamped_size,
                target_time: current_time,
                actual_time: None,
                order: None,
                status: SliceStatus::Pending,
                market_conditions: self.get_market_conditions(&parent_order.symbol)?,
            };

            slices.push_back(slice);
            current_time += chrono::Duration::seconds(params.slice_interval_seconds as i64);
        }

        info!("Generated {} slices for TWAP strategy", num_slices);
        Ok(slices)
    }

    fn calculate_adaptive_slice_size(
        &self,
        base_size: f64,
        symbol: &str,
        target_time: DateTime<Utc>,
        params: &TwapParams,
    ) -> TradingResult<f64> {
        let volume_tracker = self.volume_tracker.read();
        
        if let Some(volume_data) = volume_tracker.symbol_volumes.get(symbol) {
            // Adjust size based on expected volume at target time
            let expected_volume = self.forecast_volume_at_time(volume_data, target_time);
            let max_participation = expected_volume * params.participation_rate;
            
            // Adjust for market conditions
            let volatility_adjustment = self.calculate_volatility_adjustment(symbol)?;
            let liquidity_adjustment = self.calculate_liquidity_adjustment(symbol)?;
            
            let adjusted_size = base_size * volatility_adjustment * liquidity_adjustment;
            Ok(adjusted_size.min(max_participation))
        } else {
            Ok(base_size)
        }
    }

    fn forecast_volume_at_time(&self, volume_data: &VolumeData, target_time: DateTime<Utc>) -> f64 {
        // Simple time-of-day volume pattern
        let hour = target_time.hour();
        let volume_multiplier = match hour {
            9..=10 => 1.8,   // Opening hours - high volume
            11..=14 => 0.8,  // Mid-day - lower volume
            15..=16 => 1.5,  // Closing hours - high volume
            _ => 0.3,        // After hours - very low volume
        };
        
        volume_data.avg_volume_per_minute * volume_multiplier
    }

    fn calculate_volatility_adjustment(&self, symbol: &str) -> TradingResult<f64> {
        let market_data_cache = self.market_data_cache.read();
        
        if let Some(market_data) = market_data_cache.get(symbol) {
            // Placeholder volatility calculation
            // In production, this would use realized volatility
            let volatility = 0.02; // 2% assumed volatility
            
            // Reduce size in high volatility environments
            Ok(1.0 / (1.0 + volatility * 10.0))
        } else {
            Ok(1.0)
        }
    }

    fn calculate_liquidity_adjustment(&self, symbol: &str) -> TradingResult<f64> {
        let market_data_cache = self.market_data_cache.read();
        
        if let Some(market_data) = market_data_cache.get(symbol) {
            // Adjust based on bid-ask spread as liquidity proxy
            if let (Some(bid), Some(ask)) = (market_data.bid, market_data.ask) {
                let spread_bps = ((ask - bid) / ((ask + bid) / 2.0)) * 10000.0;
                
                // Reduce size in low liquidity (high spread) conditions
                if spread_bps > 20.0 {
                    Ok(0.7)
                } else if spread_bps > 10.0 {
                    Ok(0.85)
                } else {
                    Ok(1.0)
                }
            } else {
                Ok(1.0)
            }
        } else {
            Ok(1.0)
        }
    }

    fn get_market_conditions(&self, symbol: &str) -> TradingResult<MarketConditions> {
        let market_data_cache = self.market_data_cache.read();
        let volume_tracker = self.volume_tracker.read();

        let (spread_bps, liquidity_score) = if let Some(market_data) = market_data_cache.get(symbol) {
            let spread = if let (Some(bid), Some(ask)) = (market_data.bid, market_data.ask) {
                ((ask - bid) / market_data.price) * 10000.0
            } else {
                5.0 // Default spread
            };
            
            let liquidity = if spread < 5.0 { 1.0 } else if spread < 15.0 { 0.7 } else { 0.3 };
            (spread, liquidity)
        } else {
            (10.0, 0.5)
        };

        let volume_rate = volume_tracker.symbol_volumes
            .get(symbol)
            .map(|v| v.avg_volume_per_minute)
            .unwrap_or(1000.0);

        Ok(MarketConditions {
            volume_rate,
            volatility: 0.02, // Placeholder
            spread_bps,
            market_impact_bps: self.estimate_market_impact(symbol, 1000.0)?, // Base size
            liquidity_score,
        })
    }

    fn estimate_market_impact(&self, symbol: &str, quantity: f64) -> TradingResult<f64> {
        // Simplified market impact model
        // In production, this would be much more sophisticated
        let base_impact = (quantity / 10000.0).sqrt() * 2.0; // Square root law
        Ok(base_impact)
    }

    pub async fn start_execution(&self) -> TradingResult<()> {
        info!("Starting TWAP execution engine");
        
        // Start slice execution loop
        let engine = self.clone();
        tokio::spawn(async move {
            engine.slice_execution_loop().await;
        });

        // Start performance monitoring
        let engine = self.clone();
        tokio::spawn(async move {
            engine.performance_monitoring_loop().await;
        });

        Ok(())
    }

    async fn slice_execution_loop(&self) {
        let mut interval = interval(Duration::from_secs(1));

        loop {
            interval.tick().await;
            
            if let Err(e) = self.process_pending_slices().await {
                warn!("Error processing TWAP slices: {}", e);
            }
        }
    }

    async fn process_pending_slices(&self) -> TradingResult<()> {
        let now = Utc::now();
        let mut strategies = self.active_strategies.write();

        for strategy in strategies.iter_mut().filter(|s| s.is_active) {
            while let Some(slice) = strategy.slices.front() {
                if slice.target_time <= now && slice.status == SliceStatus::Pending {
                    self.execute_slice(strategy, slice).await?;
                    break;
                } else {
                    break;
                }
            }

            // Check if strategy is complete
            if strategy.slices.iter().all(|s| matches!(s.status, SliceStatus::Filled | SliceStatus::Cancelled)) {
                strategy.is_active = false;
                self.finalize_strategy_performance(strategy);
                info!("TWAP strategy {} completed", strategy.strategy_id);
            }
        }

        Ok(())
    }

    async fn execute_slice(&self, strategy: &mut TwapStrategy, slice: &TwapSlice) -> TradingResult<()> {
        // Create child order for slice execution
        let mut slice_order = strategy.parent_order.clone();
        slice_order.id = Uuid::new_v4();
        slice_order.quantity = slice.quantity;
        slice_order.parent_order_id = Some(strategy.parent_order.id);

        // Adjust price based on market conditions if needed
        if let Some(market_data) = self.market_data_cache.read().get(&slice_order.symbol) {
            slice_order.price = Some(self.calculate_slice_price(
                &slice_order,
                market_data,
                &slice.market_conditions,
            )?);
        }

        // Mark slice as submitted
        if let Some(mut current_slice) = strategy.slices.pop_front() {
            current_slice.status = SliceStatus::Submitted;
            current_slice.actual_time = Some(Utc::now());
            current_slice.order = Some(slice_order.clone());
            strategy.slices.push_front(current_slice);
        }

        info!("Executed TWAP slice: {} shares of {} at {:.4}", 
              slice_order.quantity, slice_order.symbol, slice_order.price.unwrap_or(0.0));

        Ok(())
    }

    fn calculate_slice_price(
        &self,
        order: &Order,
        market_data: &MarketData,
        conditions: &MarketConditions,
    ) -> TradingResult<f64> {
        let mid_price = market_data.price;
        
        // Adjust for market impact
        let impact_adjustment = match order.side {
            OrderSide::Buy => conditions.market_impact_bps / 10000.0,
            OrderSide::Sell => -conditions.market_impact_bps / 10000.0,
        };

        // Add small improvement to increase fill probability
        let improvement = match order.side {
            OrderSide::Buy => 0.0001, // Pay slightly more
            OrderSide::Sell => -0.0001, // Accept slightly less
        };

        Ok(mid_price * (1.0 + impact_adjustment + improvement))
    }

    async fn performance_monitoring_loop(&self) {
        let mut interval = interval(Duration::from_secs(30));

        loop {
            interval.tick().await;
            self.update_performance_metrics();
        }
    }

    fn update_performance_metrics(&self) {
        let mut strategies = self.active_strategies.write();
        
        for strategy in strategies.iter_mut() {
            self.calculate_strategy_performance(strategy);
        }
    }

    fn calculate_strategy_performance(&self, strategy: &mut TwapStrategy) {
        let executed_slices: Vec<_> = strategy.slices.iter()
            .filter(|s| s.status == SliceStatus::Filled)
            .collect();

        if !executed_slices.is_empty() {
            // Calculate VWAP of executed trades
            let total_value: f64 = executed_slices.iter()
                .filter_map(|s| s.order.as_ref())
                .map(|o| o.quantity * o.price.unwrap_or(0.0))
                .sum();
            
            let total_quantity: f64 = executed_slices.iter()
                .filter_map(|s| s.order.as_ref())
                .map(|o| o.quantity)
                .sum();

            if total_quantity > 0.0 {
                strategy.average_price = total_value / total_quantity;
                
                // Calculate implementation shortfall vs arrival price
                let arrival_price = strategy.parent_order.price.unwrap_or(strategy.average_price);
                let price_diff = match strategy.parent_order.side {
                    OrderSide::Buy => strategy.average_price - arrival_price,
                    OrderSide::Sell => arrival_price - strategy.average_price,
                };
                
                strategy.performance_metrics.implementation_shortfall = 
                    (price_diff / arrival_price) * 10000.0; // In BPS
            }

            // Calculate participation rate achieved
            strategy.performance_metrics.participation_rate_achieved = 
                self.calculate_achieved_participation_rate(strategy);

            // Calculate schedule adherence
            strategy.performance_metrics.schedule_adherence = 
                self.calculate_schedule_adherence(strategy);
        }
    }

    fn calculate_achieved_participation_rate(&self, strategy: &TwapStrategy) -> f64 {
        // Placeholder calculation
        // In production, this would compare executed volume vs market volume
        0.15 // 15% participation rate
    }

    fn calculate_schedule_adherence(&self, strategy: &TwapStrategy) -> f64 {
        let on_time_slices = strategy.slices.iter()
            .filter(|s| {
                if let Some(actual_time) = s.actual_time {
                    let delay = (actual_time - s.target_time).num_seconds().abs();
                    delay <= 30 // Within 30 seconds is considered on-time
                } else {
                    false
                }
            })
            .count();

        let total_slices = strategy.slices.len();
        if total_slices > 0 {
            on_time_slices as f64 / total_slices as f64
        } else {
            0.0
        }
    }

    fn finalize_strategy_performance(&self, strategy: &mut TwapStrategy) {
        info!("TWAP Strategy {} Performance Summary:", strategy.strategy_id);
        info!("  Implementation Shortfall: {:.2} bps", strategy.performance_metrics.implementation_shortfall);
        info!("  Average Price: {:.4}", strategy.average_price);
        info!("  Participation Rate: {:.2}%", strategy.performance_metrics.participation_rate_achieved * 100.0);
        info!("  Schedule Adherence: {:.2}%", strategy.performance_metrics.schedule_adherence * 100.0);
    }

    pub fn update_market_data(&self, symbol: String, market_data: MarketData) {
        let mut cache = self.market_data_cache.write();
        cache.insert(symbol.clone(), market_data.clone());

        // Update volume tracking
        let mut volume_tracker = self.volume_tracker.write();
        let volume_data = volume_tracker.symbol_volumes
            .entry(symbol)
            .or_insert_with(|| VolumeData {
                volume_history: VecDeque::new(),
                current_session_volume: 0,
                avg_volume_per_minute: 1000.0,
                last_updated: Utc::now(),
            });

        volume_data.volume_history.push_back(VolumePoint {
            timestamp: market_data.timestamp,
            volume: market_data.volume,
            price: market_data.price,
        });

        // Keep only recent history
        while volume_data.volume_history.len() > 1000 {
            volume_data.volume_history.pop_front();
        }

        volume_data.last_updated = Utc::now();
    }

    pub fn get_strategy_status(&self, strategy_id: Uuid) -> Option<TwapStrategy> {
        let strategies = self.active_strategies.read();
        strategies.iter()
            .find(|s| s.strategy_id == strategy_id)
            .cloned()
    }

    pub fn cancel_strategy(&self, strategy_id: Uuid) -> TradingResult<()> {
        let mut strategies = self.active_strategies.write();
        
        if let Some(strategy) = strategies.iter_mut().find(|s| s.strategy_id == strategy_id) {
            strategy.is_active = false;
            
            // Cancel pending slices
            for slice in &mut strategy.slices {
                if slice.status == SliceStatus::Pending {
                    slice.status = SliceStatus::Cancelled;
                }
            }
            
            info!("Cancelled TWAP strategy {}", strategy_id);
            Ok(())
        } else {
            Err(crate::trading::TradingError::InvalidOrder(
                format!("TWAP strategy {} not found", strategy_id)
            ))
        }
    }
}

// Implement Clone to allow spawning async tasks
impl Clone for TwapEngine {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            active_strategies: Arc::clone(&self.active_strategies),
            market_data_cache: Arc::clone(&self.market_data_cache),
            volume_tracker: Arc::clone(&self.volume_tracker),
            impact_model: MarketImpactModel {
                temporary_impact_coefficient: self.impact_model.temporary_impact_coefficient,
                permanent_impact_coefficient: self.impact_model.permanent_impact_coefficient,
                volatility_adjustment: self.impact_model.volatility_adjustment,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::OrderType;

    #[test]
    fn test_twap_engine_creation() {
        let config = Arc::new(crate::config::TradingConfig::default());
        let twap_engine = TwapEngine::new(config);
        
        let strategies = twap_engine.active_strategies.read();
        assert!(strategies.is_empty());
    }

    #[test]
    fn test_twap_strategy_creation() {
        let config = Arc::new(crate::config::TradingConfig::default());
        let twap_engine = TwapEngine::new(config);
        
        let order = Order {
            id: Uuid::new_v4(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            quantity: 1000.0,
            price: Some(150.0),
            ..Default::default()
        };

        let params = TwapParams {
            total_quantity: 1000.0,
            duration_minutes: 60,
            slice_interval_seconds: 300, // 5 minutes
            min_slice_size: 50.0,
            max_slice_size: 200.0,
            participation_rate: 0.1,
            price_improvement_threshold: 2.0,
            enable_market_impact_model: true,
            enable_adaptive_scheduling: false,
        };

        let result = twap_engine.create_twap_strategy(order, params);
        assert!(result.is_ok());

        let strategies = twap_engine.active_strategies.read();
        assert_eq!(strategies.len(), 1);
        assert_eq!(strategies[0].slices.len(), 12); // 60 minutes / 5 minute intervals
    }
}
