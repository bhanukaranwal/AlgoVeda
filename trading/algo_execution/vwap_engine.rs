/*!
 * Volume-Weighted Average Price (VWAP) Execution Engine
 * 
 * Advanced VWAP implementation with historical volume curves,
 * real-time optimization, and market microstructure awareness.
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::time::{Duration, interval};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Timelike};
use tracing::{info, warn, instrument};
use uuid::Uuid;

use crate::trading::{Order, OrderSide, TradingResult, MarketData};
use crate::trading::algo_execution::twap_engine::{SliceStatus, MarketConditions};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapParams {
    pub total_quantity: f64,
    pub target_participation_rate: f64,
    pub max_participation_rate: f64,
    pub volume_curve_lookback_days: u32,
    pub aggression_factor: f64, // 0.0 = passive, 1.0 = aggressive
    pub enable_volume_forecasting: bool,
    pub enable_catch_up: bool,
    pub price_improvement_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct VolumeProfile {
    pub symbol: String,
    pub intraday_curve: Vec<f64>, // Minute-by-minute volume percentages
    pub total_daily_volume: f64,
    pub last_updated: DateTime<Utc>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone)]
pub struct VwapSlice {
    pub slice_id: Uuid,
    pub target_quantity: f64,
    pub actual_quantity: f64,
    pub target_minute: u32,
    pub volume_allocation: f64,
    pub participation_rate: f64,
    pub status: SliceStatus,
    pub orders: Vec<Order>,
}

/// VWAP Engine with sophisticated volume modeling
#[derive(Debug)]
pub struct VwapEngine {
    active_strategies: Arc<RwLock<HashMap<Uuid, VwapStrategy>>>,
    volume_profiles: Arc<RwLock<HashMap<String, VolumeProfile>>>,
    market_data_cache: Arc<RwLock<HashMap<String, MarketData>>>,
    historical_volumes: Arc<RwLock<HashMap<String, VecDeque<DailyVolumeData>>>>,
}

#[derive(Debug)]
pub struct VwapStrategy {
    pub strategy_id: Uuid,
    pub parent_order: Order,
    pub params: VwapParams,
    pub volume_profile: VolumeProfile,
    pub slices: Vec<VwapSlice>,
    pub executed_quantity: f64,
    pub vwap_price: f64,
    pub market_vwap: f64,
    pub start_minute: u32,
    pub current_minute: u32,
    pub is_active: bool,
    pub performance: VwapPerformance,
}

#[derive(Debug, Clone, Default)]
pub struct VwapPerformance {
    pub vwap_variance_bps: f64,
    pub volume_participation_achieved: f64,
    pub schedule_adherence: f64,
    pub market_impact_bps: f64,
    pub slippage_bps: f64,
}

#[derive(Debug, Clone)]
struct DailyVolumeData {
    date: chrono::NaiveDate,
    minute_volumes: Vec<u64>,
    total_volume: u64,
}

impl VwapEngine {
    pub fn new() -> Self {
        Self {
            active_strategies: Arc::new(RwLock::new(HashMap::new())),
            volume_profiles: Arc::new(RwLock::new(HashMap::new())),
            market_data_cache: Arc::new(RwLock::new(HashMap::new())),
            historical_volumes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[instrument(skip(self))]
    pub async fn create_vwap_strategy(
        &self, 
        parent_order: Order, 
        params: VwapParams
    ) -> TradingResult<Uuid> {
        let strategy_id = Uuid::new_v4();
        
        // Get or create volume profile
        let volume_profile = self.get_or_create_volume_profile(&parent_order.symbol, &params).await?;
        
        // Generate execution schedule based on volume curve
        let slices = self.generate_vwap_schedule(&parent_order, &params, &volume_profile)?;
        
        let strategy = VwapStrategy {
            strategy_id,
            parent_order: parent_order.clone(),
            params,
            volume_profile,
            slices,
            executed_quantity: 0.0,
            vwap_price: 0.0,
            market_vwap: 0.0,
            start_minute: self.get_current_market_minute(),
            current_minute: self.get_current_market_minute(),
            is_active: true,
            performance: VwapPerformance::default(),
        };

        {
            let mut strategies = self.active_strategies.write();
            strategies.insert(strategy_id, strategy);
        }

        info!("Created VWAP strategy {} for {}", strategy_id, parent_order.symbol);
        Ok(strategy_id)
    }

    async fn get_or_create_volume_profile(
        &self, 
        symbol: &str, 
        params: &VwapParams
    ) -> TradingResult<VolumeProfile> {
        {
            let profiles = self.volume_profiles.read();
            if let Some(profile) = profiles.get(symbol) {
                if profile.confidence_score > 0.7 {
                    return Ok(profile.clone());
                }
            }
        }

        // Build new volume profile from historical data
        self.build_volume_profile(symbol, params).await
    }

    async fn build_volume_profile(
        &self, 
        symbol: &str, 
        params: &VwapParams
    ) -> TradingResult<VolumeProfile> {
        let historical_data = {
            let historical = self.historical_volumes.read();
            historical.get(symbol).cloned().unwrap_or_default()
        };

        if historical_data.is_empty() {
            // Fallback to default U-shaped curve
            return Ok(self.create_default_volume_profile(symbol));
        }

        // Calculate average intraday volume distribution
        let mut minute_averages = vec![0.0; 375]; // Market hours: 9:15 AM - 3:30 PM
        let mut valid_days = 0;

        for day_data in historical_data.iter().take(params.volume_curve_lookback_days as usize) {
            if day_data.minute_volumes.len() == 375 && day_data.total_volume > 0 {
                for (i, &volume) in day_data.minute_volumes.iter().enumerate() {
                    minute_averages[i] += volume as f64 / day_data.total_volume as f64;
                }
                valid_days += 1;
            }
        }

        if valid_days > 0 {
            for avg in &mut minute_averages {
                *avg /= valid_days as f64;
            }
        }

        let confidence_score = (valid_days as f64 / params.volume_curve_lookback_days as f64)
            .min(1.0);

        let total_daily_volume = historical_data.iter()
            .take(5) // Last 5 days
            .map(|d| d.total_volume as f64)
            .sum::<f64>() / historical_data.len().min(5) as f64;

        Ok(VolumeProfile {
            symbol: symbol.to_string(),
            intraday_curve: minute_averages,
            total_daily_volume,
            last_updated: Utc::now(),
            confidence_score,
        })
    }

    fn create_default_volume_profile(&self, symbol: &str) -> VolumeProfile {
        let mut curve = vec![0.0; 375];
        
        // U-shaped volume pattern (high at open/close, low mid-day)
        for (i, weight) in curve.iter_mut().enumerate() {
            let minute_ratio = i as f64 / 375.0;
            *weight = if minute_ratio < 0.1 || minute_ratio > 0.9 {
                0.004 // High volume at open/close
            } else if minute_ratio < 0.3 || minute_ratio > 0.7 {
                0.002 // Medium volume
            } else {
                0.001 // Low volume mid-day
            };
        }

        VolumeProfile {
            symbol: symbol.to_string(),
            intraday_curve: curve,
            total_daily_volume: 1000000.0, // Default assumption
            last_updated: Utc::now(),
            confidence_score: 0.3, // Low confidence for default
        }
    }

    fn generate_vwap_schedule(
        &self,
        parent_order: &Order,
        params: &VwapParams,
        volume_profile: &VolumeProfile,
    ) -> TradingResult<Vec<VwapSlice>> {
        let mut slices = Vec::new();
        let current_minute = self.get_current_market_minute();
        let remaining_minutes = 375 - current_minute as usize;

        for minute_offset in 0..remaining_minutes {
            let minute_index = current_minute as usize + minute_offset;
            if minute_index >= volume_profile.intraday_curve.len() {
                break;
            }

            let volume_weight = volume_profile.intraday_curve[minute_index];
            let target_quantity = params.total_quantity * volume_weight;

            if target_quantity > 0.1 { // Skip very small slices
                let slice = VwapSlice {
                    slice_id: Uuid::new_v4(),
                    target_quantity,
                    actual_quantity: 0.0,
                    target_minute: minute_index as u32,
                    volume_allocation: volume_weight,
                    participation_rate: params.target_participation_rate,
                    status: SliceStatus::Pending,
                    orders: Vec::new(),
                };
                slices.push(slice);
            }
        }

        info!("Generated {} VWAP slices for execution", slices.len());
        Ok(slices)
    }

    pub async fn start_execution(&self) -> TradingResult<()> {
        info!("Starting VWAP execution engine");
        
        let engine = self.clone();
        tokio::spawn(async move {
            engine.execution_loop().await;
        });

        let engine = self.clone();
        tokio::spawn(async move {
            engine.performance_monitoring_loop().await;
        });

        Ok(())
    }

    async fn execution_loop(&self) {
        let mut interval = interval(Duration::from_secs(60)); // Execute every minute

        loop {
            interval.tick().await;
            
            let current_minute = self.get_current_market_minute();
            if let Err(e) = self.process_minute_execution(current_minute).await {
                warn!("VWAP execution error: {}", e);
            }
        }
    }

    async fn process_minute_execution(&self, current_minute: u32) -> TradingResult<()> {
        let mut strategies = self.active_strategies.write();
        
        for strategy in strategies.values_mut().filter(|s| s.is_active) {
            strategy.current_minute = current_minute;
            
            // Find slices for current minute
            let current_slices: Vec<_> = strategy.slices.iter_mut()
                .filter(|s| s.target_minute == current_minute && s.status == SliceStatus::Pending)
                .collect();

            for slice in current_slices {
                self.execute_vwap_slice(strategy, slice).await?;
            }

            // Check if strategy is complete
            if strategy.slices.iter().all(|s| matches!(s.status, SliceStatus::Filled | SliceStatus::Cancelled)) {
                strategy.is_active = false;
                self.finalize_vwap_performance(strategy);
            }
        }

        Ok(())
    }

    async fn execute_vwap_slice(
        &self,
        strategy: &mut VwapStrategy,
        slice: &mut VwapSlice,
    ) -> TradingResult<()> {
        // Calculate expected volume for this minute
        let expected_minute_volume = strategy.volume_profile.total_daily_volume * slice.volume_allocation;
        let max_participation_volume = expected_minute_volume * strategy.params.max_participation_rate;
        
        // Adjust quantity based on catch-up logic
        let adjusted_quantity = if strategy.params.enable_catch_up {
            self.calculate_catch_up_quantity(strategy, slice, max_participation_volume)
        } else {
            slice.target_quantity.min(max_participation_volume)
        };

        // Create and submit orders
        if adjusted_quantity > 0.0 {
            let order = self.create_vwap_order(strategy, adjusted_quantity)?;
            slice.orders.push(order);
            slice.status = SliceStatus::Submitted;
            
            info!("Submitted VWAP slice: {} shares for minute {}", 
                  adjusted_quantity, slice.target_minute);
        } else {
            slice.status = SliceStatus::Cancelled;
        }

        Ok(())
    }

    fn calculate_catch_up_quantity(
        &self,
        strategy: &VwapStrategy,
        slice: &VwapSlice,
        max_participation: f64,
    ) -> f64 {
        // Calculate how far behind schedule we are
        let scheduled_quantity: f64 = strategy.slices.iter()
            .filter(|s| s.target_minute < strategy.current_minute)
            .map(|s| s.target_quantity)
            .sum();
        
        let shortfall = scheduled_quantity - strategy.executed_quantity;
        
        if shortfall > 0.0 {
            // Add catch-up quantity with aggression factor
            let catch_up_quantity = shortfall * strategy.params.aggression_factor;
            (slice.target_quantity + catch_up_quantity).min(max_participation)
        } else {
            slice.target_quantity.min(max_participation)
        }
    }

    fn create_vwap_order(&self, strategy: &VwapStrategy, quantity: f64) -> TradingResult<Order> {
        let mut order = strategy.parent_order.clone();
        order.id = Uuid::new_v4();
        order.quantity = quantity;
        order.parent_order_id = Some(strategy.parent_order.id);

        // Set competitive price based on current market
        if let Some(market_data) = self.market_data_cache.read().get(&order.symbol) {
            let price_adjustment = strategy.params.price_improvement_threshold / 10000.0;
            order.price = Some(match order.side {
                OrderSide::Buy => market_data.price * (1.0 + price_adjustment),
                OrderSide::Sell => market_data.price * (1.0 - price_adjustment),
            });
        }

        Ok(order)
    }

    async fn performance_monitoring_loop(&self) {
        let mut interval = interval(Duration::from_secs(30));

        loop {
            interval.tick().await;
            self.update_vwap_performance().await;
        }
    }

    async fn update_vwap_performance(&self) {
        let mut strategies = self.active_strategies.write();
        
        for strategy in strategies.values_mut() {
            self.calculate_vwap_performance(strategy);
        }
    }

    fn calculate_vwap_performance(&self, strategy: &mut VwapStrategy) {
        let filled_orders: Vec<_> = strategy.slices.iter()
            .flat_map(|s| &s.orders)
            .filter(|o| o.status == crate::trading::OrderStatus::Filled)
            .collect();

        if !filled_orders.is_empty() {
            // Calculate execution VWAP
            let total_value: f64 = filled_orders.iter()
                .map(|o| o.quantity * o.price.unwrap_or(0.0))
                .sum();
            let total_quantity: f64 = filled_orders.iter()
                .map(|o| o.quantity)
                .sum();

            if total_quantity > 0.0 {
                strategy.vwap_price = total_value / total_quantity;
                strategy.executed_quantity = total_quantity;

                // Calculate market VWAP (simplified)
                strategy.market_vwap = self.calculate_market_vwap(&strategy.parent_order.symbol);

                // Performance metrics
                let vwap_diff = strategy.vwap_price - strategy.market_vwap;
                strategy.performance.vwap_variance_bps = (vwap_diff / strategy.market_vwap) * 10000.0;
                
                strategy.performance.volume_participation_achieved = 
                    total_quantity / strategy.volume_profile.total_daily_volume;
            }
        }
    }

    fn calculate_market_vwap(&self, symbol: &str) -> f64 {
        // Simplified market VWAP calculation
        // In production, this would use actual market volume data
        if let Some(market_data) = self.market_data_cache.read().get(symbol) {
            market_data.price
        } else {
            0.0
        }
    }

    fn finalize_vwap_performance(&self, strategy: &VwapStrategy) {
        info!("VWAP Strategy {} Performance Summary:", strategy.strategy_id);
        info!("  Execution VWAP: {:.4}", strategy.vwap_price);
        info!("  Market VWAP: {:.4}", strategy.market_vwap);
        info!("  VWAP Variance: {:.2} bps", strategy.performance.vwap_variance_bps);
        info!("  Volume Participation: {:.4}%", strategy.performance.volume_participation_achieved * 100.0);
    }

    fn get_current_market_minute(&self) -> u32 {
        let now = Utc::now().with_timezone(&chrono::FixedOffset::east_opt(5 * 3600 + 30 * 60).unwrap());
        let market_start = now.date_naive().and_hms_opt(9, 15, 0).unwrap();
        let current_time = now.time();
        
        if current_time >= chrono::NaiveTime::from_hms_opt(9, 15, 0).unwrap() && 
           current_time <= chrono::NaiveTime::from_hms_opt(15, 30, 0).unwrap() {
            let elapsed = current_time - chrono::NaiveTime::from_hms_opt(9, 15, 0).unwrap();
            elapsed.num_minutes() as u32
        } else {
            0
        }
    }

    pub fn update_historical_volume(&self, symbol: String, date: chrono::NaiveDate, volumes: Vec<u64>) {
        let mut historical = self.historical_volumes.write();
        let symbol_data = historical.entry(symbol).or_insert_with(VecDeque::new);
        
        let daily_data = DailyVolumeData {
            date,
            total_volume: volumes.iter().sum(),
            minute_volumes: volumes,
        };
        
        symbol_data.push_back(daily_data);
        
        // Keep only recent data
        while symbol_data.len() > 60 { // 60 days
            symbol_data.pop_front();
        }
    }

    pub fn update_market_data(&self, symbol: String, market_data: MarketData) {
        let mut cache = self.market_data_cache.write();
        cache.insert(symbol, market_data);
    }
}

impl Clone for VwapEngine {
    fn clone(&self) -> Self {
        Self {
            active_strategies: Arc::clone(&self.active_strategies),
            volume_profiles: Arc::clone(&self.volume_profiles),
            market_data_cache: Arc::clone(&self.market_data_cache),
            historical_volumes: Arc::clone(&self.historical_volumes),
        }
    }
}

impl Default for VwapEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vwap_engine_creation() {
        let engine = VwapEngine::new();
        let strategies = engine.active_strategies.read();
        assert!(strategies.is_empty());
    }

    #[test]
    fn test_default_volume_profile() {
        let engine = VwapEngine::new();
        let profile = engine.create_default_volume_profile("AAPL");
        
        assert_eq!(profile.symbol, "AAPL");
        assert_eq!(profile.intraday_curve.len(), 375);
        assert!(profile.confidence_score < 0.5); // Low confidence for default
    }
}
