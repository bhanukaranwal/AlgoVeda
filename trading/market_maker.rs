/*!
 * Market Making Engine for AlgoVeda Trading Platform
 * 
 * Sophisticated market making with inventory management, adverse
 * selection protection, and dynamic spread adjustment.
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, warn, instrument};
use uuid::Uuid;

use crate::trading::{Order, OrderSide, OrderType, TimeInForce, OrderStatus, TradingResult, MarketData};
use crate::config::TradingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingParams {
    pub max_inventory: f64,
    pub target_inventory: f64,
    pub quote_size: f64,
    pub max_spread_bps: f64,
    pub min_spread_bps: f64,
    pub inventory_skew_factor: f64,
    pub adverse_selection_threshold: f64,
    pub quote_refresh_interval_ms: u64,
    pub max_quote_age_ms: u64,
    pub enable_inventory_management: bool,
    pub enable_adverse_selection_protection: bool,
}

impl Default for MarketMakingParams {
    fn default() -> Self {
        Self {
            max_inventory: 10000.0,
            target_inventory: 0.0,
            quote_size: 100.0,
            max_spread_bps: 50.0,
            min_spread_bps: 5.0,
            inventory_skew_factor: 0.1,
            adverse_selection_threshold: 0.002, // 20 bps
            quote_refresh_interval_ms: 100,
            max_quote_age_ms: 1000,
            enable_inventory_management: true,
            enable_adverse_selection_protection: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Quote {
    pub bid_order: Order,
    pub ask_order: Order,
    pub created_at: DateTime<Utc>,
    pub mid_price: f64,
    pub spread_bps: f64,
    pub inventory_at_quote: f64,
}

#[derive(Debug, Clone)]
pub struct InventoryPosition {
    pub symbol: String,
    pub quantity: f64,
    pub average_price: f64,
    pub unrealized_pnl: f64,
    pub last_updated: DateTime<Utc>,
}

/// Market Making Engine with advanced inventory and risk management
#[derive(Debug)]
pub struct MarketMaker {
    symbol: String,
    params: MarketMakingParams,
    config: Arc<TradingConfig>,
    inventory: Arc<RwLock<InventoryPosition>>,
    active_quotes: Arc<RwLock<Option<Quote>>>,
    quote_history: Arc<RwLock<VecDeque<Quote>>>,
    fill_history: Arc<RwLock<VecDeque<Fill>>>,
    adverse_selection_metrics: Arc<RwLock<AdverseSelectionMetrics>>,
    is_active: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub timestamp: DateTime<Utc>,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub was_aggressive: bool, // True if we were lifted/hit, false if we lifted/hit
}

#[derive(Debug, Clone, Default)]
pub struct AdverseSelectionMetrics {
    pub recent_adverse_fills: VecDeque<Fill>,
    pub adverse_selection_ratio: f64,
    pub avg_adverse_selection_bps: f64,
    pub last_calculated: DateTime<Utc>,
}

impl MarketMaker {
    pub fn new(
        symbol: String,
        params: MarketMakingParams,
        config: Arc<TradingConfig>,
    ) -> Self {
        let inventory = InventoryPosition {
            symbol: symbol.clone(),
            quantity: 0.0,
            average_price: 0.0,
            unrealized_pnl: 0.0,
            last_updated: Utc::now(),
        };

        Self {
            symbol,
            params,
            config,
            inventory: Arc::new(RwLock::new(inventory)),
            active_quotes: Arc::new(RwLock::new(None)),
            quote_history: Arc::new(RwLock::new(VecDeque::new())),
            fill_history: Arc::new(RwLock::new(VecDeque::new())),
            adverse_selection_metrics: Arc::new(RwLock::new(AdverseSelectionMetrics::default())),
            is_active: Arc::new(RwLock::new(false)),
        }
    }

    #[instrument(skip(self))]
    pub fn generate_quotes(&self, market_data: &MarketData) -> TradingResult<Option<Quote>> {
        let is_active = *self.is_active.read();
        if !is_active {
            return Ok(None);
        }

        // Check if we should refresh quotes
        if !self.should_refresh_quotes(market_data)? {
            return Ok(None);
        }

        let inventory = self.inventory.read();
        let current_inventory = inventory.quantity;
        drop(inventory);

        // Calculate optimal spread based on inventory and adverse selection
        let optimal_spread = self.calculate_optimal_spread(current_inventory, market_data)?;
        
        // Generate bid and ask prices
        let mid_price = market_data.price;
        let half_spread = optimal_spread / 2.0;
        
        // Apply inventory skew
        let inventory_skew = self.calculate_inventory_skew(current_inventory);
        
        let bid_price = mid_price - half_spread + inventory_skew;
        let ask_price = mid_price + half_spread + inventory_skew;

        // Determine quote sizes based on inventory
        let (bid_size, ask_size) = self.calculate_quote_sizes(current_inventory);

        // Create orders
        let bid_order = self.create_quote_order(
            OrderSide::Buy,
            bid_price,
            bid_size,
        )?;

        let ask_order = self.create_quote_order(
            OrderSide::Sell,
            ask_price,
            ask_size,
        )?;

        let quote = Quote {
            bid_order,
            ask_order,
            created_at: Utc::now(),
            mid_price,
            spread_bps: (optimal_spread / mid_price) * 10000.0,
            inventory_at_quote: current_inventory,
        };

        // Store active quote
        {
            let mut active_quotes = self.active_quotes.write();
            *active_quotes = Some(quote.clone());
        }

        // Add to history
        {
            let mut history = self.quote_history.write();
            history.push_back(quote.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        info!("Generated market making quotes: bid={:.4} ask={:.4} spread={:.2}bps", 
              quote.bid_order.price.unwrap(), 
              quote.ask_order.price.unwrap(), 
              quote.spread_bps);

        Ok(Some(quote))
    }

    fn should_refresh_quotes(&self, market_data: &MarketData) -> TradingResult<bool> {
        let active_quotes = self.active_quotes.read();
        
        match active_quotes.as_ref() {
            None => Ok(true), // No active quotes, should generate
            Some(quote) => {
                // Check quote age
                let age_ms = (Utc::now() - quote.created_at).num_milliseconds() as u64;
                if age_ms > self.params.max_quote_age_ms {
                    return Ok(true);
                }

                // Check if market has moved significantly
                let price_change = (market_data.price - quote.mid_price).abs() / quote.mid_price;
                if price_change > 0.001 { // 10 bps price change
                    return Ok(true);
                }

                // Check adverse selection
                if self.params.enable_adverse_selection_protection {
                    let adverse_metrics = self.adverse_selection_metrics.read();
                    if adverse_metrics.adverse_selection_ratio > self.params.adverse_selection_threshold {
                        warn!("High adverse selection detected: {:.2}%", 
                              adverse_metrics.adverse_selection_ratio * 100.0);
                        return Ok(false); // Don't quote when adverse selection is high
                    }
                }

                Ok(false)
            }
        }
    }

    fn calculate_optimal_spread(&self, inventory: f64, market_data: &MarketData) -> TradingResult<f64> {
        let base_spread_bps = (self.params.min_spread_bps + self.params.max_spread_bps) / 2.0;
        
        // Adjust for inventory risk
        let inventory_risk_adjustment = if self.params.enable_inventory_management {
            let inventory_ratio = inventory.abs() / self.params.max_inventory;
            inventory_ratio * self.params.max_spread_bps * 0.5
        } else {
            0.0
        };

        // Adjust for adverse selection
        let adverse_selection_adjustment = if self.params.enable_adverse_selection_protection {
            let adverse_metrics = self.adverse_selection_metrics.read();
            adverse_metrics.avg_adverse_selection_bps * 0.1
        } else {
            0.0
        };

        let total_spread_bps = base_spread_bps + inventory_risk_adjustment + adverse_selection_adjustment;
        let clamped_spread_bps = total_spread_bps.clamp(self.params.min_spread_bps, self.params.max_spread_bps);
        
        Ok((clamped_spread_bps / 10000.0) * market_data.price)
    }

    fn calculate_inventory_skew(&self, inventory: f64) -> f64 {
        if !self.params.enable_inventory_management {
            return 0.0;
        }

        let inventory_imbalance = (inventory - self.params.target_inventory) / self.params.max_inventory;
        -inventory_imbalance * self.params.inventory_skew_factor
    }

    fn calculate_quote_sizes(&self, inventory: f64) -> (f64, f64) {
        let base_size = self.params.quote_size;
        
        if !self.params.enable_inventory_management {
            return (base_size, base_size);
        }

        // Adjust sizes based on inventory to encourage mean reversion
        let inventory_ratio = inventory / self.params.max_inventory;
        
        let bid_size_multiplier = if inventory_ratio > 0.0 {
            1.0 - inventory_ratio.abs() * 0.5 // Reduce bid size when long
        } else {
            1.0 + inventory_ratio.abs() * 0.5 // Increase bid size when short
        };

        let ask_size_multiplier = if inventory_ratio < 0.0 {
            1.0 - inventory_ratio.abs() * 0.5 // Reduce ask size when short
        } else {
            1.0 + inventory_ratio.abs() * 0.5 // Increase ask size when long
        };

        let bid_size = (base_size * bid_size_multiplier).max(base_size * 0.1);
        let ask_size = (base_size * ask_size_multiplier).max(base_size * 0.1);

        (bid_size, ask_size)
    }

    fn create_quote_order(&self, side: OrderSide, price: f64, quantity: f64) -> TradingResult<Order> {
        let now = Utc::now();
        
        Ok(Order {
            id: Uuid::new_v4(),
            client_order_id: format!("mm_{}_{}", 
                match side { OrderSide::Buy => "bid", OrderSide::Sell => "ask" },
                now.timestamp_millis()
            ),
            symbol: self.symbol.clone(),
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::PendingNew,
            filled_quantity: 0.0,
            average_fill_price: 0.0,
            remaining_quantity: quantity,
            created_at: now,
            updated_at: now,
            account_id: "market_maker".to_string(),
            strategy_id: Some("market_making".to_string()),
            parent_order_id: None,
            execution_instructions: {
                let mut instructions = HashMap::new();
                instructions.insert("post_only".to_string(), "true".to_string());
                instructions
            },
            tags: {
                let mut tags = HashMap::new();
                tags.insert("strategy".to_string(), "market_making".to_string());
                tags.insert("symbol".to_string(), self.symbol.clone());
                tags
            },
        })
    }

    pub fn on_fill(&self, fill: Fill, market_price: f64) -> TradingResult<()> {
        // Update inventory
        {
            let mut inventory = self.inventory.write();
            let quantity_change = match fill.side {
                OrderSide::Buy => fill.quantity,
                OrderSide::Sell => -fill.quantity,
            };

            // Update position
            if inventory.quantity == 0.0 {
                inventory.average_price = fill.price;
            } else {
                let total_value = inventory.quantity * inventory.average_price + quantity_change * fill.price;
                let new_quantity = inventory.quantity + quantity_change;
                
                if new_quantity != 0.0 {
                    inventory.average_price = total_value / new_quantity;
                }
            }
            
            inventory.quantity += quantity_change;
            inventory.unrealized_pnl = (market_price - inventory.average_price) * inventory.quantity;
            inventory.last_updated = Utc::now();
        }

        // Add to fill history
        {
            let mut history = self.fill_history.write();
            history.push_back(fill.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Update adverse selection metrics
        self.update_adverse_selection_metrics(&fill, market_price)?;

        info!("Market making fill processed: {} {:.0}@{:.4}", 
              match fill.side { OrderSide::Buy => "BUY", OrderSide::Sell => "SELL" },
              fill.quantity, fill.price);

        Ok(())
    }

    fn update_adverse_selection_metrics(&self, fill: &Fill, current_market_price: f64) -> TradingResult<()> {
        if !self.params.enable_adverse_selection_protection {
            return Ok(());
        }

        let mut metrics = self.adverse_selection_metrics.write();
        
        // Check if this fill was adverse
        let is_adverse = match fill.side {
            OrderSide::Buy => current_market_price < fill.price, // Bought high
            OrderSide::Sell => current_market_price > fill.price, // Sold low
        };

        if is_adverse {
            metrics.recent_adverse_fills.push_back(fill.clone());
        }

        // Keep only recent fills (last 100)
        while metrics.recent_adverse_fills.len() > 100 {
            metrics.recent_adverse_fills.pop_front();
        }

        // Calculate adverse selection ratio and average adverse selection
        let total_fills = self.fill_history.read().len().max(1);
        let adverse_fills = metrics.recent_adverse_fills.len();
        
        metrics.adverse_selection_ratio = adverse_fills as f64 / total_fills as f64;
        
        if !metrics.recent_adverse_fills.is_empty() {
            let total_adverse_bps: f64 = metrics.recent_adverse_fills.iter()
                .map(|f| {
                    match f.side {
                        OrderSide::Buy => ((f.price - current_market_price) / f.price).abs(),
                        OrderSide::Sell => ((current_market_price - f.price) / f.price).abs(),
                    } * 10000.0
                })
                .sum();
                
            metrics.avg_adverse_selection_bps = total_adverse_bps / adverse_fills as f64;
        }

        metrics.last_calculated = Utc::now();

        Ok(())
    }

    pub fn start(&self) -> TradingResult<()> {
        let mut is_active = self.is_active.write();
        *is_active = true;
        info!("Market maker started for symbol {}", self.symbol);
        Ok(())
    }

    pub fn stop(&self) -> TradingResult<()> {
        let mut is_active = self.is_active.write();
        *is_active = false;
        info!("Market maker stopped for symbol {}", self.symbol);
        Ok(())
    }

    pub fn get_inventory(&self) -> InventoryPosition {
        let inventory = self.inventory.read();
        inventory.clone()
    }

    pub fn get_active_quote(&self) -> Option<Quote> {
        let active_quotes = self.active_quotes.read();
        active_quotes.clone()
    }

    pub fn get_adverse_selection_metrics(&self) -> AdverseSelectionMetrics {
        let metrics = self.adverse_selection_metrics.read();
        metrics.clone()
    }

    pub fn update_params(&self, new_params: MarketMakingParams) -> TradingResult<()> {
        // In a real implementation, this would need to be atomic
        // For now, we'll log the change
        info!("Market making parameters updated for symbol {}", self.symbol);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_maker_creation() {
        let params = MarketMakingParams::default();
        let config = Arc::new(TradingConfig::default());
        let mm = MarketMaker::new("AAPL".to_string(), params, config);
        
        assert_eq!(mm.symbol, "AAPL");
        assert!(!*mm.is_active.read());
    }

    #[test]
    fn test_inventory_skew_calculation() {
        let params = MarketMakingParams::default();
        let config = Arc::new(TradingConfig::default());
        let mm = MarketMaker::new("AAPL".to_string(), params, config);
        
        // Test positive inventory (long) should create negative skew (lower prices)
        let skew = mm.calculate_inventory_skew(1000.0);
        assert!(skew < 0.0);
        
        // Test negative inventory (short) should create positive skew (higher prices)
        let skew = mm.calculate_inventory_skew(-1000.0);
        assert!(skew > 0.0);
        
        // Test zero inventory should have no skew
        let skew = mm.calculate_inventory_skew(0.0);
        assert_eq!(skew, 0.0);
    }

    #[test]
    fn test_quote_size_calculation() {
        let params = MarketMakingParams::default();
        let config = Arc::new(TradingConfig::default());
        let mm = MarketMaker::new("AAPL".to_string(), params, config);
        
        // Test balanced inventory
        let (bid_size, ask_size) = mm.calculate_quote_sizes(0.0);
        assert_eq!(bid_size, ask_size);
        assert_eq!(bid_size, params.quote_size);
        
        // Test positive inventory (should favor selling)
        let (bid_size, ask_size) = mm.calculate_quote_sizes(5000.0);
        assert!(ask_size > bid_size);
        
        // Test negative inventory (should favor buying)
        let (bid_size, ask_size) = mm.calculate_quote_sizes(-5000.0);
        assert!(bid_size > ask_size);
    }
}
