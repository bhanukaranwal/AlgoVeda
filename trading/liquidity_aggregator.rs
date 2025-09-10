/*!
 * Liquidity Aggregator for AlgoVeda Trading Platform
 * 
 * Aggregates order book data from multiple venues to provide
 * a consolidated view of available liquidity.
 */

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, instrument};

use crate::trading::{Venue, TradingResult, MarketData};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub venue: Option<Venue>,
    pub bids: BTreeMap<OrderedFloat, f64>, // Price -> Quantity
    pub asks: BTreeMap<OrderedFloat, f64>, // Price -> Quantity
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
}

// Wrapper for f64 to make it Ord for BTreeMap
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat(f)
    }
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            symbol: String::new(),
            venue: None,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: Utc::now(),
            sequence_number: 0,
        }
    }

    pub fn with_symbol(symbol: String) -> Self {
        Self {
            symbol,
            venue: None,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: Utc::now(),
            sequence_number: 0,
        }
    }

    pub fn add_bid(&mut self, price: f64, quantity: f64) {
        if quantity > 0.0 {
            self.bids.insert(OrderedFloat(price), quantity);
        } else {
            self.bids.remove(&OrderedFloat(price));
        }
        self.timestamp = Utc::now();
    }

    pub fn add_ask(&mut self, price: f64, quantity: f64) {
        if quantity > 0.0 {
            self.asks.insert(OrderedFloat(price), quantity);
        } else {
            self.asks.remove(&OrderedFloat(price));
        }
        self.timestamp = Utc::now();
    }

    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().last().map(|(price, qty)| (price.0, *qty))
    }

    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().first().map(|(price, qty)| (price.0, *qty))
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    pub fn total_bid_quantity(&self, max_levels: usize) -> f64 {
        self.bids.iter()
            .rev()
            .take(max_levels)
            .map(|(_, qty)| qty)
            .sum()
    }

    pub fn total_ask_quantity(&self, max_levels: usize) -> f64 {
        self.asks.iter()
            .take(max_levels)
            .map(|(_, qty)| qty)
            .sum()
    }
}

/// Liquidity Aggregator consolidates order books from multiple venues
#[derive(Debug)]
pub struct LiquidityAggregator {
    venue_books: Arc<RwLock<HashMap<Venue, OrderBook>>>,
    aggregated_books: Arc<RwLock<HashMap<String, OrderBook>>>, // Symbol -> Aggregated Book
    venue_weights: HashMap<Venue, f64>,
    max_staleness_ms: u64,
}

impl LiquidityAggregator {
    pub fn new() -> Self {
        Self {
            venue_books: Arc::new(RwLock::new(HashMap::new())),
            aggregated_books: Arc::new(RwLock::new(HashMap::new())),
            venue_weights: HashMap::new(),
            max_staleness_ms: 5000, // 5 seconds max staleness
        }
    }

    pub fn with_staleness_threshold(max_staleness_ms: u64) -> Self {
        Self {
            venue_books: Arc::new(RwLock::new(HashMap::new())),
            aggregated_books: Arc::new(RwLock::new(HashMap::new())),
            venue_weights: HashMap::new(),
            max_staleness_ms,
        }
    }

    #[instrument(skip(self))]
    pub fn add_orderbook(&self, venue: Venue, orderbook: OrderBook) -> TradingResult<()> {
        // Check if orderbook is not stale
        let now = Utc::now();
        let age_ms = (now - orderbook.timestamp).num_milliseconds() as u64;
        
        if age_ms > self.max_staleness_ms {
            info!("Rejecting stale orderbook from venue {}: {}ms old", venue.name, age_ms);
            return Ok(()); // Don't error, just skip stale data
        }

        // Store venue orderbook
        {
            let mut venue_books = self.venue_books.write();
            venue_books.insert(venue.clone(), orderbook.clone());
        }

        // Reaggregate for this symbol
        self.reaggregate_symbol(&orderbook.symbol)?;
        
        Ok(())
    }

    #[instrument(skip(self))]
    pub fn aggregate(&self, symbol: &str) -> TradingResult<OrderBook> {
        let aggregated_books = self.aggregated_books.read();
        
        aggregated_books
            .get(symbol)
            .cloned()
            .ok_or_else(|| crate::trading::TradingError::ExecutionFailed(
                format!("No aggregated orderbook for symbol {}", symbol)
            ))
    }

    pub fn set_venue_weight(&mut self, venue: Venue, weight: f64) {
        self.venue_weights.insert(venue, weight.max(0.0).min(1.0));
    }

    fn reaggregate_symbol(&self, symbol: &str) -> TradingResult<()> {
        let venue_books = self.venue_books.read();
        let now = Utc::now();
        
        // Get all fresh orderbooks for this symbol
        let relevant_books: Vec<_> = venue_books
            .iter()
            .filter(|(venue, book)| {
                book.symbol == symbol && 
                (now - book.timestamp).num_milliseconds() as u64 <= self.max_staleness_ms
            })
            .collect();

        if relevant_books.is_empty() {
            return Ok(());
        }

        // Create aggregated orderbook
        let mut aggregated = OrderBook::with_symbol(symbol.to_string());
        aggregated.timestamp = now;

        // Aggregate bids (highest prices first)
        let mut all_bids: Vec<(f64, f64, &Venue)> = Vec::new();
        for (venue, book) in &relevant_books {
            let weight = self.venue_weights.get(venue).unwrap_or(&1.0);
            for (price, quantity) in &book.bids {
                all_bids.push((price.0, quantity * weight, venue));
            }
        }
        all_bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Group by price and sum quantities
        let mut bid_prices: BTreeMap<OrderedFloat, f64> = BTreeMap::new();
        for (price, quantity, _venue) in all_bids {
            *bid_prices.entry(OrderedFloat(price)).or_insert(0.0) += quantity;
        }
        aggregated.bids = bid_prices;

        // Aggregate asks (lowest prices first)
        let mut all_asks: Vec<(f64, f64, &Venue)> = Vec::new();
        for (venue, book) in &relevant_books {
            let weight = self.venue_weights.get(venue).unwrap_or(&1.0);
            for (price, quantity) in &book.asks {
                all_asks.push((price.0, quantity * weight, venue));
            }
        }
        all_asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Group by price and sum quantities
        let mut ask_prices: BTreeMap<OrderedFloat, f64> = BTreeMap::new();
        for (price, quantity, _venue) in all_asks {
            *ask_prices.entry(OrderedFloat(price)).or_insert(0.0) += quantity;
        }
        aggregated.asks = ask_prices;

        // Store aggregated book
        {
            let mut aggregated_books = self.aggregated_books.write();
            aggregated_books.insert(symbol.to_string(), aggregated);
        }

        info!("Reaggregated orderbook for {} from {} venues", symbol, relevant_books.len());
        Ok(())
    }

    pub fn get_market_depth(&self, symbol: &str, levels: usize) -> TradingResult<MarketDepth> {
        let aggregated = self.aggregate(symbol)?;
        
        let bid_levels: Vec<(f64, f64)> = aggregated.bids.iter()
            .rev()
            .take(levels)
            .map(|(price, qty)| (price.0, *qty))
            .collect();

        let ask_levels: Vec<(f64, f64)> = aggregated.asks.iter()
            .take(levels)
            .map(|(price, qty)| (price.0, *qty))
            .collect();

        Ok(MarketDepth {
            symbol: symbol.to_string(),
            bid_levels,
            ask_levels,
            timestamp: aggregated.timestamp,
        })
    }

    pub fn calculate_vwap(&self, symbol: &str, quantity: f64, side: VwapSide) -> TradingResult<f64> {
        let aggregated = self.aggregate(symbol)?;
        
        let mut remaining_quantity = quantity;
        let mut total_value = 0.0;
        let mut total_quantity = 0.0;

        match side {
            VwapSide::Buy => {
                // Walk through asks (we're buying)
                for (price, available_qty) in &aggregated.asks {
                    let trade_qty = remaining_quantity.min(*available_qty);
                    total_value += price.0 * trade_qty;
                    total_quantity += trade_qty;
                    remaining_quantity -= trade_qty;

                    if remaining_quantity <= 0.0 {
                        break;
                    }
                }
            }
            VwapSide::Sell => {
                // Walk through bids (we're selling)
                for (price, available_qty) in aggregated.bids.iter().rev() {
                    let trade_qty = remaining_quantity.min(*available_qty);
                    total_value += price.0 * trade_qty;
                    total_quantity += trade_qty;
                    remaining_quantity -= trade_qty;

                    if remaining_quantity <= 0.0 {
                        break;
                    }
                }
            }
        }

        if total_quantity > 0.0 {
            Ok(total_value / total_quantity)
        } else {
            Err(crate::trading::TradingError::ExecutionFailed(
                "Insufficient liquidity for VWAP calculation".to_string()
            ))
        }
    }

    pub fn get_available_venues(&self) -> Vec<Venue> {
        let venue_books = self.venue_books.read();
        venue_books.keys().cloned().collect()
    }

    pub fn get_venue_count(&self) -> usize {
        let venue_books = self.venue_books.read();
        venue_books.len()
    }
}

#[derive(Debug, Clone)]
pub struct MarketDepth {
    pub symbol: String,
    pub bid_levels: Vec<(f64, f64)>, // (price, quantity)
    pub ask_levels: Vec<(f64, f64)>, // (price, quantity)
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy)]
pub enum VwapSide {
    Buy,
    Sell,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LiquidityAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_operations() {
        let mut book = OrderBook::with_symbol("AAPL".to_string());
        
        book.add_bid(100.0, 50.0);
        book.add_bid(99.5, 75.0);
        book.add_ask(101.0, 25.0);
        book.add_ask(101.5, 40.0);
        
        assert_eq!(book.best_bid(), Some((100.0, 50.0)));
        assert_eq!(book.best_ask(), Some((101.0, 25.0)));
        assert_eq!(book.spread(), Some(1.0));
        assert_eq!(book.mid_price(), Some(100.5));
    }

    #[test]
    fn test_liquidity_aggregation() {
        let aggregator = LiquidityAggregator::new();
        
        let venue1 = Venue {
            name: "NYSE".to_string(),
            exchange: "NYSE".to_string(),
            mic_code: "XNYS".to_string(),
            country: "US".to_string(),
            currency: "USD".to_string(),
            supports_dark_pool: false,
        };

        let mut book1 = OrderBook::with_symbol("AAPL".to_string());
        book1.add_bid(100.0, 50.0);
        book1.add_ask(101.0, 25.0);

        let result = aggregator.add_orderbook(venue1, book1);
        assert!(result.is_ok());

        let aggregated = aggregator.aggregate("AAPL");
        assert!(aggregated.is_ok());
        
        let book = aggregated.unwrap();
        assert_eq!(book.symbol, "AAPL");
        assert_eq!(book.best_bid(), Some((100.0, 50.0)));
    }

    #[test]
    fn test_vwap_calculation() {
        let aggregator = LiquidityAggregator::new();
        
        let venue1 = Venue {
            name: "NYSE".to_string(),
            exchange: "NYSE".to_string(),
            mic_code: "XNYS".to_string(),
            country: "US".to_string(),
            currency: "USD".to_string(),
            supports_dark_pool: false,
        };

        let mut book1 = OrderBook::with_symbol("AAPL".to_string());
        book1.add_ask(101.0, 50.0);
        book1.add_ask(102.0, 25.0);

        let _ = aggregator.add_orderbook(venue1, book1);
        
        let vwap = aggregator.calculate_vwap("AAPL", 60.0, VwapSide::Buy);
        assert!(vwap.is_ok());
        
        // Should be (101.0 * 50.0 + 102.0 * 10.0) / 60.0 = 101.167
        let vwap_price = vwap.unwrap();
        assert!((vwap_price - 101.167).abs() < 0.01);
    }
}
