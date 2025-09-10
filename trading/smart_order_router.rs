/*!
 * Smart Order Router for AlgoVeda Trading Platform
 * 
 * Advanced order routing with venue selection, latency optimization,
 * and intelligent order fragmentation across multiple venues.
 */

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, instrument};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::trading::{Order, TradingError, TradingResult, MarketData};
use crate::config::TradingConfig;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Venue {
    pub name: String,
    pub exchange: String,
    pub mic_code: String, // Market Identifier Code
    pub country: String,
    pub currency: String,
    pub supports_dark_pool: bool,
}

#[derive(Debug, Clone)]
pub struct VenueMetrics {
    pub venue: Venue,
    pub avg_latency_micros: u64,
    pub fill_rate: f64,
    pub average_spread_bps: f64,
    pub market_impact_bps: f64,
    pub available_liquidity: f64,
    pub last_updated: DateTime<Utc>,
    pub is_active: bool,
    pub daily_volume: f64,
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub venue: Venue,
    pub quantity: f64,
    pub price: Option<f64>,
    pub routing_reason: String,
    pub expected_fill_rate: f64,
    pub expected_latency_micros: u64,
}

/// Smart Order Router with advanced venue selection algorithms
#[derive(Debug)]
pub struct SmartOrderRouter {
    config: Arc<TradingConfig>,
    venue_metrics: Arc<RwLock<HashMap<Venue, VenueMetrics>>>,
    routing_strategies: HashMap<String, Box<dyn RoutingStrategy + Send + Sync>>,
    dark_pool_threshold: f64,
    max_venue_latency_micros: u64,
    min_fill_rate: f64,
}

pub trait RoutingStrategy: std::fmt::Debug {
    fn route_order(&self, order: &Order, venues: &[VenueMetrics]) -> TradingResult<Vec<RoutingDecision>>;
    fn name(&self) -> &str;
}

#[derive(Debug)]
pub struct BestExecutionRouter;

impl RoutingStrategy for BestExecutionRouter {
    fn route_order(&self, order: &Order, venues: &[VenueMetrics]) -> TradingResult<Vec<RoutingDecision>> {
        let mut decisions = Vec::new();
        
        // Sort venues by a composite score considering fill rate, latency, and spread
        let mut sorted_venues = venues.to_vec();
        sorted_venues.sort_by(|a, b| {
            let score_a = self.calculate_venue_score(a);
            let score_b = self.calculate_venue_score(b);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Route to best venue
        if let Some(best_venue) = sorted_venues.first() {
            decisions.push(RoutingDecision {
                venue: best_venue.venue.clone(),
                quantity: order.quantity,
                price: order.price,
                routing_reason: "Best execution venue".to_string(),
                expected_fill_rate: best_venue.fill_rate,
                expected_latency_micros: best_venue.avg_latency_micros,
            });
        }

        Ok(decisions)
    }

    fn name(&self) -> &str {
        "BestExecution"
    }
}

impl BestExecutionRouter {
    fn calculate_venue_score(&self, metrics: &VenueMetrics) -> f64 {
        // Composite score: higher fill rate, lower latency, lower spread
        let fill_score = metrics.fill_rate * 0.4;
        let latency_score = (1.0 / (metrics.avg_latency_micros as f64 + 1.0)) * 0.3;
        let spread_score = (1.0 / (metrics.average_spread_bps + 1.0)) * 0.3;
        
        fill_score + latency_score + spread_score
    }
}

#[derive(Debug)]
pub struct FragmentationRouter {
    max_fragment_size: f64,
}

impl FragmentationRouter {
    pub fn new(max_fragment_size: f64) -> Self {
        Self { max_fragment_size }
    }
}

impl RoutingStrategy for FragmentationRouter {
    fn route_order(&self, order: &Order, venues: &[VenueMetrics]) -> TradingResult<Vec<RoutingDecision>> {
        let mut decisions = Vec::new();
        let mut remaining_quantity = order.quantity;

        // Sort venues by available liquidity
        let mut sorted_venues = venues.to_vec();
        sorted_venues.sort_by(|a, b| b.available_liquidity.partial_cmp(&a.available_liquidity).unwrap());

        for venue_metrics in &sorted_venues {
            if remaining_quantity <= 0.0 {
                break;
            }

            let fragment_size = remaining_quantity.min(self.max_fragment_size)
                .min(venue_metrics.available_liquidity * 0.1); // Max 10% of available liquidity

            if fragment_size > 0.0 {
                decisions.push(RoutingDecision {
                    venue: venue_metrics.venue.clone(),
                    quantity: fragment_size,
                    price: order.price,
                    routing_reason: format!("Fragment {:.0} of {:.0}", fragment_size, order.quantity),
                    expected_fill_rate: venue_metrics.fill_rate,
                    expected_latency_micros: venue_metrics.avg_latency_micros,
                });

                remaining_quantity -= fragment_size;
            }
        }

        Ok(decisions)
    }

    fn name(&self) -> &str {
        "Fragmentation"
    }
}

impl SmartOrderRouter {
    pub fn new(
        config: Arc<TradingConfig>,
        max_venue_latency_micros: u64,
        min_fill_rate: f64,
    ) -> Self {
        let mut routing_strategies: HashMap<String, Box<dyn RoutingStrategy + Send + Sync>> = HashMap::new();
        routing_strategies.insert("best_execution".to_string(), Box::new(BestExecutionRouter));
        routing_strategies.insert("fragmentation".to_string(), Box::new(FragmentationRouter::new(10000.0)));

        Self {
            config,
            venue_metrics: Arc::new(RwLock::new(HashMap::new())),
            routing_strategies,
            dark_pool_threshold: 50000.0, // Orders above this size consider dark pools
            max_venue_latency_micros,
            min_fill_rate,
        }
    }

    #[instrument(skip(self))]
    pub fn route_order(&self, order: &Order) -> TradingResult<Vec<RoutingDecision>> {
        info!("Routing order {} for symbol {}", order.id, order.symbol);

        // Get available venues for this symbol
        let available_venues = self.get_available_venues(&order.symbol)?;
        
        if available_venues.is_empty() {
            return Err(TradingError::ExecutionFailed("No available venues".to_string()));
        }

        // Select routing strategy based on order characteristics
        let strategy_name = self.select_routing_strategy(order);
        let strategy = self.routing_strategies.get(&strategy_name)
            .ok_or_else(|| TradingError::ExecutionFailed("Routing strategy not found".to_string()))?;

        // Route the order
        let decisions = strategy.route_order(order, &available_venues)?;
        
        info!("Order routed to {} venues using {} strategy", decisions.len(), strategy_name);
        Ok(decisions)
    }

    pub fn add_venue(&self, venue: Venue, metrics: VenueMetrics) {
        let mut venue_metrics = self.venue_metrics.write();
        venue_metrics.insert(venue, metrics);
        info!("Added venue to routing table");
    }

    pub fn update_venue_metrics(&self, venue: &Venue, metrics: VenueMetrics) {
        let mut venue_metrics = self.venue_metrics.write();
        venue_metrics.insert(venue.clone(), metrics);
    }

    fn get_available_venues(&self, symbol: &str) -> TradingResult<Vec<VenueMetrics>> {
        let venue_metrics = self.venue_metrics.read();
        
        let available: Vec<VenueMetrics> = venue_metrics
            .values()
            .filter(|metrics| {
                metrics.is_active &&
                metrics.fill_rate >= self.min_fill_rate &&
                metrics.avg_latency_micros <= self.max_venue_latency_micros
            })
            .cloned()
            .collect();

        Ok(available)
    }

    fn select_routing_strategy(&self, order: &Order) -> String {
        // Select strategy based on order characteristics
        if order.quantity > self.dark_pool_threshold {
            "fragmentation".to_string()
        } else {
            "best_execution".to_string()
        }
    }

    pub fn get_venue_statistics(&self) -> HashMap<Venue, VenueMetrics> {
        let venue_metrics = self.venue_metrics.read();
        venue_metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_creation() {
        let venue = Venue {
            name: "NYSE".to_string(),
            exchange: "NYSE".to_string(),
            mic_code: "XNYS".to_string(),
            country: "US".to_string(),
            currency: "USD".to_string(),
            supports_dark_pool: false,
        };

        assert_eq!(venue.name, "NYSE");
        assert_eq!(venue.mic_code, "XNYS");
    }

    #[test]
    fn test_best_execution_routing() {
        let strategy = BestExecutionRouter;
        let order = Order::new();
        
        let venue_metrics = vec![
            VenueMetrics {
                venue: Venue {
                    name: "NYSE".to_string(),
                    exchange: "NYSE".to_string(),
                    mic_code: "XNYS".to_string(),
                    country: "US".to_string(),
                    currency: "USD".to_string(),
                    supports_dark_pool: false,
                },
                avg_latency_micros: 100,
                fill_rate: 0.95,
                average_spread_bps: 2.0,
                market_impact_bps: 1.0,
                available_liquidity: 1000000.0,
                last_updated: Utc::now(),
                is_active: true,
                daily_volume: 50000000.0,
            }
        ];

        let result = strategy.route_order(&order, &venue_metrics);
        assert!(result.is_ok());
        
        let decisions = result.unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].venue.name, "NYSE");
    }
}
