/*!
 * Smart Order Router
 * Intelligent order routing with latency and cost optimization
 */

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use uuid::Uuid;
use rust_decimal::Decimal;
use tracing::{info, debug, warn, instrument};
use serde::{Serialize, Deserialize};

use crate::{
    config::TradingConfig,
    trading::{Order, OrderSide, OrderType},
    market_data::{MarketData, Quote, Level2Book},
    networking::VenueConnector,
    monitoring::MetricsCollector,
    error::{Result, AlgoVedaError},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub venue: String,
    pub expected_cost: Decimal,
    pub expected_latency_ms: u64,
    pub fill_probability: f64,
    pub market_impact_bps: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueMetrics {
    pub venue: String,
    pub avg_latency_ms: f64,
    pub fill_rate: f64,
    pub avg_spread_bps: f64,
    pub market_depth: Decimal,
    pub recent_volume: Decimal,
    pub uptime_percentage: f64,
    pub last_updated: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConstraints {
    pub max_latency_ms: u64,
    pub max_cost_bps: u32,
    pub min_fill_probability: f64,
    pub preferred_venues: Vec<String>,
    pub blacklisted_venues: Vec<String>,
    pub dark_pool_preference: f64,
}

pub struct SmartOrderRouter {
    config: Arc<TradingConfig>,
    venue_connectors: Arc<RwLock<HashMap<String, Arc<VenueConnector>>>>,
    venue_metrics: Arc<RwLock<HashMap<String, VenueMetrics>>>,
    market_data: Arc<RwLock<HashMap<String, MarketData>>>,
    level2_books: Arc<RwLock<HashMap<String, Level2Book>>>,
    metrics_collector: Arc<MetricsCollector>,
    routing_history: Arc<Mutex<Vec<RoutingDecision>>>,
}

impl SmartOrderRouter {
    pub async fn new(
        config: Arc<TradingConfig>,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            venue_connectors: Arc::new(RwLock::new(HashMap::new())),
            venue_metrics: Arc::new(RwLock::new(HashMap::new())),
            market_data: Arc::new(RwLock::new(HashMap::new())),
            level2_books: Arc::new(RwLock::new(HashMap::new())),
            metrics_collector: metrics,
            routing_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    #[instrument(skip(self))]
    pub async fn route_order(
        &self,
        order: &Order,
        constraints: &RoutingConstraints,
    ) -> Result<RoutingDecision> {
        debug!("Routing order: {} for {}", order.id, order.symbol);
        
        let start_time = Instant::now();
        
        // Get available venues for this order
        let available_venues = self.get_available_venues(order).await?;
        
        if available_venues.is_empty() {
            return Err(AlgoVedaError::Trading(
                "No venues available for order".to_string()
            ));
        }
        
        // Score each venue
        let mut venue_scores = Vec::new();
        
        for venue in available_venues {
            match self.score_venue(order, &venue, constraints).await {
                Ok(score) => venue_scores.push((venue, score)),
                Err(e) => {
                    warn!("Failed to score venue {}: {}", venue, e);
                }
            }
        }
        
        if venue_scores.is_empty() {
            return Err(AlgoVedaError::Trading(
                "No suitable venues found".to_string()
            ));
        }
        
        // Sort by score (higher is better)
        venue_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let best_venue = &venue_scores[0].0;
        let best_score = venue_scores[0].1;
        
        // Create routing decision
        let decision = self.create_routing_decision(order, best_venue, best_score).await?;
        
        // Store routing decision
        let mut history = self.routing_history.lock().await;
        history.push(decision.clone());
        
        // Keep only recent history
        if history.len() > 10000 {
            history.drain(0..5000);
        }
        
        // Update metrics
        let routing_time = start_time.elapsed();
        self.metrics_collector.record_histogram(
            "routing_decision_time_ms",
            routing_time.as_millis() as f64,
            &[("venue", best_venue)],
        );
        
        self.metrics_collector.increment_counter(
            "orders_routed",
            &[("venue", best_venue)],
        );
        
        debug!("Routed order {} to venue {} with score {}", 
               order.id, best_venue, best_score);
        
        Ok(decision)
    }

    async fn get_available_venues(&self, order: &Order) -> Result<Vec<String>> {
        let mut venues = Vec::new();
        
        // Get enabled venues from config
        let enabled_venues = self.config.get_enabled_venues();
        
        for venue_config in enabled_venues {
            // Check if venue supports this instrument
            if venue_config.supported_instruments.contains(&self.get_instrument_type(order)) {
                // Check order size limits
                if order.quantity >= venue_config.min_order_size && 
                   order.quantity <= venue_config.max_order_size {
                    venues.push(venue_config.name.clone());
                }
            }
        }
        
        Ok(venues)
    }

    async fn score_venue(
        &self,
        order: &Order,
        venue: &str,
        constraints: &RoutingConstraints,
    ) -> Result<f64> {
        // Get venue metrics
        let venue_metrics = self.get_venue_metrics(venue)?;
        
        // Check hard constraints first
        if venue_metrics.avg_latency_ms > constraints.max_latency_ms as f64 {
            return Ok(0.0); // Venue fails latency constraint
        }
        
        if venue_metrics.fill_rate < constraints.min_fill_probability {
            return Ok(0.0); // Venue fails fill probability constraint
        }
        
        if constraints.blacklisted_venues.contains(&venue.to_string()) {
            return Ok(0.0); // Venue is blacklisted
        }
        
        // Calculate composite score
        let mut score = 0.0;
        
        // Latency component (lower is better)
        let latency_score = 1.0 - (venue_metrics.avg_latency_ms / 1000.0).min(1.0);
        score += latency_score * 0.3;
        
        // Fill rate component (higher is better)
        score += venue_metrics.fill_rate * 0.3;
        
        // Cost component (lower spread is better)
        let cost_score = 1.0 - (venue_metrics.avg_spread_bps / 100.0).min(1.0);
        score += cost_score * 0.2;
        
        // Market depth component (higher is better for large orders)
        let depth_ratio = (venue_metrics.market_depth / order.quantity).min(Decimal::new(10, 0));
        let depth_score = depth_ratio.to_f64().unwrap() / 10.0;
        score += depth_score * 0.1;
        
        // Uptime component (higher is better)
        score += (venue_metrics.uptime_percentage / 100.0) * 0.1;
        
        // Preferred venue bonus
        if constraints.preferred_venues.contains(&venue.to_string()) {
            score *= 1.2;
        }
        
        // Dark pool preference for large orders
        if self.is_dark_pool(venue) && order.quantity > self.get_large_order_threshold() {
            score *= 1.0 + constraints.dark_pool_preference;
        }
        
        Ok(score.max(0.0).min(1.0))
    }

    async fn create_routing_decision(
        &self,
        order: &Order,
        venue: &str,
        score: f64,
    ) -> Result<RoutingDecision> {
        let venue_metrics = self.get_venue_metrics(venue)?;
        
        // Estimate costs
        let expected_cost = self.estimate_execution_cost(order, venue).await?;
        
        // Estimate market impact
        let market_impact_bps = self.estimate_market_impact(order, venue).await?;
        
        Ok(RoutingDecision {
            venue: venue.to_string(),
            expected_cost,
            expected_latency_ms: venue_metrics.avg_latency_ms as u64,
            fill_probability: venue_metrics.fill_rate,
            market_impact_bps,
            reasoning: format!(
                "Selected based on score {:.3} considering latency ({:.1}ms), fill rate ({:.1}%), and cost",
                score, venue_metrics.avg_latency_ms, venue_metrics.fill_rate * 100.0
            ),
        })
    }

    fn get_venue_metrics(&self, venue: &str) -> Result<VenueMetrics> {
        let metrics = self.venue_metrics.read().unwrap();
        
        metrics.get(venue)
            .cloned()
            .or_else(|| {
                // Return default metrics if none available
                Some(VenueMetrics {
                    venue: venue.to_string(),
                    avg_latency_ms: 100.0,
                    fill_rate: 0.8,
                    avg_spread_bps: 5.0,
                    market_depth: Decimal::new(10000, 0),
                    recent_volume: Decimal::new(1000000, 0),
                    uptime_percentage: 99.0,
                    last_updated: Instant::now(),
                })
            })
            .ok_or_else(|| AlgoVedaError::Trading(
                format!("No metrics available for venue: {}", venue)
            ))
    }

    pub async fn update_venue_metrics(&self, venue: &str, metrics: VenueMetrics) {
        let mut venue_metrics = self.venue_metrics.write().unwrap();
        venue_metrics.insert(venue.to_string(), metrics);
        
        // Update monitoring metrics
        self.metrics_collector.set_gauge(
            "venue_latency_ms",
            metrics.avg_latency_ms,
            &[("venue", venue)],
        );
        
        self.metrics_collector.set_gauge(
            "venue_fill_rate",
            metrics.fill_rate,
            &[("venue", venue)],
        );
        
        self.metrics_collector.set_gauge(
            "venue_spread_bps",
            metrics.avg_spread_bps,
            &[("venue", venue)],
        );
    }

    pub async fn update_market_data(&self, symbol: &str, data: MarketData) {
        let mut market_data = self.market_data.write().unwrap();
        market_data.insert(symbol.to_string(), data);
    }

    pub async fn update_level2_book(&self, symbol: &str, book: Level2Book) {
        let mut books = self.level2_books.write().unwrap();
        books.insert(symbol.to_string(), book);
    }

    async fn estimate_execution_cost(&self, order: &Order, venue: &str) -> Result<Decimal> {
        // Get venue config for fee calculation
        let venue_config = self.config.get_venue(venue)
            .ok_or_else(|| AlgoVedaError::Trading(format!("Unknown venue: {}", venue)))?;
        
        let order_value = order.quantity * order.price.unwrap_or(Decimal::new(100, 0));
        
        // Calculate trading fees
        let fee_bps = match order.order_type {
            OrderType::Market => venue_config.trading_fees.taker_fee_bps,
            OrderType::Limit => venue_config.trading_fees.maker_fee_bps,
            _ => venue_config.trading_fees.taker_fee_bps,
        };
        
        let trading_fee = order_value * Decimal::new(fee_bps as i64, 4); // bps to decimal
        
        // Add estimated spread cost for market orders
        let spread_cost = if matches!(order.order_type, OrderType::Market) {
            let venue_metrics = self.get_venue_metrics(venue)?;
            order_value * Decimal::new(venue_metrics.avg_spread_bps as i64, 4) / Decimal::new(2, 0) // half spread
        } else {
            Decimal::ZERO
        };
        
        Ok(trading_fee + spread_cost)
    }

    async fn estimate_market_impact(&self, order: &Order, venue: &str) -> Result<f64> {
        // Get market data for the symbol
        let market_data = {
            let data = self.market_data.read().unwrap();
            data.get(&order.symbol).cloned()
        };
        
        let level2_book = {
            let books = self.level2_books.read().unwrap();
            books.get(&order.symbol).cloned()
        };
        
        // Simple market impact model based on order size vs market depth
        if let (Some(market_data), Some(book)) = (market_data, level2_book) {
            let relevant_side = match order.side {
                OrderSide::Buy => &book.asks,
                OrderSide::Sell => &book.bids,
            };
            
            if !relevant_side.is_empty() {
                let total_depth: Decimal = relevant_side.iter()
                    .take(5) // Top 5 levels
                    .map(|level| level.quantity)
                    .sum();
                
                let impact_ratio = order.quantity.to_f64().unwrap() / 
                                 total_depth.to_f64().unwrap();
                
                // Square root model for market impact
                return Ok((impact_ratio.sqrt() * 10.0).min(50.0)); // Cap at 50bps
            }
        }
        
        // Default impact estimate
        Ok(5.0) // 5 basis points
    }

    fn get_instrument_type(&self, order: &Order) -> String {
        // Simple classification - in practice would be more sophisticated
        if order.symbol.len() <= 4 {
            "EQUITY".to_string()
        } else if order.symbol.contains("USD") || order.symbol.contains("BTC") {
            "CRYPTO".to_string()
        } else {
            "OTHER".to_string()
        }
    }

    fn is_dark_pool(&self, venue: &str) -> bool {
        venue.to_lowercase().contains("dark") || 
        venue.to_lowercase().contains("hidden")
    }

    fn get_large_order_threshold(&self) -> Decimal {
        Decimal::new(10000, 0) // $10,000 threshold
    }

    pub async fn get_routing_statistics(&self) -> HashMap<String, f64> {
        let history = self.routing_history.lock().await;
        let mut stats = HashMap::new();
        
        if history.is_empty() {
            return stats;
        }
        
        // Calculate venue distribution
        let mut venue_counts = HashMap::new();
        for decision in history.iter() {
            *venue_counts.entry(decision.venue.clone()).or_insert(0) += 1;
        }
        
        let total_decisions = history.len() as f64;
        for (venue, count) in venue_counts {
            stats.insert(
                format!("venue_{}_percentage", venue),
                (count as f64 / total_decisions) * 100.0
            );
        }
        
        // Calculate average metrics
        if !history.is_empty() {
            let avg_latency = history.iter()
                .map(|d| d.expected_latency_ms as f64)
                .sum::<f64>() / total_decisions;
            
            let avg_fill_probability = history.iter()
                .map(|d| d.fill_probability)
                .sum::<f64>() / total_decisions;
            
            let avg_market_impact = history.iter()
                .map(|d| d.market_impact_bps)
                .sum::<f64>() / total_decisions;
            
            stats.insert("avg_expected_latency_ms".to_string(), avg_latency);
            stats.insert("avg_fill_probability".to_string(), avg_fill_probability);
            stats.insert("avg_market_impact_bps".to_string(), avg_market_impact);
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TradingConfig;

    #[tokio::test]
    async fn test_smart_order_router_creation() {
        let config = Arc::new(TradingConfig::default());
        let metrics = Arc::new(MetricsCollector::new_test());
        
        let router = SmartOrderRouter::new(config, metrics).await;
        assert!(router.is_ok());
    }

    #[tokio::test]
    async fn test_venue_scoring() {
        let config = Arc::new(TradingConfig::default());
        let metrics = Arc::new(MetricsCollector::new_test());
        let router = SmartOrderRouter::new(config, metrics).await.unwrap();
        
        // Add test venue metrics
        let test_metrics = VenueMetrics {
            venue: "TEST_VENUE".to_string(),
            avg_latency_ms: 50.0,
            fill_rate: 0.95,
            avg_spread_bps: 3.0,
            market_depth: Decimal::new(100000, 0),
            recent_volume: Decimal::new(5000000, 0),
            uptime_percentage: 99.5,
            last_updated: Instant::now(),
        };
        
        router.update_venue_metrics("TEST_VENUE", test_metrics).await;
        
        let order = Order {
            id: Uuid::new_v4(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: Decimal::new(1000, 0),
            price: Some(Decimal::new(150, 0)),
            order_type: OrderType::Limit,
            status: crate::trading::OrderStatus::New,
            created_at: Instant::now(),
            updated_at: Instant::now(),
        };
        
        let constraints = RoutingConstraints {
            max_latency_ms: 100,
            max_cost_bps: 10,
            min_fill_probability: 0.8,
            preferred_venues: vec![],
            blacklisted_venues: vec![],
            dark_pool_preference: 0.1,
        };
        
        let score = router.score_venue(&order, "TEST_VENUE", &constraints).await;
        assert!(score.is_ok());
        assert!(score.unwrap() > 0.5); // Should be a good score
    }
}
