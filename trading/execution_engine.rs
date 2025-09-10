/*!
 * Execution Engine for AlgoVeda Trading Platform
 * 
 * Handles order execution, routing, and fill management with
 * ultra-low latency and sophisticated execution algorithms.
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::Mutex;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{Duration, Instant};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, instrument};

use crate::trading::{Order, OrderEvent, Fill, TradingError, TradingResult, MarketData};
use crate::config::TradingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub id: Uuid,
    pub order_id: Uuid,
    pub execution_type: ExecutionType,
    pub status: ExecutionStatus,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub execution_id: String,
    pub latency_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionType {
    New,
    PartialFill,
    Fill,
    DoneForDay,
    Cancelled,
    Replace,
    PendingCancel,
    Stopped,
    Rejected,
    Suspended,
    PendingNew,
    Calculated,
    Expired,
    Restated,
    PendingReplace,
    Trade,
    TradeCorrect,
    TradeCancel,
    OrderStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    New,
    PartiallyFilled,
    Filled,
    DoneForDay,
    Cancelled,
    Replaced,
    PendingCancel,
    Stopped,
    Rejected,
    Suspended,
    PendingNew,
    Calculated,
    Expired,
    AcceptedForBidding,
    PendingReplace,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub id: Uuid,
    pub order_id: Uuid,
    pub symbol: String,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub execution_id: String,
    pub commission: f64,
    pub fees: f64,
    pub counterparty: Option<String>,
}

/// Execution Engine coordinates order execution across multiple venues
#[derive(Debug)]
pub struct ExecutionEngine {
    config: Arc<TradingConfig>,
    order_queue: Arc<Mutex<VecDeque<Order>>>,
    execution_reports: Arc<RwLock<HashMap<Uuid, ExecutionReport>>>,
    trades: Arc<RwLock<HashMap<Uuid, Trade>>>,
    venue_connections: Arc<RwLock<HashMap<String, VenueConnection>>>,
    execution_latencies: Arc<Mutex<VecDeque<u64>>>,
    total_trades_executed: Arc<Mutex<u64>>,
    order_events_tx: mpsc::UnboundedSender<OrderEvent>,
    execution_events_tx: mpsc::UnboundedSender<ExecutionEvent>,
    market_data: Arc<RwLock<HashMap<String, MarketData>>>,
    is_running: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    OrderReceived(Order),
    OrderRouted(Order, String), // Order and venue
    OrderExecuted(ExecutionReport),
    TradeGenerated(Trade),
    ExecutionFailed(Order, String),
}

#[derive(Debug, Clone)]
struct VenueConnection {
    name: String,
    latency_micros: u64,
    fill_rate: f64,
    available: bool,
    last_ping: DateTime<Utc>,
}

impl ExecutionEngine {
    pub fn new(
        config: Arc<TradingConfig>,
        order_events_tx: mpsc::UnboundedSender<OrderEvent>,
        execution_events_tx: mpsc::UnboundedSender<ExecutionEvent>,
    ) -> Self {
        Self {
            config,
            order_queue: Arc::new(Mutex::new(VecDeque::new())),
            execution_reports: Arc::new(RwLock::new(HashMap::new())),
            trades: Arc::new(RwLock::new(HashMap::new())),
            venue_connections: Arc::new(RwLock::new(HashMap::new())),
            execution_latencies: Arc::new(Mutex::new(VecDeque::new())),
            total_trades_executed: Arc::new(Mutex::new(0)),
            order_events_tx,
            execution_events_tx,
            market_data: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start_execution(&self) -> TradingResult<()> {
        {
            let mut running = self.is_running.write().await;
            *running = true;
        }

        info!("Execution engine started");
        
        // Start execution loop
        let engine = self.clone();
        tokio::spawn(async move {
            engine.execution_loop().await;
        });

        // Start venue monitoring
        let engine = self.clone();
        tokio::spawn(async move {
            engine.venue_monitoring_loop().await;
        });

        Ok(())
    }

    async fn execution_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_millis(1));
        
        loop {
            interval.tick().await;
            
            let is_running = {
                let running = self.is_running.read().await;
                *running
            };
            
            if !is_running {
                break;
            }

            // Process pending orders
            if let Some(order) = self.dequeue_order() {
                if let Err(e) = self.process_order(order).await {
                    error!("Failed to process order: {}", e);
                }
            }
        }
        
        info!("Execution loop stopped");
    }

    async fn venue_monitoring_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        
        loop {
            interval.tick().await;
            
            let is_running = {
                let running = self.is_running.read().await;
                *running
            };
            
            if !is_running {
                break;
            }

            self.monitor_venues().await;
        }
    }

    #[instrument(skip(self))]
    pub async fn submit_order(&self, order: Order) -> TradingResult<()> {
        info!("Received order for execution: {}", order.id);
        
        // Add to execution queue
        {
            let mut queue = self.order_queue.lock();
            queue.push_back(order.clone());
        }

        // Send event
        let _ = self.execution_events_tx.send(ExecutionEvent::OrderReceived(order));
        
        Ok(())
    }

    fn dequeue_order(&self) -> Option<Order> {
        let mut queue = self.order_queue.lock();
        queue.pop_front()
    }

    async fn process_order(&self, order: Order) -> TradingResult<()> {
        let start_time = Instant::now();
        
        // Route order to best venue
        let venue = self.route_order(&order).await?;
        
        // Send order to venue
        let execution_result = self.execute_on_venue(&order, &venue).await?;
        
        // Record latency
        let latency_micros = start_time.elapsed().as_micros() as u64;
        self.record_latency(latency_micros);
        
        // Process execution result
        self.process_execution_result(order, execution_result, venue, latency_micros).await?;
        
        Ok(())
    }

    async fn route_order(&self, order: &Order) -> TradingResult<String> {
        let venues = self.venue_connections.read().await;
        
        // Simple routing: pick first available venue
        // In production, this would use sophisticated routing algorithms
        let best_venue = venues
            .values()
            .filter(|v| v.available)
            .min_by_key(|v| v.latency_micros)
            .map(|v| v.name.clone())
            .ok_or_else(|| TradingError::ExecutionFailed("No available venues".to_string()))?;

        // Send routing event
        let _ = self.execution_events_tx.send(ExecutionEvent::OrderRouted(order.clone(), best_venue.clone()));
        
        Ok(best_venue)
    }

    async fn execute_on_venue(&self, order: &Order, venue: &str) -> TradingResult<ExecutionResult> {
        // Simulate venue execution
        // In production, this would send orders to actual venues
        
        let market_data = {
            let data = self.market_data.read().await;
            data.get(&order.symbol).cloned()
        };

        let execution_price = match &order.order_type {
            crate::trading::OrderType::Market => {
                // Use current market price
                market_data
                    .map(|md| md.price)
                    .unwrap_or(100.0) // Default price for simulation
            }
            crate::trading::OrderType::Limit => {
                order.price.unwrap_or(100.0)
            }
            _ => order.price.unwrap_or(100.0),
        };

        // Simulate execution with some randomness
        let fill_probability = 0.95; // 95% fill probability
        let random_value: f64 = rand::random();
        
        if random_value < fill_probability {
            Ok(ExecutionResult::Filled {
                quantity: order.quantity,
                price: execution_price,
                venue: venue.to_string(),
                execution_id: format!("exec_{}", Uuid::new_v4()),
                commission: order.quantity * self.config.default_commission,
                fees: 0.0,
            })
        } else {
            Ok(ExecutionResult::Rejected {
                reason: "Market conditions".to_string(),
            })
        }
    }

    async fn process_execution_result(
        &self,
        order: Order,
        result: ExecutionResult,
        venue: String,
        latency_micros: u64,
    ) -> TradingResult<()> {
        match result {
            ExecutionResult::Filled { quantity, price, venue, execution_id, commission, fees } => {
                // Create trade
                let trade = Trade {
                    id: Uuid::new_v4(),
                    order_id: order.id,
                    symbol: order.symbol.clone(),
                    quantity,
                    price,
                    timestamp: Utc::now(),
                    venue: venue.clone(),
                    execution_id: execution_id.clone(),
                    commission,
                    fees,
                    counterparty: None,
                };

                // Create execution report
                let execution_report = ExecutionReport {
                    id: Uuid::new_v4(),
                    order_id: order.id,
                    execution_type: ExecutionType::Fill,
                    status: ExecutionStatus::Filled,
                    quantity,
                    price,
                    timestamp: Utc::now(),
                    venue: venue.clone(),
                    execution_id,
                    latency_micros,
                };

                // Store trade and execution report
                {
                    let mut trades = self.trades.write().await;
                    trades.insert(trade.id, trade.clone());
                    
                    let mut reports = self.execution_reports.write().await;
                    reports.insert(execution_report.id, execution_report.clone());
                }

                // Update trade count
                {
                    let mut count = self.total_trades_executed.lock();
                    *count += 1;
                }

                // Create fill
                let fill = Fill {
                    id: Uuid::new_v4(),
                    order_id: order.id,
                    quantity,
                    price,
                    timestamp: Utc::now(),
                    venue,
                    execution_id: execution_report.execution_id,
                    commission,
                    fees,
                };

                // Send events
                let _ = self.order_events_tx.send(OrderEvent::OrderFilled(order, fill));
                let _ = self.execution_events_tx.send(ExecutionEvent::OrderExecuted(execution_report));
                let _ = self.execution_events_tx.send(ExecutionEvent::TradeGenerated(trade));
                
                info!("Order executed successfully");
            }
            ExecutionResult::Rejected { reason } => {
                let _ = self.order_events_tx.send(OrderEvent::OrderRejected(order.clone(), reason.clone()));
                let _ = self.execution_events_tx.send(ExecutionEvent::ExecutionFailed(order, reason));
                
                warn!("Order execution rejected");
            }
        }

        Ok(())
    }

    async fn monitor_venues(&self) {
        // Update venue health and latency
        let mut venues = self.venue_connections.write().await;
        
        for venue in venues.values_mut() {
            // Simulate venue health check
            venue.available = true; // In production, would ping actual venues
            venue.last_ping = Utc::now();
            
            // Update latency (simulated)
            venue.latency_micros = 50 + (rand::random::<u64>() % 100); // 50-150 microseconds
        }
    }

    fn record_latency(&self, latency_micros: u64) {
        let mut latencies = self.execution_latencies.lock();
        latencies.push_back(latency_micros);
        
        // Keep only last 1000 latency measurements
        if latencies.len() > 1000 {
            latencies.pop_front();
        }
    }

    pub async fn add_venue(&self, name: String) {
        let venue = VenueConnection {
            name: name.clone(),
            latency_micros: 100,
            fill_rate: 0.95,
            available: true,
            last_ping: Utc::now(),
        };

        let mut venues = self.venue_connections.write().await;
        venues.insert(name, venue);
    }

    pub async fn get_average_latency(&self) -> Duration {
        let latencies = self.execution_latencies.lock();
        if latencies.is_empty() {
            return Duration::from_micros(0);
        }

        let sum: u64 = latencies.iter().sum();
        let avg = sum / latencies.len() as u64;
        Duration::from_micros(avg)
    }

    pub async fn get_trades_executed(&self) -> u64 {
        let count = self.total_trades_executed.lock();
        *count
    }

    pub async fn complete_pending_orders(&self) -> TradingResult<()> {
        let pending_count = {
            let queue = self.order_queue.lock();
            queue.len()
        };
        
        info!("Completing {} pending orders", pending_count);
        
        // Process remaining orders
        while let Some(order) = self.dequeue_order() {
            if let Err(e) = self.process_order(order).await {
                error!("Failed to complete pending order: {}", e);
            }
        }
        
        Ok(())
    }

    pub async fn health_check(&self) -> bool {
        let venues = self.venue_connections.read().await;
        let available_venues = venues.values().filter(|v| v.available).count();
        
        available_venues > 0
    }

    pub async fn set_market_data_engine(&self, _market_data_engine: Arc<dyn Send + Sync>) -> TradingResult<()> {
        // Implementation would connect to market data engine
        Ok(())
    }
}

#[derive(Debug)]
enum ExecutionResult {
    Filled {
        quantity: f64,
        price: f64,
        venue: String,
        execution_id: String,
        commission: f64,
        fees: f64,
    },
    Rejected {
        reason: String,
    },
}

// Implement Clone for ExecutionEngine to allow spawning tasks
impl Clone for ExecutionEngine {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            order_queue: Arc::clone(&self.order_queue),
            execution_reports: Arc::clone(&self.execution_reports),
            trades: Arc::clone(&self.trades),
            venue_connections: Arc::clone(&self.venue_connections),
            execution_latencies: Arc::clone(&self.execution_latencies),
            total_trades_executed: Arc::clone(&self.total_trades_executed),
            order_events_tx: self.order_events_tx.clone(),
            execution_events_tx: self.execution_events_tx.clone(),
            market_data: Arc::clone(&self.market_data),
            is_running: Arc::clone(&self.is_running),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_execution_engine() {
        let config = Arc::new(TradingConfig::default());
        let (order_tx, _order_rx) = mpsc::unbounded_channel();
        let (exec_tx, _exec_rx) = mpsc::unbounded_channel();
        
        let engine = ExecutionEngine::new(config, order_tx, exec_tx);
        engine.add_venue("TEST_VENUE".to_string()).await;
        
        let mut order = Order::new();
        order.symbol = "AAPL".to_string();
        order.quantity = 100.0;
        order.order_type = crate::trading::OrderType::Market;
        
        let result = engine.submit_order(order).await;
        assert!(result.is_ok());
        
        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
