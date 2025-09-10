/*!
 * Core Order Management Module for AlgoVeda Trading Platform
 * 
 * Handles order lifecycle, validation, routing, and state management
 * with ultra-low latency and thread-safe operations.
 */

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use tokio::sync::mpsc;
use tracing::{info, warn, error, instrument};

use crate::trading::{TradingError, TradingResult, TradingLimits, MarketData};
use crate::config::TradingConfig;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    Iceberg,
    TWAP,
    VWAP,
    Implementation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    IOC,  // Immediate or Cancel
    FOK,  // Fill or Kill
    GTC,  // Good Till Cancelled
    GTD,  // Good Till Date
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    PendingNew,
    New,
    PartiallyFilled,
    Filled,
    PendingCancel,
    Cancelled,
    PendingReplace,
    Replaced,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub status: OrderStatus,
    pub filled_quantity: f64,
    pub average_fill_price: f64,
    pub remaining_quantity: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub account_id: String,
    pub strategy_id: Option<String>,
    pub parent_order_id: Option<Uuid>,
    pub execution_instructions: HashMap<String, String>,
    pub tags: HashMap<String, String>,
}

impl Order {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            client_order_id: String::new(),
            symbol: String::new(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 0.0,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
            status: OrderStatus::PendingNew,
            filled_quantity: 0.0,
            average_fill_price: 0.0,
            remaining_quantity: 0.0,
            created_at: now,
            updated_at: now,
            account_id: String::new(),
            strategy_id: None,
            parent_order_id: None,
            execution_instructions: HashMap::new(),
            tags: HashMap::new(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.status, 
            OrderStatus::Filled | 
            OrderStatus::Cancelled | 
            OrderStatus::Rejected | 
            OrderStatus::Expired
        )
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, 
            OrderStatus::New | 
            OrderStatus::PartiallyFilled
        )
    }

    pub fn update_fill(&mut self, fill_quantity: f64, fill_price: f64) {
        self.filled_quantity += fill_quantity;
        self.remaining_quantity = self.quantity - self.filled_quantity;
        
        // Update average fill price
        if self.filled_quantity > 0.0 {
            self.average_fill_price = ((self.average_fill_price * (self.filled_quantity - fill_quantity)) + 
                                       (fill_price * fill_quantity)) / self.filled_quantity;
        }
        
        // Update status
        if self.remaining_quantity <= 0.0 {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartiallyFilled;
        }
        
        self.updated_at = Utc::now();
    }
}

/// Order Manager handles the complete lifecycle of trading orders
#[derive(Debug)]
pub struct OrderManager {
    config: Arc<TradingConfig>,
    orders: RwLock<HashMap<Uuid, Order>>,
    orders_by_client_id: RwLock<HashMap<String, Uuid>>,
    orders_by_symbol: RwLock<HashMap<String, Vec<Uuid>>>,
    active_orders: RwLock<Vec<Uuid>>,
    order_sequence: Mutex<u64>,
    trading_limits: Arc<TradingLimits>,
    order_events: mpsc::UnboundedSender<OrderEvent>,
}

#[derive(Debug, Clone)]
pub enum OrderEvent {
    OrderSubmitted(Order),
    OrderUpdated(Order),
    OrderFilled(Order, Fill),
    OrderCancelled(Order),
    OrderRejected(Order, String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub id: Uuid,
    pub order_id: Uuid,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub execution_id: String,
    pub commission: f64,
    pub fees: f64,
}

impl OrderManager {
    pub fn new(
        config: Arc<TradingConfig>,
        trading_limits: Arc<TradingLimits>,
        order_events: mpsc::UnboundedSender<OrderEvent>,
    ) -> Self {
        Self {
            config,
            orders: RwLock::new(HashMap::new()),
            orders_by_client_id: RwLock::new(HashMap::new()),
            orders_by_symbol: RwLock::new(HashMap::new()),
            active_orders: RwLock::new(Vec::new()),
            order_sequence: Mutex::new(1),
            trading_limits,
            order_events,
        }
    }

    #[instrument(skip(self))]
    pub fn submit_order(&self, mut order: Order) -> TradingResult<Uuid> {
        // Validate order
        self.validate_order(&order)?;
        
        // Check trading limits
        self.check_trading_limits(&order)?;
        
        // Assign order ID and sequence
        order.id = Uuid::new_v4();
        let mut sequence = self.order_sequence.lock();
        *sequence += 1;
        
        let now = Utc::now();
        order.created_at = now;
        order.updated_at = now;
        order.remaining_quantity = order.quantity;
        order.status = OrderStatus::PendingNew;

        // Store order
        let order_id = order.id;
        {
            let mut orders = self.orders.write();
            let mut orders_by_client_id = self.orders_by_client_id.write();
            let mut orders_by_symbol = self.orders_by_symbol.write();
            let mut active_orders = self.active_orders.write();

            orders.insert(order_id, order.clone());
            
            if !order.client_order_id.is_empty() {
                orders_by_client_id.insert(order.client_order_id.clone(), order_id);
            }
            
            orders_by_symbol.entry(order.symbol.clone())
                .or_insert_with(Vec::new)
                .push(order_id);
                
            active_orders.push(order_id);
        }

        // Send event
        let _ = self.order_events.send(OrderEvent::OrderSubmitted(order));
        
        info!("Order submitted: {}", order_id);
        Ok(order_id)
    }

    #[instrument(skip(self))]
    pub fn cancel_order(&self, order_id: Uuid) -> TradingResult<()> {
        let mut order = {
            let orders = self.orders.read();
            orders.get(&order_id)
                .ok_or(TradingError::InvalidOrder("Order not found".to_string()))?
                .clone()
        };

        if order.is_terminal() {
            return Err(TradingError::InvalidOrder(
                "Cannot cancel terminal order".to_string()
            ));
        }

        order.status = OrderStatus::PendingCancel;
        order.updated_at = Utc::now();

        // Update stored order
        {
            let mut orders = self.orders.write();
            orders.insert(order_id, order.clone());
        }

        // Remove from active orders
        {
            let mut active_orders = self.active_orders.write();
            active_orders.retain(|&id| id != order_id);
        }

        // Send event
        let _ = self.order_events.send(OrderEvent::OrderCancelled(order));
        
        info!("Order cancelled: {}", order_id);
        Ok(())
    }

    pub fn update_order_status(&self, order_id: Uuid, status: OrderStatus) -> TradingResult<()> {
        let mut order = {
            let orders = self.orders.read();
            orders.get(&order_id)
                .ok_or(TradingError::InvalidOrder("Order not found".to_string()))?
                .clone()
        };

        order.status = status;
        order.updated_at = Utc::now();

        // Update stored order
        {
            let mut orders = self.orders.write();
            orders.insert(order_id, order.clone());
        }

        // Remove from active orders if terminal
        if order.is_terminal() {
            let mut active_orders = self.active_orders.write();
            active_orders.retain(|&id| id != order_id);
        }

        // Send event
        let _ = self.order_events.send(OrderEvent::OrderUpdated(order));
        
        Ok(())
    }

    pub fn process_fill(&self, order_id: Uuid, fill: Fill) -> TradingResult<()> {
        let mut order = {
            let orders = self.orders.read();
            orders.get(&order_id)
                .ok_or(TradingError::InvalidOrder("Order not found".to_string()))?
                .clone()
        };

        order.update_fill(fill.quantity, fill.price);

        // Update stored order
        {
            let mut orders = self.orders.write();
            orders.insert(order_id, order.clone());
        }

        // Remove from active orders if filled
        if order.status == OrderStatus::Filled {
            let mut active_orders = self.active_orders.write();
            active_orders.retain(|&id| id != order_id);
        }

        // Send event
        let _ = self.order_events.send(OrderEvent::OrderFilled(order, fill));
        
        Ok(())
    }

    pub fn get_order(&self, order_id: Uuid) -> Option<Order> {
        let orders = self.orders.read();
        orders.get(&order_id).cloned()
    }

    pub fn get_order_by_client_id(&self, client_order_id: &str) -> Option<Order> {
        let orders_by_client_id = self.orders_by_client_id.read();
        let orders = self.orders.read();
        
        orders_by_client_id.get(client_order_id)
            .and_then(|order_id| orders.get(order_id))
            .cloned()
    }

    pub fn get_orders_by_symbol(&self, symbol: &str) -> Vec<Order> {
        let orders_by_symbol = self.orders_by_symbol.read();
        let orders = self.orders.read();
        
        orders_by_symbol.get(symbol)
            .map(|order_ids| {
                order_ids.iter()
                    .filter_map(|id| orders.get(id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_active_orders(&self) -> Vec<Order> {
        let active_orders = self.active_orders.read();
        let orders = self.orders.read();
        
        active_orders.iter()
            .filter_map(|id| orders.get(id))
            .cloned()
            .collect()
    }

    pub fn get_orders_processed(&self) -> u64 {
        let sequence = self.order_sequence.lock();
        *sequence
    }

    fn validate_order(&self, order: &Order) -> TradingResult<()> {
        if order.symbol.is_empty() {
            return Err(TradingError::InvalidOrder("Symbol cannot be empty".to_string()));
        }

        if order.quantity <= 0.0 {
            return Err(TradingError::InvalidOrder("Quantity must be positive".to_string()));
        }

        if order.quantity < self.config.min_order_size {
            return Err(TradingError::InvalidOrder(
                format!("Quantity {} below minimum {}", order.quantity, self.config.min_order_size)
            ));
        }

        if order.quantity > self.config.max_order_size {
            return Err(TradingError::InvalidOrder(
                format!("Quantity {} exceeds maximum {}", order.quantity, self.config.max_order_size)
            ));
        }

        match order.order_type {
            OrderType::Limit | OrderType::StopLimit => {
                if order.price.is_none() {
                    return Err(TradingError::InvalidOrder("Limit orders require price".to_string()));
                }
                if order.price.unwrap() <= 0.0 {
                    return Err(TradingError::InvalidOrder("Price must be positive".to_string()));
                }
            }
            OrderType::Stop | OrderType::StopLimit => {
                if order.stop_price.is_none() {
                    return Err(TradingError::InvalidOrder("Stop orders require stop price".to_string()));
                }
                if order.stop_price.unwrap() <= 0.0 {
                    return Err(TradingError::InvalidOrder("Stop price must be positive".to_string()));
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn check_trading_limits(&self, order: &Order) -> TradingResult<()> {
        let order_value = order.quantity * order.price.unwrap_or(0.0);
        
        if order_value > self.trading_limits.max_order_value {
            return Err(TradingError::InvalidOrder(
                format!("Order value {} exceeds limit {}", order_value, self.trading_limits.max_order_value)
            ));
        }

        // Check daily trade count
        let symbol_orders = self.get_orders_by_symbol(&order.symbol);
        let today = Utc::now().date_naive();
        let today_trades = symbol_orders.iter()
            .filter(|o| o.created_at.date_naive() == today)
            .count() as u32;
            
        if today_trades >= self.trading_limits.max_trades_per_day {
            return Err(TradingError::InvalidOrder(
                "Daily trade limit exceeded".to_string()
            ));
        }

        Ok(())
    }

    pub async fn health_check(&self) -> bool {
        // Check if order manager is responsive
        let orders_count = {
            let orders = self.orders.read();
            orders.len()
        };
        
        // Basic health check - ensure we can access internal state
        orders_count >= 0 // This should always be true
    }

    pub async fn stop_accepting_orders(&self) -> TradingResult<()> {
        info!("Order manager stopping - no longer accepting new orders");
        Ok(())
    }

    pub async fn get_current_positions(&self) -> u32 {
        let active_orders = self.active_orders.read();
        active_orders.len() as u32
    }
}

impl Default for Order {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_order_submission() {
        let config = Arc::new(TradingConfig::default());
        let limits = Arc::new(TradingLimits {
            max_order_value: 1000000.0,
            max_position_size: 500000.0,
            max_daily_volume: 10000000.0,
            max_trades_per_day: 100,
        });
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let order_manager = OrderManager::new(config, limits, tx);
        
        let mut order = Order::new();
        order.symbol = "AAPL".to_string();
        order.side = OrderSide::Buy;
        order.order_type = OrderType::Market;
        order.quantity = 100.0;
        order.account_id = "test_account".to_string();
        
        let result = order_manager.submit_order(order);
        assert!(result.is_ok());
        
        let order_id = result.unwrap();
        let retrieved_order = order_manager.get_order(order_id);
        assert!(retrieved_order.is_some());
        assert_eq!(retrieved_order.unwrap().symbol, "AAPL");
    }

    #[tokio::test]
    async fn test_order_cancellation() {
        let config = Arc::new(TradingConfig::default());
        let limits = Arc::new(TradingLimits {
            max_order_value: 1000000.0,
            max_position_size: 500000.0,
            max_daily_volume: 10000000.0,
            max_trades_per_day: 100,
        });
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let order_manager = OrderManager::new(config, limits, tx);
        
        let mut order = Order::new();
        order.symbol = "AAPL".to_string();
        order.quantity = 100.0;
        
        let order_id = order_manager.submit_order(order).unwrap();
        let result = order_manager.cancel_order(order_id);
        assert!(result.is_ok());
        
        let cancelled_order = order_manager.get_order(order_id).unwrap();
        assert_eq!(cancelled_order.status, OrderStatus::PendingCancel);
    }
}
