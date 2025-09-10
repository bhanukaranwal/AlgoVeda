/*!
 * Ultra-Fast Backtester in Rust
 * High-performance event-driven backtesting engine
 */

use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage_rate: f64,
    pub benchmark_symbol: Option<String>,
    pub enable_margin: bool,
    pub margin_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub action: SignalAction,
    pub quantity: f64,
    pub price_limit: Option<f64>,
    pub strategy_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalAction {
    Buy,
    Sell,
    Hold,
    Close,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub strategy_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub average_cost: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cash: f64,
    pub total_value: f64,
    pub positions: Vec<Position>,
    pub total_pnl: f64,
    pub daily_pnl: f64,
}

pub struct UltraFastBacktester {
    config: BacktestConfig,
    market_data: Arc<RwLock<BTreeMap<DateTime<Utc>, Vec<MarketDataPoint>>>>,
    signals: Arc<RwLock<BTreeMap<DateTime<Utc>, Vec<Signal>>>>,
    trades: Arc<RwLock<Vec<Trade>>>,
    portfolio_history: Arc<RwLock<Vec<PortfolioSnapshot>>>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    cash: Arc<RwLock<f64>>,
    trade_id_counter: Arc<RwLock<u64>>,
}

impl UltraFastBacktester {
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config: config.clone(),
            market_data: Arc::new(RwLock::new(BTreeMap::new())),
            signals: Arc::new(RwLock::new(BTreeMap::new())),
            trades: Arc::new(RwLock::new(Vec::new())),
            portfolio_history: Arc::new(RwLock::new(Vec::new())),
            positions: Arc::new(RwLock::new(HashMap::new())),
            cash: Arc::new(RwLock::new(config.initial_capital)),
            trade_id_counter: Arc::new(RwLock::new(0)),
        }
    }

    pub fn add_market_data(&self, data: Vec<MarketDataPoint>) {
        let mut market_data = self.market_data.write();
        
        for point in data {
            market_data
                .entry(point.timestamp)
                .or_insert_with(Vec::new)
                .push(point);
        }
    }

    pub fn add_signals(&self, signals: Vec<Signal>) {
        let mut signal_map = self.signals.write();
        
        for signal in signals {
            signal_map
                .entry(signal.timestamp)
                .or_insert_with(Vec::new)
                .push(signal);
        }
    }

    pub fn run_backtest(&self) -> BacktestResult {
        // Reset state
        *self.cash.write() = self.config.initial_capital;
        self.positions.write().clear();
        self.trades.write().clear();
        self.portfolio_history.write().clear();

        let market_data = self.market_data.read();
        let signals = self.signals.read();

        // Process each timestamp in chronological order
        for (&timestamp, data_points) in market_data.iter() {
            if timestamp < self.config.start_date || timestamp > self.config.end_date {
                continue;
            }

            // Update market prices
            self.update_market_prices(data_points);

            // Process signals for this timestamp
            if let Some(timestamp_signals) = signals.get(&timestamp) {
                self.process_signals(timestamp_signals, data_points);
            }

            // Create portfolio snapshot
            self.create_portfolio_snapshot(timestamp, data_points);
        }

        // Generate backtest results
        self.generate_results()
    }

    fn update_market_prices(&self, data_points: &[MarketDataPoint]) {
        let mut positions = self.positions.write();
        
        for data_point in data_points {
            if let Some(position) = positions.get_mut(&data_point.symbol) {
                let previous_value = position.market_value;
                position.market_value = position.quantity * data_point.close;
                position.unrealized_pnl = position.market_value - (position.quantity * position.average_cost);
            }
        }
    }

    fn process_signals(&self, signals: &[Signal], market_data: &[MarketDataPoint]) {
        let market_prices: HashMap<String, f64> = market_data
            .iter()
            .map(|data| (data.symbol.clone(), data.close))
            .collect();

        for signal in signals {
            if let Some(&current_price) = market_prices.get(&signal.symbol) {
                self.execute_signal(signal, current_price);
            }
        }
    }

    fn execute_signal(&self, signal: &Signal, current_price: f64) {
        match signal.action {
            SignalAction::Buy => self.execute_buy_order(signal, current_price),
            SignalAction::Sell => self.execute_sell_order(signal, current_price),
            SignalAction::Close => self.close_position(&signal.symbol, current_price, &signal.strategy_id),
            SignalAction::Hold => {}, // No action
        }
    }

    fn execute_buy_order(&self, signal: &Signal, current_price: f64) {
        let execution_price = if let Some(limit_price) = signal.price_limit {
            if current_price > limit_price {
                return; // Price too high, skip order
            }
            limit_price.min(current_price)
        } else {
            current_price
        };

        // Apply slippage
        let slippage = execution_price * self.config.slippage_rate;
        let final_price = execution_price + slippage;

        let trade_value = signal.quantity * final_price;
        let commission = trade_value * self.config.commission_rate;
        let total_cost = trade_value + commission;

        // Check if we have enough cash
        let cash = *self.cash.read();
        if cash < total_cost {
            return; // Insufficient funds
        }

        // Execute trade
        *self.cash.write() -= total_cost;
        
        let trade_id = {
            let mut counter = self.trade_id_counter.write();
            *counter += 1;
            *counter
        };

        let trade = Trade {
            id: trade_id,
            timestamp: signal.timestamp,
            symbol: signal.symbol.clone(),
            side: TradeSide::Buy,
            quantity: signal.quantity,
            price: final_price,
            commission,
            slippage,
            strategy_id: signal.strategy_id.clone(),
        };

        self.trades.write().push(trade);

        // Update position
        self.update_position(&signal.symbol, signal.quantity, final_price);
    }

    fn execute_sell_order(&self, signal: &Signal, current_price: f64) {
        let execution_price = if let Some(limit_price) = signal.price_limit {
            if current_price < limit_price {
                return; // Price too low, skip order
            }
            limit_price.max(current_price)
        } else {
            current_price
        };

        // Apply slippage
        let slippage = execution_price * self.config.slippage_rate;
        let final_price = execution_price - slippage;

        // Check if we have enough position to sell
        let positions = self.positions.read();
        if let Some(position) = positions.get(&signal.symbol) {
            if position.quantity < signal.quantity {
                return; // Insufficient position
            }
        } else {
            return; // No position to sell
        }
        drop(positions);

        let trade_value = signal.quantity * final_price;
        let commission = trade_value * self.config.commission_rate;
        let net_proceeds = trade_value - commission;

        // Execute trade
        *self.cash.write() += net_proceeds;

        let trade_id = {
            let mut counter = self.trade_id_counter.write();
            *counter += 1;
            *counter
        };

        let trade = Trade {
            id: trade_id,
            timestamp: signal.timestamp,
            symbol: signal.symbol.clone(),
            side: TradeSide::Sell,
            quantity: signal.quantity,
            price: final_price,
            commission,
            slippage,
            strategy_id: signal.strategy_id.clone(),
        };

        self.trades.write().push(trade);

        // Update position
        self.update_position(&signal.symbol, -signal.quantity, final_price);
    }

    fn close_position(&self, symbol: &str, current_price: f64, strategy_id: &str) {
        let positions = self.positions.read();
        if let Some(position) = positions.get(symbol) {
            let quantity_to_sell = position.quantity;
            drop(positions);
            
            if quantity_to_sell > 0.0 {
                let close_signal = Signal {
                    timestamp: Utc::now(), // This should be the current backtest timestamp
                    symbol: symbol.to_string(),
                    action: SignalAction::Sell,
                    quantity: quantity_to_sell,
                    price_limit: None,
                    strategy_id: strategy_id.to_string(),
                };
                
                self.execute_sell_order(&close_signal, current_price);
            }
        }
    }

    fn update_position(&self, symbol: &str, quantity_change: f64, price: f64) {
        let mut positions = self.positions.write();
        
        if let Some(position) = positions.get_mut(symbol) {
            let old_quantity = position.quantity;
            let new_quantity = old_quantity + quantity_change;
            
            if new_quantity.abs() < 1e-8 {
                // Position closed
                position.realized_pnl += position.unrealized_pnl;
                positions.remove(symbol);
            } else {
                // Update average cost
                if (old_quantity > 0.0 && quantity_change > 0.0) || (old_quantity < 0.0 && quantity_change < 0.0) {
                    // Adding to position
                    let total_cost = old_quantity * position.average_cost + quantity_change * price;
                    position.average_cost = total_cost / new_quantity;
                }
                
                position.quantity = new_quantity;
                position.market_value = new_quantity * price;
                position.unrealized_pnl = new_quantity * (price - position.average_cost);
            }
        } else if quantity_change.abs() > 1e-8 {
            // New position
            positions.insert(symbol.to_string(), Position {
                symbol: symbol.to_string(),
                quantity: quantity_change,
                average_cost: price,
                market_value: quantity_change * price,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
            });
        }
    }

    fn create_portfolio_snapshot(&self, timestamp: DateTime<Utc>, market_data: &[MarketDataPoint]) {
        let cash = *self.cash.read();
        let positions = self.positions.read().clone();
        
        let total_market_value: f64 = positions.values()
            .map(|pos| pos.market_value)
            .sum();
        
        let total_value = cash + total_market_value;
        
        let total_pnl: f64 = positions.values()
            .map(|pos| pos.unrealized_pnl + pos.realized_pnl)
            .sum();

        // Calculate daily P&L (simplified)
        let daily_pnl = {
            let history = self.portfolio_history.read();
            if let Some(previous) = history.last() {
                total_value - previous.total_value
            } else {
                total_pnl
            }
        };

        let snapshot = PortfolioSnapshot {
            timestamp,
            cash,
            total_value,
            positions: positions.into_values().collect(),
            total_pnl,
            daily_pnl,
        };

        self.portfolio_history.write().push(snapshot);
    }

    fn generate_results(&self) -> BacktestResult {
        let portfolio_history = self.portfolio_history.read();
        let trades = self.trades.read();

        if portfolio_history.is_empty() {
            return BacktestResult::default();
        }

        let initial_value = self.config.initial_capital;
        let final_value = portfolio_history.last().unwrap().total_value;
        let total_return = (final_value - initial_value) / initial_value;

        // Calculate performance metrics
        let returns: Vec<f64> = portfolio_history
            .windows(2)
            .map(|window| (window[1].total_value - window[0].total_value) / window[0].total_value)
            .collect();

        let volatility = self.calculate_volatility(&returns);
        let sharpe_ratio = self.calculate_sharpe_ratio(&returns);
        let max_drawdown = self.calculate_max_drawdown(&portfolio_history);

        BacktestResult {
            initial_capital: initial_value,
            final_value,
            total_return,
            annualized_return: total_return * 252.0 / portfolio_history.len() as f64, // Simplified
            volatility,
            sharpe_ratio,
            max_drawdown,
            total_trades: trades.len(),
            winning_trades: trades.iter().filter(|t| self.is_winning_trade(t)).count(),
            portfolio_history: portfolio_history.clone(),
            trades: trades.clone(),
        }
    }

    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt() * (252.0_f64).sqrt() // Annualized
    }

    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = self.calculate_volatility(returns) / (252.0_f64).sqrt(); // Daily vol
        
        if volatility == 0.0 {
            0.0
        } else {
            mean_return / volatility * (252.0_f64).sqrt() // Annualized Sharpe
        }
    }

    fn calculate_max_drawdown(&self, portfolio_history: &[PortfolioSnapshot]) -> f64 {
        let mut max_value = 0.0;
        let mut max_drawdown = 0.0;

        for snapshot in portfolio_history {
            if snapshot.total_value > max_value {
                max_value = snapshot.total_value;
            }
            
            let drawdown = (max_value - snapshot.total_value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    fn is_winning_trade(&self, trade: &Trade) -> bool {
        // Simplified - in practice would need to track individual trade P&L
        match trade.side {
            TradeSide::Buy => true,  // Placeholder
            TradeSide::Sell => true, // Placeholder
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BacktestResult {
    pub initial_capital: f64,
    pub final_value: f64,
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub portfolio_history: Vec<PortfolioSnapshot>,
    pub trades: Vec<Trade>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_basic() {
        let config = BacktestConfig {
            start_date: Utc::now(),
            end_date: Utc::now(),
            initial_capital: 100000.0,
            commission_rate: 0.001,
            slippage_rate: 0.0001,
            benchmark_symbol: None,
            enable_margin: false,
            margin_ratio: 1.0,
        };

        let backtester = UltraFastBacktester::new(config);
        let result = backtester.run_backtest();
        
        assert_eq!(result.initial_capital, 100000.0);
    }
}
