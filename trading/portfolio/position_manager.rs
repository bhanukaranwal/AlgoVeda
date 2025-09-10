/*!
 * Position Manager for AlgoVeda Trading Platform
 * 
 * Real-time position tracking, P&L calculation, and portfolio analytics
 * with sophisticated risk metrics and attribution analysis.
 */

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Date, NaiveDate};
use tracing::{info, warn, instrument};
use uuid::Uuid;

use crate::trading::{Fill, TradingResult, MarketData, OrderSide};
use crate::config::TradingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub average_cost: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub daily_pnl: f64,
    pub cost_basis: f64,
    pub first_trade_date: DateTime<Utc>,
    pub last_trade_date: DateTime<Utc>,
    pub trade_count: u32,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub currency: String,
}

impl Position {
    pub fn new(symbol: String, currency: String) -> Self {
        let now = Utc::now();
        Self {
            symbol,
            quantity: 0.0,
            average_cost: 0.0,
            market_value: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            daily_pnl: 0.0,
            cost_basis: 0.0,
            first_trade_date: now,
            last_trade_date: now,
            trade_count: 0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            currency,
        }
    }

    pub fn is_long(&self) -> bool {
        self.quantity > 0.0
    }

    pub fn is_short(&self) -> bool {
        self.quantity < 0.0
    }

    pub fn is_flat(&self) -> bool {
        self.quantity == 0.0
    }

    pub fn update_market_value(&mut self, market_price: f64) {
        self.market_value = self.quantity * market_price;
        self.unrealized_pnl = (market_price - self.average_cost) * self.quantity;
        self.gross_exposure = self.market_value.abs();
        self.net_exposure = self.market_value;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub positions: HashMap<String, Position>,
    pub total_value: f64,
    pub total_unrealized_pnl: f64,
    pub total_realized_pnl: f64,
    pub total_daily_pnl: f64,
    pub cash_balance: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub long_exposure: f64,
    pub short_exposure: f64,
    pub position_count: u32,
    pub currency_exposures: HashMap<String, f64>,
    pub sector_exposures: HashMap<String, f64>,
    pub last_updated: DateTime<Utc>,
}

/// Position Manager handles all position tracking and portfolio analytics
#[derive(Debug)]
pub struct PositionManager {
    positions: Arc<RwLock<HashMap<String, Position>>>,
    fill_history: Arc<RwLock<Vec<Fill>>>,
    daily_pnl_snapshots: Arc<RwLock<HashMap<NaiveDate, f64>>>,
    cash_balance: Arc<RwLock<f64>>,
    initial_capital: Arc<RwLock<f64>>,
    config: Arc<TradingConfig>,
    market_data: Arc<RwLock<HashMap<String, MarketData>>>,
}

impl PositionManager {
    pub fn new(config: Arc<TradingConfig>, initial_capital: f64) -> Self {
        Self {
            positions: Arc::new(RwLock::new(HashMap::new())),
            fill_history: Arc::new(RwLock::new(Vec::new())),
            daily_pnl_snapshots: Arc::new(RwLock::new(HashMap::new())),
            cash_balance: Arc::new(RwLock::new(initial_capital)),
            initial_capital: Arc::new(RwLock::new(initial_capital)),
            config,
            market_data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[instrument(skip(self))]
    pub fn process_fill(&self, fill: &Fill) -> TradingResult<()> {
        let mut positions = self.positions.write();
        let mut fill_history = self.fill_history.write();
        let mut cash_balance = self.cash_balance.write();

        // Add fill to history
        fill_history.push(fill.clone());

        // Update or create position
        let position = positions.entry(fill.symbol.clone())
            .or_insert_with(|| Position::new(fill.symbol.clone(), "USD".to_string()));

        // Calculate new position after fill
        let quantity_change = match fill.side {
            OrderSide::Buy => fill.quantity,
            OrderSide::Sell => -fill.quantity,
        };

        let old_quantity = position.quantity;
        let new_quantity = old_quantity + quantity_change;

        // Handle different scenarios for P&L calculation
        if old_quantity == 0.0 {
            // Opening new position
            position.average_cost = fill.price;
            position.first_trade_date = fill.timestamp;
        } else if old_quantity.signum() == quantity_change.signum() {
            // Adding to existing position
            let total_cost = (position.average_cost * old_quantity) + (fill.price * quantity_change);
            position.average_cost = total_cost / new_quantity;
        } else {
            // Reducing or closing position
            if new_quantity.abs() < old_quantity.abs() {
                // Partial close - realize some P&L
                let closed_quantity = quantity_change.abs();
                let realized_pnl = (fill.price - position.average_cost) * closed_quantity * old_quantity.signum();
                position.realized_pnl += realized_pnl;
            } else if new_quantity == 0.0 {
                // Full close - realize all P&L
                let realized_pnl = (fill.price - position.average_cost) * old_quantity;
                position.realized_pnl += realized_pnl;
            } else {
                // Flip position - close old and open new
                let close_pnl = (fill.price - position.average_cost) * old_quantity;
                position.realized_pnl += close_pnl;
                
                // New position starts fresh
                position.average_cost = fill.price;
            }
        }

        // Update position fields
        position.quantity = new_quantity;
        position.last_trade_date = fill.timestamp;
        position.trade_count += 1;
        position.cost_basis = position.average_cost * position.quantity.abs();

        // Update cash balance
        let cash_change = match fill.side {
            OrderSide::Buy => -(fill.quantity * fill.price + fill.commission + fill.fees),
            OrderSide::Sell => fill.quantity * fill.price - fill.commission - fill.fees,
        };
        *cash_balance += cash_change;

        info!("Processed fill: {} {} {} @ {:.4}, new position: {:.0} @ {:.4}",
              fill.symbol,
              match fill.side { OrderSide::Buy => "BUY", OrderSide::Sell => "SELL" },
              fill.quantity,
              fill.price,
              position.quantity,
              position.average_cost);

        Ok(())
    }

    pub fn update_market_data(&self, symbol: String, market_data: MarketData) {
        let mut data_map = self.market_data.write();
        data_map.insert(symbol.clone(), market_data.clone());
        
        // Update position market values
        let mut positions = self.positions.write();
        if let Some(position) = positions.get_mut(&symbol) {
            position.update_market_value(market_data.price);
        }
    }

    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        let positions = self.positions.read();
        positions.get(symbol).cloned()
    }

    pub fn get_all_positions(&self) -> HashMap<String, Position> {
        let positions = self.positions.read();
        positions.clone()
    }

    pub fn get_portfolio_summary(&self) -> PortfolioSummary {
        let positions = self.positions.read();
        let cash_balance = *self.cash_balance.read();
        
        let mut total_value = cash_balance;
        let mut total_unrealized_pnl = 0.0;
        let mut total_realized_pnl = 0.0;
        let mut total_daily_pnl = 0.0;
        let mut gross_exposure = 0.0;
        let mut net_exposure = 0.0;
        let mut long_exposure = 0.0;
        let mut short_exposure = 0.0;
        let mut position_count = 0;
        let mut currency_exposures: HashMap<String, f64> = HashMap::new();

        for position in positions.values() {
            if !position.is_flat() {
                total_value += position.market_value;
                total_unrealized_pnl += position.unrealized_pnl;
                total_realized_pnl += position.realized_pnl;
                total_daily_pnl += position.daily_pnl;
                gross_exposure += position.gross_exposure;
                net_exposure += position.net_exposure;
                
                if position.is_long() {
                    long_exposure += position.market_value;
                } else {
                    short_exposure += position.market_value.abs();
                }
                
                position_count += 1;
                
                *currency_exposures.entry(position.currency.clone()).or_insert(0.0) += position.market_value;
            }
        }

        PortfolioSummary {
            positions: positions.clone(),
            total_value,
            total_unrealized_pnl,
            total_realized_pnl,
            total_daily_pnl,
            cash_balance,
            gross_exposure,
            net_exposure,
            long_exposure,
            short_exposure,
            position_count,
            currency_exposures,
            sector_exposures: HashMap::new(), // Would be populated with sector mapping
            last_updated: Utc::now(),
        }
    }

    pub fn calculate_daily_pnl(&self, date: NaiveDate) -> f64 {
        let daily_snapshots = self.daily_pnl_snapshots.read();
        daily_snapshots.get(&date).copied().unwrap_or(0.0)
    }

    pub fn take_daily_snapshot(&self) {
        let today = Utc::now().date_naive();
        let portfolio = self.get_portfolio_summary();
        let daily_pnl = portfolio.total_unrealized_pnl + portfolio.total_realized_pnl;
        
        let mut snapshots = self.daily_pnl_snapshots.write();
        snapshots.insert(today, daily_pnl);
        
        // Keep only last 365 days
        let cutoff_date = today - chrono::Duration::days(365);
        snapshots.retain(|&date, _| date > cutoff_date);
        
        info!("Daily P&L snapshot: {:.2} on {}", daily_pnl, today);
    }

    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let portfolio = self.get_portfolio_summary();
        let initial_capital = *self.initial_capital.read();
        
        let total_return = if initial_capital > 0.0 {
            (portfolio.total_value - initial_capital) / initial_capital
        } else {
            0.0
        };

        let daily_snapshots = self.daily_pnl_snapshots.read();
        let returns: Vec<f64> = daily_snapshots.values().cloned().collect();
        
        let (volatility, sharpe_ratio) = if returns.len() > 1 {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized
            
            let risk_free_rate = 0.05; // 5% annual risk-free rate
            let sharpe_ratio = if volatility > 0.0 {
                (mean_return * 252.0 - risk_free_rate) / volatility
            } else {
                0.0
            };
            
            (volatility, sharpe_ratio)
        } else {
            (0.0, 0.0)
        };

        let max_drawdown = self.calculate_max_drawdown(&returns);

        PerformanceMetrics {
            total_return,
            annualized_return: total_return, // Simplified
            volatility,
            sharpe_ratio,
            max_drawdown,
            win_rate: self.calculate_win_rate(),
            profit_factor: self.calculate_profit_factor(),
            average_win: self.calculate_average_win(),
            average_loss: self.calculate_average_loss(),
            last_updated: Utc::now(),
        }
    }

    fn calculate_max_drawdown(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut peak = 0.0;
        let mut max_drawdown = 0.0;
        let mut cumulative = 0.0;

        for &ret in returns {
            cumulative += ret;
            peak = peak.max(cumulative);
            let drawdown = peak - cumulative;
            max_drawdown = max_drawdown.max(drawdown);
        }

        max_drawdown
    }

    fn calculate_win_rate(&self) -> f64 {
        let fill_history = self.fill_history.read();
        if fill_history.is_empty() {
            return 0.0;
        }

        let winning_trades = fill_history.iter()
            .filter(|fill| {
                // Simplified: assume any trade with positive commission-adjusted P&L is winning
                match fill.side {
                    OrderSide::Buy => false, // Would need more complex logic to determine P&L
                    OrderSide::Sell => false,
                }
            })
            .count();

        winning_trades as f64 / fill_history.len() as f64
    }

    fn calculate_profit_factor(&self) -> f64 {
        // Simplified implementation
        let positions = self.positions.read();
        let total_profit: f64 = positions.values()
            .map(|p| p.realized_pnl.max(0.0))
            .sum();
        let total_loss: f64 = positions.values()
            .map(|p| p.realized_pnl.min(0.0).abs())
            .sum();

        if total_loss > 0.0 {
            total_profit / total_loss
        } else {
            f64::INFINITY
        }
    }

    fn calculate_average_win(&self) -> f64 {
        let positions = self.positions.read();
        let wins: Vec<f64> = positions.values()
            .filter_map(|p| if p.realized_pnl > 0.0 { Some(p.realized_pnl) } else { None })
            .collect();

        if wins.is_empty() {
            0.0
        } else {
            wins.iter().sum::<f64>() / wins.len() as f64
        }
    }

    fn calculate_average_loss(&self) -> f64 {
        let positions = self.positions.read();
        let losses: Vec<f64> = positions.values()
            .filter_map(|p| if p.realized_pnl < 0.0 { Some(p.realized_pnl.abs()) } else { None })
            .collect();

        if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub last_updated: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::Fill;

    #[test]
    fn test_position_creation() {
        let position = Position::new("AAPL".to_string(), "USD".to_string());
        assert_eq!(position.symbol, "AAPL");
        assert!(position.is_flat());
        assert_eq!(position.quantity, 0.0);
    }

    #[test]
    fn test_portfolio_manager() {
        let config = Arc::new(TradingConfig::default());
        let pm = PositionManager::new(config, 100000.0);
        
        let summary = pm.get_portfolio_summary();
        assert_eq!(summary.cash_balance, 100000.0);
        assert_eq!(summary.position_count, 0);
    }

    #[test]
    fn test_fill_processing() {
        let config = Arc::new(TradingConfig::default());
        let pm = PositionManager::new(config, 100000.0);
        
        let fill = Fill {
            id: Uuid::new_v4(),
            order_id: Uuid::new_v4(),
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            quantity: 100.0,
            price: 150.0,
            timestamp: Utc::now(),
            venue: "NYSE".to_string(),
            execution_id: "exec_1".to_string(),
            commission: 1.0,
            fees: 0.0,
        };
        
        let result = pm.process_fill(&fill);
        assert!(result.is_ok());
        
        let position = pm.get_position("AAPL");
        assert!(position.is_some());
        
        let pos = position.unwrap();
        assert_eq!(pos.quantity, 100.0);
        assert_eq!(pos.average_cost, 150.0);
    }
}
