/*!
 * Real-Time Risk Engine for AlgoVeda Trading Platform
 * 
 * Continuous risk monitoring with real-time calculations of VaR,
 * position limits, concentration risk, and automatic risk controls.
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant, interval};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, warn, error, instrument};

use crate::trading::{Order, OrderEvent, Position, TradingResult, TradingError, MarketData};
use crate::config::RiskConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheck {
    pub check_type: RiskCheckType,
    pub passed: bool,
    pub message: String,
    pub severity: RiskSeverity,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskCheckType {
    PositionLimit,
    ConcentrationLimit,
    DailyLoss,
    VaRLimit,
    Leverage,
    Drawdown,
    Correlation,
    Liquidity,
    Volatility,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Info,
    Warning,
    Critical,
    Fatal,
}

#[derive(Debug, Clone)]
pub struct RiskViolation {
    pub violation_type: RiskCheckType,
    pub symbol: String,
    pub current_value: f64,
    pub limit_value: f64,
    pub severity: RiskSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub order_id: Option<uuid::Uuid>,
}

#[derive(Debug, Clone)]
pub struct PortfolioRisk {
    pub total_value: f64,
    pub total_pnl: f64,
    pub daily_pnl: f64,
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall_95: f64,
    pub max_drawdown: f64,
    pub current_drawdown: f64,
    pub leverage: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub beta: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub last_updated: DateTime<Utc>,
}

/// Real-Time Risk Engine with continuous monitoring
#[derive(Debug)]
pub struct RealTimeRiskEngine {
    config: Arc<RiskConfig>,
    positions: Arc<RwLock<HashMap<String, Position>>>,
    market_data: Arc<RwLock<HashMap<String, MarketData>>>,
    portfolio_risk: Arc<RwLock<PortfolioRisk>>,
    risk_violations: Arc<RwLock<VecDeque<RiskViolation>>>,
    daily_pnl_history: Arc<RwLock<VecDeque<f64>>>,
    return_history: Arc<RwLock<VecDeque<f64>>>,
    risk_events_tx: mpsc::UnboundedSender<RiskEvent>,
    is_running: Arc<RwLock<bool>>,
    last_risk_check: Arc<RwLock<DateTime<Utc>>>,
}

#[derive(Debug, Clone)]
pub enum RiskEvent {
    RiskViolation(RiskViolation),
    PortfolioRiskUpdate(PortfolioRisk),
    EmergencyStop(String),
    RiskLimitBreach(RiskCheckType, f64, f64),
}

impl RealTimeRiskEngine {
    pub fn new(
        config: Arc<RiskConfig>,
        risk_events_tx: mpsc::UnboundedSender<RiskEvent>,
    ) -> Self {
        let portfolio_risk = PortfolioRisk {
            total_value: 0.0,
            total_pnl: 0.0,
            daily_pnl: 0.0,
            var_95: 0.0,
            var_99: 0.0,
            expected_shortfall_95: 0.0,
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            leverage: 0.0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            beta: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            last_updated: Utc::now(),
        };

        Self {
            config,
            positions: Arc::new(RwLock::new(HashMap::new())),
            market_data: Arc::new(RwLock::new(HashMap::new())),
            portfolio_risk: Arc::new(RwLock::new(portfolio_risk)),
            risk_violations: Arc::new(RwLock::new(VecDeque::new())),
            daily_pnl_history: Arc::new(RwLock::new(VecDeque::new())),
            return_history: Arc::new(RwLock::new(VecDeque::new())),
            risk_events_tx,
            is_running: Arc::new(RwLock::new(false)),
            last_risk_check: Arc::new(RwLock::new(Utc::now())),
        }
    }

    pub async fn start_monitoring(&self) -> TradingResult<()> {
        {
            let mut running = self.is_running.write();
            *running = true;
        }

        info!("Real-time risk engine started");

        // Start continuous risk monitoring
        let engine = self.clone();
        tokio::spawn(async move {
            engine.risk_monitoring_loop().await;
        });

        // Start daily PnL tracking
        let engine = self.clone();
        tokio::spawn(async move {
            engine.daily_pnl_tracking_loop().await;
        });

        Ok(())
    }

    async fn risk_monitoring_loop(&self) {
        let check_interval = Duration::from_millis(self.config.risk_check_interval);
        let mut interval_timer = interval(check_interval);

        while *self.is_running.read() {
            interval_timer.tick().await;

            if let Err(e) = self.perform_risk_checks().await {
                error!("Risk check failed: {}", e);
            }
        }

        info!("Risk monitoring loop stopped");
    }

    async fn daily_pnl_tracking_loop(&self) {
        let mut interval_timer = interval(Duration::from_secs(60)); // Check every minute

        let mut last_day = Utc::now().date_naive();

        while *self.is_running.read() {
            interval_timer.tick().await;

            let current_day = Utc::now().date_naive();
            if current_day != last_day {
                // New day - reset daily PnL tracking
                self.reset_daily_pnl().await;
                last_day = current_day;
            }

            // Update current daily PnL
            if let Err(e) = self.update_daily_pnl().await {
                error!("Failed to update daily PnL: {}", e);
            }
        }
    }

    #[instrument(skip(self))]
    pub async fn check_order(&self, order: &Order) -> TradingResult<Vec<RiskCheck>> {
        let mut checks = Vec::new();

        // Pre-trade risk checks
        checks.push(self.check_position_limit(order).await?);
        checks.push(self.check_order_value_limit(order).await?);
        checks.push(self.check_concentration_limit(order).await?);
        checks.push(self.check_leverage_limit(order).await?);

        // Check if any critical checks failed
        let critical_failures: Vec<_> = checks.iter()
            .filter(|check| !check.passed && check.severity == RiskSeverity::Critical)
            .collect();

        if !critical_failures.is_empty() {
            let violation = RiskViolation {
                violation_type: critical_failures[0].check_type.clone(),
                symbol: order.symbol.clone(),
                current_value: 0.0, // Would be populated with actual values
                limit_value: 0.0,
                severity: RiskSeverity::Critical,
                message: format!("Order blocked due to risk violation: {}", critical_failures[0].message),
                timestamp: Utc::now(),
                order_id: Some(order.id),
            };

            self.record_violation(violation).await;
            
            return Err(TradingError::RiskCheckFailed(
                format!("Critical risk checks failed: {}", 
                        critical_failures.iter()
                            .map(|c| &c.message)
                            .collect::<Vec<_>>()
                            .join(", "))
            ));
        }

        Ok(checks)
    }

    async fn check_position_limit(&self, order: &Order) -> TradingResult<RiskCheck> {
        let positions = self.positions.read();
        let current_position = positions.get(&order.symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0);

        let new_position = match order.side {
            crate::trading::OrderSide::Buy => current_position + order.quantity,
            crate::trading::OrderSide::Sell => current_position - order.quantity,
        };

        let position_value = new_position * order.price.unwrap_or(100.0); // Default price for market orders
        let passed = position_value.abs() <= self.config.position_limits.max_single_position;

        Ok(RiskCheck {
            check_type: RiskCheckType::PositionLimit,
            passed,
            message: if passed {
                "Position limit check passed".to_string()
            } else {
                format!("Position limit exceeded: {:.2} > {:.2}", 
                        position_value.abs(), 
                        self.config.position_limits.max_single_position)
            },
            severity: if passed { RiskSeverity::Info } else { RiskSeverity::Critical },
            timestamp: Utc::now(),
        })
    }

    async fn check_order_value_limit(&self, order: &Order) -> TradingResult<RiskCheck> {
        let order_value = order.quantity * order.price.unwrap_or(100.0);
        let passed = order_value <= self.config.max_position_size;

        Ok(RiskCheck {
            check_type: RiskCheckType::PositionLimit,
            passed,
            message: if passed {
                "Order value limit check passed".to_string()
            } else {
                format!("Order value exceeds limit: {:.2} > {:.2}", 
                        order_value, self.config.max_position_size)
            },
            severity: if passed { RiskSeverity::Info } else { RiskSeverity::Critical },
            timestamp: Utc::now(),
        })
    }

    async fn check_concentration_limit(&self, order: &Order) -> TradingResult<RiskCheck> {
        let portfolio_risk = self.portfolio_risk.read();
        let portfolio_value = portfolio_risk.total_value;
        drop(portfolio_risk);

        let position_value = order.quantity * order.price.unwrap_or(100.0);
        let concentration = if portfolio_value > 0.0 {
            position_value / portfolio_value
        } else {
            0.0
        };

        let passed = concentration <= self.config.max_concentration;

        Ok(RiskCheck {
            check_type: RiskCheckType::ConcentrationLimit,
            passed,
            message: if passed {
                "Concentration limit check passed".to_string()
            } else {
                format!("Concentration limit exceeded: {:.2}% > {:.2}%", 
                        concentration * 100.0, self.config.max_concentration * 100.0)
            },
            severity: if passed { RiskSeverity::Info } else { RiskSeverity::Warning },
            timestamp: Utc::now(),
        })
    }

    async fn check_leverage_limit(&self, order: &Order) -> TradingResult<RiskCheck> {
        let portfolio_risk = self.portfolio_risk.read();
        let current_leverage = portfolio_risk.leverage;
        drop(portfolio_risk);

        let passed = current_leverage <= self.config.max_leverage;

        Ok(RiskCheck {
            check_type: RiskCheckType::Leverage,
            passed,
            message: if passed {
                "Leverage limit check passed".to_string()
            } else {
                format!("Leverage limit exceeded: {:.2}x > {:.2}x", 
                        current_leverage, self.config.max_leverage)
            },
            severity: if passed { RiskSeverity::Info } else { RiskSeverity::Critical },
            timestamp: Utc::now(),
        })
    }

    async fn perform_risk_checks(&self) -> TradingResult<()> {
        {
            let mut last_check = self.last_risk_check.write();
            *last_check = Utc::now();
        }

        // Update portfolio risk metrics
        self.calculate_portfolio_risk().await?;

        // Check daily loss limit
        self.check_daily_loss_limit().await?;

        // Check VaR limits
        self.check_var_limits().await?;

        // Check drawdown limits
        self.check_drawdown_limits().await?;

        Ok(())
    }

    async fn calculate_portfolio_risk(&self) -> TradingResult<()> {
        let positions = self.positions.read();
        let market_data = self.market_data.read();

        let mut total_value = 0.0;
        let mut total_pnl = 0.0;
        let mut gross_exposure = 0.0;
        let mut net_exposure = 0.0;

        // Calculate basic portfolio metrics
        for (symbol, position) in positions.iter() {
            if let Some(market_price) = market_data.get(symbol).map(|md| md.price) {
                let market_value = position.quantity * market_price;
                let pnl = (market_price - position.average_cost) * position.quantity;

                total_value += market_value.abs();
                total_pnl += pnl;
                gross_exposure += market_value.abs();
                net_exposure += market_value;
            }
        }

        // Calculate VaR (simplified historical simulation)
        let var_95 = self.calculate_var(0.95).await?;
        let var_99 = self.calculate_var(0.99).await?;
        let expected_shortfall_95 = self.calculate_expected_shortfall(0.95).await?;

        // Calculate drawdown
        let (max_drawdown, current_drawdown) = self.calculate_drawdown().await?;

        // Calculate volatility and Sharpe ratio
        let volatility = self.calculate_portfolio_volatility().await?;
        let sharpe_ratio = self.calculate_sharpe_ratio().await?;

        let portfolio_risk = PortfolioRisk {
            total_value,
            total_pnl,
            daily_pnl: self.get_current_daily_pnl().await?,
            var_95,
            var_99,
            expected_shortfall_95,
            max_drawdown,
            current_drawdown,
            leverage: if total_value > 0.0 { gross_exposure / total_value } else { 0.0 },
            gross_exposure,
            net_exposure,
            beta: 1.0, // Simplified - would calculate vs benchmark
            volatility,
            sharpe_ratio,
            last_updated: Utc::now(),
        };

        {
            let mut portfolio_risk_guard = self.portfolio_risk.write();
            *portfolio_risk_guard = portfolio_risk.clone();
        }

        // Send risk update event
        let _ = self.risk_events_tx.send(RiskEvent::PortfolioRiskUpdate(portfolio_risk));

        Ok(())
    }

    async fn calculate_var(&self, confidence_level: f64) -> TradingResult<f64> {
        let return_history = self.return_history.read();
        
        if return_history.len() < 30 {
            return Ok(0.0); // Need minimum history for VaR calculation
        }

        let mut sorted_returns: Vec<f64> = return_history.iter().cloned().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var = -sorted_returns[index.min(sorted_returns.len() - 1)];

        Ok(var)
    }

    async fn calculate_expected_shortfall(&self, confidence_level: f64) -> TradingResult<f64> {
        let return_history = self.return_history.read();
        
        if return_history.len() < 30 {
            return Ok(0.0);
        }

        let mut sorted_returns: Vec<f64> = return_history.iter().cloned().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let cutoff_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let tail_returns: Vec<f64> = sorted_returns.iter().take(cutoff_index).cloned().collect();

        if tail_returns.is_empty() {
            return Ok(0.0);
        }

        let expected_shortfall = -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
        Ok(expected_shortfall)
    }

    async fn calculate_drawdown(&self) -> TradingResult<(f64, f64)> {
        let daily_pnl_history = self.daily_pnl_history.read();
        
        if daily_pnl_history.is_empty() {
            return Ok((0.0, 0.0));
        }

        let mut cumulative_pnl = 0.0;
        let mut peak_pnl = 0.0;
        let mut max_drawdown = 0.0;

        for &daily_pnl in daily_pnl_history.iter() {
            cumulative_pnl += daily_pnl;
            peak_pnl = peak_pnl.max(cumulative_pnl);
            let current_drawdown = peak_pnl - cumulative_pnl;
            max_drawdown = max_drawdown.max(current_drawdown);
        }

        let current_drawdown = peak_pnl - cumulative_pnl;

        Ok((max_drawdown, current_drawdown))
    }

    async fn calculate_portfolio_volatility(&self) -> TradingResult<f64> {
        let return_history = self.return_history.read();
        
        if return_history.len() < 2 {
            return Ok(0.0);
        }

        let mean_return = return_history.iter().sum::<f64>() / return_history.len() as f64;
        let variance = return_history.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (return_history.len() - 1) as f64;

        Ok(variance.sqrt() * (252.0_f64).sqrt()) // Annualized volatility
    }

    async fn calculate_sharpe_ratio(&self) -> TradingResult<f64> {
        let return_history = self.return_history.read();
        
        if return_history.len() < 2 {
            return Ok(0.0);
        }

        let mean_return = return_history.iter().sum::<f64>() / return_history.len() as f64;
        let volatility = self.calculate_portfolio_volatility().await?;

        if volatility > 0.0 {
            let risk_free_rate = self.config.risk_free_rate / 252.0; // Daily risk-free rate
            Ok((mean_return * 252.0 - self.config.risk_free_rate) / volatility)
        } else {
            Ok(0.0)
        }
    }

    async fn check_daily_loss_limit(&self) -> TradingResult<()> {
        let daily_pnl = self.get_current_daily_pnl().await?;
        
        if daily_pnl < -self.config.max_daily_loss {
            let violation = RiskViolation {
                violation_type: RiskCheckType::DailyLoss,
                symbol: "PORTFOLIO".to_string(),
                current_value: -daily_pnl,
                limit_value: self.config.max_daily_loss,
                severity: RiskSeverity::Fatal,
                message: format!("Daily loss limit exceeded: {:.2} > {:.2}", 
                                -daily_pnl, self.config.max_daily_loss),
                timestamp: Utc::now(),
                order_id: None,
            };

            self.record_violation(violation).await;
            let _ = self.risk_events_tx.send(RiskEvent::EmergencyStop(
                "Daily loss limit exceeded".to_string()
            ));
        }

        Ok(())
    }

    async fn check_var_limits(&self) -> TradingResult<()> {
        let portfolio_risk = self.portfolio_risk.read();
        let var_95 = portfolio_risk.var_95;
        drop(portfolio_risk);

        if var_95 > self.config.max_portfolio_var {
            let violation = RiskViolation {
                violation_type: RiskCheckType::VaRLimit,
                symbol: "PORTFOLIO".to_string(),
                current_value: var_95,
                limit_value: self.config.max_portfolio_var,
                severity: RiskSeverity::Critical,
                message: format!("VaR limit exceeded: {:.2} > {:.2}", 
                                var_95, self.config.max_portfolio_var),
                timestamp: Utc::now(),
                order_id: None,
            };

            self.record_violation(violation).await;
        }

        Ok(())
    }

    async fn check_drawdown_limits(&self) -> TradingResult<()> {
        let portfolio_risk = self.portfolio_risk.read();
        let current_drawdown = portfolio_risk.current_drawdown;
        drop(portfolio_risk);

        if current_drawdown > self.config.max_drawdown * self.portfolio_risk.read().total_value {
            let violation = RiskViolation {
                violation_type: RiskCheckType::Drawdown,
                symbol: "PORTFOLIO".to_string(),
                current_value: current_drawdown,
                limit_value: self.config.max_drawdown * self.portfolio_risk.read().total_value,
                severity: RiskSeverity::Critical,
                message: format!("Drawdown limit exceeded: {:.2} > {:.2}", 
                                current_drawdown, 
                                self.config.max_drawdown * self.portfolio_risk.read().total_value),
                timestamp: Utc::now(),
                order_id: None,
            };

            self.record_violation(violation).await;
        }

        Ok(())
    }

    async fn record_violation(&self, violation: RiskViolation) {
        warn!("Risk violation: {}", violation.message);

        {
            let mut violations = self.risk_violations.write();
            violations.push_back(violation.clone());
            
            // Keep only recent violations
            if violations.len() > 1000 {
                violations.pop_front();
            }
        }

        let _ = self.risk_events_tx.send(RiskEvent::RiskViolation(violation));
    }

    async fn reset_daily_pnl(&self) {
        info!("Resetting daily PnL tracking for new trading day");
        // Implementation would reset daily tracking
    }

    async fn update_daily_pnl(&self) -> TradingResult<()> {
        // Implementation would calculate current daily PnL
        Ok(())
    }

    async fn get_current_daily_pnl(&self) -> TradingResult<f64> {
        // Implementation would return current daily PnL
        Ok(0.0)
    }

    pub fn update_position(&self, symbol: String, position: Position) {
        let mut positions = self.positions.write();
        positions.insert(symbol, position);
    }

    pub fn update_market_data(&self, symbol: String, market_data: MarketData) {
        let mut market_data_guard = self.market_data.write();
        market_data_guard.insert(symbol, market_data);
    }

    pub async fn get_current_pnl(&self) -> f64 {
        let portfolio_risk = self.portfolio_risk.read();
        portfolio_risk.total_pnl
    }

    pub async fn get_risk_violations(&self) -> u32 {
        let violations = self.risk_violations.read();
        violations.len() as u32
    }

    pub async fn health_check(&self) -> bool {
        let last_check = self.last_risk_check.read();
        let now = Utc::now();
        let time_since_last_check = (now - *last_check).num_milliseconds() as u64;
        
        // Consider healthy if risk checks ran within the last 10 seconds
        time_since_last_check < 10000
    }

    pub async fn final_risk_report(&self) -> TradingResult<()> {
        let portfolio_risk = self.portfolio_risk.read();
        info!("Final Risk Report:");
        info!("  Total P&L: {:.2}", portfolio_risk.total_pnl);
        info!("  Max Drawdown: {:.2}", portfolio_risk.max_drawdown);
        info!("  VaR 95%: {:.2}", portfolio_risk.var_95);
        info!("  Sharpe Ratio: {:.2}", portfolio_risk.sharpe_ratio);
        
        let violations = self.risk_violations.read();
        info!("  Total Risk Violations: {}", violations.len());
        
        Ok(())
    }

    pub async fn set_order_manager(&self, _order_manager: Arc<dyn Send + Sync>) -> TradingResult<()> {
        // Implementation would connect to order manager
        Ok(())
    }
}

// Implement Clone to allow spawning async tasks
impl Clone for RealTimeRiskEngine {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            positions: Arc::clone(&self.positions),
            market_data: Arc::clone(&self.market_data),
            portfolio_risk: Arc::clone(&self.portfolio_risk),
            risk_violations: Arc::clone(&self.risk_violations),
            daily_pnl_history: Arc::clone(&self.daily_pnl_history),
            return_history: Arc::clone(&self.return_history),
            risk_events_tx: self.risk_events_tx.clone(),
            is_running: Arc::clone(&self.is_running),
            last_risk_check: Arc::clone(&self.last_risk_check),
        }
    }
}

// Define Position struct if not already defined
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub average_cost: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub last_updated: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_risk_engine_creation() {
        let config = Arc::new(RiskConfig::default());
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let engine = RealTimeRiskEngine::new(config, tx);
        assert!(!*engine.is_running.read());
    }

    #[tokio::test]
    async fn test_position_limit_check() {
        let config = Arc::new(RiskConfig::default());
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let engine = RealTimeRiskEngine::new(config, tx);
        
        let order = Order {
            id: uuid::Uuid::new_v4(),
            symbol: "AAPL".to_string(),
            side: crate::trading::OrderSide::Buy,
            quantity: 100.0,
            price: Some(150.0),
            ..Default::default()
        };
        
        let check = engine.check_position_limit(&order).await.unwrap();
        assert!(check.passed); // Should pass with default limits
    }

    #[tokio::test]
    async fn test_var_calculation() {
        let config = Arc::new(RiskConfig::default());
        let (tx, _rx) = mpsc::unbounded_channel();
        
        let engine = RealTimeRiskEngine::new(config, tx);
        
        // Add some test return data
        {
            let mut return_history = engine.return_history.write();
            for i in 0..100 {
                return_history.push_back((i as f64 - 50.0) / 1000.0); // Returns from -5% to +5%
            }
        }
        
        let var_95 = engine.calculate_var(0.95).await.unwrap();
        assert!(var_95 > 0.0);
    }
}
