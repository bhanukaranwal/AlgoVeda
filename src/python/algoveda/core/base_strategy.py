"""
Base Strategy Class for AlgoVeda Backtesting Engine
Provides comprehensive framework for strategy implementation with
performance optimization, risk management, and portfolio analytics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd
import numpy as np
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by the backtesting engine."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    
class OrderSide(Enum):
    """Order sides for buy/sell operations."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"

class TimeInForce(Enum):
    """Time in force options for orders."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date

class OrderStatus(Enum):
    """Order status types."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """
    Comprehensive order representation with all necessary attributes
    for institutional-grade order management.
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    commission: float = 0.0
    slippage: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.order_id is None:
            self.order_id = f"order_{id(self)}_{int(self.created_at.timestamp())}"

@dataclass
class Position:
    """
    Position tracking with comprehensive metrics and risk management.
    """
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    cost_basis: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    max_position_size: float = 0.0
    min_position_size: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-10
    
    def update_market_value(self, current_price: float):
        """Update position market value and unrealized P&L."""
        self.market_value = self.quantity * current_price
        if not self.is_flat:
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.updated_at = datetime.now()

@dataclass
class Trade:
    """
    Individual trade record with comprehensive execution details.
    """
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    executed_at: datetime
    order_id: str
    strategy_id: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    venue: Optional[str] = None
    execution_algo: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> float:
        return abs(self.quantity * self.price)
    
    @property
    def net_proceeds(self) -> float:
        """Net proceeds after commission and slippage."""
        gross = self.quantity * self.price
        costs = self.commission + abs(self.slippage * self.quantity)
        return gross - costs if self.side == OrderSide.SELL else gross + costs

class RiskMetrics:
    """
    Comprehensive risk metrics calculator for strategy evaluation.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio with risk-free rate adjustment."""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio focusing on downside deviation."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        drawdown_start = None
        drawdown_end = None
        max_drawdown_duration = 0
        current_drawdown_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_drawdown_duration += 1
            else:
                if drawdown_start is not None:
                    drawdown_end = i - 1
                    max_drawdown_duration = max(max_drawdown_duration, current_drawdown_duration)
                    drawdown_start = None
                    current_drawdown_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0.0,
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = returns.mean() * 252
        max_dd = RiskMetrics.calculate_max_drawdown(equity_curve)['max_drawdown']
        return annual_return / abs(max_dd) if max_dd != 0 else 0.0
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk at given confidence level."""
        return returns.quantile(confidence_level)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies in AlgoVeda.
    
    Provides comprehensive framework including:
    - Portfolio management
    - Risk management
    - Performance analytics
    - Order management
    - Real-time processing capabilities
    - Extensible architecture
    """
    
    def __init__(
        self,
        name: str,
        initial_capital: float = 1000000.0,
        max_position_size: float = 0.1,
        max_leverage: float = 1.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        enable_short_selling: bool = True,
        risk_free_rate: float = 0.02,
        benchmark_symbol: str = 'SPY',
        **kwargs
    ):
        """
        Initialize strategy with comprehensive configuration.
        
        Args:
            name: Strategy name
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of portfolio
            max_leverage: Maximum leverage allowed
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            enable_short_selling: Whether short selling is allowed
            risk_free_rate: Risk-free rate for calculations
            benchmark_symbol: Benchmark symbol for comparison
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.enable_short_selling = enable_short_selling
        self.risk_free_rate = risk_free_rate
        self.benchmark_symbol = benchmark_symbol
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        
        # Performance tracking
        self.equity_curve: List[float] = [initial_capital]
        self.returns: List[float] = []
        self.dates: List[datetime] = []
        
        # Strategy state
        self.current_date: Optional[datetime] = None
        self.is_initialized = False
        self.is_running = False
        self.strategy_params = kwargs
        
        # Risk management
        self.daily_loss_limit: Optional[float] = kwargs.get('daily_loss_limit')
        self.total_loss_limit: Optional[float] = kwargs.get('total_loss_limit')
        self.var_limit: Optional[float] = kwargs.get('var_limit')
        
        # Performance metrics cache
        self._performance_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Strategy '{name}' initialized with capital ${initial_capital:,.2f}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            DataFrame with trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_sizes(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position sizes based on signals and current portfolio state.
        
        Args:
        signals: Trading signals DataFrame
        data: Market data DataFrame
        
    Returns:
        DataFrame with position sizes for each signal
    """
    pass

def on_bar(self, bar_data: pd.Series) -> None:
    """
    Process single bar of market data.
    
    Args:
        bar_data: Single row of OHLCV data
    """
    self.current_date = bar_data.name if hasattr(bar_data, 'name') else datetime.now()
    
    # Update positions with current prices
    self.update_positions(bar_data)
    
    # Process pending orders
    self.process_pending_orders(bar_data)
    
    # Update equity curve
    current_equity = self.calculate_total_equity(bar_data)
    self.equity_curve.append(current_equity)
    
    # Calculate returns
    if len(self.equity_curve) > 1:
        ret = (current_equity - self.equity_curve[-2]) / self.equity_curve[-2]
        self.returns.append(ret)
    
    self.dates.append(self.current_date)
    
    # Check risk limits
    self.check_risk_limits()

def place_order(
    self,
    symbol: str,
    side: OrderSide,
    quantity: float,
    order_type: OrderType = OrderType.MARKET,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    **kwargs
) -> Order:
    """
    Place a new order with comprehensive validation and risk checks.
    
    Args:
        symbol: Trading symbol
        side: Buy or sell side
        quantity: Order quantity
        order_type: Type of order
        price: Limit price (if applicable)
        stop_price: Stop price (if applicable)
        time_in_force: Time in force
        **kwargs: Additional order parameters
        
    Returns:
        Order object
    """
    # Validate order parameters
    if quantity <= 0:
        raise ValueError("Order quantity must be positive")
    
    if not self.enable_short_selling and side in [OrderSide.SELL_SHORT]:
        raise ValueError("Short selling is not enabled")
    
    # Create order
    order = Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        price=price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        strategy_id=self.name,
        **kwargs
    )
    
    # Pre-trade risk checks
    if not self.validate_order(order):
        order.status = OrderStatus.REJECTED
        logger.warning(f"Order rejected for {symbol}: Risk validation failed")
        return order
    
    # Add to orders list
    self.orders.append(order)
    
    # Add to pending orders for processing
    if order_type == OrderType.MARKET:
        # Market orders execute immediately
        self.execute_order(order)
    else:
        # Limit/Stop orders go to pending
        self.pending_orders.append(order)
    
    logger.info(f"Order placed: {side.value} {quantity} {symbol} @ {order_type.value}")
    return order

def validate_order(self, order: Order) -> bool:
    """
    Comprehensive order validation including risk checks.
    
    Args:
        order: Order to validate
        
    Returns:
        True if order passes all validations
    """
    # Position size validation
    current_position = self.positions.get(order.symbol, Position(order.symbol))
    new_quantity = current_position.quantity
    
    if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
        new_quantity += order.quantity
    else:
        new_quantity -= order.quantity
    
    # Check maximum position size
    portfolio_value = self.calculate_total_equity()
    max_position_value = portfolio_value * self.max_position_size
    
    if order.price:
        order_value = abs(new_quantity * order.price)
        if order_value > max_position_value:
            logger.warning(f"Order would exceed max position size: {order_value} > {max_position_value}")
            return False
    
    # Check leverage limits
    total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
    if order.price:
        total_exposure += abs(order.quantity * order.price)
    
    leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
    if leverage > self.max_leverage:
        logger.warning(f"Order would exceed max leverage: {leverage:.2f} > {self.max_leverage}")
        return False
    
    # Capital requirement check
    if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER] and order.price:
        required_capital = order.quantity * order.price * (1 + self.commission_rate + self.slippage_rate)
        if required_capital > self.current_capital:
            logger.warning(f"Insufficient capital: {required_capital} > {self.current_capital}")
            return False
    
    return True

def execute_order(self, order: Order) -> None:
    """
    Execute order with realistic market simulation.
    
    Args:
        order: Order to execute
    """
    if order.status != OrderStatus.PENDING:
        return
    
    # Simulate execution price and slippage
    execution_price = self.calculate_execution_price(order)
    if execution_price is None:
        order.status = OrderStatus.REJECTED
        return
    
    # Calculate commission and slippage
    commission = abs(order.quantity * execution_price * self.commission_rate)
    slippage = abs(order.quantity * execution_price * self.slippage_rate)
    
    # Update order
    order.filled_quantity = order.quantity
    order.filled_price = execution_price
    order.status = OrderStatus.FILLED
    order.filled_at = self.current_date
    order.commission = commission
    order.slippage = slippage
    
    # Create trade record
    trade = Trade(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        price=execution_price,
        executed_at=self.current_date,
        order_id=order.order_id,
        strategy_id=self.name,
        commission=commission,
        slippage=slippage
    )
    self.trades.append(trade)
    
    # Update position
    self.update_position_from_trade(trade)
    
    # Update capital
    if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
        self.current_capital -= (trade.quantity * execution_price + commission + slippage)
    else:
        self.current_capital += (trade.quantity * execution_price - commission - slippage)
    
    # Remove from pending orders
    if order in self.pending_orders:
        self.pending_orders.remove(order)
    
    logger.info(f"Order executed: {trade.side.value} {trade.quantity} {trade.symbol} @ ${execution_price:.4f}")

def calculate_execution_price(self, order: Order) -> Optional[float]:
    """
    Calculate realistic execution price based on order type and market conditions.
    
    Args:
        order: Order to price
        
    Returns:
        Execution price or None if order cannot be filled
    """
    # This would be implemented based on current market data
    # For now, return the order price or current market price
    if order.order_type == OrderType.MARKET:
        # Use current market price (would need current bar data)
        return order.price if order.price else 100.0  # Placeholder
    elif order.order_type == OrderType.LIMIT:
        return order.price
    else:
        return order.price

def update_position_from_trade(self, trade: Trade) -> None:
    """
    Update position based on executed trade.
    
    Args:
        trade: Executed trade
    """
    if trade.symbol not in self.positions:
        self.positions[trade.symbol] = Position(
            symbol=trade.symbol,
            opened_at=trade.executed_at
        )
    
    position = self.positions[trade.symbol]
    
    # Calculate new position
    old_quantity = position.quantity
    trade_quantity = trade.quantity if trade.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER] else -trade.quantity
    
    new_quantity = old_quantity + trade_quantity
    
    # Update average price
    if (old_quantity >= 0 and trade_quantity > 0) or (old_quantity <= 0 and trade_quantity < 0):
        # Adding to position in same direction
        total_cost = (old_quantity * position.average_price) + (trade_quantity * trade.price)
        position.average_price = total_cost / new_quantity if new_quantity != 0 else 0
    elif old_quantity * trade_quantity < 0:
        # Reducing or reversing position
        if abs(trade_quantity) < abs(old_quantity):
            # Partial close - average price stays same
            pass
        else:
            # Full close or reversal
            position.average_price = trade.price
    
    # Update realized P&L for position reduction
    if old_quantity * trade_quantity < 0:  # Reducing position
        closed_quantity = min(abs(old_quantity), abs(trade_quantity))
        if old_quantity > 0:  # Closing long
            realized_pnl = closed_quantity * (trade.price - position.average_price)
        else:  # Closing short
            realized_pnl = closed_quantity * (position.average_price - trade.price)
        
        position.realized_pnl += realized_pnl - trade.commission - trade.slippage
    
    position.quantity = new_quantity
    position.total_commission += trade.commission
    position.total_slippage += trade.slippage
    position.updated_at = trade.executed_at
    
    # Track position size extremes
    position.max_position_size = max(position.max_position_size, abs(new_quantity))
    position.min_position_size = min(position.min_position_size, abs(new_quantity))

def update_positions(self, bar_data: pd.Series) -> None:
    """
    Update all positions with current market prices.
    
    Args:
        bar_data: Current market data
    """
    for symbol, position in self.positions.items():
        if symbol in bar_data.index:
            current_price = bar_data[symbol] if isinstance(bar_data[symbol], (int, float)) else bar_data[symbol].get('close', position.average_price)
            position.update_market_value(current_price)

def process_pending_orders(self, bar_data: pd.Series) -> None:
    """
    Process pending limit and stop orders.
    
    Args:
        bar_data: Current market data
    """
    executed_orders = []
    
    for order in self.pending_orders[:]:  # Copy list to avoid modification during iteration
        should_execute = False
        
        if order.symbol in bar_data.index:
            current_price = bar_data[order.symbol]
            high_price = getattr(current_price, 'high', current_price)
            low_price = getattr(current_price, 'low', current_price)
            
            if order.order_type == OrderType.LIMIT:
                if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                    should_execute = low_price <= order.price
                else:
                    should_execute = high_price >= order.price
            
            elif order.order_type == OrderType.STOP:
                if order.side in [OrderSide.BUY, OrderSide.BUY_TO_COVER]:
                    should_execute = high_price >= order.stop_price
                else:
                    should_execute = low_price <= order.stop_price
            
            if should_execute:
                self.execute_order(order)
                executed_orders.append(order)

def calculate_total_equity(self, bar_data: Optional[pd.Series] = None) -> float:
    """
    Calculate total portfolio equity including cash and positions.
    
    Args:
        bar_data: Current market data for position valuation
        
    Returns:
        Total portfolio equity
    """
    total_equity = self.current_capital
    
    for position in self.positions.values():
        if bar_data is not None and position.symbol in bar_data.index:
            current_price = bar_data[position.symbol]
            if hasattr(current_price, 'close'):
                current_price = current_price.close
            position.update_market_value(current_price)
        
        total_equity += position.market_value
    
    return total_equity

def check_risk_limits(self) -> None:
    """Check various risk limits and take action if violated."""
    current_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
    
    # Daily loss limit
    if self.daily_loss_limit and len(self.equity_curve) > 1:
        daily_pnl = current_equity - self.equity_curve[-2]
        if daily_pnl < -self.daily_loss_limit:
            logger.warning(f"Daily loss limit exceeded: ${daily_pnl:.2f}")
            self.emergency_liquidation()
    
    # Total loss limit
    if self.total_loss_limit:
        total_pnl = current_equity - self.initial_capital
        if total_pnl < -self.total_loss_limit:
            logger.warning(f"Total loss limit exceeded: ${total_pnl:.2f}")
            self.emergency_liquidation()

def emergency_liquidation(self) -> None:
    """Emergency liquidation of all positions."""
    logger.critical("EMERGENCY LIQUIDATION TRIGGERED")
    
    for symbol, position in self.positions.items():
        if not position.is_flat:
            side = OrderSide.SELL if position.is_long else OrderSide.BUY_TO_COVER
            self.place_order(
                symbol=symbol,
                side=side,
                quantity=abs(position.quantity),
                order_type=OrderType.MARKET
            )

def get_performance_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        force_refresh: Force recalculation of cached metrics
        
    Returns:
        Dictionary of performance metrics
    """
    # Check cache validity
    if (not force_refresh and 
        self._cache_timestamp and 
        self._performance_cache and
        (datetime.now() - self._cache_timestamp).seconds < 60):
        return self._performance_cache
    
    if len(self.returns) < 2:
        return {}
    
    returns_series = pd.Series(self.returns)
    equity_series = pd.Series(self.equity_curve)
    
    # Basic metrics
    total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
    annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
    volatility = returns_series.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = RiskMetrics.calculate_sharpe_ratio(returns_series, self.risk_free_rate)
    sortino_ratio = RiskMetrics.calculate_sortino_ratio(returns_series, self.risk_free_rate)
    calmar_ratio = RiskMetrics.calculate_calmar_ratio(returns_series, equity_series)
    
    # Drawdown analysis
    drawdown_metrics = RiskMetrics.calculate_max_drawdown(equity_series)
    
    # Risk metrics
    var_95 = RiskMetrics.calculate_var(returns_series, 0.05)
    cvar_95 = RiskMetrics.calculate_cvar(returns_series, 0.05)
    
    # Trade statistics
    winning_trades = [t for t in self.trades if t.net_proceeds > 0]
    losing_trades = [t for t in self.trades if t.net_proceeds <= 0]
    
    win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
    avg_win = np.mean([t.net_proceeds for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.net_proceeds for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(sum(t.net_proceeds for t in winning_trades) / 
                       sum(t.net_proceeds for t in losing_trades)) if losing_trades else float('inf')
    
    metrics = {
        # Returns
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        
        # Risk
        'max_drawdown': drawdown_metrics['max_drawdown'],
        'max_drawdown_duration': drawdown_metrics['max_drawdown_duration'],
        'current_drawdown': drawdown_metrics['current_drawdown'],
        'var_95': var_95,
        'cvar_95': cvar_95,
        
        # Trading
        'total_trades': len(self.trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'largest_win': max([t.net_proceeds for t in self.trades]) if self.trades else 0,
        'largest_loss': min([t.net_proceeds for t in self.trades]) if self.trades else 0,
        
        # Portfolio
        'current_equity': equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital,
        'cash': self.current_capital,
        'positions': len([p for p in self.positions.values() if not p.is_flat]),
        'total_commission': sum(t.commission for t in self.trades),
        'total_slippage': sum(t.slippage for t in self.trades),
    }
    
    # Cache results
    self._performance_cache = metrics
    self._cache_timestamp = datetime.now()
    
    return metrics

def generate_tear_sheet(self, save_path: Optional[str] = None) -> str:
    """
    Generate comprehensive strategy tear sheet.
    
    Args:
        save_path: Path to save tear sheet
        
    Returns:
        Tear sheet as formatted string
    """
    metrics = self.get_performance_metrics(force_refresh=True)
    
    tear_sheet = f"""
{'='*80}
ALGOVEDA STRATEGY TEAR SHEET
{'='*80}

Strategy Name: {self.name}
Analysis Period: {self.dates[0].strftime('%Y-%m-%d') if self.dates else 'N/A'} to {self.dates[-1].strftime('%Y-%m-%d') if self.dates else 'N/A'}
Initial Capital: ${self.initial_capital:,.2f}
Final Equity: ${metrics.get('current_equity', 0):,.2f}

{'='*40} RETURNS {'='*40}
Total Return: {metrics.get('total_return', 0)*100:.2f}%
Annualized Return: {metrics.get('annualized_return', 0)*100:.2f}%
Volatility: {metrics.get('volatility', 0)*100:.2f}%
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}
Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}

{'='*40} RISK ANALYSIS {'='*40}
Maximum Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%
Max DD Duration: {metrics.get('max_drawdown_duration', 0)} days
Current Drawdown: {metrics.get('current_drawdown', 0)*100:.2f}%
VaR (95%): {metrics.get('var_95', 0)*100:.2f}%
CVaR (95%): {metrics.get('cvar_95', 0)*100:.2f}%

{'='*40} TRADING STATS {'='*40}
Total Trades: {metrics.get('total_trades', 0)}
Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
Average Win: ${metrics.get('avg_win', 0):.2f}
Average Loss: ${metrics.get('avg_loss', 0):.2f}
Profit Factor: {metrics.get('profit_factor', 0):.2f}
Largest Win: ${metrics.get('largest_win', 0):.2f}
Largest Loss: ${metrics.get('largest_loss', 0):.2f}

{'='*40} PORTFOLIO {'='*40}
Current Cash: ${metrics.get('cash', 0):,.2f}
Active Positions: {metrics.get('positions', 0)}
Total Commission: ${metrics.get('total_commission', 0):,.2f}
Total Slippage: ${metrics.get('total_slippage', 0):,.2f}

{'='*80}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(tear_sheet)
        logger.info(f"Tear sheet saved to {save_path}")
    
    return tear_sheet

def __repr__(self) -> str:
    """String representation of strategy."""
    return f"BaseStrategy(name='{self.name}', equity=${self.equity_curve[-1] if self.equity_curve else self.initial_capital:,.2f})"

def __del__(self):
    """Cleanup when strategy is destroyed."""
    if hasattr(self, 'executor'):
        self.executor.shutdown(wait=True)
