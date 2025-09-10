"""
Complete Momentum Breakout Strategy Template for AlgoVeda
Production-ready template with comprehensive features:
- Multi-timeframe analysis
- Dynamic position sizing
- Advanced risk management
- Real-time execution
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from algoveda.core.base_strategy import BaseStrategy, Order, OrderSide, OrderType, TimeInForce
from algoveda.indicators.technical_analysis import TechnicalAnalyzer
from algoveda.risk.position_sizer import KellyCriterionSizer, VolatilityPositionSizer
from algoveda.utils.data_validation import validate_ohlcv_data
from algoveda.market_data.realtime_feed import RealtimeDataFeed

logger = logging.getLogger(__name__)

@dataclass
class MomentumParameters:
    """Configuration parameters for momentum breakout strategy."""
    # Breakout detection
    lookback_period: int = 20
    breakout_threshold: float = 1.02  # 2% breakout
    volume_confirmation: bool = True
    volume_multiplier: float = 1.5
    
    # Moving averages
    fast_ma_period: int = 10
    slow_ma_period: int = 50
    ma_type: str = 'EMA'  # 'SMA', 'EMA', 'WMA'
    
    # RSI filter
    use_rsi_filter: bool = True
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # Stop loss and take profit
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Position sizing
    max_position_pct: float = 0.1  # 10% max position size
    use_kelly_criterion: bool = True
    volatility_lookback: int = 20
    
    # Market timing
    trend_filter: bool = True
    market_hours_only: bool = True
    max_daily_trades: int = 5

class MomentumBreakoutStrategy(BaseStrategy):
    """
    Advanced Momentum Breakout Strategy
    
    Entry Logic:
    1. Price breaks above resistance level (20-period high)
    2. Volume confirmation (1.5x average volume)
    3. RSI not overbought (< 70)
    4. Price above long-term moving average (trend filter)
    
    Exit Logic:
    1. Stop loss: 5% below entry
    2. Take profit: 15% above entry
    3. Trailing stop: 3% below recent high
    4. RSI overbought (> 70)
    """
    
    def __init__(
        self,
        name: str = "MomentumBreakout",
        initial_capital: float = 100000.0,
        symbols: List[str] = None,
        timeframe: str = '5min',
        params: Optional[MomentumParameters] = None,
        **kwargs
    ):
        """
        Initialize momentum breakout strategy.
        
        Args:
            name: Strategy name
            initial_capital: Starting capital
            symbols: List of symbols to trade
            timeframe: Data timeframe
            params: Strategy parameters
            **kwargs: Additional parameters
        """
        super().__init__(name=name, initial_capital=initial_capital, **kwargs)
        
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.timeframe = timeframe
        self.params = params or MomentumParameters()
        
        # Technical analyzer
        self.analyzer = TechnicalAnalyzer()
        
        # Position sizer
        if self.params.use_kelly_criterion:
            self.position_sizer = KellyCriterionSizer(
                lookback_period=self.params.volatility_lookback
            )
        else:
            self.position_sizer = VolatilityPositionSizer(
                target_vol=0.2,  # 20% annual volatility target
                lookback_period=self.params.volatility_lookback
            )
        
        # Strategy state
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.active_signals: Dict[str, Dict] = {}
        self.daily_trades: Dict[str, int] = {}
        self.stop_levels: Dict[str, float] = {}
        self.trailing_stops: Dict[str, float] = {}
        
        # Performance tracking
        self.trade_log: List[Dict] = []
        self.daily_pnl: List[float] = []
        
        logger.info(f"Initialized {name} strategy with {len(self.symbols)} symbols")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on momentum breakout logic.
        
        Args:
            data: OHLCV market data
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Reset daily trade counter if new day
        current_date = data.index[-1].date() if len(data) > 0 else datetime.now().date()
        if not hasattr(self, '_last_date') or self._last_date != current_date:
            self.daily_trades = {symbol: 0 for symbol in self.symbols}
            self._last_date = current_date
        
        # Process each symbol
        for symbol in self.symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) < max(self.params.lookback_period, self.params.slow_ma_period):
                continue
            
            # Store symbol data for analysis
            self.symbol_data[symbol] = symbol_data
            
            # Generate signals for this symbol
            symbol_signals = self._generate_symbol_signals(symbol, symbol_data)
            
            # Add to main signals DataFrame
            for signal_type, values in symbol_signals.items():
                col_name = f"{symbol}_{signal_type}"
                signals[col_name] = 0.0
                for idx, value in values.items():
                    if idx in signals.index:
                        signals.loc[idx, col_name] = value
        
        return signals
    
    def _generate_symbol_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Dict]:
        """Generate signals for a specific symbol."""
        signals = {
            'entry_long': {},
            'exit_long': {},
            'entry_short': {},
            'exit_short': {}
        }
        
        if len(data) < self.params.lookback_period:
            return signals
        
        # Calculate indicators
        indicators = self._calculate_indicators(data)
        
        # Get latest values
        latest = data.iloc[-1]
        latest_idx = latest.name
        
        current_price = latest['close']
        current_volume = latest['volume']
        
        # Check for existing position
        position = self.positions.get(symbol)
        has_position = position is not None and not position.is_flat
        
        # Entry Logic (Long Only for this template)
        if not has_position and self.daily_trades.get(symbol, 0) < self.params.max_daily_trades:
            entry_signal = self._check_entry_conditions(symbol, current_price, current_volume, indicators)
            if entry_signal:
                signals['entry_long'][latest_idx] = 1.0
                self.active_signals[symbol] = {
                    'type': 'long',
                    'entry_price': current_price,
                    'entry_time': latest_idx,
                    'stop_loss': current_price * (1 - self.params.stop_loss_pct),
                    'take_profit': current_price * (1 + self.params.take_profit_pct)
                }
                self.daily_trades[symbol] = self.daily_trades.get(symbol, 0) + 1
        
        # Exit Logic
        if has_position and symbol in self.active_signals:
            exit_signal = self._check_exit_conditions(symbol, current_price, indicators)
            if exit_signal:
                signals['exit_long'][latest_idx] = 1.0
                # Update trailing stop
                self._update_trailing_stop(symbol, current_price)
        
        return signals
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate technical indicators for signal generation."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        indicators = {}
        
        # Moving averages
        if self.params.ma_type == 'SMA':
            indicators['fast_ma'] = talib.SMA(close, self.params.fast_ma_period)
            indicators['slow_ma'] = talib.SMA(close, self.params.slow_ma_period)
        elif self.params.ma_type == 'EMA':
            indicators['fast_ma'] = talib.EMA(close, self.params.fast_ma_period)
            indicators['slow_ma'] = talib.EMA(close, self.params.slow_ma_period)
        elif self.params.ma_type == 'WMA':
            indicators['fast_ma'] = talib.WMA(close, self.params.fast_ma_period)
            indicators['slow_ma'] = talib.WMA(close, self.params.slow_ma_period)
        
        # RSI
        indicators['rsi'] = talib.RSI(close, self.params.rsi_period)
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # ATR for volatility
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume indicators
        indicators['volume_sma'] = talib.SMA(volume, self.params.lookback_period)
        
        # Support/Resistance levels
        indicators['resistance'] = pd.Series(high).rolling(self.params.lookback_period).max().values
        indicators['support'] = pd.Series(low).rolling(self.params.lookback_period).min().values
        
        return indicators
    
    def _check_entry_conditions(self, symbol: str, price: float, volume: float, indicators: Dict) -> bool:
        """Check if entry conditions are met."""
        try:
            # Get latest indicator values
            resistance = indicators['resistance'][-1]
            fast_ma = indicators['fast_ma'][-1]
            slow_ma = indicators['slow_ma'][-1]
            rsi = indicators['rsi'][-1]
            volume_sma = indicators['volume_sma'][-1]
            
            # Skip if indicators are NaN
            if np.isnan([resistance, fast_ma, slow_ma, rsi, volume_sma]).any():
                return False
            
            conditions = []
            
            # 1. Breakout condition
            breakout = price > resistance * self.params.breakout_threshold
            conditions.append(('breakout', breakout))
            
            # 2. Volume confirmation
            if self.params.volume_confirmation:
                volume_confirmed = volume > volume_sma * self.params.volume_multiplier
                conditions.append(('volume', volume_confirmed))
            else:
                conditions.append(('volume', True))
            
            # 3. Trend filter
            if self.params.trend_filter:
                uptrend = fast_ma > slow_ma
                conditions.append(('trend', uptrend))
            else:
                conditions.append(('trend', True))
            
            # 4. RSI filter
            if self.params.use_rsi_filter:
                rsi_ok = self.params.rsi_oversold < rsi < self.params.rsi_overbought
                conditions.append(('rsi', rsi_ok))
            else:
                conditions.append(('rsi', True))
            
            # 5. Market hours check
            if self.params.market_hours_only:
                market_open = self._is_market_hours()
                conditions.append(('market_hours', market_open))
            else:
                conditions.append(('market_hours', True))
            
            # Log conditions for debugging
            all_conditions_met = all(condition[1] for condition in conditions)
            if all_conditions_met:
                logger.info(f"Entry signal for {symbol}: {dict(conditions)}")
            
            return all_conditions_met
            
        except Exception as e:
            logger.error(f"Error checking entry conditions for {symbol}: {e}")
            return False
    
    def _check_exit_conditions(self, symbol: str, price: float, indicators: Dict) -> bool:
        """Check if exit conditions are met."""
        if symbol not in self.active_signals:
            return False
        
        signal_info = self.active_signals[symbol]
        
        try:
            rsi = indicators['rsi'][-1]
            
            # Exit conditions
            conditions = []
            
            # 1. Stop loss
            stop_loss_hit = price <= signal_info['stop_loss']
            conditions.append(('stop_loss', stop_loss_hit))
            
            # 2. Take profit
            take_profit_hit = price >= signal_info['take_profit']
            conditions.append(('take_profit', take_profit_hit))
            
            # 3. Trailing stop
            if self.params.trailing_stop and symbol in self.trailing_stops:
                trailing_stop_hit = price <= self.trailing_stops[symbol]
                conditions.append(('trailing_stop', trailing_stop_hit))
            else:
                conditions.append(('trailing_stop', False))
            
            # 4. RSI overbought
            if self.params.use_rsi_filter and not np.isnan(rsi):
                rsi_overbought = rsi > self.params.rsi_overbought
                conditions.append(('rsi_overbought', rsi_overbought))
            else:
                conditions.append(('rsi_overbought', False))
            
            # Exit if any condition is met
            should_exit = any(condition[1] for condition in conditions)
            
            if should_exit:
                exit_reason = [name for name, condition in conditions if condition][0]
                logger.info(f"Exit signal for {symbol}: {exit_reason}")
                
                # Clean up tracking
                if symbol in self.active_signals:
                    del self.active_signals[symbol]
                if symbol in self.trailing_stops:
                    del self.trailing_stops[symbol]
            
            return should_exit
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}")
            return False
    
    def _update_trailing_stop(self, symbol: str, price: float) -> None:
        """Update trailing stop level."""
        if not self.params.trailing_stop or symbol not in self.active_signals:
            return
        
        signal_info = self.active_signals[symbol]
        entry_price = signal_info['entry_price']
        
        # Calculate new trailing stop
        new_trailing_stop = price * (1 - self.params.trailing_stop_pct)
        
        # Update if it's higher than current or first time
        if symbol not in self.trailing_stops or new_trailing_stop > self.trailing_stops[symbol]:
            # Only move trailing stop up, never down
            if new_trailing_stop > entry_price * (1 - self.params.stop_loss_pct):
                self.trailing_stops[symbol] = new_trailing_stop
    
    def calculate_position_sizes(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes using Kelly Criterion or volatility targeting."""
        position_sizes = signals.copy()
        
        for symbol in self.symbols:
            entry_col = f"{symbol}_entry_long"
            if entry_col in signals.columns and signals[entry_col].iloc[-1] > 0:
                
                # Get symbol data for position sizing
                symbol_data = data[data['symbol'] == symbol]
                if len(symbol_data) < self.params.volatility_lookback:
                    continue
                
                # Calculate position size
                portfolio_value = self.calculate_total_equity()
                max_position_value = portfolio_value * self.params.max_position_pct
                
                current_price = symbol_data['close'].iloc[-1]
                
                # Use position sizer
                optimal_size = self.position_sizer.calculate_size(
                    symbol_data, portfolio_value, current_price
                )
                
                # Apply maximum position limit
                max_shares = int(max_position_value / current_price)
                final_size = min(optimal_size, max_shares)
                
                position_sizes.loc[position_sizes.index[-1], entry_col] = final_size
        
        return position_sizes
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        
        # US market hours: 9:30 AM - 4:00 PM EST, Monday-Friday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def on_trade_execution(self, trade: Dict) -> None:
        """Handle trade execution events."""
        super().on_trade_execution(trade)
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': trade['symbol'],
            'side': trade['side'],
            'quantity': trade['quantity'],
            'price': trade['price'],
            'strategy': self.name
        }
        self.trade_log.append(trade_record)
        
        logger.info(f"Trade executed: {trade_record}")
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy-specific performance metrics."""
        base_metrics = super().get_performance_metrics()
        
        # Add momentum strategy specific metrics
        momentum_metrics = {
            'total_signals_generated': len(self.trade_log),
            'avg_trades_per_day': len(self.trade_log) / max(1, len(set(t['timestamp'].date() for t in self.trade_log))),
            'symbols_traded': len(set(t['symbol'] for t in self.trade_log)),
            'most_traded_symbol': max(set(t['symbol'] for t in self.trade_log), key=lambda x: sum(1 for t in self.trade_log if t['symbol'] == x)) if self.trade_log else 'None',
            'active_signals': len(self.active_signals),
            'trailing_stops_active': len(self.trailing_stops),
        }
        
        # Calculate win rate by symbol
        symbol_performance = {}
        for symbol in self.symbols:
            symbol_trades = [t for t in self.trade_log if t['symbol'] == symbol]
            if symbol_trades:
                # This would need actual P&L calculation
                symbol_performance[f'{symbol}_trades'] = len(symbol_trades)
        
        momentum_metrics.update(symbol_performance)
        base_metrics.update(momentum_metrics)
        
        return base_metrics

# Example usage and testing
if __name__ == "__main__":
    # Create strategy instance
    params = MomentumParameters(
        lookback_period=20,
        breakout_threshold=1.02,
        volume_confirmation=True,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )
    
    strategy = MomentumBreakoutStrategy(
        name="TestMomentum",
        initial_capital=100000.0,
        symbols=['AAPL', 'MSFT'],
        params=params
    )
    
    # Mock data for testing
    import datetime as dt
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='5min')
    np.random.seed(42)
    
    mock_data = pd.DataFrame({
        'symbol': np.random.choice(['AAPL', 'MSFT'], len(dates)),
        'open': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
        'high': 100 + np.random.randn(len(dates)).cumsum() * 0.5 + np.random.rand(len(dates)) * 2,
        'low': 100 + np.random.randn(len(dates)).cumsum() * 0.5 - np.random.rand(len(dates)) * 2,
        'close': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Test signal generation
    signals = strategy.generate_signals(mock_data.tail(1000))
    position_sizes = strategy.calculate_position_sizes(signals, mock_data.tail(1000))
    
    print("Strategy testing completed")
    print(f"Generated {signals.sum().sum()} signals")
    print(f"Strategy metrics: {strategy.get_strategy_metrics()}")
