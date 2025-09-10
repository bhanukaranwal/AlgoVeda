"""
Comprehensive Equity Trading Strategies for Backtesting
Multi-factor, momentum, mean-reversion, and statistical arbitrage strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

from .base_strategy import BaseStrategy
from ..risk_management import RiskManager
from ..portfolio import Portfolio
from ..utils import technical_indicators as ti
from ..utils.performance import calculate_sharpe_ratio, calculate_max_drawdown

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class EquitySignal:
    symbol: str
    timestamp: pd.Timestamp
    signal: SignalStrength
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    factors: Optional[Dict[str, float]] = None

class MomentumStrategy(BaseStrategy):
    """
    Multi-timeframe momentum strategy with risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Strategy parameters
        self.lookback_periods = config.get('lookback_periods', [20, 50, 200])
        self.momentum_threshold = config.get('momentum_threshold', 0.02)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.5)
        self.volatility_filter = config.get('volatility_filter', True)
        self.max_volatility = config.get('max_volatility', 0.4)
        self.sector_neutrality = config.get('sector_neutrality', False)
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% max per position
        self.stop_loss_pct = config.get('stop_loss_pct', 0.08)  # 8% stop loss
        self.take_profit_pct = config.get('take_profit_pct', 0.15)  # 15% take profit
        
        # State tracking
        self.momentum_scores = {}
        self.volatility_scores = {}
        self.volume_scores = {}
        
        logger.info(f"Initialized MomentumStrategy with lookback periods: {self.lookback_periods}")

    def calculate_momentum_score(self, prices: pd.Series, volumes: pd.Series) -> float:
        """Calculate multi-timeframe momentum score"""
        if len(prices) < max(self.lookback_periods):
            return 0.0
        
        momentum_scores = []
        
        for period in self.lookback_periods:
            if len(prices) >= period:
                # Price momentum
                price_momentum = (prices.iloc[-1] / prices.iloc[-period] - 1)
                
                # Volume-weighted momentum
                recent_volume = volumes.iloc[-period:].mean()
                historical_volume = volumes.iloc[-period*3:-period].mean()
                volume_ratio = recent_volume / (historical_volume + 1e-8)
                
                # Adjust momentum by volume
                adjusted_momentum = price_momentum * min(volume_ratio, 3.0)
                momentum_scores.append(adjusted_momentum)
        
        # Weighted average (shorter periods get higher weights)
        weights = np.array([1/period for period in self.lookback_periods])
        weights = weights / weights.sum()
        
        return np.average(momentum_scores, weights=weights)

    def calculate_volatility_score(self, returns: pd.Series) -> float:
        """Calculate volatility-adjusted score"""
        if len(returns) < 20:
            return 0.0
        
        # Use different volatility measures
        historical_vol = returns.std() * np.sqrt(252)
        garch_vol = self._estimate_garch_volatility(returns)
        
        # Penalize high volatility
        vol_score = 1.0 / (1.0 + historical_vol * 2.0)
        
        return vol_score

    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """Simple GARCH(1,1) volatility estimation"""
        if len(returns) < 30:
            return returns.std()
        
        # Simple EWMA volatility as GARCH proxy
        lambda_param = 0.94
        var_ewma = returns.var()
        
        for i in range(1, len(returns)):
            var_ewma = lambda_param * var_ewma + (1 - lambda_param) * returns.iloc[i]**2
        
        return np.sqrt(var_ewma * 252)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[EquitySignal]:
        """Generate trading signals for all symbols"""
        signals = []
        
        for symbol, data in market_data.items():
            try:
                signal = self._generate_symbol_signal(symbol, data)
                if signal and signal.signal != SignalStrength.HOLD:
                    signals.append(signal)
                    
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Apply sector neutrality if enabled
        if self.sector_neutrality:
            signals = self._apply_sector_neutrality(signals, market_data)
        
        return signals

    def _generate_symbol_signal(self, symbol: str, data: pd.DataFrame) -> Optional[EquitySignal]:
        """Generate signal for individual symbol"""
        if len(data) < max(self.lookback_periods) + 50:
            return None
        
        prices = data['close']
        volumes = data['volume']
        returns = prices.pct_change().dropna()
        
        # Calculate component scores
        momentum_score = self.calculate_momentum_score(prices, volumes)
        volatility_score = self.calculate_volatility_score(returns)
        
        # Volume analysis
        avg_volume_20 = volumes.rolling(20).mean()
        volume_ratio = volumes.iloc[-1] / avg_volume_20.iloc[-1]
        volume_score = min(volume_ratio / self.min_volume_ratio, 2.0)
        
        # Technical indicators
        rsi = ti.calculate_rsi(prices, 14).iloc[-1]
        bb_position = ti.calculate_bollinger_position(prices, 20).iloc[-1]
        macd_signal = ti.calculate_macd_signal(prices, 12, 26, 9).iloc[-1]
        
        # Composite score
        technical_score = self._calculate_technical_score(rsi, bb_position, macd_signal)
        
        # Final signal strength
        signal_strength = (momentum_score * 0.4 + 
                         technical_score * 0.3 + 
                         volume_score * 0.2 + 
                         volatility_score * 0.1)
        
        # Apply volatility filter
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        if self.volatility_filter and current_vol > self.max_volatility:
            signal_strength *= 0.5  # Reduce signal strength for high volatility stocks
        
        # Determine signal direction and strength
        signal = SignalStrength.HOLD
        confidence = abs(signal_strength)
        
        if signal_strength > self.momentum_threshold:
            signal = SignalStrength.STRONG_BUY if signal_strength > self.momentum_threshold * 2 else SignalStrength.BUY
        elif signal_strength < -self.momentum_threshold:
            signal = SignalStrength.STRONG_SELL if signal_strength < -self.momentum_threshold * 2 else SignalStrength.SELL
        
        # Calculate target and stop prices
        current_price = prices.iloc[-1]
        target_price = None
        stop_loss = None
        
        if signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]:
            target_price = current_price * (1 + self.take_profit_pct)
            stop_loss = current_price * (1 - self.stop_loss_pct)
        elif signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL]:
            target_price = current_price * (1 - self.take_profit_pct)
            stop_loss = current_price * (1 + self.stop_loss_pct)
        
        return EquitySignal(
            symbol=symbol,
            timestamp=data.index[-1],
            signal=signal,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return=signal_strength,
            risk_score=current_vol,
            factors={
                'momentum': momentum_score,
                'technical': technical_score,
                'volume': volume_score,
                'volatility': volatility_score
            }
        )

    def _calculate_technical_score(self, rsi: float, bb_position: float, macd_signal: float) -> float:
        """Calculate technical indicator composite score"""
        # RSI score (oversold/overbought)
        rsi_score = 0.0
        if rsi < 30:
            rsi_score = (30 - rsi) / 30  # Bullish when oversold
        elif rsi > 70:
            rsi_score = -(rsi - 70) / 30  # Bearish when overbought
        
        # Bollinger Band position score
        bb_score = 0.0
        if bb_position < 0.2:
            bb_score = (0.2 - bb_position) * 2.5  # Bullish near lower band
        elif bb_position > 0.8:
            bb_score = -(bb_position - 0.8) * 2.5  # Bearish near upper band
        
        # MACD signal score
        macd_score = np.tanh(macd_signal * 100)  # Normalize MACD signal
        
        # Weighted combination
        return (rsi_score * 0.4 + bb_score * 0.3 + macd_score * 0.3)

    def _apply_sector_neutrality(self, signals: List[EquitySignal], 
                                market_data: Dict[str, pd.DataFrame]) -> List[EquitySignal]:
        """Apply sector neutrality by balancing long/short signals per sector"""
        # This would require sector classification data
        # Simplified implementation
        
        buy_signals = [s for s in signals if s.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL]]
        
        # Sort by confidence and limit to top signals
        buy_signals.sort(key=lambda x: x.confidence, reverse=True)
        sell_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        max_signals_per_side = min(len(buy_signals), len(sell_signals), 20)
        
        return buy_signals[:max_signals_per_side] + sell_signals[:max_signals_per_side]

class MeanReversionStrategy(BaseStrategy):
    """
    Statistical mean reversion strategy with pairs trading elements
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.lookback_window = config.get('lookback_window', 60)
        self.entry_threshold = config.get('entry_threshold', 2.0)  # Z-score threshold
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self.max_holding_period = config.get('max_holding_period', 20)  # Days
        self.min_observations = config.get('min_observations', 100)
        
        # Statistical tests
        self.adf_pvalue_threshold = config.get('adf_pvalue_threshold', 0.05)
        self.half_life_max = config.get('half_life_max', 30)  # Days
        
        # Tracking
        self.mean_reversion_stats = {}
        self.positions_entered = {}

    def calculate_mean_reversion_signal(self, prices: pd.Series) -> Tuple[float, dict]:
        """Calculate mean reversion signal with statistical tests"""
        if len(prices) < self.min_observations:
            return 0.0, {}
        
        # Calculate rolling statistics
        rolling_mean = prices.rolling(self.lookback_window).mean()
        rolling_std = prices.rolling(self.lookback_window).std()
        
        # Z-score
        z_score = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        # Stationarity test (simplified ADF test proxy)
        returns = prices.pct_change().dropna()
        is_stationary = self._test_stationarity(returns.iloc[-self.min_observations:])
        
        # Half-life of mean reversion
        half_life = self._calculate_half_life(prices.iloc[-self.min_observations:])
        
        # Hurst exponent (measure of mean reversion vs momentum)
        hurst = self._calculate_hurst_exponent(prices.iloc[-self.min_observations:])
        
        stats_dict = {
            'z_score': z_score,
            'is_stationary': is_stationary,
            'half_life': half_life,
            'hurst_exponent': hurst,
            'rolling_mean': rolling_mean.iloc[-1],
            'rolling_std': rolling_std.iloc[-1]
        }
        
        # Signal strength based on statistical evidence
        if not is_stationary or half_life > self.half_life_max:
            return 0.0, stats_dict
        
        # Mean reversion signal (opposite to momentum)
        signal = -np.tanh(z_score / 2.0)  # Normalize between -1 and 1
        
        # Adjust by statistical confidence
        confidence_multiplier = (1.0 - hurst) * (1.0 if half_life < 10 else 0.5)
        signal *= confidence_multiplier
        
        return signal, stats_dict

    def _test_stationarity(self, returns: pd.Series) -> bool:
        """Simplified stationarity test"""
        # Use variance ratio test as ADF proxy
        n = len(returns)
        if n < 30:
            return False
        
        # Calculate variance ratios for different periods
        var_ratios = []
        for k in [2, 4, 8]:
            if n > k * 10:
                var_k = returns.rolling(k).sum().var() / k
                var_1 = returns.var()
                var_ratio = var_k / var_1 if var_1 > 0 else 1.0
                var_ratios.append(abs(var_ratio - 1.0))
        
        # If variance ratios are close to 1, series is likely stationary
        return np.mean(var_ratios) < 0.3 if var_ratios else False

    def _calculate_half_life(self, prices: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        if len(prices) < 20:
            return float('inf')
        
        # Use AR(1) model to estimate half-life
        returns = prices.pct_change().dropna()
        lagged_returns = returns.shift(1).dropna()
        current_returns = returns[1:]
        
        if len(current_returns) < 10:
            return float('inf')
        
        # Linear regression: r_t = alpha + beta * r_{t-1} + epsilon
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(lagged_returns, current_returns)
            
            # Half-life = -log(2) / log(beta)
            if slope > 0 and slope < 1:
                half_life = -np.log(2) / np.log(slope)
                return max(1, min(half_life, 365))  # Cap between 1 day and 1 year
            else:
                return float('inf')
                
        except Exception:
            return float('inf')

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(prices) < 50:
            return 0.5
        
        log_prices = np.log(prices)
        n = len(log_prices)
        
        # Calculate different lag periods
        lags = range(2, min(n//4, 100))
        rs_values = []
        
        for lag in lags:
            # Split series into non-overlapping periods
            num_periods = n // lag
            rs_period = []
            
            for i in range(num_periods):
                start_idx = i * lag
                end_idx = (i + 1) * lag
                period_data = log_prices.iloc[start_idx:end_idx]
                
                if len(period_data) < lag:
                    continue
                
                # Calculate mean
                mean_val = period_data.mean()
                
                # Calculate cumulative deviations
                cum_devs = (period_data - mean_val).cumsum()
                
                # Calculate range
                r_val = cum_devs.max() - cum_devs.min()
                
                # Calculate standard deviation
                s_val = period_data.std()
                
                if s_val > 0:
                    rs_period.append(r_val / s_val)
            
            if rs_period:
                rs_values.append(np.mean(rs_period))
            else:
                rs_values.append(1.0)
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression of log(R/S) vs log(lag)
        try:
            log_lags = np.log(list(lags[:len(rs_values)]))
            log_rs = np.log(rs_values)
            
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            return max(0.0, min(slope, 1.0))  # Clamp between 0 and 1
            
        except Exception:
            return 0.5

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[EquitySignal]:
        """Generate mean reversion signals"""
        signals = []
        
        for symbol, data in market_data.items():
            try:
                if len(data) < self.min_observations:
                    continue
                
                prices = data['close']
                signal_strength, stats = self.calculate_mean_reversion_signal(prices)
                
                if abs(signal_strength) < self.exit_threshold:
                    continue
                
                # Determine signal direction
                signal = SignalStrength.HOLD
                if signal_strength > self.entry_threshold:
                    signal = SignalStrength.BUY
                elif signal_strength < -self.entry_threshold:
                    signal = SignalStrength.SELL
                elif abs(signal_strength) > self.exit_threshold:
                    # Check if we have existing position to exit
                    if symbol in self.positions_entered:
                        if self.positions_entered[symbol]['signal'] > 0 and signal_strength < 0:
                            signal = SignalStrength.SELL
                        elif self.positions_entered[symbol]['signal'] < 0 and signal_strength > 0:
                            signal = SignalStrength.BUY
                
                if signal != SignalStrength.HOLD:
                    current_price = prices.iloc[-1]
                    
                    # Calculate target and stop based on statistical levels
                    target_price = stats['rolling_mean']  # Target is the mean
                    
                    # Stop loss at 3 standard deviations
                    if signal == SignalStrength.BUY:
                        stop_loss = current_price - 3 * stats['rolling_std']
                    else:
                        stop_loss = current_price + 3 * stats['rolling_std']
                    
                    signal_obj = EquitySignal(
                        symbol=symbol,
                        timestamp=data.index[-1],
                        signal=signal,
                        confidence=abs(signal_strength),
                        target_price=target_price,
                        stop_loss=stop_loss,
                        expected_return=signal_strength * stats['rolling_std'] / current_price,
                        risk_score=1.0 - stats.get('hurst_exponent', 0.5),
                        factors={
                            'z_score': stats['z_score'],
                            'half_life': stats['half_life'],
                            'hurst_exponent': stats['hurst_exponent'],
                            'mean_reversion_strength': abs(signal_strength)
                        }
                    )
                    
                    signals.append(signal_obj)
                    
                    # Track position entry
                    if signal in [SignalStrength.BUY, SignalStrength.SELL]:
                        self.positions_entered[symbol] = {
                            'signal': 1 if signal == SignalStrength.BUY else -1,
                            'entry_date': data.index[-1],
                            'entry_price': current_price
                        }
                        
            except Exception as e:
                logger.warning(f"Error in mean reversion signal for {symbol}: {e}")
                continue
        
        return signals

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy using PCA and cointegration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.min_correlation = config.get('min_correlation', 0.7)
        self.lookback_period = config.get('lookback_period', 252)  # 1 year
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self.max_pairs = config.get('max_pairs', 20)
        
        # PCA parameters
        self.n_components = config.get('n_components', 10)
        self.explained_variance_threshold = config.get('explained_variance_threshold', 0.8)
        
        # State tracking
        self.pairs = []
        self.pca_model = None
        self.scaler = StandardScaler()

    def find_pairs(self, market_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """Find cointegrated pairs using statistical tests"""
        symbols = list(market_data.keys())
        if len(symbols) < 2:
            return []
        
        # Get price data
        price_data = {}
        min_length = float('inf')
        
        for symbol in symbols:
            if len(market_data[symbol]) >= self.lookback_period:
                price_data[symbol] = market_data[symbol]['close'].iloc[-self.lookback_period:]
                min_length = min(min_length, len(price_data[symbol]))
        
        if len(price_data) < 2 or min_length < self.lookback_period * 0.8:
            return []
        
        # Align all series to same length
        for symbol in price_data:
            price_data[symbol] = price_data[symbol].iloc[-min_length:]
        
        # Create correlation matrix
        price_df = pd.DataFrame(price_data)
        correlation_matrix = price_df.corr()
        
        # Find highly correlated pairs
        pairs = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = correlation_matrix.loc[symbol1, symbol2]
                
                if abs(correlation) >= self.min_correlation:
                    # Test for cointegration
                    if self._test_cointegration(price_data[symbol1], price_data[symbol2]):
                        pairs.append((symbol1, symbol2))
        
        return pairs[:self.max_pairs]

    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Test for cointegration between two series"""
        # Simplified cointegration test using correlation of differences
        try:
            # Calculate log prices
            log_s1 = np.log(series1)
            log_s2 = np.log(series2)
            
            # Linear regression to find cointegrating relationship
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_s1, log_s2)
            
            # Calculate spread
            spread = log_s2 - slope * log_s1 - intercept
            
            # Test if spread is stationary (simplified)
            # Use variance ratio test
            spread_returns = spread.diff().dropna()
            if len(spread_returns) < 30:
                return False
            
            # Calculate variance ratio for lag 2
            var_2 = spread_returns.rolling(2).sum().var() / 2
            var_1 = spread_returns.var()
            
            if var_1 <= 0:
                return False
            
            variance_ratio = var_2 / var_1
            
            # If variance ratio close to 1, spread is likely stationary
            return abs(variance_ratio - 1.0) < 0.3 and abs(r_value) > 0.5
            
        except Exception:
            return False

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[EquitySignal]:
        """Generate statistical arbitrage signals"""
        signals = []
        
        # Update pairs periodically
        if not self.pairs or len(self.pairs) == 0:
            self.pairs = self.find_pairs(market_data)
            logger.info(f"Found {len(self.pairs)} cointegrated pairs")
        
        # Generate signals for each pair
        for symbol1, symbol2 in self.pairs:
            try:
                if symbol1 not in market_data or symbol2 not in market_data:
                    continue
                
                pair_signals = self._generate_pair_signals(
                    symbol1, symbol2, 
                    market_data[symbol1], market_data[symbol2]
                )
                signals.extend(pair_signals)
                
            except Exception as e:
                logger.warning(f"Error generating signals for pair {symbol1}-{symbol2}: {e}")
                continue
        
        return signals

    def _generate_pair_signals(self, symbol1: str, symbol2: str, 
                              data1: pd.DataFrame, data2: pd.DataFrame) -> List[EquitySignal]:
        """Generate signals for a specific pair"""
        if len(data1) < self.lookback_period or len(data2) < self.lookback_period:
            return []
        
        # Get aligned price series
        prices1 = data1['close'].iloc[-self.lookback_period:]
        prices2 = data2['close'].iloc[-self.lookback_period:]
        
        # Ensure same length
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1.iloc[-min_len:]
        prices2 = prices2.iloc[-min_len:]
        
        # Calculate spread
        log_prices1 = np.log(prices1)
        log_prices2 = np.log(prices2)
        
        # Rolling cointegration
        window = min(60, len(prices1) // 4)
        spreads = []
        hedge_ratios = []
        
        for i in range(window, len(log_prices1)):
            window_p1 = log_prices1.iloc[i-window:i]
            window_p2 = log_prices2.iloc[i-window:i]
            
            # Calculate hedge ratio
            slope, intercept, _, _, _ = stats.linregress(window_p1, window_p2)
            hedge_ratios.append(slope)
            
            # Calculate spread
            spread = window_p2.iloc[-1] - slope * window_p1.iloc[-1] - intercept
            spreads.append(spread)
        
        if len(spreads) < 20:
            return []
        
        spreads = pd.Series(spreads, index=log_prices1.index[-len(spreads):])
        
        # Calculate z-score of spread
        spread_mean = spreads.rolling(20).mean()
        spread_std = spreads.rolling(20).std()
        z_score = (spreads - spread_mean) / spread_std
        
        current_z = z_score.iloc[-1]
        current_hedge_ratio = hedge_ratios[-1]
        
        signals = []
        
        # Generate signals based on z-score
        if abs(current_z) > self.entry_threshold:
            current_price1 = prices1.iloc[-1]
            current_price2 = prices2.iloc[-1]
            
            if current_z > self.entry_threshold:
                # Spread is too high: short symbol2, long symbol1
                signal1 = EquitySignal(
                    symbol=symbol1,
                    timestamp=data1.index[-1],
                    signal=SignalStrength.BUY,
                    confidence=min(abs(current_z) / 5.0, 1.0),
                    target_price=current_price1 * (1 + spread_std.iloc[-1]),
                    stop_loss=current_price1 * (1 - 3 * spread_std.iloc[-1]),
                    expected_return=spread_std.iloc[-1],
                    factors={
                        'pair_symbol': symbol2,
                        'z_score': current_z,
                        'hedge_ratio': current_hedge_ratio,
                        'strategy_type': 'statistical_arbitrage'
                    }
                )
                
                signal2 = EquitySignal(
                    symbol=symbol2,
                    timestamp=data2.index[-1],
                    signal=SignalStrength.SELL,
                    confidence=min(abs(current_z) / 5.0, 1.0),
                    target_price=current_price2 * (1 - spread_std.iloc[-1]),
                    stop_loss=current_price2 * (1 + 3 * spread_std.iloc[-1]),
                    expected_return=-spread_std.iloc[-1],
                    factors={
                        'pair_symbol': symbol1,
                        'z_score': current_z,
                        'hedge_ratio': 1.0 / current_hedge_ratio,
                        'strategy_type': 'statistical_arbitrage'
                    }
                )
                
                signals.extend([signal1, signal2])
                
            elif current_z < -self.entry_threshold:
                # Spread is too low: long symbol2, short symbol1
                signal1 = EquitySignal(
                    symbol=symbol1,
                    timestamp=data1.index[-1],
                    signal=SignalStrength.SELL,
                    confidence=min(abs(current_z) / 5.0, 1.0),
                    target_price=current_price1 * (1 - spread_std.iloc[-1]),
                    stop_loss=current_price1 * (1 + 3 * spread_std.iloc[-1]),
                    expected_return=-spread_std.iloc[-1],
                    factors={
                        'pair_symbol': symbol2,
                        'z_score': current_z,
                        'hedge_ratio': current_hedge_ratio,
                        'strategy_type': 'statistical_arbitrage'
                    }
                )
                
                signal2 = EquitySignal(
                    symbol=symbol2,
                    timestamp=data2.index[-1],
                    signal=SignalStrength.BUY,
                    confidence=min(abs(current_z) / 5.0, 1.0),
                    target_price=current_price2 * (1 + spread_std.iloc[-1]),
                    stop_loss=current_price2 * (1 - 3 * spread_std.iloc[-1]),
                    expected_return=spread_std.iloc[-1],
                    factors={
                        'pair_symbol': symbol1,
                        'z_score': current_z,
                        'hedge_ratio': 1.0 / current_hedge_ratio,
                        'strategy_type': 'statistical_arbitrage'
                    }
                )
                
                signals.extend([signal1, signal2])
        
        return signals
