"""
Comprehensive Options Trading Strategies for Backtesting
Advanced options strategies with Greeks hedging and volatility trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, optimize
from scipy.stats import norm
import logging
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy
from ..risk_management import OptionsRiskManager
from ..portfolio import OptionsPortfolio
from ..utils import options_pricing as op
from ..utils.greeks import calculate_all_greeks
from ..utils.volatility import estimate_implied_volatility

logger = logging.getLogger(__name__)

class OptionsStrategy(Enum):
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    RATIO_SPREAD = "ratio_spread"
    JADE_LIZARD = "jade_lizard"
    VOLATILITY_ARBITRAGE = "volatility_arbitrage"
    GAMMA_SCALPING = "gamma_scalping"
    DELTA_HEDGING = "delta_hedging"

@dataclass
class OptionsLeg:
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiry: datetime
    quantity: int  # Positive for long, negative for short
    premium: Optional[float] = None
    implied_vol: Optional[float] = None
    greeks: Optional[Dict[str, float]] = None

@dataclass
class OptionsPosition:
    strategy_type: OptionsStrategy
    underlying_symbol: str
    legs: List[OptionsLeg]
    underlying_quantity: int = 0  # For strategies involving stock
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    target_profit: Optional[float] = None
    max_loss: Optional[float] = None
    expected_return: Optional[float] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)

class VolatilityTradingStrategy(BaseStrategy):
    """
    Advanced volatility trading strategies using options
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Strategy parameters
        self.vol_lookback = config.get('vol_lookback', 30)
        self.vol_threshold = config.get('vol_threshold', 0.2)  # 20%
        self.min_time_to_expiry = config.get('min_time_to_expiry', 7)  # Days
        self.max_time_to_expiry = config.get('max_time_to_expiry', 60)  # Days
        self.delta_neutral_threshold = config.get('delta_neutral_threshold', 0.05)
        
        # Risk management
        self.max_vega_exposure = config.get('max_vega_exposure', 1000.0)
        self.max_gamma_exposure = config.get('max_gamma_exposure', 500.0)
        self.profit_target = config.get('profit_target', 0.5)  # 50% of max profit
        self.loss_limit = config.get('loss_limit', 2.0)  # 2x credit received
        
        # Volatility models
        self.vol_model = config.get('vol_model', 'garch')
        self.vol_forecast_horizon = config.get('vol_forecast_horizon', 5)  # Days
        
        # State tracking
        self.current_positions = {}
        self.vol_forecasts = {}
        self.realized_vol_history = {}

    def estimate_realized_volatility(self, prices: pd.Series, window: int = 30) -> float:
        """Estimate realized volatility using close-to-close returns"""
        if len(prices) < window + 1:
            return 0.2  # Default 20% volatility
        
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return 0.2
        
        # Use exponentially weighted moving average
        weights = np.exp(np.linspace(-1, 0, window))
        weights = weights / weights.sum()
        
        recent_returns = returns.iloc[-window:]
        weighted_var = np.average(recent_returns ** 2, weights=weights)
        
        return np.sqrt(weighted_var * 252)  # Annualized

    def forecast_volatility(self, returns: pd.Series, horizon: int = 5) -> float:
        """Forecast volatility using GARCH-like model"""
        if len(returns) < 50:
            return self.estimate_realized_volatility(returns.cumsum().apply(np.exp))
        
        # Simple EWMA forecast
        lambda_param = 0.94
        forecast_var = returns.var()
        
        for i in range(len(returns)):
            forecast_var = lambda_param * forecast_var + (1 - lambda_param) * returns.iloc[i]**2
        
        # Project forward
        return np.sqrt(forecast_var * 252)

    def identify_volatility_opportunities(self, market_data: Dict[str, pd.DataFrame], 
                                        options_data: Dict[str, pd.DataFrame]) -> List[OptionsPosition]:
        """Identify volatility trading opportunities"""
        opportunities = []
        
        for symbol in market_data.keys():
            if symbol not in options_data:
                continue
                
            try:
                stock_data = market_data[symbol]
                option_chain = options_data[symbol]
                
                if len(stock_data) < self.vol_lookback + 10:
                    continue
                
                # Calculate realized and implied volatilities
                realized_vol = self.estimate_realized_volatility(
                    stock_data['close'], self.vol_lookback
                )
                
                # Get ATM implied volatility
                current_price = stock_data['close'].iloc[-1]
                atm_options = self.get_atm_options(option_chain, current_price)
                
                if not atm_options:
                    continue
                
                implied_vol = np.mean([opt['implied_vol'] for opt in atm_options])
                
                # Volatility differential
                vol_diff = implied_vol - realized_vol
                vol_diff_pct = vol_diff / realized_vol
                
                # Forecast volatility
                returns = stock_data['close'].pct_change().dropna()
                forecast_vol = self.forecast_volatility(returns, self.vol_forecast_horizon)
                
                # Identify opportunity type
                if abs(vol_diff_pct) > self.vol_threshold:
                    if vol_diff_pct > self.vol_threshold:
                        # Implied > Realized: Sell volatility
                        position = self.create_short_volatility_position(
                            symbol, current_price, atm_options, stock_data.index[-1]
                        )
                    else:
                        # Implied < Realized: Buy volatility
                        position = self.create_long_volatility_position(
                            symbol, current_price, atm_options, stock_data.index[-1]
                        )
                    
                    if position:
                        position.risk_metrics.update({
                            'realized_vol': realized_vol,
                            'implied_vol': implied_vol,
                            'vol_diff': vol_diff,
                            'vol_diff_pct': vol_diff_pct,
                            'forecast_vol': forecast_vol
                        })
                        opportunities.append(position)
                        
            except Exception as e:
                logger.warning(f"Error analyzing volatility for {symbol}: {e}")
                continue
        
        return opportunities

    def get_atm_options(self, option_chain: pd.DataFrame, spot_price: float) -> List[Dict]:
        """Get at-the-money options"""
        # Filter by time to expiry
        current_date = option_chain.index[-1] if hasattr(option_chain, 'index') else datetime.now()
        
        atm_options = []
        
        for _, option in option_chain.iterrows():
            try:
                time_to_expiry = (option['expiry'] - current_date).days
                
                if (self.min_time_to_expiry <= time_to_expiry <= self.max_time_to_expiry and
                    abs(option['strike'] - spot_price) / spot_price < 0.1):  # Within 10% of ATM
                    
                    atm_options.append({
                        'strike': option['strike'],
                        'expiry': option['expiry'],
                        'option_type': option['type'],
                        'bid': option['bid'],
                        'ask': option['ask'],
                        'implied_vol': option.get('implied_vol', 0.2),
                        'delta': option.get('delta', 0.5),
                        'gamma': option.get('gamma', 0.01),
                        'vega': option.get('vega', 0.1),
                        'theta': option.get('theta', -0.01)
                    })
                    
            except Exception as e:
                continue
        
        return atm_options

    def create_short_volatility_position(self, symbol: str, spot_price: float, 
                                       atm_options: List[Dict], entry_date: datetime) -> Optional[OptionsPosition]:
        """Create short volatility position (short straddle/strangle)"""
        call_options = [opt for opt in atm_options if opt['option_type'] == 'CALL']
        put_options = [opt for opt in atm_options if opt['option_type'] == 'PUT']
        
        if not call_options or not put_options:
            return None
        
        # Select ATM call and put
        atm_call = min(call_options, key=lambda x: abs(x['strike'] - spot_price))
        atm_put = min(put_options, key=lambda x: abs(x['strike'] - spot_price))
        
        # Create short straddle
        legs = [
            OptionsLeg(
                option_type='CALL',
                strike=atm_call['strike'],
                expiry=atm_call['expiry'],
                quantity=-1,  # Short
                premium=(atm_call['bid'] + atm_call['ask']) / 2,
                implied_vol=atm_call['implied_vol'],
                greeks={
                    'delta': -atm_call['delta'],
                    'gamma': -atm_call['gamma'],
                    'vega': -atm_call['vega'],
                    'theta': -atm_call['theta']
                }
            ),
            OptionsLeg(
                option_type='PUT',
                strike=atm_put['strike'],
                expiry=atm_put['expiry'],
                quantity=-1,  # Short
                premium=(atm_put['bid'] + atm_put['ask']) / 2,
                implied_vol=atm_put['implied_vol'],
                greeks={
                    'delta': -atm_put['delta'],
                    'gamma': -atm_put['gamma'],
                    'vega': -atm_put['vega'],
                    'theta': -atm_put['theta']
                }
            )
        ]
        
        # Calculate position metrics
        max_profit = sum(leg.premium for leg in legs)
        max_loss = max(legs[0].strike - max_profit, legs[1].strike - max_profit)
        
        return OptionsPosition(
            strategy_type=OptionsStrategy.STRADDLE,
            underlying_symbol=symbol,
            legs=legs,
            entry_date=entry_date,
            entry_price=spot_price,
            target_profit=max_profit * self.profit_target,
            max_loss=max_loss,
            expected_return=max_profit * 0.3  # Expected 30% of max profit
        )

    def create_long_volatility_position(self, symbol: str, spot_price: float, 
                                      atm_options: List[Dict], entry_date: datetime) -> Optional[OptionsPosition]:
        """Create long volatility position (long straddle/strangle)"""
        call_options = [opt for opt in atm_options if opt['option_type'] == 'CALL']
        put_options = [opt for opt in atm_options if opt['option_type'] == 'PUT']
        
        if not call_options or not put_options:
            return None
        
        # Select ATM call and put
        atm_call = min(call_options, key=lambda x: abs(x['strike'] - spot_price))
        atm_put = min(put_options, key=lambda x: abs(x['strike'] - spot_price))
        
        # Create long straddle
        legs = [
            OptionsLeg(
                option_type='CALL',
                strike=atm_call['strike'],
                expiry=atm_call['expiry'],
                quantity=1,  # Long
                premium=(atm_call['bid'] + atm_call['ask']) / 2,
                implied_vol=atm_call['implied_vol'],
                greeks={
                    'delta': atm_call['delta'],
                    'gamma': atm_call['gamma'],
                    'vega': atm_call['vega'],
                    'theta': atm_call['theta']
                }
            ),
            OptionsLeg(
                option_type='PUT',
                strike=atm_put['strike'],
                expiry=atm_put['expiry'],
                quantity=1,  # Long
                premium=(atm_put['bid'] + atm_put['ask']) / 2,
                implied_vol=atm_put['implied_vol'],
                greeks={
                    'delta': atm_put['delta'],
                    'gamma': atm_put['gamma'],
                    'vega': atm_put['vega'],
                    'theta': atm_put['theta']
                }
            )
        ]
        
        # Calculate position metrics
        max_loss = sum(leg.premium for leg in legs)
        breakeven_up = legs[0].strike + max_loss
        breakeven_down = legs[1].strike - max_loss
        
        return OptionsPosition(
            strategy_type=OptionsStrategy.STRADDLE,
            underlying_symbol=symbol,
            legs=legs,
            entry_date=entry_date,
            entry_price=spot_price,
            target_profit=max_loss * 2.0,  # 2x premium paid
            max_loss=max_loss,
            expected_return=max_loss * 0.5  # Expected 50% of premium
        )

class GammaScalpingStrategy(BaseStrategy):
    """
    Gamma scalping strategy for delta-neutral trading
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.gamma_threshold = config.get('gamma_threshold', 0.01)
        self.delta_rebalance_threshold = config.get('delta_rebalance_threshold', 0.05)
        self.min_stock_move = config.get('min_stock_move', 0.5)  # Minimum % move to rebalance
        self.transaction_cost = config.get('transaction_cost', 0.01)  # 1 cent per share
        self.rebalance_frequency = config.get('rebalance_frequency', 'intraday')  # or 'daily'
        
        # Position tracking
        self.option_positions = {}
        self.stock_positions = {}
        self.last_rebalance_prices = {}

    def calculate_hedge_ratio(self, position: OptionsPosition, current_price: float) -> float:
        """Calculate required stock hedge ratio"""
        total_delta = 0.0
        
        for leg in position.legs:
            # Recalculate delta based on current price
            current_delta = self.calculate_option_delta(
                current_price, leg.strike, leg.expiry, 
                0.05, leg.implied_vol, leg.option_type
            )
            total_delta += leg.quantity * current_delta * 100  # Convert to shares
        
        return -total_delta  # Hedge ratio is negative of delta

    def calculate_option_delta(self, spot: float, strike: float, expiry: datetime, 
                             risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Calculate option delta using Black-Scholes"""
        try:
            time_to_expiry = max((expiry - datetime.now()).days / 365.0, 1/365)
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            
            if option_type.upper() == 'CALL':
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1.0
                
        except Exception:
            return 0.5 if option_type.upper() == 'CALL' else -0.5

    def should_rebalance(self, symbol: str, current_price: float, position: OptionsPosition) -> bool:
        """Determine if position should be rebalanced"""
        if symbol not in self.last_rebalance_prices:
            return True
        
        last_price = self.last_rebalance_prices[symbol]
        price_move = abs(current_price - last_price) / last_price
        
        # Check if price moved enough to warrant rebalancing
        if price_move < self.min_stock_move / 100:
            return False
        
        # Calculate current delta
        current_delta = sum(
            leg.quantity * self.calculate_option_delta(
                current_price, leg.strike, leg.expiry, 0.05, leg.implied_vol, leg.option_type
            ) for leg in position.legs
        )
        
        return abs(current_delta) > self.delta_rebalance_threshold

    def generate_rebalancing_signals(self, market_data: Dict[str, pd.DataFrame], 
                                   current_positions: Dict[str, OptionsPosition]) -> List[Dict]:
        """Generate rebalancing signals for gamma scalping"""
        signals = []
        
        for symbol, position in current_positions.items():
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['close'].iloc[-1]
            
            if not self.should_rebalance(symbol, current_price, position):
                continue
            
            # Calculate required hedge
            required_hedge = self.calculate_hedge_ratio(position, current_price)
            current_stock_position = self.stock_positions.get(symbol, 0)
            
            stock_adjustment = required_hedge - current_stock_position
            
            # Only rebalance if adjustment is significant
            if abs(stock_adjustment) < 10:  # Less than 10 shares
                continue
            
            # Calculate expected profit from gamma scalping
            total_gamma = sum(
                leg.quantity * leg.greeks.get('gamma', 0.01) 
                for leg in position.legs if leg.greeks
            )
            
            price_move = current_price - self.last_rebalance_prices.get(symbol, current_price)
            expected_gamma_pnl = 0.5 * total_gamma * (price_move ** 2) * 100  # Convert to dollars
            
            # Account for transaction costs
            transaction_cost = abs(stock_adjustment) * self.transaction_cost
            net_expected_pnl = expected_gamma_pnl - transaction_cost
            
            if net_expected_pnl > 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY' if stock_adjustment > 0 else 'SELL',
                    'quantity': abs(stock_adjustment),
                    'reason': 'gamma_scalping_rebalance',
                    'expected_pnl': net_expected_pnl,
                    'current_delta': sum(leg.quantity * leg.greeks.get('delta', 0.5) for leg in position.legs),
                    'target_delta': 0.0,
                    'transaction_cost': transaction_cost
                })
                
                # Update tracking
                self.last_rebalance_prices[symbol] = current_price
                self.stock_positions[symbol] = required_hedge
        
        return signals

class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor strategy for range-bound markets
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.wing_width = config.get('wing_width', 5.0)  # Strike width between wings
        self.min_credit = config.get('min_credit', 1.0)  # Minimum net credit
        self.max_days_to_expiry = config.get('max_days_to_expiry', 45)
        self.min_days_to_expiry = config.get('min_days_to_expiry', 15)
        self.profit_target_pct = config.get('profit_target_pct', 0.5)  # 50% of max profit
        self.loss_limit_pct = config.get('loss_limit_pct', 2.0)  # 2x max profit
        
        # Range detection parameters
        self.range_lookback = config.get('range_lookback', 20)
        self.range_threshold = config.get('range_threshold', 0.15)  # 15% range
        self.volatility_percentile_threshold = config.get('volatility_percentile_threshold', 30)

    def detect_range_bound_market(self, prices: pd.Series) -> Tuple[bool, Dict[str, float]]:
        """Detect if market is range-bound"""
        if len(prices) < self.range_lookback:
            return False, {}
        
        recent_prices = prices.iloc[-self.range_lookback:]
        
        # Calculate range statistics
        high = recent_prices.max()
        low = recent_prices.min()
        range_pct = (high - low) / recent_prices.mean()
        
        # Calculate trend strength
        returns = recent_prices.pct_change().dropna()
        trend_strength = abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
        
        # Calculate volatility percentile
        vol_window = min(100, len(prices))
        volatilities = prices.rolling(self.range_lookback).std().dropna()
        if len(volatilities) >= vol_window:
            current_vol = volatilities.iloc[-1]
            vol_percentile = (volatilities.iloc[-vol_window:] < current_vol).mean() * 100
        else:
            vol_percentile = 50
        
        # Range-bound conditions
        is_range_bound = (
            range_pct < self.range_threshold and
            trend_strength < 0.5 and
            vol_percentile < self.volatility_percentile_threshold
        )
        
        stats = {
            'range_pct': range_pct,
            'trend_strength': trend_strength,
            'vol_percentile': vol_percentile,
            'support_level': low,
            'resistance_level': high,
            'range_midpoint': (high + low) / 2
        }
        
        return is_range_bound, stats

    def create_iron_condor(self, symbol: str, current_price: float, 
                          option_chain: pd.DataFrame, entry_date: datetime) -> Optional[OptionsPosition]:
        """Create iron condor position"""
        
        # Filter options by expiry
        suitable_options = []
        for _, option in option_chain.iterrows():
            days_to_expiry = (option['expiry'] - entry_date).days
            if self.min_days_to_expiry <= days_to_expiry <= self.max_days_to_expiry:
                suitable_options.append(option)
        
        if len(suitable_options) < 4:
            return None
        
        suitable_df = pd.DataFrame(suitable_options)
        
        # Select strikes for iron condor
        # Structure: Short PUT (lower), Long PUT (lower-wing), Long CALL (upper+wing), Short CALL (upper)
        
        # Find ATM strikes
        put_strikes = suitable_df[suitable_df['type'] == 'PUT']['strike'].values
        call_strikes = suitable_df[suitable_df['type'] == 'CALL']['strike'].values
        
        # Select short put strike (below current price)
        short_put_candidates = put_strikes[put_strikes < current_price * 0.95]  # 5% OTM
        if len(short_put_candidates) == 0:
            return None
        short_put_strike = max(short_put_candidates)
        
        # Select long put strike (further OTM)
        long_put_strike = short_put_strike - self.wing_width
        if long_put_strike not in put_strikes:
            return None
        
        # Select short call strike (above current price)
        short_call_candidates = call_strikes[call_strikes > current_price * 1.05]  # 5% OTM
        if len(short_call_candidates) == 0:
            return None
        short_call_strike = min(short_call_candidates)
        
        # Select long call strike (further OTM)
        long_call_strike = short_call_strike + self.wing_width
        if long_call_strike not in call_strikes:
            return None
        
        # Get option prices
        try:
            short_put = suitable_df[(suitable_df['strike'] == short_put_strike) & 
                                   (suitable_df['type'] == 'PUT')].iloc[0]
            long_put = suitable_df[(suitable_df['strike'] == long_put_strike) & 
                                  (suitable_df['type'] == 'PUT')].iloc[0]
            short_call = suitable_df[(suitable_df['strike'] == short_call_strike) & 
                                    (suitable_df['type'] == 'CALL')].iloc[0]
            long_call = suitable_df[(suitable_df['strike'] == long_call_strike) & 
                                   (suitable_df['type'] == 'CALL')].iloc[0]
        except IndexError:
            return None
        
        # Create legs
        legs = [
            # Short put (collect premium)
            OptionsLeg(
                option_type='PUT',
                strike=short_put_strike,
                expiry=short_put['expiry'],
                quantity=-1,
                premium=(short_put['bid'] + short_put['ask']) / 2,
                implied_vol=short_put.get('implied_vol', 0.2)
            ),
            # Long put (pay premium)
            OptionsLeg(
                option_type='PUT',
                strike=long_put_strike,
                expiry=long_put['expiry'],
                quantity=1,
                premium=(long_put['bid'] + long_put['ask']) / 2,
                implied_vol=long_put.get('implied_vol', 0.2)
            ),
            # Short call (collect premium)
            OptionsLeg(
                option_type='CALL',
                strike=short_call_strike,
                expiry=short_call['expiry'],
                quantity=-1,
                premium=(short_call['bid'] + short_call['ask']) / 2,
                implied_vol=short_call.get('implied_vol', 0.2)
            ),
            # Long call (pay premium)
            OptionsLeg(
                option_type='CALL',
                strike=long_call_strike,
                expiry=long_call['expiry'],
                quantity=1,
                premium=(long_call['bid'] + long_call['ask']) / 2,
                implied_vol=long_call.get('implied_vol', 0.2)
            )
        ]
        
        # Calculate net credit
        net_credit = sum(leg.premium * (-1 if leg.quantity < 0 else 1) for leg in legs)
        
        if net_credit < self.min_credit:
            return None
        
        # Calculate max profit/loss
        max_profit = net_credit
        max_loss = self.wing_width - net_credit
        
        return OptionsPosition(
            strategy_type=OptionsStrategy.IRON_CONDOR,
            underlying_symbol=symbol,
            legs=legs,
            entry_date=entry_date,
            entry_price=current_price,
            target_profit=max_profit * self.profit_target_pct,
            max_loss=max_loss,
            expected_return=max_profit * 0.4,  # Expected 40% of max profit
            risk_metrics={
                'max_profit': max_profit,
                'max_loss': max_loss,
                'net_credit': net_credit,
                'breakeven_lower': short_put_strike - net_credit,
                'breakeven_upper': short_call_strike + net_credit,
                'profit_range': short_call_strike - short_put_strike + 2 * net_credit
            }
        )

    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        options_data: Dict[str, pd.DataFrame]) -> List[OptionsPosition]:
        """Generate iron condor signals for range-bound markets"""
        signals = []
        
        for symbol in market_data.keys():
            if symbol not in options_data:
                continue
            
            try:
                stock_data = market_data[symbol]
                option_chain = options_data[symbol]
                
                if len(stock_data) < self.range_lookback + 10:
                    continue
                
                current_price = stock_data['close'].iloc[-1]
                
                # Check if market is range-bound
                is_range_bound, range_stats = self.detect_range_bound_market(stock_data['close'])
                
                if not is_range_bound:
                    continue
                
                # Create iron condor position
                position = self.create_iron_condor(
                    symbol, current_price, option_chain, stock_data.index[-1]
                )
                
                if position:
                    position.risk_metrics.update(range_stats)
                    signals.append(position)
                    
                    logger.info(f"Iron Condor signal for {symbol}: "
                              f"Range {range_stats['range_pct']:.1%}, "
                              f"Vol percentile {range_stats['vol_percentile']:.1f}")
                    
            except Exception as e:
                logger.warning(f"Error generating iron condor signal for {symbol}: {e}")
                continue
        
        return signals
