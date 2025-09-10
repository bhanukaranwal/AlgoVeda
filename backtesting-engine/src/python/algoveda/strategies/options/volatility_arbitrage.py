"""
Advanced Volatility Arbitrage Strategy for AlgoVeda
Trades volatility surface inefficiencies with delta-neutral hedging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from algoveda.core.base_strategy import BaseStrategy
from algoveda.calculations.options_pricing.black_scholes import BlackScholes
from algoveda.calculations.greeks.analytical_greeks import GreeksCalculator
from algoveda.risk_management.dynamic_hedging.delta_hedging import DeltaHedger

class VolatilityArbitrageStrategy(BaseStrategy):
    """
    Advanced volatility arbitrage strategy that:
    1. Identifies mispriced options using implied vs realized volatility
    2. Constructs delta-neutral portfolios
    3. Dynamically hedges to maintain neutrality
    4. Captures volatility risk premium
    """
    
    def __init__(self, 
                 lookback_period: int = 252,
                 vol_threshold: float = 0.02,
                 min_time_to_expiry: int = 7,
                 max_position_size: float = 100000,
                 rehedge_threshold: float = 0.1,
                 transaction_cost: float = 0.001):
        super().__init__()
        
        self.lookback_period = lookback_period
        self.vol_threshold = vol_threshold
        self.min_time_to_expiry = min_time_to_expiry
        self.max_position_size = max_position_size
        self.rehedge_threshold = rehedge_threshold
        self.transaction_cost = transaction_cost
        
        # Initialize calculators
        self.bs_calculator = BlackScholes()
        self.greeks_calculator = GreeksCalculator()
        self.delta_hedger = DeltaHedger()
        
        # Strategy state
        self.positions = {}
        self.hedge_positions = {}
        self.volatility_forecasts = {}
        self.last_hedge_time = {}
        
    def calculate_realized_volatility(self, 
                                    prices: pd.Series, 
                                    window: int = None) -> float:
        """Calculate realized volatility using close-to-close returns"""
        if window is None:
            window = min(len(prices), self.lookback_period)
            
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < window:
            return np.nan
            
        recent_returns = returns.tail(window)
        return recent_returns.std() * np.sqrt(252)
    
    def calculate_garch_volatility(self, 
                                 returns: pd.Series, 
                                 horizon: int = 30) -> float:
        """Enhanced volatility forecast using GARCH(1,1) model"""
        try:
            from arch import arch_model
            
            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            
            # Forecast volatility
            forecasts = results.forecast(horizon=horizon)
            vol_forecast = np.sqrt(forecasts.variance.iloc[-1].mean() / 100) * np.sqrt(252)
            
            return vol_forecast
            
        except Exception:
            # Fallback to EWMA
            return self.calculate_ewma_volatility(returns)
    
    def calculate_ewma_volatility(self, 
                                returns: pd.Series, 
                                lambda_param: float = 0.94) -> float:
        """Exponentially weighted moving average volatility"""
        squared_returns = returns ** 2
        weights = [(1 - lambda_param) * (lambda_param ** i) 
                  for i in range(len(squared_returns))]
        weights.reverse()
        
        ewma_var = np.average(squared_returns, weights=weights[:len(squared_returns)])
        return np.sqrt(ewma_var * 252)
    
    def identify_mispriced_options(self, 
                                 options_data: pd.DataFrame,
                                 underlying_price: float,
                                 risk_free_rate: float) -> Dict:
        """Identify options with significant IV vs RV discrepancies"""
        mispriced = {
            'undervalued': [],  # IV < RV (buy volatility)
            'overvalued': []    # IV > RV (sell volatility)
        }
        
        for _, option in options_data.iterrows():
            if option['time_to_expiry'] < self.min_time_to_expiry:
                continue
                
            # Get realized and implied volatility
            realized_vol = self.volatility_forecasts.get(
                option['underlying'], 
                self.calculate_realized_volatility(option['price_history'])
            )
            
            implied_vol = option['implied_volatility']
            vol_diff = implied_vol - realized_vol
            
            # Check for significant mispricing
            if abs(vol_diff) > self.vol_threshold:
                option_info = {
                    'symbol': option['symbol'],
                    'strike': option['strike'],
                    'expiry': option['expiry'],
                    'option_type': option['type'],
                    'iv': implied_vol,
                    'rv': realized_vol,
                    'vol_diff': vol_diff,
                    'market_price': option['market_price'],
                    'theoretical_price': self.bs_calculator.price(
                        underlying_price,
                        option['strike'],
                        option['time_to_expiry'] / 365.0,
                        risk_free_rate,
                        realized_vol,
                        option['dividend_yield'],
                        option['type']
                    )
                }
                
                if vol_diff > self.vol_threshold:
                    mispriced['overvalued'].append(option_info)
                else:
                    mispriced['undervalued'].append(option_info)
        
        return mispriced
    
    def construct_volatility_trade(self, 
                                 option_info: Dict,
                                 underlying_price: float,
                                 trade_type: str) -> Dict:
        """Construct delta-neutral volatility trade"""
        
        # Calculate option Greeks
        greeks = self.greeks_calculator.calculate_all_greeks(
            underlying_price,
            option_info['strike'],
            option_info['expiry'],
            0.05,  # risk-free rate
            option_info['iv'],
            0.0,   # dividend yield
            option_info['option_type']
        )
        
        # Determine position size based on vol edge and risk limits
        vol_edge = abs(option_info['vol_diff'])
        base_position_size = min(
            self.max_position_size * vol_edge / 0.1,  # Scale with edge
            self.max_position_size
        )
        
        if trade_type == 'buy_vol':
            # Buy undervalued options, sell underlying for delta neutrality
            option_position = base_position_size / option_info['market_price']
            hedge_position = -option_position * greeks['delta']
            
        else:  # sell_vol
            # Sell overvalued options, buy underlying for delta neutrality
            option_position = -base_position_size / option_info['market_price']
            hedge_position = -option_position * greeks['delta']
        
        trade = {
            'option_symbol': option_info['symbol'],
            'option_position': option_position,
            'hedge_position': hedge_position,
            'entry_delta': greeks['delta'],
            'entry_gamma': greeks['gamma'],
            'entry_vega': greeks['vega'],
            'entry_theta': greeks['theta'],
            'expected_pnl': self.calculate_expected_pnl(option_info, trade_type),
            'max_risk': self.calculate_max_risk(option_info, base_position_size),
            'trade_type': trade_type
        }
        
        return trade
    
    def calculate_expected_pnl(self, option_info: Dict, trade_type: str) -> float:
        """Calculate expected P&L from volatility trade"""
        vol_edge = option_info['vol_diff']
        vega_exposure = option_info.get('vega', 1.0)
        
        if trade_type == 'buy_vol':
            return -vol_edge * vega_exposure * 100  # Convert to P&L
        else:
            return vol_edge * vega_exposure * 100
    
    def calculate_max_risk(self, option_info: Dict, position_size: float) -> float:
        """Calculate maximum risk for the trade"""
        # Simplified risk calculation - in practice would be more sophisticated
        return position_size * 0.1  # 10% of position size
    
    def manage_dynamic_hedging(self, 
                             current_positions: Dict,
                             current_underlying_price: float,
                             current_time: pd.Timestamp) -> List[Dict]:
        """Manage dynamic delta hedging of volatility positions"""
        
        hedge_orders = []
        
        for symbol, position in current_positions.items():
            if symbol not in self.last_hedge_time:
                self.last_hedge_time[symbol] = current_time
                continue
            
            # Calculate current delta
            current_greeks = self.greeks_calculator.calculate_all_greeks(
                current_underlying_price,
                position['strike'],
                position['time_to_expiry'],
                0.05,
                position['current_iv'],
                0.0,
                position['option_type']
            )
            
            current_portfolio_delta = (position['option_position'] * 
                                     current_greeks['delta'] + 
                                     position['hedge_position'])
            
            # Check if rehedging is needed
            if abs(current_portfolio_delta) > self.rehedge_threshold:
                hedge_adjustment = -current_portfolio_delta
                
                hedge_order = {
                    'symbol': position['underlying_symbol'],
                    'side': 'buy' if hedge_adjustment > 0 else 'sell',
                    'quantity': abs(hedge_adjustment),
                    'order_type': 'market',
                    'reason': f'delta_hedge_{symbol}'
                }
                
                hedge_orders.append(hedge_order)
                self.last_hedge_time[symbol] = current_time
                
                # Update hedge position
                self.hedge_positions[symbol] = (
                    self.hedge_positions.get(symbol, 0) + hedge_adjustment
                )
        
        return hedge_orders
    
    def calculate_portfolio_greeks(self, current_positions: Dict) -> Dict:
        """Calculate portfolio-level Greeks"""
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        
        for position in current_positions.values():
            total_delta += position['option_position'] * position['current_delta']
            total_gamma += position['option_position'] * position['current_gamma']
            total_vega += position['option_position'] * position['current_vega']
            total_theta += position['option_position'] * position['current_theta']
        
        return {
            'portfolio_delta': total_delta,
            'portfolio_gamma': total_gamma,
            'portfolio_vega': total_vega,
            'portfolio_theta': total_theta
        }
    
    def on_data(self, data):
        """Main strategy logic called on each data update"""
        current_time = data.current_time
        
        orders = []
        
        # Update volatility forecasts
        for symbol in self.get_universe():
            price_history = data.history(symbol, 'close', self.lookback_period)
            returns = np.log(price_history / price_history.shift(1)).dropna()
            
            self.volatility_forecasts[symbol] = self.calculate_garch_volatility(returns)
        
        # Get options data
        options_data = data.get_options_data()
        if options_data.empty:
            return orders
        
        # Identify mispriced options
        for underlying in self.get_universe():
            underlying_price = data.current(underlying, 'close')
            underlying_options = options_data[options_data['underlying'] == underlying]
            
            mispriced = self.identify_mispriced_options(
                underlying_options, 
                underlying_price, 
                0.05
            )
            
            # Create new volatility trades
            for option_info in mispriced['undervalued'][:3]:  # Limit to top 3
                trade = self.construct_volatility_trade(
                    option_info, underlying_price, 'buy_vol'
                )
                
                if self.validate_trade(trade):
                    orders.extend(self.create_trade_orders(trade))
            
            for option_info in mispriced['overvalued'][:3]:  # Limit to top 3
                trade = self.construct_volatility_trade(
                    option_info, underlying_price, 'sell_vol'
                )
                
                if self.validate_trade(trade):
                    orders.extend(self.create_trade_orders(trade))
        
        # Manage existing positions
        current_positions = self.get_current_positions()
        hedge_orders = self.manage_dynamic_hedging(
            current_positions, 
            underlying_price, 
            current_time
        )
        orders.extend(hedge_orders)
        
        # Risk management
        portfolio_greeks = self.calculate_portfolio_greeks(current_positions)
        if abs(portfolio_greeks['portfolio_delta']) > 1.0:
            # Emergency delta hedge if portfolio delta gets too large
            emergency_hedge = self.create_emergency_hedge_order(
                portfolio_greeks['portfolio_delta']
            )
            if emergency_hedge:
                orders.append(emergency_hedge)
        
        return orders
    
    def validate_trade(self, trade: Dict) -> bool:
        """Validate trade meets risk and position limits"""
        if trade['max_risk'] > self.max_position_size * 0.2:
            return False
        
        if trade['expected_pnl'] < 0:
            return False
            
        return True
    
    def create_trade_orders(self, trade: Dict) -> List[Dict]:
        """Create orders for volatility trade"""
        orders = []
        
        # Option order
        option_order = {
            'symbol': trade['option_symbol'],
            'side': 'buy' if trade['option_position'] > 0 else 'sell',
            'quantity': abs(trade['option_position']),
            'order_type': 'limit',
            'price': None,  # Would set appropriate limit price
            'reason': f"vol_arb_{trade['trade_type']}"
        }
        orders.append(option_order)
        
        # Hedge order
        if trade['hedge_position'] != 0:
            hedge_order = {
                'symbol': trade['option_symbol'].split('_')[0],  # Extract underlying
                'side': 'buy' if trade['hedge_position'] > 0 else 'sell',
                'quantity': abs(trade['hedge_position']),
                'order_type': 'market',
                'reason': f"initial_hedge_{trade['trade_type']}"
            }
            orders.append(hedge_order)
        
        return orders
