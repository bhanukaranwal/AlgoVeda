"""
Advanced Volatility Arbitrage Strategy for AlgoVeda
Implements sophisticated volatility trading including:
- Implied vs Realized volatility arbitrage
- Volatility surface modeling
- Greeks-based hedging
- Multi-leg options strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import scipy.optimize as optimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from algoveda.core.base_strategy import BaseStrategy, OrderSide, OrderType, TimeInForce
from algoveda.calculations.options_pricing.black_scholes_gpu import BlackScholesGPU, OptionsParameters, OptionType
from algoveda.calculations.volatility.implied_volatility import ImpliedVolatilityCalculator
from algoveda.calculations.greeks.analytical_greeks import GreeksCalculator

class VolatilityArbitrageStrategy(BaseStrategy):
    """
    Advanced volatility arbitrage strategy that trades the difference
    between implied and realized volatility across the options surface.
    """
    
    def __init__(
        self,
        name: str = "VolatilityArbitrage",
        initial_capital: float = 5000000.0,  # $5M for options trading
        vol_threshold: float = 0.05,  # 5% vol difference threshold
        min_time_to_expiry: int = 7,  # Minimum days to expiration
        max_time_to_expiry: int = 90,  # Maximum days to expiration
        delta_neutral_threshold: float = 0.01,  # Delta neutrality threshold
        gamma_threshold: float = 0.1,  # Gamma exposure limit
        vega_threshold: float = 1000,  # Vega exposure limit
        realized_vol_window: int = 20,  # Lookback for realized vol calculation
        rebalance_frequency: int = 1,  # Daily rebalancing
        max_positions_per_expiry: int = 5,  # Max positions per expiration
        commission_per_contract: float = 1.5,  # Commission per options contract
        **kwargs
    ):
        """
        Initialize volatility arbitrage strategy.
        
        Args:
            vol_threshold: Minimum vol difference to trigger trade
            min_time_to_expiry: Minimum days to expiration for trades
            max_time_to_expiry: Maximum days to expiration for trades
            delta_neutral_threshold: Maximum delta exposure allowed
            gamma_threshold: Maximum gamma exposure per position
            vega_threshold: Maximum vega exposure per position
            realized_vol_window: Lookback window for realized vol
            rebalance_frequency: Days between rebalancing
            max_positions_per_expiry: Maximum positions per expiration
            commission_per_contract: Commission cost per contract
        """
        super().__init__(
            name=name,
            initial_capital=initial_capital,
            commission_rate=0.0,  # We'll handle options commissions separately
            **kwargs
        )
        
        self.vol_threshold = vol_threshold
        self.min_time_to_expiry = min_time_to_expiry
        self.max_time_to_expiry = max_time_to_expiry
        self.delta_neutral_threshold = delta_neutral_threshold
        self.gamma_threshold = gamma_threshold
        self.vega_threshold = vega_threshold
        self.realized_vol_window = realized_vol_window
        self.rebalance_frequency = rebalance_frequency
        self.max_positions_per_expiry = max_positions_per_expiry
        self.commission_per_contract = commission_per_contract
        
        # Initialize calculation engines
        self.bs_engine = None  # Will be initialized when needed
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.greeks_calculator = GreeksCalculator()
        
        # Strategy state
        self.options_data: pd.DataFrame = pd.DataFrame()
        self.underlying_data: pd.DataFrame = pd.DataFrame()
        self.volatility_surface: Dict[str, pd.DataFrame] = {}
        self.realized_volatilities: Dict[str, float] = {}
        self.portfolio_greeks: Dict[str, float] = {
            'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0
        }
        
        # Position tracking for options
        self.option_positions: Dict[str, Dict] = {}  # symbol -> position details
        self.hedging_positions: Dict[str, float] = {}  # underlying positions for hedging
        
        # Performance tracking
        self.pnl_breakdown: Dict[str, List[float]] = {
            'volatility_pnl': [],
            'gamma_pnl': [],
            'theta_pnl': [],
            'hedging_pnl': [],
            'total_pnl': []
        }
        
        self.last_rebalance_date: Optional[datetime] = None
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility arbitrage signals based on implied vs realized vol.
        
        Args:
            data: Combined options and underlying market data
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Separate options and underlying data
        self.underlying_data = data[data['instrument_type'] == 'stock'].copy()
        self.options_data = data[data['instrument_type'] == 'option'].copy()
        
        if self.options_data.empty or self.underlying_data.empty:
            return signals
        
        # Calculate realized volatilities
        self._calculate_realized_volatilities()
        
        # Build volatility surface
        self._build_volatility_surface()
        
        # Generate arbitrage signals
        arbitrage_signals = self._identify_volatility_arbitrage_opportunities()
        
        # Generate delta hedging signals
        hedging_signals = self._generate_delta_hedging_signals()
        
        # Combine signals
        signals = pd.concat([arbitrage_signals, hedging_signals], axis=1)
        signals.fillna(0, inplace=True)
        
        return signals
    
    def calculate_position_sizes(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position sizes based on volatility arbitrage signals and risk limits.
        
        Args:
            signals: Trading signals
            data: Market data
            
        Returns:
            DataFrame with position sizes
        """
        position_sizes = signals.copy()
        
        # Calculate current portfolio greeks
        self._update_portfolio_greeks()
        
        # Size positions based on vega exposure limits
        for symbol in signals.columns:
            if 'option' in symbol.lower():
                # Get option details
                option_info = self._parse_option_symbol(symbol)
                if option_info is None:
                    continue
                
                # Calculate vega per contract
                vega_per_contract = self._calculate_option_vega(symbol, data)
                
                if vega_per_contract > 0:
                    # Limit position size based on vega exposure
                    max_contracts = self.vega_threshold / vega_per_contract
                    current_signal = signals.loc[signals.index[-1], symbol]
                    
                    if abs(current_signal) > 0:
                        # Scale position size
                        position_sizes.loc[position_sizes.index[-1], symbol] = np.sign(current_signal) * min(abs(current_signal), max_contracts)
        
        return position_sizes
    
    def _calculate_realized_volatilities(self) -> None:
        """Calculate realized volatilities for underlying assets."""
        for symbol in self.underlying_data['symbol'].unique():
            symbol_data = self.underlying_data[self.underlying_data['symbol'] == symbol].copy()
            
            if len(symbol_data) >= self.realized_vol_window:
                # Calculate log returns
                symbol_data['log_return'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
                
                # Calculate realized volatility (annualized)
                realized_vol = symbol_data['log_return'].rolling(self.realized_vol_window).std() * np.sqrt(252)
                
                self.realized_volatilities[symbol] = realized_vol.iloc[-1] if not pd.isna(realized_vol.iloc[-1]) else 0.2
    
    def _build_volatility_surface(self) -> None:
        """Build implied volatility surface for each underlying."""
        for symbol in self.underlying_data['symbol'].unique():
            # Get options for this underlying
            underlying_options = self.options_data[
                self.options_data['underlying_symbol'] == symbol
            ].copy()
            
            if underlying_options.empty:
                continue
            
            # Get current underlying price
            current_price = self.underlying_data[
                self.underlying_data['symbol'] == symbol
            ]['close'].iloc[-1]
            
            # Calculate moneyness and time to expiry
            underlying_options['moneyness'] = underlying_options['strike'] / current_price
            underlying_options['time_to_expiry'] = (
                pd.to_datetime(underlying_options['expiration']) - self.current_date
            ).dt.days / 365.0
            
            # Filter by time to expiry
            valid_options = underlying_options[
                (underlying_options['time_to_expiry'] >= self.min_time_to_expiry/365.0) &
                (underlying_options['time_to_expiry'] <= self.max_time_to_expiry/365.0)
            ].copy()
            
            if not valid_options.empty:
                # Calculate implied volatilities
                implied_vols = []
                for _, option in valid_options.iterrows():
                    iv = self.iv_calculator.calculate_iv(
                        market_price=option['mid_price'],
                        spot_price=current_price,
                        strike_price=option['strike'],
                        time_to_expiry=option['time_to_expiry'],
                        risk_free_rate=0.02,  # Assume 2% risk-free rate
                        dividend_yield=0.0,
                        option_type=option['option_type']
                    )
                    implied_vols.append(iv)
                
                valid_options['implied_vol'] = implied_vols
                self.volatility_surface[symbol] = valid_options
    
    def _identify_volatility_arbitrage_opportunities(self) -> pd.DataFrame:
        """Identify volatility arbitrage opportunities."""
        signals = pd.DataFrame()
        
        for symbol, vol_surface in self.volatility_surface.items():
            realized_vol = self.realized_volatilities.get(symbol, 0.2)
            
            for _, option in vol_surface.iterrows():
                option_symbol = self._create_option_symbol(
                    option['underlying_symbol'],
                    option['expiration'],
                    option['strike'],
                    option['option_type']
                )
                
                implied_vol = option['implied_vol']
                vol_diff = implied_vol - realized_vol
                
                # Generate signal based on volatility differential
                if abs(vol_diff) > self.vol_threshold:
                    if vol_diff > 0:  # IV > RV, sell volatility
                        signal = -1.0
                    else:  # IV < RV, buy volatility
                        signal = 1.0
                    
                    # Create straddle position (buy/sell both call and put)
                    call_symbol = self._create_option_symbol(
                        option['underlying_symbol'],
                        option['expiration'],
                        option['strike'],
                        'call'
                    )
                    put_symbol = self._create_option_symbol(
                        option['underlying_symbol'],
                        option['expiration'],
                        option['strike'],
                        'put'
                    )
                    
                    if call_symbol not in signals.columns:
                        signals[call_symbol] = 0.0
                    if put_symbol not in signals.columns:
                        signals[put_symbol] = 0.0
                    
                    signals.loc[signals.index[-1] if not signals.empty else 0, call_symbol] = signal
                    signals.loc[signals.index[-1] if not signals.empty else 0, put_symbol] = signal
        
        return signals
    
    def _generate_delta_hedging_signals(self) -> pd.DataFrame:
        """Generate delta hedging signals for portfolio neutrality."""
        hedging_signals = pd.DataFrame()
        
        # Calculate net delta exposure for each underlying
        for symbol in self.underlying_data['symbol'].unique():
            net_delta = 0.0
            
            # Sum delta from all option positions
            for option_symbol, position in self.option_positions.items():
                if position['underlying'] == symbol:
                    option_delta = self._calculate_option_delta(option_symbol)
                    net_delta += option_delta * position['quantity']
            
            # Generate hedging signal for underlying
            if abs(net_delta) > self.delta_neutral_threshold:
                hedging_signals[f"{symbol}_hedge"] = -net_delta
        
        return hedging_signals
    
    def _calculate_option_delta(self, option_symbol: str) -> float:
        """Calculate delta for a specific option."""
        option_info = self._parse_option_symbol(option_symbol)
        if option_info is None:
            return 0.0
        
        # Get current market data
        underlying_price = self.underlying_data[
            self.underlying_data['symbol'] == option_info['underlying']
        ]['close'].iloc[-1] if not self.underlying_data.empty else 100.0
        
        # Calculate delta using Black-Scholes
        delta = self.greeks_calculator.calculate_delta(
            spot_price=underlying_price,
            strike_price=option_info['strike'],
            time_to_expiry=option_info['time_to_expiry'],
            risk_free_rate=0.02,
            volatility=0.2,  # Use appropriate volatility
            dividend_yield=0.0,
            option_type=option_info['type']
        )
        
        return delta
    
    def _calculate_option_vega(self, option_symbol: str, data: pd.DataFrame) -> float:
        """Calculate vega for a specific option."""
        option_info = self._parse_option_symbol(option_symbol)
        if option_info is None:
            return 0.0
        
        # Get current market data
        underlying_price = self.underlying_data[
            self.underlying_data['symbol'] == option_info['underlying']
        ]['close'].iloc[-1] if not self.underlying_data.empty else 100.0
        
        # Calculate vega using Black-Scholes
        vega = self.greeks_calculator.calculate_vega(
            spot_price=underlying_price,
            strike_price=option_info['strike'],
            time_to_expiry=option_info['time_to_expiry'],
            risk_free_rate=0.02,
            volatility=0.2,  # Use appropriate volatility
            dividend_yield=0.0
        )
        
        return vega
    
    def _update_portfolio_greeks(self) -> None:
        """Update portfolio-level Greeks."""
        total_greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        for option_symbol, position in self.option_positions.items():
            # Calculate Greeks for each position
            greeks = self._calculate_all_greeks(option_symbol)
            
            for greek_name, greek_value in greeks.items():
                total_greeks[greek_name] += greek_value * position['quantity']
        
        self.portfolio_greeks = total_greeks
    
    def _calculate_all_greeks(self, option_symbol: str) -> Dict[str, float]:
        """Calculate all Greeks for a specific option."""
        option_info = self._parse_option_symbol(option_symbol)
        if option_info is None:
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        # Get current market data
        underlying_price = self.underlying_data[
            self.underlying_data['symbol'] == option_info['underlying']
        ]['close'].iloc[-1] if not self.underlying_data.empty else 100.0
        
        # Calculate all Greeks
        greeks = self.greeks_calculator.calculate_all_greeks(
            spot_price=underlying_price,
            strike_price=option_info['strike'],
            time_to_expiry=option_info['time_to_expiry'],
            risk_free_rate=0.02,
            volatility=0.2,  # Use appropriate volatility
            dividend_yield=0.0,
            option_type=option_info['type']
        )
        
        return greeks
    
    def _parse_option_symbol(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """Parse option symbol to extract components."""
        # This would parse a symbol like "AAPL_20231215_150_C" 
        # Format: UNDERLYING_YYYYMMDD_STRIKE_TYPE
        try:
            parts = option_symbol.split('_')
            if len(parts) != 4:
                return None
            
            underlying = parts[0]
            expiration_str = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3].upper() == 'C' else 'put'
            
            expiration = datetime.strptime(expiration_str, '%Y%m%d')
            time_to_expiry = (expiration - self.current_date).days / 365.0
            
            return {
                'underlying': underlying,
                'expiration': expiration,
                'strike': strike,
                'type': option_type,
                'time_to_expiry': time_to_expiry
            }
        except:
            return None
    
    def _create_option_symbol(self, underlying: str, expiration: datetime, strike: float, option_type: str) -> str:
        """Create standardized option symbol."""
        exp_str = expiration.strftime('%Y%m%d')
        type_code = 'C' if option_type.lower() == 'call' else 'P'
        return f"{underlying}_{exp_str}_{strike}_{type_code}"
    
    def on_bar(self, bar_data: pd.Series) -> None:
        """Process bar data with volatility strategy logic."""
        super().on_bar(bar_data)
        
        # Check if rebalancing is needed
        if (self.last_rebalance_date is None or 
            (self.current_date - self.last_rebalance_date).days >= self.rebalance_frequency):
            
            self._rebalance_portfolio()
            self.last_rebalance_date = self.current_date
        
        # Update portfolio Greeks and P&L attribution
        self._update_portfolio_greeks()
        self._calculate_pnl_attribution()
    
    def _rebalance_portfolio(self) -> None:
        """Rebalance portfolio to maintain delta neutrality and risk limits."""
        # Check delta exposure
        if abs(self.portfolio_greeks['delta']) > self.delta_neutral_threshold:
            # Generate hedging trades for underlying
            for underlying in self.underlying_data['symbol'].unique():
                underlying_delta = self._calculate_underlying_delta_exposure(underlying)
                
                if abs(underlying_delta) > self.delta_neutral_threshold:
                    # Place hedging order
                    side = OrderSide.SELL if underlying_delta > 0 else OrderSide.BUY
                    quantity = abs(underlying_delta) * 100  # Convert to shares
                    
                    self.place_order(
                        symbol=underlying,
                        side=side,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )
        
        # Check and manage risk limits
        self._manage_risk_limits()
    
    def _calculate_underlying_delta_exposure(self, underlying: str) -> float:
        """Calculate delta exposure for a specific underlying."""
        total_delta = 0.0
        
        for option_symbol, position in self.option_positions.items():
            option_info = self._parse_option_symbol(option_symbol)
            if option_info and option_info['underlying'] == underlying:
                option_delta = self._calculate_option_delta(option_symbol)
                total_delta += option_delta * position['quantity']
        
        return total_delta
    
    def _manage_risk_limits(self) -> None:
        """Manage portfolio risk limits."""
        # Check gamma exposure
        if abs(self.portfolio_greeks['gamma']) > self.gamma_threshold:
            self._reduce_gamma_exposure()
        
        # Check vega exposure
        if abs(self.portfolio_greeks['vega']) > self.vega_threshold:
            self._reduce_vega_exposure()
    
    def _reduce_gamma_exposure(self) -> None:
        """Reduce gamma exposure by closing high-gamma positions."""
        # Find positions with highest gamma contribution
        gamma_positions = []
        
        for option_symbol, position in self.option_positions.items():
            option_gamma = self._calculate_all_greeks(option_symbol)['gamma']
            gamma_contribution = abs(option_gamma * position['quantity'])
            gamma_positions.append((option_symbol, gamma_contribution))
        
        # Sort by gamma contribution and close highest contributors
        gamma_positions.sort(key=lambda x: x[1], reverse=True)
        
        for option_symbol, _ in gamma_positions[:3]:  # Close top 3 contributors
            position = self.option_positions[option_symbol]
            side = OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY
            
            self.place_order(
                symbol=option_symbol,
                side=side,
                quantity=abs(position['quantity']),
                order_type=OrderType.MARKET
            )
    
    def _reduce_vega_exposure(self) -> None:
        """Reduce vega exposure by closing high-vega positions."""
        # Similar logic to gamma reduction but for vega
        vega_positions = []
        
        for option_symbol, position in self.option_positions.items():
            option_vega = self._calculate_all_greeks(option_symbol)['vega']
            vega_contribution = abs(option_vega * position['quantity'])
            vega_positions.append((option_symbol, vega_contribution))
        
        # Sort by vega contribution and close highest contributors
        vega_positions.sort(key=lambda x: x[1], reverse=True)
        
        for option_symbol, _ in vega_positions[:3]:  # Close top 3 contributors
            position = self.option_positions[option_symbol]
            side = OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY
            
            self.place_order(
                symbol=option_symbol,
                side=side,
                quantity=abs(position['quantity']),
                order_type=OrderType.MARKET
            )
    
    def _calculate_pnl_attribution(self) -> None:
        """Calculate P&L attribution by source."""
        current_equity = self.calculate_total_equity()
        
        if len(self.equity_curve) > 1:
            total_pnl = current_equity - self.equity_curve[-2]
            
            # Attribute P&L to different sources
            volatility_pnl = self._calculate_volatility_pnl()
            gamma_pnl = self._calculate_gamma_pnl()
            theta_pnl = self._calculate_theta_pnl()
            hedging_pnl = self._calculate_hedging_pnl()
            
            # Store attribution
            self.pnl_breakdown['volatility_pnl'].append(volatility_pnl)
            self.pnl_breakdown['gamma_pnl'].append(gamma_pnl)
            self.pnl_breakdown['theta_pnl'].append(theta_pnl)
            self.pnl_breakdown['hedging_pnl'].append(hedging_pnl)
            self.pnl_breakdown['total_pnl'].append(total_pnl)
    
    def _calculate_volatility_pnl(self) -> float:
        """Calculate P&L from volatility changes."""
        # This would calculate P&L from changes in implied volatility
        # Simplified calculation for demonstration
        return self.portfolio_greeks['vega'] * 0.01  # Assume 1% vol change
    
    def _calculate_gamma_pnl(self) -> float:
        """Calculate P&L from gamma effects."""
        # Calculate P&L from underlying price movements and gamma
        total_gamma_pnl = 0.0
        
        for underlying in self.underlying_data['symbol'].unique():
            if len(self.underlying_data) > 1:
                price_change = (
                    self.underlying_data[self.underlying_data['symbol'] == underlying]['close'].iloc[-1] -
                    self.underlying_data[self.underlying_data['symbol'] == underlying]['close'].iloc[-2]
                )
                underlying_gamma = self._calculate_underlying_gamma_exposure(underlying)
                gamma_pnl = 0.5 * underlying_gamma * (price_change ** 2)
                total_gamma_pnl += gamma_pnl
        
        return total_gamma_pnl
    
    def _calculate_theta_pnl(self) -> float:
        """Calculate P&L from time decay."""
        return self.portfolio_greeks['theta'] / 365.0  # Daily theta decay
    
    def _calculate_hedging_pnl(self) -> float:
        """Calculate P&L from hedging positions."""
        hedging_pnl = 0.0
        
        for symbol, hedge_quantity in self.hedging_positions.items():
            if len(self.underlying_data) > 1:
                price_change = (
                    self.underlying_data[self.underlying_data['symbol'] == symbol]['close'].iloc[-1] -
                    self.underlying_data[self.underlying_data['symbol'] == symbol]['close'].iloc[-2]
                )
                hedging_pnl += hedge_quantity * price_change
        
        return hedging_pnl
    
    def _calculate_underlying_gamma_exposure(self, underlying: str) -> float:
        """Calculate gamma exposure for a specific underlying."""
        total_gamma = 0.0
        
        for option_symbol, position in self.option_positions.items():
            option_info = self._parse_option_symbol(option_symbol)
            if option_info and option_info['underlying'] == underlying:
                option_gamma = self._calculate_all_greeks(option_symbol)['gamma']
                total_gamma += option_gamma * position['quantity']
        
        return total_gamma
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific performance metrics."""
        base_metrics = self.get_performance_metrics()
        
        # Add volatility strategy specific metrics
        vol_metrics = {
            'avg_realized_vol': np.mean(list(self.realized_volatilities.values())) if self.realized_volatilities else 0.0,
            'portfolio_delta': self.portfolio_greeks['delta'],
            'portfolio_gamma': self.portfolio_greeks['gamma'],
            'portfolio_vega': self.portfolio_greeks['vega'],
            'portfolio_theta': self.portfolio_greeks['theta'],
            'active_option_positions': len([p for p in self.option_positions.values() if p['quantity'] != 0]),
            'volatility_pnl_total': sum(self.pnl_breakdown['volatility_pnl']) if self.pnl_breakdown['volatility_pnl'] else 0.0,
            'gamma_pnl_total': sum(self.pnl_breakdown['gamma_pnl']) if self.pnl_breakdown['gamma_pnl'] else 0.0,
            'theta_pnl_total': sum(self.pnl_breakdown['theta_pnl']) if self.pnl_breakdown['theta_pnl'] else 0.0,
        }
        
        # Combine metrics
        base_metrics.update(vol_metrics)
        return base_metrics
