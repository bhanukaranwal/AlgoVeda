"""
Advanced Options Market Making Strategy for AlgoVeda
Implements sophisticated market making with:
- Real-time Greeks management
- Inventory optimization  
- Adverse selection protection
- Multi-venue execution
- Volatility smile trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from scipy import optimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from algoveda.core.base_strategy import BaseStrategy, Order, OrderSide, OrderType, TimeInForce
from algoveda.calculations.options_pricing.black_scholes_gpu import BlackScholesGPU, OptionsParameters
from algoveda.calculations.greeks.analytical_greeks import GreeksCalculator
from algoveda.market_data.level2_book import Level2OrderBook
from algoveda.utils.mathematics.statistics import StatisticalAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class MarketMakingParameters:
    """Configuration for options market making strategy."""
    max_spread_bps: float = 50.0  # Maximum spread in basis points
    min_spread_bps: float = 5.0   # Minimum spread in basis points
    target_delta_neutral: float = 0.01  # Target delta neutrality
    max_gamma_exposure: float = 1000.0   # Maximum gamma exposure
    max_vega_exposure: float = 5000.0    # Maximum vega exposure
    inventory_half_life: float = 3600.0  # Inventory decay half-life in seconds
    adverse_selection_alpha: float = 0.1  # Adverse selection protection factor
    quote_size_base: int = 10            # Base quote size in contracts
    max_position_size: int = 500         # Maximum position size per strike
    volatility_lookback: int = 20        # Volatility calculation lookback periods
    skew_adjustment_factor: float = 0.5  # Volatility skew adjustment
    smile_interpolation: str = "cubic"   # Volatility smile interpolation method

@dataclass
class Quote:
    """Market making quote representation."""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    mid_price: float
    spread_bps: float
    theoretical_value: float
    edge: float
    inventory_adjustment: float
    adverse_selection_adjustment: float
    created_at: datetime
    expires_at: datetime
    confidence: float = 0.95

@dataclass
class InventoryPosition:
    """Inventory tracking for market making."""
    symbol: str
    net_position: int
    vega_weighted_position: float
    delta_weighted_position: float
    gamma_weighted_position: float
    inventory_cost: float
    last_trade_time: Optional[datetime] = None
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    
class OptionsMarketMakingStrategy(BaseStrategy):
    """
    Advanced options market making strategy with sophisticated
    risk management and inventory optimization.
    """
    
    def __init__(
        self,
        name: str = "OptionsMarketMaking",
        initial_capital: float = 10000000.0,  # $10M for market making
        params: Optional[MarketMakingParameters] = None,
        target_symbols: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize options market making strategy.
        
        Args:
            name: Strategy name
            initial_capital: Starting capital
            params: Market making parameters
            target_symbols: List of symbols to make markets in
            **kwargs: Additional strategy parameters
        """
        super().__init__(
            name=name,
            initial_capital=initial_capital,
            **kwargs
        )
        
        self.params = params or MarketMakingParameters()
        self.target_symbols = target_symbols or []
        
        # Initialize calculation engines
        self.bs_engine = BlackScholesGPU()
        self.greeks_calculator = GreeksCalculator()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Market making state
        self.current_quotes: Dict[str, Quote] = {}
        self.inventory_positions: Dict[str, InventoryPosition] = {}
        self.order_book_snapshots: Dict[str, Level2OrderBook] = {}
        self.volatility_estimates: Dict[str, float] = {}
        self.volatility_surface: Dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self.market_making_metrics = {
            'quotes_sent': 0,
            'fills_received': 0,
            'inventory_turns': 0,
            'edge_captured': 0.0,
            'adverse_selection_cost': 0.0,
            'spread_revenue': 0.0,
            'inventory_cost': 0.0,
            'delta_pnl': 0.0,
            'gamma_pnl': 0.0,
            'vega_pnl': 0.0,
            'theta_pnl': 0.0,
        }
        
        # Risk management
        self.portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        # Execution pools for parallel processing
        self.pricing_executor = ThreadPoolExecutor(max_workers=4)
        self.quote_executor = ThreadPoolExecutor(max_workers=8)
        
    async def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market making signals based on market conditions.
        
        Args:
            data: Market data with options chains and underlying prices
            
        Returns:
            DataFrame with market making signals
        """
        signals = pd.DataFrame(index=data.index)
        
        if data.empty:
            return signals
        
        # Update market state
        await self._update_market_state(data)
        
        # Calculate theoretical prices and Greeks
        await self._calculate_theoretical_values(data)
        
        # Generate quotes for each target symbol
        for symbol in self.target_symbols:
            symbol_data = data[data['symbol'] == symbol]
            if not symbol_data.empty:
                quotes = await self._generate_symbol_quotes(symbol, symbol_data)
                
                for quote in quotes:
                    # Convert quotes to signals
                    if quote.bid_size > 0:
                        signals.loc[signals.index[-1], f"{symbol}_bid"] = quote.bid_size
                        signals.loc[signals.index[-1], f"{symbol}_bid_price"] = quote.bid_price
                    
                    if quote.ask_size > 0:
                        signals.loc[signals.index[-1], f"{symbol}_ask"] = -quote.ask_size
                        signals.loc[signals.index[-1], f"{symbol}_ask_price"] = quote.ask_price
        
        return signals
    
    async def calculate_position_sizes(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate optimal position sizes based on inventory and risk limits.
        
        Args:
            signals: Trading signals
            data: Market data
            
        Returns:
            DataFrame with position sizes
        """
        position_sizes = signals.copy()
        
        # Apply inventory-based position sizing
        for symbol in self.target_symbols:
            inventory = self.inventory_positions.get(symbol, InventoryPosition(symbol, 0, 0.0, 0.0, 0.0, 0.0))
            
            # Calculate inventory adjustment
            inventory_adjustment = self._calculate_inventory_adjustment(inventory)
            
            # Apply position limits
            for col in position_sizes.columns:
                if symbol in col:
                    current_size = position_sizes.loc[position_sizes.index[-1], col]
                    adjusted_size = current_size * inventory_adjustment
                    
                    # Apply maximum position limits
                    if abs(inventory.net_position + adjusted_size) > self.params.max_position_size:
                        adjusted_size = max(0, self.params.max_position_size - abs(inventory.net_position))
                    
                    position_sizes.loc[position_sizes.index[-1], col] = adjusted_size
        
        return position_sizes
    
    async def _update_market_state(self, data: pd.DataFrame) -> None:
        """Update internal market state with new data."""
        # Update order book snapshots
        for symbol in self.target_symbols:
            symbol_data = data[data['symbol'] == symbol]
            if not symbol_data.empty:
                # Update order book (simplified)
                self.order_book_snapshots[symbol] = Level2OrderBook(
                    symbol=symbol,
                    bids=[(symbol_data['bid'].iloc[-1], symbol_data['bid_size'].iloc[-1])],
                    asks=[(symbol_data['ask'].iloc[-1], symbol_data['ask_size'].iloc[-1])],
                    timestamp=self.current_date
                )
        
        # Update volatility estimates
        await self._update_volatility_estimates(data)
        
        # Update portfolio Greeks
        await self._update_portfolio_greeks()
    
    async def _calculate_theoretical_values(self, data: pd.DataFrame) -> None:
        """Calculate theoretical option values using GPU acceleration."""
        options_data = data[data['instrument_type'] == 'option']
        
        if options_data.empty:
            return
        
        # Prepare parameters for GPU calculation
        params = OptionsParameters(
            spot_prices=options_data['underlying_price'].tolist(),
            strike_prices=options_data['strike'].tolist(),
            times_to_expiry=options_data['time_to_expiry'].tolist(),
            risk_free_rates=[0.02] * len(options_data),
            volatilities=options_data['implied_volatility'].tolist(),
            dividend_yields=[0.0] * len(options_data),
            option_types=[self._convert_option_type(ot) for ot in options_data['option_type']]
        )
        
        # Calculate using GPU
        results = await self.bs_engine.calculate_options(params)
        
        # Store results
        for i, (_, option) in enumerate(options_data.iterrows()):
            symbol = option['symbol']
            self.theoretical_values[symbol] = {
                'price': results.option_prices[i],
                'delta': results.deltas[i],
                'gamma': results.gammas[i],
                'vega': results.vegas[i],
                'theta': results.thetas[i],
                'rho': results.rhos[i],
            }
    
    async def _generate_symbol_quotes(self, symbol: str, symbol_data: pd.DataFrame) -> List[Quote]:
        """Generate quotes for a specific option symbol."""
        quotes = []
        
        if symbol not in self.theoretical_values:
            return quotes
        
        theoretical = self.theoretical_values[symbol]
        current_price = symbol_data['mid_price'].iloc[-1] if not symbol_data.empty else theoretical['price']
        
        # Calculate fair value and edge
        fair_value = theoretical['price']
        edge = fair_value - current_price
        
        # Calculate spread based on various factors
        base_spread = self._calculate_base_spread(symbol, symbol_data)
        inventory_adjustment = self._calculate_inventory_spread_adjustment(symbol)
        adverse_selection_adjustment = self._calculate_adverse_selection_adjustment(symbol, symbol_data)
        volatility_adjustment = self._calculate_volatility_spread_adjustment(symbol)
        
        total_spread = base_spread + inventory_adjustment + adverse_selection_adjustment + volatility_adjustment
        half_spread = total_spread / 2.0
        
        # Generate bid/ask prices
        mid_price = fair_value
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread
        
        # Calculate quote sizes
        bid_size, ask_size = self._calculate_quote_sizes(symbol, theoretical)
        
        # Apply inventory skewing
        bid_size, ask_size, bid_price, ask_price = self._apply_inventory_skewing(
            symbol, bid_size, ask_size, bid_price, ask_price
        )
        
        # Create quote
        quote = Quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            mid_price=mid_price,
            spread_bps=total_spread * 10000 / mid_price,
            theoretical_value=fair_value,
            edge=edge,
            inventory_adjustment=inventory_adjustment,
            adverse_selection_adjustment=adverse_selection_adjustment,
            created_at=self.current_date,
            expires_at=self.current_date + timedelta(seconds=30),  # 30-second quote life
            confidence=self._calculate_quote_confidence(symbol, symbol_data)
        )
        
        quotes.append(quote)
        self.current_quotes[symbol] = quote
        
        return quotes
    
    def _calculate_base_spread(self, symbol: str, symbol_data: pd.DataFrame) -> float:
        """Calculate base spread based on market conditions."""
        if symbol_data.empty:
            return self.params.max_spread_bps / 10000.0
        
        # Get current bid-ask spread from market
        current_bid = symbol_data['bid'].iloc[-1]
        current_ask = symbol_data['ask'].iloc[-1]
        current_spread = (current_ask - current_bid) / ((current_ask + current_bid) / 2.0)
        
        # Base spread is a fraction of current market spread
        base_spread = max(
            self.params.min_spread_bps / 10000.0,
            min(self.params.max_spread_bps / 10000.0, current_spread * 0.8)
        )
        
        return base_spread
    
    def _calculate_inventory_spread_adjustment(self, symbol: str) -> float:
        """Calculate spread adjustment based on inventory position."""
        inventory = self.inventory_positions.get(symbol, InventoryPosition(symbol, 0, 0.0, 0.0, 0.0, 0.0))
        
        if inventory.net_position == 0:
            return 0.0
        
        # Widen spread when carrying inventory
        inventory_ratio = abs(inventory.net_position) / self.params.max_position_size
        adjustment = inventory_ratio * (self.params.max_spread_bps / 10000.0) * 0.5
        
        return adjustment
    
    def _calculate_adverse_selection_adjustment(self, symbol: str, symbol_data: pd.DataFrame) -> float:
        """Calculate spread adjustment to protect against adverse selection."""
        if symbol_data.empty:
            return 0.0
        
        # Calculate recent volatility
        if len(symbol_data) >= 10:
            returns = symbol_data['mid_price'].pct_change().dropna()
            recent_vol = returns.rolling(10).std().iloc[-1] if len(returns) >= 10 else 0.02
        else:
            recent_vol = 0.02
        
        # Adjust spread based on volatility and adverse selection alpha
        adjustment = recent_vol * self.params.adverse_selection_alpha
        
        return adjustment
    
    def _calculate_volatility_spread_adjustment(self, symbol: str) -> float:
        """Calculate spread adjustment based on volatility uncertainty."""
        if symbol not in self.volatility_estimates:
            return 0.0
        
        vol_estimate = self.volatility_estimates[symbol]
        
        # Historical volatility vs implied volatility adjustment
        # This would compare historical and implied volatility
        vol_uncertainty = 0.05  # Placeholder - would be calculated from vol surface
        
        adjustment = vol_uncertainty * 0.1  # 10% of vol uncertainty
        
        return adjustment
    
    def _calculate_quote_sizes(self, symbol: str, theoretical: Dict[str, float]) -> Tuple[int, int]:
        """Calculate optimal quote sizes based on Greeks and market conditions."""
        base_size = self.params.quote_size_base
        
        # Adjust size based on gamma exposure
        gamma = abs(theoretical.get('gamma', 0.0))
        gamma_adjustment = 1.0 - min(0.5, gamma / 100.0)  # Reduce size for high gamma
        
        # Adjust size based on vega exposure
        vega = abs(theoretical.get('vega', 0.0))
        vega_adjustment = 1.0 - min(0.5, vega / 1000.0)  # Reduce size for high vega
        
        # Calculate final sizes
        bid_size = int(base_size * gamma_adjustment * vega_adjustment)
        ask_size = int(base_size * gamma_adjustment * vega_adjustment)
        
        return max(1, bid_size), max(1, ask_size)
    
    def _apply_inventory_skewing(
        self, 
        symbol: str, 
        bid_size: int, 
        ask_size: int, 
        bid_price: float, 
        ask_price: float
    ) -> Tuple[int, int, float, float]:
        """Apply inventory skewing to quotes."""
        inventory = self.inventory_positions.get(symbol, InventoryPosition(symbol, 0, 0.0, 0.0, 0.0, 0.0))
        
        if inventory.net_position == 0:
            return bid_size, ask_size, bid_price, ask_price
        
        inventory_ratio = inventory.net_position / self.params.max_position_size
        
        if inventory_ratio > 0:  # Long inventory - favor selling
            ask_size = int(ask_size * 1.5)  # Increase ask size
            bid_size = max(1, int(bid_size * 0.5))  # Decrease bid size
            ask_price *= (1 - abs(inventory_ratio) * 0.1)  # Slightly lower ask price
            bid_price *= (1 - abs(inventory_ratio) * 0.2)  # More aggressive bid price reduction
        else:  # Short inventory - favor buying
            bid_size = int(bid_size * 1.5)  # Increase bid size
            ask_size = max(1, int(ask_size * 0.5))  # Decrease ask size
            bid_price *= (1 + abs(inventory_ratio) * 0.1)  # Slightly higher bid price
            ask_price *= (1 + abs(inventory_ratio) * 0.2)  # More aggressive ask price increase
        
        return bid_size, ask_size, bid_price, ask_price
    
    def _calculate_inventory_adjustment(self, inventory: InventoryPosition) -> float:
        """Calculate inventory-based position size adjustment."""
        if inventory.net_position == 0:
            return 1.0
        
        # Apply exponential decay to inventory
        time_since_last_trade = (
            (self.current_date - inventory.last_trade_time).total_seconds() 
            if inventory.last_trade_time else 0
        )
        
        decay_factor = np.exp(-time_since_last_trade / self.params.inventory_half_life)
        inventory_pressure = (inventory.net_position / self.params.max_position_size) * decay_factor
        
        # Reduce position size when inventory is large
        adjustment = 1.0 - abs(inventory_pressure) * 0.5
        
        return max(0.1, adjustment)
    
    def _calculate_quote_confidence(self, symbol: str, symbol_data: pd.DataFrame) -> float:
        """Calculate confidence level for quotes."""
        if symbol_data.empty:
            return 0.5
        
        # Base confidence
        confidence = 0.8
        
        # Adjust based on market volatility
        if len(symbol_data) >= 10:
            returns = symbol_data['mid_price'].pct_change().dropna()
            vol = returns.std() if len(returns) > 1 else 0.02
            confidence *= (1.0 - min(0.5, vol * 10))  # Reduce confidence in high vol
        
        # Adjust based on bid-ask spread
        if 'bid' in symbol_data.columns and 'ask' in symbol_data.columns:
            spread = (symbol_data['ask'].iloc[-1] - symbol_data['bid'].iloc[-1])
            mid = (symbol_data['ask'].iloc[-1] + symbol_data['bid'].iloc[-1]) / 2.0
            spread_ratio = spread / mid if mid > 0 else 1.0
            confidence *= (1.0 - min(0.5, spread_ratio * 5))  # Reduce confidence in wide spreads
        
        return max(0.1, min(0.99, confidence))
    
    async def _update_volatility_estimates(self, data: pd.DataFrame) -> None:
        """Update volatility estimates for all symbols."""
        for symbol in self.target_symbols:
            symbol_data = data[data['symbol'] == symbol]
            if len(symbol_data) >= self.params.volatility_lookback:
                returns = symbol_data['mid_price'].pct_change().dropna()
                if len(returns) >= self.params.volatility_lookback:
                    vol = returns.rolling(self.params.volatility_lookback).std().iloc[-1] * np.sqrt(252)
                    self.volatility_estimates[symbol] = vol
    
    async def _update_portfolio_greeks(self) -> None:
        """Update portfolio-level Greeks."""
        total_greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        for symbol, inventory in self.inventory_positions.items():
            if symbol in self.theoretical_values and inventory.net_position != 0:
                theoretical = self.theoretical_values[symbol]
                
                for greek in total_greeks:
                    if greek in theoretical:
                        total_greeks[greek] += theoretical[greek] * inventory.net_position
        
        self.portfolio_greeks = total_greeks
    
    def _convert_option_type(self, option_type: str) -> Any:
        """Convert option type string to enum."""
        # This would return the appropriate option type enum
        return option_type.lower() == 'call'
    
    async def handle_fill(self, fill_event: Dict[str, Any]) -> None:
        """Handle order fill events."""
        symbol = fill_event['symbol']
        side = fill_event['side']
        quantity = fill_event['quantity']
        price = fill_event['price']
        
        # Update inventory
        if symbol not in self.inventory_positions:
            self.inventory_positions[symbol] = InventoryPosition(symbol, 0, 0.0, 0.0, 0.0, 0.0)
        
        inventory = self.inventory_positions[symbol]
        
        if side == 'buy':
            inventory.net_position += quantity
        else:
            inventory.net_position -= quantity
        
        inventory.last_trade_time = self.current_date
        
        # Update metrics
        self.market_making_metrics['fills_received'] += 1
        
        # Calculate spread revenue
        if symbol in self.current_quotes:
            quote = self.current_quotes[symbol]
            if side == 'buy':
                spread_revenue = quote.bid_price - quote.theoretical_value
            else:
                spread_revenue = quote.theoretical_value - quote.ask_price
            
            self.market_making_metrics['spread_revenue'] += spread_revenue * quantity
        
        logger.info(f"Fill received: {side} {quantity} {symbol} @ ${price:.4f}")
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific performance metrics."""
        base_metrics = self.get_performance_metrics()
        
        # Add market making specific metrics
        mm_metrics = {
            'quotes_sent_per_day': self.market_making_metrics['quotes_sent'] / max(1, len(self.equity_curve) / 252),
            'fill_ratio': self.market_making_metrics['fills_received'] / max(1, self.market_making_metrics['quotes_sent']),
            'average_spread_captured': self.market_making_metrics['spread_revenue'] / max(1, self.market_making_metrics['fills_received']),
            'inventory_turnover': self.market_making_metrics['inventory_turns'],
            'edge_capture_ratio': self.market_making_metrics['edge_captured'] / max(1, abs(self.market_making_metrics['spread_revenue'])),
            'adverse_selection_ratio': abs(self.market_making_metrics['adverse_selection_cost']) / max(1, self.market_making_metrics['spread_revenue']),
            'portfolio_delta': self.portfolio_greeks['delta'],
            'portfolio_gamma': self.portfolio_greeks['gamma'],
            'portfolio_vega': self.portfolio_greeks['vega'],
            'portfolio_theta': self.portfolio_greeks['theta'],
            'active_inventory_positions': len([inv for inv in self.inventory_positions.values() if inv.net_position != 0]),
            'delta_pnl_contribution': self.market_making_metrics['delta_pnl'] / max(1, abs(sum([self.market_making_metrics[k] for k in ['delta_pnl', 'gamma_pnl', 'vega_pnl', 'theta_pnl']]))),
            'gamma_pnl_contribution': self.market_making_metrics['gamma_pnl'] / max(1, abs(sum([self.market_making_metrics[k] for k in ['delta_pnl', 'gamma_pnl', 'vega_pnl', 'theta_pnl']]))),
        }
        
        # Combine metrics
        base_metrics.update(mm_metrics)
        return base_metrics
    
    async def generate_market_making_report(self) -> str:
        """Generate comprehensive market making performance report."""
        metrics = self.get_strategy_metrics()
        
        report = f"""
{'='*80}
ALGOVEDA OPTIONS MARKET MAKING STRATEGY REPORT
{'='*80}

Strategy: {self.name}
Analysis Period: {self.dates[0].strftime('%Y-%m-%d') if self.dates else 'N/A'} to {self.dates[-1].strftime('%Y-%m-%d') if self.dates else 'N/A'}
Initial Capital: ${self.initial_capital:,.2f}

{'='*40} MARKET MAKING METRICS {'='*40}
Quotes Sent: {self.market_making_metrics['quotes_sent']:,}
Fills Received: {self.market_making_metrics['fills_received']:,}
Fill Ratio: {metrics.get('fill_ratio', 0)*100:.2f}%
Average Spread Captured: ${metrics.get('average_spread_captured', 0):.4f}
Inventory Turnover: {metrics.get('inventory_turnover', 0):.2f}x

{'='*40} REVENUE BREAKDOWN {'='*40}
Spread Revenue: ${self.market_making_metrics['spread_revenue']:,.2f}
Edge Captured: ${self.market_making_metrics['edge_captured']:,.2f}
Adverse Selection Cost: ${self.market_making_metrics['adverse_selection_cost']:,.2f}
Net Market Making PnL: ${self.market_making_metrics['spread_revenue'] - abs(self.market_making_metrics['adverse_selection_cost']):,.2f}

{'='*40} GREEKS P&L ATTRIBUTION {'='*40}
Delta P&L: ${self.market_making_metrics['delta_pnl']:,.2f}
Gamma P&L: ${self.market_making_metrics['gamma_pnl']:,.2f}
Vega P&L: ${self.market_making_metrics['vega_pnl']:,.2f}
Theta P&L: ${self.market_making_metrics['theta_pnl']:,.2f}

{'='*40} CURRENT PORTFOLIO GREEKS {'='*40}
Portfolio Delta: {self.portfolio_greeks['delta']:,.0f}
Portfolio Gamma: {self.portfolio_greeks['gamma']:,.0f}
Portfolio Vega: ${self.portfolio_greeks['vega']:,.0f}
Portfolio Theta: ${self.portfolio_greeks['theta']:,.0f}

{'='*40} INVENTORY POSITIONS {'='*40}
"""
        
        for symbol, inventory in self.inventory_positions.items():
            if inventory.net_position != 0:
                report += f"""
{symbol}: {inventory.net_position:+d} contracts
  Average Price: ${inventory.average_price:.4f}
  Unrealized P&L: ${inventory.unrealized_pnl:+,.2f}
  Vega Weighted: ${inventory.vega_weighted_position:+,.0f}
  Delta Weighted: {inventory.delta_weighted_position:+,.0f}
"""
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def __del__(self):
        """Cleanup when strategy is destroyed."""
        if hasattr(self, 'pricing_executor'):
            self.pricing_executor.shutdown(wait=True)
        if hasattr(self, 'quote_executor'):
            self.quote_executor.shutdown(wait=True)
        super().__del__()
