# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
Ultra-High Performance Portfolio Engine in Cython
Real-time portfolio management with sub-microsecond updates
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp, fabs
import cython
from collections import defaultdict

ctypedef np.float64_t DTYPE_t

cdef class PortfolioEngine:
    """
    High-performance portfolio management engine
    Optimized for real-time trading with minimal latency
    """
    cdef:
        public double cash
        public double initial_capital
        public double commission_rate
        public double slippage_rate
        public dict positions
        public dict position_values
        public dict unrealized_pnl
        public dict realized_pnl
        public double portfolio_value
        public double total_commission_paid
        public int transaction_count
        
    def __cinit__(self, double initial_cash, double commission_rate=0.001, double slippage_rate=0.0001):
        self.cash = initial_cash
        self.initial_capital = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.positions = {}
        self.position_values = {}
        self.unrealized_pnl = {}
        self.realized_pnl = {}
        self.portfolio_value = initial_cash
        self.total_commission_paid = 0.0
        self.transaction_count = 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple update_position(self, str symbol, double target_quantity, double current_price):
        """
        Update position with ultra-fast execution
        Returns: (trade_size, commission_cost, slippage_cost)
        """
        cdef double current_position = self.positions.get(symbol, 0.0)
        cdef double trade_size = target_quantity - current_position
        cdef double trade_value, commission, slippage, net_cost
        
        if fabs(trade_size) < 1e-8:  # No trade needed
            return 0.0, 0.0, 0.0
        
        # Calculate costs
        trade_value = fabs(trade_size) * current_price
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate
        
        # Net cost including direction
        if trade_size > 0:  # Buying
            net_cost = trade_size * current_price + commission + slippage
        else:  # Selling
            net_cost = trade_size * current_price - commission - slippage
        
        # Update cash
        self.cash -= net_cost
        
        # Update position
        self.positions[symbol] = target_quantity
        self.position_values[symbol] = target_quantity * current_price
        
        # Track costs
        self.total_commission_paid += commission
        self.transaction_count += 1
        
        # Update realized PnL if closing position
        if current_position != 0.0 and ((current_position > 0 > target_quantity) or (current_position < 0 < target_quantity)):
            # Partial or full position close
            closed_quantity = min(fabs(current_position), fabs(trade_size))
            if symbol in self.realized_pnl:
                self.realized_pnl[symbol] += closed_quantity * (current_price - self.get_avg_cost(symbol))
            else:
                self.realized_pnl[symbol] = closed_quantity * (current_price - self.get_avg_cost(symbol))
        
        return trade_size, commission, slippage
    
    @cython.boundscheck(False)
    cpdef double get_avg_cost(self, str symbol):
        """Get average cost basis for symbol"""
        # Simplified - in practice would track average cost
        return 100.0  # Placeholder
    
    @cython.boundscheck(False)
    cpdef void update_market_values(self, dict prices):
        """Update all position values with current market prices"""
        cdef str symbol
        cdef double quantity, price, value
        cdef double total_position_value = 0.0
        
        for symbol in self.positions:
            quantity = self.positions[symbol]
            if quantity != 0.0 and symbol in prices:
                price = prices[symbol]
                value = quantity * price
                self.position_values[symbol] = value
                total_position_value += value
                
                # Update unrealized PnL
                avg_cost = self.get_avg_cost(symbol)
                self.unrealized_pnl[symbol] = quantity * (price - avg_cost)
        
        self.portfolio_value = self.cash + total_position_value
    
    @cython.boundscheck(False)
    cpdef dict get_portfolio_summary(self):
        """Get comprehensive portfolio summary"""
        cdef double total_unrealized = sum(self.unrealized_pnl.values())
        cdef double total_realized = sum(self.realized_pnl.values())
        cdef double total_pnl = total_unrealized + total_realized
        
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'commission_paid': self.total_commission_paid,
            'transaction_count': self.transaction_count,
            'number_of_positions': len([p for p in self.positions.values() if p != 0])
        }

@cython.boundscheck(False)
@cython.wraparound(False)
def vectorized_portfolio_backtest(
    np.ndarray[DTYPE_t, ndim=2] prices,
    np.ndarray[DTYPE_t, ndim=2] weights,
    double initial_capital,
    double commission_rate = 0.001
):
    """
    Vectorized portfolio backtesting with Cython optimization
    """
    cdef int n_periods = prices.shape[0]
    cdef int n_assets = prices.shape[1]
    cdef int i, j
    
    # Initialize output arrays
    cdef np.ndarray[DTYPE_t, ndim=1] portfolio_values = np.zeros(n_periods, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] cash_values = np.zeros(n_periods, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] position_quantities = np.zeros((n_periods, n_assets), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] total_commissions = np.zeros(n_periods, dtype=np.float64)
    
    # Initialize first period
    portfolio_values[0] = initial_capital
    cash_values[0] = initial_capital
    
    cdef double cash = initial_capital
    cdef double portfolio_value = initial_capital
    cdef double target_value, current_value, trade_value, commission
    cdef double new_quantity, old_quantity, trade_quantity
    
    # Main backtesting loop
    for i in range(1, n_periods):
        portfolio_value = cash
        
        # Calculate current portfolio value
        for j in range(n_assets):
            portfolio_value += position_quantities[i-1, j] * prices[i, j]
        
        # Rebalance portfolio
        for j in range(n_assets):
            target_value = portfolio_value * weights[i, j]
            new_quantity = target_value / prices[i, j]
            old_quantity = position_quantities[i-1, j]
            trade_quantity = new_quantity - old_quantity
            
            if fabs(trade_quantity) > 1e-8:
                trade_value = fabs(trade_quantity) * prices[i, j]
                commission = trade_value * commission_rate
                cash -= trade_quantity * prices[i, j] + commission
                total_commissions[i] += commission
            
            position_quantities[i, j] = new_quantity
        
        # Update values
        cash_values[i] = cash
        portfolio_value = cash
        for j in range(n_assets):
            portfolio_value += position_quantities[i, j] * prices[i, j]
        portfolio_values[i] = portfolio_value
    
    return portfolio_values, cash_values, position_quantities, total_commissions
