# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
Ultra-Fast Vectorized Backtester in Cython
Performance-optimized backtesting engine
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp, fabs
import cython

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_vectorized_backtest(
    np.ndarray[DTYPE_t, ndim=2] prices,
    np.ndarray[DTYPE_t, ndim=2] signals,
    np.ndarray[DTYPE_t, ndim=1] initial_capital,
    double commission_rate = 0.001
):
    """
    Ultra-fast vectorized backtesting engine
    
    Parameters:
    -----------
    prices : 2D array (time_steps x assets)
    signals : 2D array (time_steps x assets) 
    initial_capital : 1D array (assets)
    commission_rate : float
    
    Returns:
    --------
    portfolio_values : 2D array
    positions : 2D array  
    trades : 2D array
    """
    cdef int n_timesteps = prices.shape[0]
    cdef int n_assets = prices.shape[1]
    cdef int i, j
    cdef double position, trade_size, commission, new_position
    
    # Initialize arrays
    cdef np.ndarray[DTYPE_t, ndim=2] portfolio_values = np.zeros((n_timesteps, n_assets), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] positions = np.zeros((n_timesteps, n_assets), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] trades = np.zeros((n_timesteps, n_assets), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] cash = np.copy(initial_capital)
    
    # Set initial values
    for j in range(n_assets):
        portfolio_values[0, j] = initial_capital[j]
    
    # Main backtesting loop - optimized for speed
    for i in range(1, n_timesteps):
        for j in range(n_assets):
            # Current position
            position = positions[i-1, j]
            
            # Calculate new position based on signal
            new_position = signals[i, j] * (cash[j] + position * prices[i, j]) / prices[i, j]
            
            # Calculate trade size
            trade_size = new_position - position
            trades[i, j] = trade_size
            
            # Calculate commission
            commission = fabs(trade_size) * prices[i, j] * commission_rate
            
            # Update cash and positions
            cash[j] -= trade_size * prices[i, j] + commission
            positions[i, j] = new_position
            
            # Update portfolio value
            portfolio_values[i, j] = cash[j] + positions[i, j] * prices[i, j]
    
    return portfolio_values, positions, trades

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_performance_metrics(np.ndarray[DTYPE_t, ndim=1] returns):
    """Calculate comprehensive performance metrics"""
    cdef int n = returns.shape[0]
    cdef double total_return, volatility, sharpe_ratio, max_drawdown
    cdef double cumulative_return, peak, drawdown
    cdef int i
    
    # Total return
    total_return = 1.0
    for i in range(n):
        total_return *= (1.0 + returns[i])
    total_return -= 1.0
    
    # Volatility
    cdef double mean_return = 0.0
    for i in range(n):
        mean_return += returns[i]
    mean_return /= n
    
    volatility = 0.0
    for i in range(n):
        volatility += (returns[i] - mean_return) ** 2
    volatility = sqrt(volatility / (n - 1)) * sqrt(252.0)  # Annualized
    
    # Sharpe ratio
    sharpe_ratio = (mean_return * 252.0) / volatility if volatility > 0 else 0.0
    
    # Maximum drawdown
    peak = 1.0
    max_drawdown = 0.0
    cumulative_return = 1.0
    
    for i in range(n):
        cumulative_return *= (1.0 + returns[i])
        if cumulative_return > peak:
            peak = cumulative_return
        drawdown = (peak - cumulative_return) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return total_return, volatility, sharpe_ratio, max_drawdown
