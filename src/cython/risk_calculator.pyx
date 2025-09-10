# distutils: language=c++
# cython: language_level=3

"""
Ultra-Fast Risk Calculations in Cython
VaR, CVaR, and Drawdown calculations optimized for speed
"""

import numpy as np
cimport numpy as np

cdef class RiskCalculator:
    def __init__(self):
        pass

    def calculate_var(self, np.ndarray[np.float64_t, ndim=1] returns, double confidence=0.95):
        """Calculate Value at Risk (VaR) at given confidence level"""
        cdef np.ndarray[np.float64_t, ndim=1] sorted_returns = np.sort(returns)
        cdef int idx = int((1 - confidence) * len(sorted_returns))
        if idx < 0:
            idx = 0
        return -sorted_returns[idx]

    def calculate_cvar(self, np.ndarray[np.float64_t, ndim=1] returns, double confidence=0.95):
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)"""
        cdef np.ndarray[np.float64_t, ndim=1] sorted_returns = np.sort(returns)
        cdef int idx = int((1 - confidence) * len(sorted_returns))
        if idx < 0:
            idx = 0
        return -np.mean(sorted_returns[:idx + 1])

    def calculate_drawdown(self, np.ndarray[np.float64_t, ndim=1] portfolio_values):
        """Calculate maximum drawdown from portfolio values"""
        cdef int n = len(portfolio_values)
        cdef double max_peak = portfolio_values[0]
        cdef double max_drawdown = 0
        cdef double dd
        cdef int i
        
        for i in range(1, n):
            if portfolio_values[i] > max_peak:
                max_peak = portfolio_values[i]
            dd = (max_peak - portfolio_values[i]) / max_peak
            if dd > max_drawdown:
                max_drawdown = dd
        return max_drawdown

    def calculate_rolling_var(self, np.ndarray[np.float64_t, ndim=1] returns, 
                             int window, double confidence=0.95):
        """Calculate rolling VaR over specified window"""
        cdef int n = len(returns)
        cdef np.ndarray[np.float64_t, ndim=1] rolling_var = np.zeros(n - window + 1)
        cdef int i
        
        for i in range(window - 1, n):
            window_returns = returns[i - window + 1:i + 1]
            rolling_var[i - window + 1] = self.calculate_var(window_returns, confidence)
        
        return rolling_var
