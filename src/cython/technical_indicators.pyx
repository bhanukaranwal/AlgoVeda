# distutils: language=c++
# cython: language_level=3

"""
High-Performance Technical Indicators in Cython
SMA, EMA, RSI, Bollinger Bands optimized for trading systems
"""

import numpy as np
cimport numpy as np

cdef class TechnicalIndicators:
    def sma(self, np.ndarray[np.float64_t, ndim=1] prices, int period):
        """Simple Moving Average"""
        cdef int n = prices.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] output = np.zeros(n - period + 1, dtype=np.float64)
        cdef int i, j
        cdef double s
        
        for i in range(n - period + 1):
            s = 0
            for j in range(i, i + period):
                s += prices[j]
            output[i] = s / period
        return output

    def ema(self, np.ndarray[np.float64_t, ndim=1] prices, int period):
        """Exponential Moving Average"""
        cdef int n = prices.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] output = np.zeros(n, dtype=np.float64)
        cdef int i
        cdef double alpha = 2.0 / (period + 1)
        
        output[0] = prices[0]
        for i in range(1, n):
            output[i] = alpha * prices[i] + (1 - alpha) * output[i - 1]
        return output

    def rsi(self, np.ndarray[np.float64_t, ndim=1] prices, int period):
        """Relative Strength Index"""
        cdef int n = prices.shape[0]
        cdef np.ndarray[np.float64_t, ndim=1] deltas = np.diff(prices)
        cdef np.ndarray[np.float64_t, ndim=1] gains = np.zeros(n - 1, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] losses = np.zeros(n - 1, dtype=np.float64)
        cdef int i
        
        for i in range(n - 1):
            if deltas[i] > 0:
                gains[i] = deltas[i]
            else:
                losses[i] = -deltas[i]
        
        avg_gain = np.convolve(gains, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period) / period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def bollinger_bands(self, np.ndarray[np.float64_t, ndim=1] prices, int period):
        """Bollinger Bands (Upper, Lower)"""
        ma = self.sma(prices, period)
        cdef int n = len(ma)
        cdef np.ndarray[np.float64_t, ndim=1] stddevs = np.zeros(n, dtype=np.float64)
        cdef int i, j
        cdef double s
        
        for i in range(n):
            s = 0
            for j in range(i, i + period):
                s += (prices[j] - ma[i]) ** 2
            stddevs[i] = (s / period) ** 0.5
        
        upper = ma + 2 * stddevs
        lower = ma - 2 * stddevs
        return upper, lower
