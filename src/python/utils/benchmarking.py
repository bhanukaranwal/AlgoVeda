"""
Comprehensive Code Benchmarking Suite for AlgoVeda
Performance measurement and optimization analysis
"""

import time
import numpy as np
import pandas as pd
import psutil
import gc
import functools
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager
import cProfile
import pstats
import io
from memory_profiler import profile
import line_profiler

@dataclass
class BenchmarkResult:
    execution_time: float
    memory_usage: float
    cpu_usage: float
    iterations: int
    throughput: float
    latency_percentiles: Dict[str, float]

class PerformanceBenchmarker:
    """Ultra-comprehensive performance benchmarking suite"""
    
    def __init__(self, warmup_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.results = {}
        
    def benchmark(self, func: Callable, iterations: int = 1000, *args, **kwargs) -> BenchmarkResult:
        """Comprehensive function benchmarking"""
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        gc.collect()
        
        # Memory and CPU monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_times = []
        cpu_times = []
        
        # Benchmark iterations
        for _ in range(iterations):
            cpu_start = process.cpu_times()
            start_time = time.perf_counter()
            
            func(*args, **kwargs)
            
            end_time = time.perf_counter()
            cpu_end = process.cpu_times()
            
            execution_times.append(end_time - start_time)
            cpu_times.append((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # Calculate statistics
        avg_time = np.mean(execution_times)
        avg_cpu = np.mean(cpu_times)
        throughput = iterations / sum(execution_times)
        
        percentiles = {
            'p50': np.percentile(execution_times, 50) * 1000,  # ms
            'p90': np.percentile(execution_times, 90) * 1000,
            'p95': np.percentile(execution_times, 95) * 1000,
            'p99': np.percentile(execution_times, 99) * 1000,
            'p99.9': np.percentile(execution_times, 99.9) * 1000,
        }
        
        return BenchmarkResult(
            execution_time=avg_time,
            memory_usage=memory_usage,
            cpu_usage=avg_cpu,
            iterations=iterations,
            throughput=throughput,
            latency_percentiles=percentiles
        )

def benchmark_trading_functions():
    """Benchmark all critical trading functions"""
    benchmarker = PerformanceBenchmarker()
    
    # Sample trading functions to benchmark
    results = {
        'option_pricing': benchmarker.benchmark(sample_option_pricing, 10000),
        'portfolio_var': benchmarker.benchmark(sample_portfolio_var, 5000),
        'monte_carlo_simulation': benchmarker.benchmark(sample_monte_carlo, 1000),
    }
    
    return results

# Sample functions for benchmarking
def sample_option_pricing():
    """Sample Black-Scholes calculation"""
    S, K, r, T, sigma = 100, 105, 0.05, 0.25, 0.2
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-0.5*d1**2)/np.sqrt(2*np.pi) - K*np.exp(-r*T)*np.exp(-0.5*d2**2)/np.sqrt(2*np.pi)

def sample_portfolio_var():
    """Sample VaR calculation"""
    returns = np.random.normal(0.001, 0.02, 252)
    return np.percentile(returns, 5)

def sample_monte_carlo():
    """Sample Monte Carlo simulation"""
    paths = np.random.normal(0, 1, (1000, 252))
    return np.mean(np.maximum(100 * np.exp(np.cumsum(paths, axis=1)[:, -1]) - 105, 0))
