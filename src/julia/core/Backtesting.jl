"""
Advanced Backtesting Framework in Julia
Ultra-high performance backtesting with GPU acceleration
"""

module Backtesting

using DataFrames
using Statistics
using LinearAlgebra
using CUDA
using Plots
using StatsBase

export BacktestEngine, run_backtest, calculate_metrics

struct BacktestEngine{T}
    prices::Matrix{T}
    signals::Matrix{T}
    initial_capital::Vector{T}
    commission_rate::T
    use_gpu::Bool
end

function BacktestEngine(prices, signals, initial_capital; commission_rate=0.001, use_gpu=false)
    if use_gpu && CUDA.functional()
        prices_gpu = CuArray(prices)
        signals_gpu = CuArray(signals)
        return BacktestEngine(prices_gpu, signals_gpu, initial_capital, commission_rate, true)
    else
        return BacktestEngine(prices, signals, initial_capital, commission_rate, false)
    end
end

function run_backtest(engine::BacktestEngine{T}) where T
    if engine.use_gpu
        return gpu_backtest(engine)
    else
        return cpu_backtest(engine)
    end
end

function cpu_backtest(engine::BacktestEngine{T}) where T
    n_timesteps, n_assets = size(engine.prices)
    
    # Initialize arrays
    portfolio_values = zeros(T, n_timesteps, n_assets)
    positions = zeros(T, n_timesteps, n_assets)
    trades = zeros(T, n_timesteps, n_assets)
    cash = copy(engine.initial_capital)
    
    # Set initial portfolio values
    portfolio_values[1, :] .= engine.initial_capital
    
    # Main backtesting loop (vectorized where possible)
    @inbounds for t in 2:n_timesteps
        for a in 1:n_assets
            current_position = positions[t-1, a]
            current_price = engine.prices[t, a]
            signal = engine.signals[t, a]
            
            # Calculate target position
            portfolio_value = cash[a] + current_position * current_price
            target_position = signal * portfolio_value / current_price
            
            # Calculate trade
            trade_size = target_position - current_position
            trades[t, a] = trade_size
            
            # Update cash and position (with commission)
            commission = abs(trade_size) * current_price * engine.commission_rate
            cash[a] -= trade_size * current_price + commission
            positions[t, a] = target_position
            
            # Update portfolio value
            portfolio_values[t, a] = cash[a] + positions[t, a] * current_price
        end
    end
    
    return BacktestResult(portfolio_values, positions, trades)
end

function gpu_backtest(engine::BacktestEngine{T}) where T
    # GPU-accelerated backtesting using CUDA kernels
    n_timesteps, n_assets = size(engine.prices)
    
    # Transfer data to GPU
    prices_gpu = engine.prices
    signals_gpu = engine.signals
    
    # Initialize GPU arrays
    portfolio_values_gpu = CUDA.zeros(T, n_timesteps, n_assets)
    positions_gpu = CUDA.zeros(T, n_timesteps, n_assets)
    trades_gpu = CUDA.zeros(T, n_timesteps, n_assets)
    
    # Set initial values
    portfolio_values_gpu[1, :] .= CuArray(engine.initial_capital)
    
    # Launch CUDA kernel
    threads_per_block = 256
    blocks = cld(n_assets, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks backtest_kernel!(
        portfolio_values_gpu, positions_gpu, trades_gpu,
        prices_gpu, signals_gpu, 
        CuArray(engine.initial_capital), engine.commission_rate,
        n_timesteps, n_assets
    )
    
    # Transfer results back to CPU
    portfolio_values = Array(portfolio_values_gpu)
    positions = Array(positions_gpu)
    trades = Array(trades_gpu)
    
    return BacktestResult(portfolio_values, positions, trades)
end

function backtest_kernel!(portfolio_values, positions, trades, prices, signals, 
                         initial_capital, commission_rate, n_timesteps, n_assets)
    asset_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if asset_idx <= n_assets
        cash = initial_capital[asset_idx]
        
        for t in 2:n_timesteps
            current_position = positions[t-1, asset_idx]
            current_price = prices[t, asset_idx]
            signal = signals[t, asset_idx]
            
            # Calculate target position
            portfolio_value = cash + current_position * current_price
            target_position = signal * portfolio_value / current_price
            
            # Calculate trade
            trade_size = target_position - current_position
            trades[t, asset_idx] = trade_size
            
            # Update cash and position
            commission = abs(trade_size) * current_price * commission_rate
            cash -= trade_size * current_price + commission
            positions[t, asset_idx] = target_position
            
            # Update portfolio value
            portfolio_values[t, asset_idx] = cash + positions[t, asset_idx] * current_price
        end
    end
    
    return nothing
end

struct BacktestResult{T}
    portfolio_values::Matrix{T}
    positions::Matrix{T}
    trades::Matrix{T}
end

function calculate_metrics(result::BacktestResult)
    """Calculate comprehensive performance metrics"""
    
    # Calculate returns
    returns = diff(log.(result.portfolio_values[:, 1]))
    
    # Performance metrics
    total_return = result.portfolio_values[end, 1] / result.portfolio_values[1, 1] - 1
    volatility = std(returns) * sqrt(252)
    sharpe_ratio = mean(returns) * 252 / volatility
    
    # Calculate maximum drawdown
    cumulative_returns = cumprod(1 .+ returns)
    peak = cumulative_returns[1]
    max_drawdown = 0.0
    
    for i in 2:length(cumulative_returns)
        if cumulative_returns[i] > peak
            peak = cumulative_returns[i]
        end
        drawdown = (peak - cumulative_returns[i]) / peak
        max_drawdown = max(max_drawdown, drawdown)
    end
    
    return Dict(
        "total_return" => total_return,
        "volatility" => volatility,
        "sharpe_ratio" => sharpe_ratio,
        "max_drawdown" => max_drawdown,
        "calmar_ratio" => total_return / max_drawdown
    )
end

end # module
