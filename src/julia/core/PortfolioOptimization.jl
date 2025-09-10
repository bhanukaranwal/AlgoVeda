"""
Advanced Portfolio Optimization in Julia
GPU-accelerated mean-variance and beyond optimization
"""

module PortfolioOptimization

using LinearAlgebra
using Convex
using SCS
using CUDA
using Statistics
using Distributions

export optimize_portfolio, black_litterman, risk_parity, optimize_gpu

struct OptimizationResult
    weights::Vector{Float64}
    expected_return::Float64
    volatility::Float64
    sharpe_ratio::Float64
    solver_status::String
end

"""
Mean-Variance Optimization with multiple constraints
"""
function optimize_portfolio(
    expected_returns::Vector{Float64},
    covariance_matrix::Matrix{Float64};
    target_return::Union{Float64, Nothing} = nothing,
    max_weight::Float64 = 1.0,
    min_weight::Float64 = 0.0,
    transaction_costs::Vector{Float64} = zeros(length(expected_returns)),
    risk_aversion::Float64 = 1.0
)
    n_assets = length(expected_returns)
    
    # Define optimization variables
    w = Variable(n_assets)
    
    # Objective: maximize expected return - risk penalty
    if target_return === nothing
        # Mean-variance optimization
        objective = expected_returns' * w - 0.5 * risk_aversion * quadform(w, covariance_matrix)
        problem = maximize(objective)
    else
        # Minimum variance for target return
        objective = quadform(w, covariance_matrix)
        problem = minimize(objective)
        problem.constraints += [expected_returns' * w >= target_return]
    end
    
    # Standard constraints
    problem.constraints += [sum(w) == 1.0]
    problem.constraints += [w >= min_weight]
    problem.constraints += [w <= max_weight]
    
    # Solve optimization
    solve!(problem, SCS.Optimizer(verbose=false))
    
    if problem.status == Convex.OPTIMAL
        optimal_weights = evaluate(w)
        portfolio_return = expected_returns' * optimal_weights
        portfolio_variance = optimal_weights' * covariance_matrix * optimal_weights
        portfolio_volatility = sqrt(portfolio_variance)
        sharpe = portfolio_return / portfolio_volatility
        
        return OptimizationResult(
            optimal_weights,
            portfolio_return,
            portfolio_volatility,
            sharpe,
            string(problem.status)
        )
    else
        error("Optimization failed with status: $(problem.status)")
    end
end

"""
Black-Litterman Model Implementation
"""
function black_litterman(
    market_weights::Vector{Float64},
    covariance_matrix::Matrix{Float64},
    views_matrix::Matrix{Float64},
    view_returns::Vector{Float64},
    view_uncertainty::Matrix{Float64};
    risk_aversion::Float64 = 3.0,
    tau::Float64 = 0.025
)
    n_assets = length(market_weights)
    
    # Market-implied expected returns (reverse optimization)
    pi = risk_aversion * covariance_matrix * market_weights
    
    # Black-Litterman formula
    M1 = inv(tau * covariance_matrix)
    M2 = views_matrix' * inv(view_uncertainty) * views_matrix
    M3 = inv(tau * covariance_matrix) * pi
    M4 = views_matrix' * inv(view_uncertainty) * view_returns
    
    # New expected returns
    mu_bl = inv(M1 + M2) * (M3 + M4)
    
    # New covariance matrix
    sigma_bl = inv(M1 + M2)
    
    return mu_bl, sigma_bl
end

"""
Risk Parity Portfolio Optimization
"""
function risk_parity(covariance_matrix::Matrix{Float64}; max_iter::Int = 1000, tol::Float64 = 1e-8)
    n_assets = size(covariance_matrix, 1)
    
    # Initial equal weights
    weights = ones(n_assets) / n_assets
    
    for iter in 1:max_iter
        # Calculate risk contributions
        portfolio_vol = sqrt(weights' * covariance_matrix * weights)
        marginal_risk = covariance_matrix * weights / portfolio_vol
        risk_contributions = weights .* marginal_risk / portfolio_vol
        
        # Target risk contribution (equal for all assets)
        target_risk = 1.0 / n_assets
        
        # Update weights using iterative method
        weights_new = weights .* (target_risk ./ risk_contributions)
        weights_new = weights_new / sum(weights_new)  # Normalize
        
        # Check convergence
        if maximum(abs.(weights_new - weights)) < tol
            return weights_new
        end
        
        weights = weights_new
    end
    
    @warn "Risk parity optimization did not converge"
    return weights
end

"""
GPU-Accelerated Portfolio Optimization
"""
function optimize_gpu(
    expected_returns::CuArray{Float64},
    covariance_matrix::CuArray{Float64},
    num_portfolios::Int = 10000
)
    if !CUDA.functional()
        @warn "CUDA not available, falling back to CPU"
        return optimize_portfolio(Array(expected_returns), Array(covariance_matrix))
    end
    
    n_assets = length(expected_returns)
    
    # Generate random portfolio weights on GPU
    weights_matrix = CUDA.rand(Float64, n_assets, num_portfolios)
    
    # Normalize weights to sum to 1
    weights_sums = sum(weights_matrix, dims=1)
    weights_matrix = weights_matrix ./ weights_sums
    
    # Calculate portfolio returns and risks on GPU
    portfolio_returns = expected_returns' * weights_matrix
    portfolio_variances = diag(weights_matrix' * covariance_matrix * weights_matrix)
    portfolio_volatilities = sqrt.(portfolio_variances)
    
    # Calculate Sharpe ratios
    sharpe_ratios = portfolio_returns ./ portfolio_volatilities
    
    # Find optimal portfolio (maximum Sharpe ratio)
    max_sharpe_idx = argmax(Array(sharpe_ratios))
    optimal_weights = Array(weights_matrix[:, max_sharpe_idx])
    
    return OptimizationResult(
        optimal_weights,
        Array(portfolio_returns)[max_sharpe_idx],
        Array(portfolio_volatilities)[max_sharpe_idx],
        Array(sharpe_ratios)[max_sharpe_idx],
        "OPTIMAL_GPU"
    )
end

"""
Multi-Objective Portfolio Optimization
"""
function multi_objective_optimization(
    expected_returns::Vector{Float64},
    covariance_matrix::Matrix{Float64},
    esg_scores::Vector{Float64};
    return_weight::Float64 = 0.5,
    risk_weight::Float64 = 0.3,
    esg_weight::Float64 = 0.2
)
    n_assets = length(expected_returns)
    
    # Normalize ESG scores
    normalized_esg = (esg_scores .- minimum(esg_scores)) ./ (maximum(esg_scores) - minimum(esg_scores))
    
    # Define optimization variables
    w = Variable(n_assets)
    
    # Multi-objective function
    return_objective = expected_returns' * w
    risk_objective = quadform(w, covariance_matrix)
    esg_objective = normalized_esg' * w
    
    # Combined objective (maximize returns and ESG, minimize risk)
    objective = return_weight * return_objective + 
                esg_weight * esg_objective - 
                risk_weight * risk_objective
    
    problem = maximize(objective)
    problem.constraints += [sum(w) == 1.0]
    problem.constraints += [w >= 0.0]
    
    solve!(problem, SCS.Optimizer(verbose=false))
    
    if problem.status == Convex.OPTIMAL
        return evaluate(w)
    else
        error("Multi-objective optimization failed")
    end
end

"""
Dynamic Portfolio Optimization with Rebalancing
"""
function dynamic_optimization(
    returns_matrix::Matrix{Float64},  # Time x Assets
    rebalance_frequency::Int = 22;    # Monthly rebalancing
    lookback_window::Int = 252        # 1 year lookback
)
    n_periods, n_assets = size(returns_matrix)
    portfolio_weights = zeros(n_periods, n_assets)
    
    for t in lookback_window:rebalance_frequency:n_periods
        # Get historical data
        historical_returns = returns_matrix[max(1, t-lookback_window+1):t, :]
        
        # Calculate expected returns and covariance
        expected_returns = mean(historical_returns, dims=1)[1, :]
        covariance_matrix = cov(historical_returns)
        
        # Optimize portfolio
        try
            result = optimize_portfolio(expected_returns, covariance_matrix)
            
            # Apply weights for rebalancing period
            end_period = min(t + rebalance_frequency - 1, n_periods)
            for period in t:end_period
                portfolio_weights[period, :] = result.weights
            end
        catch e
            @warn "Optimization failed at period $t: $e"
            # Use equal weights as fallback
            portfolio_weights[t:min(t+rebalance_frequency-1, n_periods), :] .= 1.0 / n_assets
        end
    end
    
    return portfolio_weights
end

end # module PortfolioOptimization
