/*!
 * Backtesting Analytics React Hook
 * Comprehensive hook for managing backtest data, strategies, and comparisons
 */

import { useState, useEffect, useCallback, useMemo } from 'react';

interface BacktestStrategy {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, number>;
  universe: string[];
  rebalance_frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
}

interface BacktestResult {
  strategy_id: string;
  strategy_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  win_rate: number;
  profit_factor: number;
  trades_count: number;
  equity_curve: Array<{
    date: string;
    value: number;
    drawdown: number;
    benchmark_value?: number;
  }>;
  monthly_returns: Array<{
    month: string;
    return: number;
    benchmark_return?: number;
  }>;
  trade_analysis: Array<{
    date: string;
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
    duration_days: number;
  }>;
  risk_metrics: {
    var_95: number;
    var_99: number;
    expected_shortfall_95: number;
    expected_shortfall_99: number;
    beta: number;
    alpha: number;
    information_ratio: number;
    tracking_error: number;
  };
}

interface BacktestComparison {
  strategies: BacktestResult[];
  comparison_metrics: {
    best_return: string;
    best_sharpe: string;
    lowest_drawdown: string;
    correlation_matrix: number[][];
  };
}

interface UseBacktestingAnalyticsReturn {
  // State
  backtestResults: BacktestResult[];
  strategies: BacktestStrategy[];
  selectedStrategies: string[];
  isLoading: boolean;
  error: string | null;
  
  // Actions
  runBacktest: (strategyIds: string[], config?: BacktestConfig) => Promise<void>;
  compareStrategies: (strategyIds: string[]) => BacktestComparison | null;
  optimizeStrategy: (strategyId: string, parameters: Record<string, [number, number]>) => Promise<BacktestResult>;
  loadStrategies: () => Promise<void>;
  saveStrategy: (strategy: BacktestStrategy) => Promise<void>;
  
  // Computed values
  bestPerformingStrategy: BacktestResult | null;
  worstPerformingStrategy: BacktestResult | null;
  averageMetrics: Record<string, number>;
}

interface BacktestConfig {
  start_date: string;
  end_date: string;
  initial_capital: number;
  benchmark: string;
  commission_rate: number;
  slippage_rate: number;
  enable_margin: boolean;
  max_leverage: number;
}

const DEFAULT_CONFIG: BacktestConfig = {
  start_date: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
  end_date: new Date().toISOString().split('T')[0],
  initial_capital: 100000,
  benchmark: 'SPY',
  commission_rate: 0.001,
  slippage_rate: 0.0001,
  enable_margin: false,
  max_leverage: 1.0,
};

export const useBacktestingAnalytics = (): UseBacktestingAnalyticsReturn => {
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [strategies, setStrategies] = useState<BacktestStrategy[]>([]);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available strategies
  const loadStrategies = useCallback(async () => {
    try {
      const response = await fetch('/api/strategies');
      if (!response.ok) throw new Error('Failed to load strategies');
      
      const strategiesData = await response.json();
      setStrategies(strategiesData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  // Run backtest for selected strategies
  const runBacktest = useCallback(async (
    strategyIds: string[], 
    config: BacktestConfig = DEFAULT_CONFIG
  ) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/backtest/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_ids: strategyIds,
          config,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backtest failed: ${response.statusText}`);
      }

      const results = await response.json();
      setBacktestResults(prev => {
        // Replace existing results for the same strategies
        const filtered = prev.filter(r => !strategyIds.includes(r.strategy_id));
        return [...filtered, ...results];
      });

      setSelectedStrategies(strategyIds);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Compare multiple strategies
  const compareStrategies = useCallback((strategyIds: string[]): BacktestComparison | null => {
    const results = backtestResults.filter(r => strategyIds.includes(r.strategy_id));
    
    if (results.length < 2) return null;

    // Calculate comparison metrics
    const returns = results.map(r => r.total_return);
    const sharpeRatios = results.map(r => r.sharpe_ratio);
    const drawdowns = results.map(r => r.max_drawdown);

    const bestReturn = results[returns.indexOf(Math.max(...returns))];
    const bestSharpe = results[sharpeRatios.indexOf(Math.max(...sharpeRatios))];
    const lowestDrawdown = results[drawdowns.indexOf(Math.min(...drawdowns))];

    // Calculate correlation matrix (simplified)
    const correlationMatrix = calculateCorrelationMatrix(
      results.map(r => r.equity_curve.map(point => point.value))
    );

    return {
      strategies: results,
      comparison_metrics: {
        best_return: bestReturn.strategy_name,
        best_sharpe: bestSharpe.strategy_name,
        lowest_drawdown: lowestDrawdown.strategy_name,
        correlation_matrix: correlationMatrix,
      },
    };
  }, [backtestResults]);

  // Strategy optimization using parameter sweep
  const optimizeStrategy = useCallback(async (
    strategyId: string,
    parameters: Record<string, [number, number]>
  ): Promise<BacktestResult> => {
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/backtest/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_id: strategyId,
          parameter_ranges: parameters,
          optimization_metric: 'sharpe_ratio',
        }),
      });

      if (!response.ok) {
        throw new Error('Strategy optimization failed');
      }

      const optimizedResult = await response.json();
      
      // Update results with optimized strategy
      setBacktestResults(prev => {
        const filtered = prev.filter(r => r.strategy_id !== strategyId);
        return [...filtered, optimizedResult];
      });

      return optimizedResult;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Optimization failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Save strategy configuration
  const saveStrategy = useCallback(async (strategy: BacktestStrategy) => {
    try {
      const response = await fetch('/api/strategies', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(strategy),
      });

      if (!response.ok) {
        throw new Error('Failed to save strategy');
      }

      await loadStrategies(); // Reload strategies
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save strategy');
    }
  }, [loadStrategies]);

  // Computed values
  const bestPerformingStrategy = useMemo(() => {
    if (backtestResults.length === 0) return null;
    return backtestResults.reduce((best, current) => 
      current.sharpe_ratio > best.sharpe_ratio ? current : best
    );
  }, [backtestResults]);

  const worstPerformingStrategy = useMemo(() => {
    if (backtestResults.length === 0) return null;
    return backtestResults.reduce((worst, current) => 
      current.sharpe_ratio < worst.sharpe_ratio ? current : worst
    );
  }, [backtestResults]);

  const averageMetrics = useMemo(() => {
    if (backtestResults.length === 0) return {};

    const metrics = backtestResults.reduce((acc, result) => {
      return {
        total_return: acc.total_return + result.total_return,
        sharpe_ratio: acc.sharpe_ratio + result.sharpe_ratio,
        max_drawdown: acc.max_drawdown + result.max_drawdown,
        volatility: acc.volatility + result.volatility,
        win_rate: acc.win_rate + result.win_rate,
      };
    }, {
      total_return: 0,
      sharpe_ratio: 0,
      max_drawdown: 0,
      volatility: 0,
      win_rate: 0,
    });

    const count = backtestResults.length;
    return {
      total_return: metrics.total_return / count,
      sharpe_ratio: metrics.sharpe_ratio / count,
      max_drawdown: metrics.max_drawdown / count,
      volatility: metrics.volatility / count,
      win_rate: metrics.win_rate / count,
    };
  }, [backtestResults]);

  // Load strategies on mount
  useEffect(() => {
    loadStrategies();
  }, [loadStrategies]);

  return {
    backtestResults,
    strategies,
    selectedStrategies,
    isLoading,
    error,
    runBacktest,
    compareStrategies,
    optimizeStrategy,
    loadStrategies,
    saveStrategy,
    bestPerformingStrategy,
    worstPerformingStrategy,
    averageMetrics,
  };
};

// Helper function to calculate correlation matrix
function calculateCorrelationMatrix(series: number[][]): number[][] {
  const n = series.length;
  const correlations = Array(n).fill(null).map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        correlations[i][j] = 1;
      } else {
        correlations[i][j] = calculateCorrelation(series[i], series[j]);
      }
    }
  }

  return correlations;
}

function calculateCorrelation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  const meanX = x.slice(0, n).reduce((a, b) => a + b) / n;
  const meanY = y.slice(0, n).reduce((a, b) => a + b) / n;

  let numerator = 0;
  let denomX = 0;
  let denomY = 0;

  for (let i = 0; i < n; i++) {
    const diffX = x[i] - meanX;
    const diffY = y[i] - meanY;
    
    numerator += diffX * diffY;
    denomX += diffX * diffX;
    denomY += diffY * diffY;
  }

  const denominator = Math.sqrt(denomX * denomY);
  return denominator === 0 ? 0 : numerator / denominator;
}

export default useBacktestingAnalytics;
