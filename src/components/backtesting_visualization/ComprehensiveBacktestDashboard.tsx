/*!
 * Comprehensive Backtesting Dashboard
 * Interactive visualization of backtesting results with performance analytics
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Chip
} from '@mui/material';
import {
  Timeline,
  TrendingUp,
  Assessment,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon,
  ShowChart
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
  ReferenceLine
} from 'recharts';
import { styled } from '@mui/material/styles';
import { useBacktestingAnalytics } from '../../hooks/useBacktestingAnalytics';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  background: 'linear-gradient(145deg, #1e1e1e 0%, #2d2d2d 100%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 12,
  color: '#ffffff',
}));

const MetricCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 8,
  color: '#ffffff',
}));

interface BacktestResult {
  strategy_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  win_rate: number;
  profit_factor: number;
  trades_count: number;
  equity_curve: Array<{date: string, value: number, drawdown: number}>;
  monthly_returns: Array<{month: string, return: number}>;
  trade_analysis: Array<{
    date: string;
    symbol: string;
    side: string;
    quantity: number;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
  }>;
}

export const ComprehensiveBacktestDashboard: React.FC = () => {
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  const [showBenchmark, setShowBenchmark] = useState(true);
  const [viewMode, setViewMode] = useState('returns'); // 'returns' | 'drawdown' | 'underwater'

  const {
    backtestResults,
    strategies,
    isLoading,
    runBacktest,
    compareStrategies
  } = useBacktestingAnalytics();

  // Performance metrics calculation
  const performanceMetrics = useMemo(() => {
    if (!backtestResults || backtestResults.length === 0) return null;

    const result = backtestResults.find(r => r.strategy_name === selectedStrategy) || backtestResults[0];
    
    return {
      totalReturn: result.total_return,
      annualizedReturn: result.annualized_return,
      volatility: result.volatility,
      sharpeRatio: result.sharpe_ratio,
      maxDrawdown: result.max_drawdown,
      calmarRatio: result.calmar_ratio,
      winRate: result.win_rate,
      profitFactor: result.profit_factor,
      tradesCount: result.trades_count,
      finalValue: result.final_value,
      initialCapital: result.initial_capital
    };
  }, [backtestResults, selectedStrategy]);

  // Risk metrics
  const riskMetrics = useMemo(() => {
    if (!performanceMetrics) return null;

    return {
      var95: performanceMetrics.volatility * 1.645, // Simplified VaR
      sortino: performanceMetrics.sharpeRatio * 1.2, // Approximation
      ulcerIndex: performanceMetrics.maxDrawdown * 0.8,
      sterlingRatio: performanceMetrics.annualizedReturn / Math.abs(performanceMetrics.maxDrawdown),
      burkeRatio: performanceMetrics.annualizedReturn / Math.sqrt(performanceMetrics.maxDrawdown)
    };
  }, [performanceMetrics]);

  // Chart colors
  const colors = {
    primary: '#2196f3',
    secondary: '#ff9800',
    success: '#4caf50',
    error: '#f44336',
    warning: '#ff5722',
    info: '#00bcd4'
  };

  const TabPanel = ({ children, value, index }: any) => (
    <div hidden={value !== index}>
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );

  return (
    <Box>
      {/* Header Controls */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Backtesting Analytics Dashboard
        </Typography>
        
        <Box display="flex" gap={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Strategy</InputLabel>
            <Select
              value={selectedStrategy}
              label="Strategy"
              onChange={(e) => setSelectedStrategy(e.target.value)}
            >
              {strategies.map((strategy) => (
                <MenuItem key={strategy.id} value={strategy.name}>
                  {strategy.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControlLabel
            control={
              <Switch
                checked={showBenchmark}
                onChange={(e) => setShowBenchmark(e.target.checked)}
              />
            }
            label="Show Benchmark"
          />
          
          <Button
            variant="contained"
            onClick={() => runBacktest(selectedStrategy)}
            disabled={isLoading}
          >
            Run Backtest
          </Button>
        </Box>
      </Box>

      {/* Performance Metrics Cards */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} md={3}>
          <MetricCard>
            <CardContent>
              <Typography variant="h6" color="primary">Total Return</Typography>
              <Typography variant="h4">
                {performanceMetrics?.totalReturn.toFixed(2)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                ${performanceMetrics?.finalValue.toLocaleString()}
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <MetricCard>
            <CardContent>
              <Typography variant="h6" color="primary">Sharpe Ratio</Typography>
              <Typography variant="h4">
                {performanceMetrics?.sharpeRatio.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Risk-adjusted return
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <MetricCard>
            <CardContent>
              <Typography variant="h6" color="primary">Max Drawdown</Typography>
              <Typography variant="h4" color="error">
                {performanceMetrics?.maxDrawdown.toFixed(2)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Peak to trough decline
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <MetricCard>
            <CardContent>
              <Typography variant="h6" color="primary">Win Rate</Typography>
              <Typography variant="h4" color="success">
                {performanceMetrics?.winRate.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {performanceMetrics?.tradesCount} total trades
              </Typography>
            </CardContent>
          </MetricCard>
        </Grid>
      </Grid>

      {/* Main Charts Section */}
      <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)} mb={2}>
        <Tab label="Equity Curve" icon={<ShowChart />} />
        <Tab label="Drawdown" icon={<Timeline />} />
        <Tab label="Monthly Returns" icon={<BarChartIcon />} />
        <Tab label="Trade Analysis" icon={<Assessment />} />
        <Tab label="Risk Analysis" icon={<PieChartIcon />} />
      </Tabs>

      <TabPanel value={activeTab} index={0}>
        {/* Equity Curve Chart */}
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Equity Curve & Drawdown
          </Typography>
          <Box height={500}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={backtestResults[0]?.equity_curve || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="#ffffff" />
                <YAxis yAxisId="left" stroke="#ffffff" />
                <YAxis yAxisId="right" orientation="right" stroke="#ffffff" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d2d',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff'
                  }}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="value"
                  stroke={colors.primary}
                  fill={colors.primary}
                  fillOpacity={0.3}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="drawdown"
                  stroke={colors.error}
                  strokeWidth={2}
                />
                <ReferenceLine yAxisId="left" y={performanceMetrics?.initialCapital} stroke="#666" strokeDasharray="5 5" />
              </ComposedChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        {/* Drawdown Chart */}
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Underwater Plot (Drawdown from Peak)
          </Typography>
          <Box height={400}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={backtestResults[0]?.equity_curve || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="#ffffff" />
                <YAxis stroke="#ffffff" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d2d',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  stroke={colors.error}
                  fill={colors.error}
                  fillOpacity={0.6}
                />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="5 5" />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        {/* Monthly Returns Heatmap */}
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Monthly Returns Distribution
          </Typography>
          <Box height={400}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={backtestResults[0]?.monthly_returns || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="month" stroke="#ffffff" />
                <YAxis stroke="#ffffff" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d2d',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff'
                  }}
                />
                <Bar dataKey="return" fill={colors.primary} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        {/* Trade Analysis */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Trade Returns Scatter Plot
              </Typography>
              <Box height={400}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={backtestResults[0]?.trade_analysis || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="entry_price" stroke="#ffffff" />
                    <YAxis dataKey="return_pct" stroke="#ffffff" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#2d2d2d',
                        border: '1px solid rgba(255,255,255,0.1)',
                        color: '#ffffff'
                      }}
                    />
                    <Scatter dataKey="return_pct" fill={colors.secondary} />
                  </ScatterChart>
                </ResponsiveContainer>
              </Box>
            </StyledPaper>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Trade Statistics
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Win Rate:</Typography>
                  <Chip
                    label={`${performanceMetrics?.winRate.toFixed(1)}%`}
                    color="success"
                    size="small"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Profit Factor:</Typography>
                  <Chip
                    label={performanceMetrics?.profitFactor.toFixed(2)}
                    color="primary"
                    size="small"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Total Trades:</Typography>
                  <Typography>{performanceMetrics?.tradesCount}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Avg. Return:</Typography>
                  <Typography>
                    {((performanceMetrics?.totalReturn || 0) / (performanceMetrics?.tradesCount || 1)).toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={4}>
        {/* Risk Analysis */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Risk Metrics
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Value at Risk (95%):</Typography>
                  <Typography color="error">{riskMetrics?.var95.toFixed(2)}%</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Sortino Ratio:</Typography>
                  <Typography>{riskMetrics?.sortino.toFixed(2)}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Ulcer Index:</Typography>
                  <Typography>{riskMetrics?.ulcerIndex.toFixed(2)}%</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Sterling Ratio:</Typography>
                  <Typography>{riskMetrics?.sterlingRatio.toFixed(2)}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography>Burke Ratio:</Typography>
                  <Typography>{riskMetrics?.burkeRatio.toFixed(2)}</Typography>
                </Box>
              </Box>
            </StyledPaper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Return vs Risk Comparison
              </Typography>
              <Box height={300}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                      dataKey="volatility"
                      type="number"
                      domain={['dataMin - 1', 'dataMax + 1']}
                      stroke="#ffffff"
                      label={{ value: 'Volatility (%)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      dataKey="return"
                      type="number"
                      domain={['dataMin - 1', 'dataMax + 1']}
                      stroke="#ffffff"
                      label={{ value: 'Return (%)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#2d2d2d',
                        border: '1px solid rgba(255,255,255,0.1)',
                        color: '#ffffff'
                      }}
                    />
                    <Scatter
                      data={[{
                        volatility: performanceMetrics?.volatility,
                        return: performanceMetrics?.annualizedReturn,
                        strategy: selectedStrategy
                      }]}
                      fill={colors.primary}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </Box>
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default ComprehensiveBacktestDashboard;
