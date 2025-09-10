/**
 * Main Trading Dashboard for AlgoVeda Platform
 * Features comprehensive real-time data visualization, portfolio monitoring,
 * risk management, and strategy performance analytics with WebGL acceleration.
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Tab,
  Tabs,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  Switch,
  FormControlLabel,
  Tooltip,
  IconButton,
  Badge,
} from '@mui/material';
import {
  Timeline,
  TrendingUp,
  TrendingDown,
  AccountBalance,
  Speed,
  Security,
  Assessment,
  Settings,
  Fullscreen,
  Refresh,
  NotificationsActive,
} from '@mui/icons-material';
import { styled, useTheme } from '@mui/material/styles';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';

// Custom hooks and services
import { useRealtimeData } from '../../hooks/useRealtimeData';
import { usePerformanceAnalytics } from '../../hooks/usePerformanceAnalytics';
import { useRiskAnalytics } from '../../hooks/useRiskAnalytics';
import { useWebSocket } from '../../hooks/useWebSocket';
import { chartRenderingService } from '../../services/chart-rendering-service';

// Chart components with WebGL acceleration
import { EquityCurveChart } from '../charts/EquityCurveChart';
import { PortfolioCompositionChart } from '../charts/PortfolioCompositionChart';
import { RiskMetricsChart } from '../charts/RiskMetricsChart';
import { VolatilitySurface3D } from '../options_analytics/VolatilitySurface3D';
import { OrderBookHeatmap } from '../charts/OrderBookHeatmap';
import { PerformanceMetricsTable } from '../tables/PerformanceMetricsTable';
import { ActivePositionsTable } from '../tables/ActivePositionsTable';
import { RecentTradesTable } from '../tables/RecentTradesTable';
import { RiskMonitorPanel } from '../risk/RiskMonitorPanel';
import { AlertsPanel } from '../alerts/AlertsPanel';

// Types
interface DashboardProps {
  strategyId?: string;
  portfolioId?: string;
  customLayout?: LayoutConfig[];
  theme?: 'light' | 'dark' | 'auto';
}

interface LayoutConfig {
  id: string;
  component: string;
  props: Record<string, any>;
  position: { x: number; y: number; w: number; h: number };
}

interface MarketDataPoint {
  timestamp: number;
  symbol: string;
  price: number;
  volume: number;
  bid: number;
  ask: number;
  change: number;
  changePercent: number;
}

interface PortfolioSummary {
  totalEquity: number;
  totalPnL: number;
  dailyPnL: number;
  positions: number;
  cash: number;
  margin: number;
  buyingPower: number;
  leverage: number;
}

interface RiskMetrics {
  var95: number;
  expectedShortfall: number;
  maxDrawdown: number;
  sharpeRatio: number;
  volatility: number;
  beta: number;
  correlation: number;
  concentration: number;
}

// Styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
  height: '100vh',
  width: '100vw',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.default,
  overflow: 'hidden',
}));

const HeaderBar = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1, 2),
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  minHeight: 64,
}));

const ContentArea = styled(Box)({
  flex: 1,
  display: 'flex',
  overflow: 'hidden',
});

const LeftPanel = styled(Paper)(({ theme }) => ({
  width: 300,
  display: 'flex',
  flexDirection: 'column',
  borderRight: `1px solid ${theme.palette.divider}`,
}));

const MainContent = styled(Box)({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
});

const ChartContainer = styled(Paper)(({ theme }) => ({
  margin: theme.spacing(1),
  padding: theme.spacing(2),
  height: 'calc(100% - 16px)',
  display: 'flex',
  flexDirection: 'column',
}));

const MetricsCard = styled(Card)(({ theme }) => ({
  margin: theme.spacing(1),
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[8],
  },
}));

const PnLText = styled(Typography)<{ profit: boolean }>(({ theme, profit }) => ({
  color: profit ? theme.palette.success.main : theme.palette.error.main,
  fontWeight: 'bold',
}));

// Main Dashboard Component
export const MainTradingDashboard: React.FC<DashboardProps> = ({
  strategyId,
  portfolioId,
  customLayout,
  theme = 'auto',
}) => {
  const muiTheme = useTheme();
  const [selectedTab, setSelectedTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(1000); // 1 second
  const [alertsCount, setAlertsCount] = useState(0);

  // WebSocket connection for real-time data
  const { 
    isConnected, 
    subscribe, 
    unsubscribe, 
    sendMessage 
  } = useWebSocket('ws://localhost:8080/ws');

  // Real-time data hooks
  const {
    marketData,
    portfolioData,
    isLoading: marketDataLoading,
    error: marketDataError,
  } = useRealtimeData({
    symbols: ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    interval: refreshInterval,
    enabled: autoRefresh,
  });

  const {
    performanceMetrics,
    equityCurve,
    returns,
    isLoading: performanceLoading,
  } = usePerformanceAnalytics({
    strategyId,
    portfolioId,
    timeframe: '1D',
    refreshInterval,
  });

  const {
    riskMetrics,
    varData,
    stressTestResults,
    isLoading: riskLoading,
  } = useRiskAnalytics({
    portfolioId,
    confidenceLevel: 0.95,
    refreshInterval,
  });

  // Refs for chart performance optimization
  const equityChartRef = useRef<HTMLCanvasElement>(null);
  const portfolioChartRef = useRef<HTMLCanvasElement>(null);
  const riskChartRef = useRef<HTMLCanvasElement>(null);

  // Memoized portfolio summary
  const portfolioSummary = useMemo((): PortfolioSummary => {
    if (!portfolioData) {
      return {
        totalEquity: 0,
        totalPnL: 0,
        dailyPnL: 0,
        positions: 0,
        cash: 0,
        margin: 0,
        buyingPower: 0,
        leverage: 0,
      };
    }

    return {
      totalEquity: portfolioData.totalEquity || 0,
      totalPnL: portfolioData.totalPnL || 0,
      dailyPnL: portfolioData.dailyPnL || 0,
      positions: portfolioData.positions?.length || 0,
      cash: portfolioData.cash || 0,
      margin: portfolioData.margin || 0,
      buyingPower: portfolioData.buyingPower || 0,
      leverage: portfolioData.leverage || 1,
    };
  }, [portfolioData]);

  // WebSocket subscriptions
  useEffect(() => {
    if (isConnected) {
      // Subscribe to real-time market data
      subscribe('market_data', (data: MarketDataPoint) => {
        // Handle real-time market data updates
        console.log('Market data update:', data);
      });

      // Subscribe to portfolio updates
      subscribe('portfolio_updates', (data: any) => {
        // Handle portfolio updates
        console.log('Portfolio update:', data);
      });

      // Subscribe to risk alerts
      subscribe('risk_alerts', (data: any) => {
        setAlertsCount(prev => prev + 1);
        console.log('Risk alert:', data);
      });

      // Subscribe to trade executions
      subscribe('trade_executions', (data: any) => {
        console.log('Trade execution:', data);
      });

      return () => {
        unsubscribe('market_data');
        unsubscribe('portfolio_updates');
        unsubscribe('risk_alerts');
        unsubscribe('trade_executions');
      };
    }
  }, [isConnected, subscribe, unsubscribe]);

  // Auto-refresh logic
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (autoRefresh) {
      interval = setInterval(() => {
        // Trigger data refresh
        sendMessage({
          type: 'refresh_request',
          timestamp: Date.now(),
        });
      }, refreshInterval);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, refreshInterval, sendMessage]);

  // Chart rendering optimization
  useEffect(() => {
    if (equityCurve && equityChartRef.current) {
      chartRenderingService.renderEquityCurve(
        equityChartRef.current,
        equityCurve,
        { 
          animate: true,
          showGrid: true,
          showTooltips: true,
        }
      );
    }
  }, [equityCurve]);

  const handleTabChange = useCallback((_: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  }, []);

  const handleFullscreenToggle = useCallback(() => {
    setIsFullscreen(!isFullscreen);
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }, [isFullscreen]);

  const handleRefreshData = useCallback(() => {
    sendMessage({
      type: 'manual_refresh',
      timestamp: Date.now(),
    });
  }, [sendMessage]);

  const formatCurrency = useCallback((value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  }, []);

  const formatPercentage = useCallback((value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value / 100);
  }, []);

  // Loading state
  if (marketDataLoading || performanceLoading || riskLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading AlgoVeda Dashboard...
        </Typography>
      </Box>
    );
  }

  // Error state
  if (marketDataError) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <Alert severity="error" sx={{ maxWidth: 600 }}>
          <Typography variant="h6">Dashboard Error</Typography>
          <Typography>{marketDataError.message}</Typography>
        </Alert>
      </Box>
    );
  }

  return (
    <DashboardContainer>
      {/* Header Bar */}
      <HeaderBar elevation={2}>
        <Box display="flex" alignItems="center">
          <Timeline sx={{ mr: 2 }} />
          <Typography variant="h5" fontWeight="bold">
            AlgoVeda Trading Platform
          </Typography>
          <Box
            sx={{
              ml: 2,
              px: 1,
              py: 0.5,
              borderRadius: 1,
              backgroundColor: isConnected ? 'success.main' : 'error.main',
              color: 'white',
            }}
          >
            <Typography variant="caption">
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </Typography>
          </Box>
        </Box>

        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="h6">
            {formatCurrency(portfolioSummary.totalEquity)}
          </Typography>
          <PnLText profit={portfolioSummary.dailyPnL >= 0} variant="body1">
            {portfolioSummary.dailyPnL >= 0 ? '+' : ''}
            {formatCurrency(portfolioSummary.dailyPnL)}
          </PnLText>
          
          <Badge badgeContent={alertsCount} color="error">
            <IconButton color="inherit">
              <NotificationsActive />
            </IconButton>
          </Badge>

          <Tooltip title="Refresh Data">
            <IconButton color="inherit" onClick={handleRefreshData}>
              <Refresh />
            </IconButton>
          </Tooltip>

          <Tooltip title="Fullscreen">
            <IconButton color="inherit" onClick={handleFullscreenToggle}>
              <Fullscreen />
            </IconButton>
          </Tooltip>

          <Tooltip title="Settings">
            <IconButton color="inherit" onClick={() => setShowSettings(true)}>
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </HeaderBar>

      <ContentArea>
        {/* Left Panel - Portfolio Summary and Controls */}
        <LeftPanel elevation={1}>
          <Box p={2}>
            <Typography variant="h6" gutterBottom>
              Portfolio Summary
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <MetricsCard>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={1}>
                      <AccountBalance color="primary" sx={{ mr: 1 }} />
                      <Typography variant="subtitle2">Total Equity</Typography>
                    </Box>
                    <Typography variant="h5" fontWeight="bold">
                      {formatCurrency(portfolioSummary.totalEquity)}
                    </Typography>
                  </CardContent>
                </MetricsCard>
              </Grid>

              <Grid item xs={6}>
                <MetricsCard>
                  <CardContent>
                    <Typography variant="caption" color="textSecondary">
                      Daily P&L
                    </Typography>
                    <PnLText profit={portfolioSummary.dailyPnL >= 0} variant="h6">
                      {formatCurrency(portfolioSummary.dailyPnL)}
                    </PnLText>
                  </CardContent>
                </MetricsCard>
              </Grid>

              <Grid item xs={6}>
                <MetricsCard>
                  <CardContent>
                    <Typography variant="caption" color="textSecondary">
                      Total P&L
                    </Typography>
                    <PnLText profit={portfolioSummary.totalPnL >= 0} variant="h6">
                      {formatCurrency(portfolioSummary.totalPnL)}
                    </PnLText>
                  </CardContent>
                </MetricsCard>
              </Grid>

              <Grid item xs={6}>
                <MetricsCard>
                  <CardContent>
                    <Typography variant="caption" color="textSecondary">
                      Positions
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {portfolioSummary.positions}
                    </Typography>
                  </CardContent>
                </MetricsCard>
              </Grid>

              <Grid item xs={6}>
                <MetricsCard>
                  <CardContent>
                    <Typography variant="caption" color="textSecondary">
                      Leverage
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {portfolioSummary.leverage.toFixed(2)}x
                    </Typography>
                  </CardContent>
                </MetricsCard>
              </Grid>
            </Grid>
          </Box>

          {/* Risk Metrics Panel */}
          <Box p={2} borderTop={1} borderColor="divider">
            <Typography variant="h6" gutterBottom>
              Risk Metrics
            </Typography>
            
            {riskMetrics && (
              <RiskMonitorPanel
                metrics={riskMetrics}
                compact={true}
              />
            )}
          </Box>

          {/* Alerts Panel */}
          <Box flex={1} p={2} borderTop={1} borderColor="divider">
            <AlertsPanel maxItems={5} />
          </Box>
        </LeftPanel>

        {/* Main Content Area */}
        <MainContent>
          <Box borderBottom={1} borderColor="divider">
            <Tabs value={selectedTab} onChange={handleTabChange}>
              <Tab label="Overview" icon={<Assessment />} />
              <Tab label="Charts" icon={<Timeline />} />
              <Tab label="Options Analytics" icon={<Speed />} />
              <Tab label="Risk Analysis" icon={<Security />} />
            </Tabs>
          </Box>

          <Box flex={1} overflow="auto">
            {selectedTab === 0 && (
              <Grid container spacing={2} p={2}>
                {/* Equity Curve Chart */}
                <Grid item xs={12} md={8}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Equity Curve
                    </Typography>
                    <Box flex={1}>
                      <EquityCurveChart
                        data={equityCurve}
                        height={300}
                        animated={true}
                        showBenchmark={true}
                      />
                    </Box>
                  </ChartContainer>
                </Grid>

                {/* Portfolio Composition */}
                <Grid item xs={12} md={4}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Portfolio Composition
                    </Typography>
                    <Box flex={1}>
                      <PortfolioCompositionChart
                        data={portfolioData?.positions || []}
                        height={300}
                      />
                    </Box>
                  </ChartContainer>
                </Grid>

                {/* Performance Metrics Table */}
                <Grid item xs={12} md={6}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <PerformanceMetricsTable
                      metrics={performanceMetrics}
                    />
                  </ChartContainer>
                </Grid>

                {/* Active Positions */}
                <Grid item xs={12} md={6}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Active Positions
                    </Typography>
                    <ActivePositionsTable
                      positions={portfolioData?.positions || []}
                      maxRows={10}
                    />
                  </ChartContainer>
                </Grid>

                {/* Recent Trades */}
                <Grid item xs={12}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Recent Trades
                    </Typography>
                    <RecentTradesTable
                      trades={portfolioData?.recentTrades || []}
                      maxRows={20}
                    />
                  </ChartContainer>
                </Grid>
              </Grid>
            )}

            {selectedTab === 1 && (
              <Grid container spacing={2} p={2}>
                {/* Advanced Charts */}
                <Grid item xs={12} md={6}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Order Book Heatmap
                    </Typography>
                    <OrderBookHeatmap
                      symbol="AAPL"
                      height={400}
                    />
                  </ChartContainer>
                </Grid>

                <Grid item xs={12} md={6}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Risk Metrics Over Time
                    </Typography>
                    <RiskMetricsChart
                      data={varData}
                      height={400}
                    />
                  </ChartContainer>
                </Grid>
              </Grid>
            )}

            {selectedTab === 2 && (
              <Grid container spacing={2} p={2}>
                {/* 3D Volatility Surface */}
                <Grid item xs={12}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      3D Volatility Surface
                    </Typography>
                    <Box height={600}>
                      <Canvas camera={{ position: [0, 0, 5] }}>
                        <ambientLight intensity={0.5} />
                        <pointLight position={[10, 10, 10]} />
                        <VolatilitySurface3D
                          symbol="AAPL"
                          data={portfolioData?.optionsData}
                        />
                        <OrbitControls />
                        <Stats />
                      </Canvas>
                    </Box>
                  </ChartContainer>
                </Grid>
              </Grid>
            )}

            {selectedTab === 3 && (
              <Grid container spacing={2} p={2}>
                {/* Risk Analysis */}
                <Grid item xs={12}>
                  <ChartContainer>
                    <Typography variant="h6" gutterBottom>
                      Comprehensive Risk Analysis
                    </Typography>
                    <RiskMonitorPanel
                      metrics={riskMetrics}
                      stressTestResults={stressTestResults}
                      compact={false}
                    />
                  </ChartContainer>
                </Grid>
              </Grid>
            )}
          </Box>
        </MainContent>
      </ContentArea>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
        <DialogTitle>Dashboard Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} p={2}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                }
                label="Auto Refresh"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography gutterBottom>
                Refresh Interval: {refreshInterval}ms
              </Typography>
              <input
                type="range"
                min="100"
                max="5000"
                step="100"
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </Grid>
          </Grid>
        </DialogContent>
      </Dialog>

      {/* Floating Action Button for Quick Actions */}
      <Fab
        color="primary"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        onClick={() => setShowSettings(true)}
      >
        <Settings />
      </Fab>
    </DashboardContainer>
  );
};

export default MainTradingDashboard;
