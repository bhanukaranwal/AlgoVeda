/**
 * Professional Trading Dashboard
 * React-based institutional trading interface with real-time data and advanced charting
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  AppBar,
  Toolbar,
  IconButton,
  Menu,
  MenuItem,
  Badge,
  Alert,
  Snackbar,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Button,
  TextField,
  Select,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Tooltip,
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  AccountBalance,
  TrendingUp,
  TrendingDown,
  Assessment,
  Settings,
  Notifications,
  Security,
  Speed,
  Timeline,
  ShowChart,
  SwapHoriz,
  AccountBalanceWallet,
  Warning,
  CheckCircle,
  Error,
  Info,
  Refresh,
  MoreVert,
  FilterList,
  Download,
  Upload,
  PlayArrow,
  Pause,
  Stop,
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  TimeScale
);

// Types and interfaces
interface Position {
  symbol: string;
  quantity: number;
  marketValue: number;
  unrealizedPnL: number;
  dailyChange: number;
  dailyChangePercent: number;
  sector: string;
  averageCost: number;
  currentPrice: number;
  weight: number;
}

interface Order {
  orderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  orderType: string;
  status: string;
  filledQuantity: number;
  averagePrice: number;
  createdAt: string;
  strategy?: string;
}

interface Trade {
  tradeId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: string;
  commission: number;
  venue: string;
}

interface PortfolioSummary {
  totalValue: number;
  totalCash: number;
  totalPnL: number;
  dailyPnL: number;
  leverage: number;
  marginUsed: number;
  availableMargin: number;
  numberOfPositions: number;
}

interface RiskMetrics {
  portfolioVaR: number;
  portfolioBeta: number;
  maxDrawdown: number;
  sharpeRatio: number;
  volatility: number;
  concentration: number;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid: number;
  ask: number;
  high: number;
  low: number;
}

interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

interface TradingStrategy {
  strategyId: string;
  name: string;
  status: 'ACTIVE' | 'PAUSED' | 'STOPPED';
  pnl: number;
  sharpeRatio: number;
  maxDrawdown: number;
  ordersToday: number;
}

// WebSocket hook for real-time data
const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket disconnected');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('disconnected');
    };
    
    setSocket(ws);
    
    return () => {
      ws.close();
    };
  }, [url]);

  return { socket, connectionStatus, lastMessage };
};

// Portfolio Overview Component
const PortfolioOverview: React.FC<{ summary: PortfolioSummary }> = ({ summary }) => {
  const formatCurrency = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

  const formatPercent = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value / 100);

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6} lg={3}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Total Portfolio Value
            </Typography>
            <Typography variant="h4" component="div">
              {formatCurrency(summary.totalValue)}
            </Typography>
            <Typography variant="body2" color={summary.dailyPnL >= 0 ? 'success.main' : 'error.main'}>
              {summary.dailyPnL >= 0 ? '+' : ''}{formatCurrency(summary.dailyPnL)} Today
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} md={6} lg={3}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Available Cash
            </Typography>
            <Typography variant="h4" component="div">
              {formatCurrency(summary.totalCash)}
            </Typography>
            <Typography variant="body2">
              Leverage: {summary.leverage.toFixed(2)}x
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} md={6} lg={3}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Margin Usage
            </Typography>
            <Typography variant="h4" component="div">
              {formatPercent(summary.marginUsed / (summary.marginUsed + summary.availableMargin))}
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={(summary.marginUsed / (summary.marginUsed + summary.availableMargin)) * 100}
              sx={{ mt: 1 }}
            />
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={12} md={6} lg={3}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Total P&L
            </Typography>
            <Typography variant="h4" component="div" color={summary.totalPnL >= 0 ? 'success.main' : 'error.main'}>
              {formatCurrency(summary.totalPnL)}
            </Typography>
            <Typography variant="body2">
              {summary.numberOfPositions} Positions
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

// Positions Table Component
const PositionsTable: React.FC<{ positions: Position[] }> = ({ positions }) => {
  const formatCurrency = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

  const formatPercent = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value);

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Market Value</TableCell>
            <TableCell align="right">Unrealized P&L</TableCell>
            <TableCell align="right">Daily Change</TableCell>
            <TableCell align="right">Weight</TableCell>
            <TableCell>Sector</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((position) => (
            <TableRow key={position.symbol} hover>
              <TableCell component="th" scope="row">
                <Typography variant="body2" fontWeight="bold">
                  {position.symbol}
                </Typography>
              </TableCell>
              <TableCell align="right">
                {new Intl.NumberFormat('en-US').format(position.quantity)}
              </TableCell>
              <TableCell align="right">
                {formatCurrency(position.marketValue)}
              </TableCell>
              <TableCell align="right">
                <Typography color={position.unrealizedPnL >= 0 ? 'success.main' : 'error.main'}>
                  {formatCurrency(position.unrealizedPnL)}
                </Typography>
              </TableCell>
              <TableCell align="right">
                <Typography color={position.dailyChange >= 0 ? 'success.main' : 'error.main'}>
                  {position.dailyChange >= 0 ? '+' : ''}{formatPercent(position.dailyChangePercent)}
                </Typography>
              </TableCell>
              <TableCell align="right">
                {formatPercent(position.weight)}
              </TableCell>
              <TableCell>
                <Chip label={position.sector} size="small" />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Orders Table Component
const OrdersTable: React.FC<{ orders: Order[] }> = ({ orders }) => {
  const formatCurrency = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'FILLED': return 'success';
      case 'PENDING': return 'warning';
      case 'CANCELLED': return 'error';
      case 'REJECTED': return 'error';
      default: return 'default';
    }
  };

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Order ID</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell>Side</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Filled</TableCell>
            <TableCell align="right">Avg Price</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Strategy</TableCell>
            <TableCell>Created</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {orders.map((order) => (
            <TableRow key={order.orderId} hover>
              <TableCell>
                <Typography variant="body2" fontFamily="monospace">
                  {order.orderId.substring(0, 8)}...
                </Typography>
              </TableCell>
              <TableCell>{order.symbol}</TableCell>
              <TableCell>
                <Chip 
                  label={order.side} 
                  color={order.side === 'BUY' ? 'success' : 'error'} 
                  size="small" 
                />
              </TableCell>
              <TableCell align="right">
                {new Intl.NumberFormat('en-US').format(order.quantity)}
              </TableCell>
              <TableCell align="right">
                {new Intl.NumberFormat('en-US').format(order.filledQuantity)}
              </TableCell>
              <TableCell align="right">
                {order.averagePrice > 0 ? formatCurrency(order.averagePrice) : '-'}
              </TableCell>
              <TableCell>
                <Chip 
                  label={order.status} 
                  color={getStatusColor(order.status) as any} 
                  size="small" 
                />
              </TableCell>
              <TableCell>
                {order.strategy ? (
                  <Chip label={order.strategy} variant="outlined" size="small" />
                ) : '-'}
              </TableCell>
              <TableCell>
                {new Date(order.createdAt).toLocaleTimeString()}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Risk Metrics Component
const RiskMetrics: React.FC<{ metrics: RiskMetrics }> = ({ metrics }) => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={6} md={4}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom variant="body2">
              Portfolio VaR (1-day)
            </Typography>
            <Typography variant="h6">
              ${new Intl.NumberFormat('en-US').format(metrics.portfolioVaR)}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={6} md={4}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom variant="body2">
              Portfolio Beta
            </Typography>
            <Typography variant="h6">
              {metrics.portfolioBeta.toFixed(2)}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={6} md={4}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom variant="body2">
              Sharpe Ratio
            </Typography>
            <Typography variant="h6">
              {metrics.sharpeRatio.toFixed(2)}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={6} md={4}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom variant="body2">
              Max Drawdown
            </Typography>
            <Typography variant="h6" color="error">
              {(metrics.maxDrawdown * 100).toFixed(1)}%
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={6} md={4}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom variant="body2">
              Volatility
            </Typography>
            <Typography variant="h6">
              {(metrics.volatility * 100).toFixed(1)}%
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      
      <Grid item xs={6} md={4}>
        <Card>
          <CardContent>
            <Typography color="textSecondary" gutterBottom variant="body2">
              Concentration Risk
            </Typography>
            <Typography variant="h6">
              {(metrics.concentration * 100).toFixed(1)}%
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

// P&L Chart Component
const PnLChart: React.FC<{ data: any[] }> = ({ data }) => {
  const chartData = {
    labels: data.map(d => new Date(d.timestamp)),
    datasets: [
      {
        label: 'Cumulative P&L',
        data: data.map(d => d.cumulativePnL),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
      {
        label: 'Daily P&L',
        data: data.map(d => d.dailyPnL),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Portfolio P&L Over Time',
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: 'day' as const,
        },
      },
      y: {
        beginAtZero: false,
        ticks: {
          callback: function(value: any) {
            return '$' + new Intl.NumberFormat('en-US').format(value);
          }
        }
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

// Strategy Performance Component
const StrategyPerformance: React.FC<{ strategies: TradingStrategy[] }> = ({ strategies }) => {
  const formatCurrency = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ACTIVE': return <PlayArrow color="success" />;
      case 'PAUSED': return <Pause color="warning" />;
      case 'STOPPED': return <Stop color="error" />;
      default: return null;
    }
  };

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Strategy Name</TableCell>
            <TableCell>Status</TableCell>
            <TableCell align="right">P&L</TableCell>
            <TableCell align="right">Sharpe Ratio</TableCell>
            <TableCell align="right">Max Drawdown</TableCell>
            <TableCell align="right">Orders Today</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {strategies.map((strategy) => (
            <TableRow key={strategy.strategyId} hover>
              <TableCell>{strategy.name}</TableCell>
              <TableCell>
                <Box display="flex" alignItems="center" gap={1}>
                  {getStatusIcon(strategy.status)}
                  <Chip 
                    label={strategy.status} 
                    size="small"
                    color={strategy.status === 'ACTIVE' ? 'success' : strategy.status === 'PAUSED' ? 'warning' : 'error'}
                  />
                </Box>
              </TableCell>
              <TableCell align="right">
                <Typography color={strategy.pnl >= 0 ? 'success.main' : 'error.main'}>
                  {formatCurrency(strategy.pnl)}
                </Typography>
              </TableCell>
              <TableCell align="right">{strategy.sharpeRatio.toFixed(2)}</TableCell>
              <TableCell align="right">
                <Typography color="error">
                  {(strategy.maxDrawdown * 100).toFixed(1)}%
                </Typography>
              </TableCell>
              <TableCell align="right">{strategy.ordersToday}</TableCell>
              <TableCell>
                <IconButton size="small">
                  <MoreVert />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Main Trading Dashboard Component
const TradingDashboard: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary>({
    totalValue: 2450000,
    totalCash: 125000,
    totalPnL: 45000,
    dailyPnL: 12500,
    leverage: 1.2,
    marginUsed: 150000,
    availableMargin: 850000,
    numberOfPositions: 15,
  });

  const [positions, setPositions] = useState<Position[]>([
    {
      symbol: 'AAPL',
      quantity: 1000,
      marketValue: 175000,
      unrealizedPnL: 5000,
      dailyChange: 2.5,
      dailyChangePercent: 1.45,
      sector: 'Technology',
      averageCost: 170.0,
      currentPrice: 175.0,
      weight: 7.14,
    },
    {
      symbol: 'MSFT',
      quantity: 500,
      marketValue: 150000,
      unrealizedPnL: -2500,
      dailyChange: -5.0,
      dailyChangePercent: -1.67,
      sector: 'Technology',
      averageCost: 305.0,
      currentPrice: 300.0,
      weight: 6.12,
    },
    // Add more mock positions...
  ]);

  const [orders, setOrders] = useState<Order[]>([
    {
      orderId: 'ORD_12345678',
      symbol: 'GOOGL',
      side: 'BUY',
      quantity: 100,
      orderType: 'LIMIT',
      status: 'PENDING',
      filledQuantity: 0,
      averagePrice: 0,
      createdAt: new Date().toISOString(),
      strategy: 'MOMENTUM',
    },
    // Add more mock orders...
  ]);

  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: 'ALERT_001',
      type: 'warning',
      message: 'Portfolio concentration exceeds 10% limit for AAPL',
      timestamp: new Date().toISOString(),
      acknowledged: false,
    },
    {
      id: 'ALERT_002',
      type: 'info',
      message: 'Daily VaR breach threshold reached',
      timestamp: new Date().toISOString(),
      acknowledged: false,
    },
  ]);

  const [riskMetrics] = useState<RiskMetrics>({
    portfolioVaR: 125000,
    portfolioBeta: 1.15,
    maxDrawdown: 0.08,
    sharpeRatio: 1.35,
    volatility: 0.18,
    concentration: 0.12,
  });

  const [strategies] = useState<TradingStrategy[]>([
    {
      strategyId: 'STRAT_001',
      name: 'Momentum Strategy',
      status: 'ACTIVE',
      pnl: 25000,
      sharpeRatio: 1.8,
      maxDrawdown: 0.05,
      ordersToday: 15,
    },
    {
      strategyId: 'STRAT_002',
      name: 'Mean Reversion',
      status: 'PAUSED',
      pnl: -5000,
      sharpeRatio: 0.9,
      maxDrawdown: 0.12,
      ordersToday: 3,
    },
  ]);

  // WebSocket connection for real-time data
  const { connectionStatus, lastMessage } = useWebSocket('ws://localhost:8080/ws');

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'PORTFOLIO_UPDATE':
          setPortfolioSummary(lastMessage.data);
          break;
        case 'POSITION_UPDATE':
          setPositions(prev => {
            const updated = [...prev];
            const index = updated.findIndex(p => p.symbol === lastMessage.data.symbol);
            if (index >= 0) {
              updated[index] = { ...updated[index], ...lastMessage.data };
            }
            return updated;
          });
          break;
        case 'ORDER_UPDATE':
          setOrders(prev => {
            const updated = [...prev];
            const index = updated.findIndex(o => o.orderId === lastMessage.data.orderId);
            if (index >= 0) {
              updated[index] = { ...updated[index], ...lastMessage.data };
            } else {
              updated.unshift(lastMessage.data);
            }
            return updated;
          });
          break;
        case 'ALERT':
          setAlerts(prev => [lastMessage.data, ...prev.slice(0, 9)]); // Keep last 10 alerts
          break;
      }
    }
  }, [lastMessage]);

  // Mock P&L data for chart
  const pnlData = Array.from({ length: 30 }, (_, i) => ({
    timestamp: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString(),
    cumulativePnL: 45000 + Math.random() * 10000 - 5000,
    dailyPnL: Math.random() * 5000 - 2500,
  }));

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleAlertAcknowledge = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* App Bar */}
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            AlgoVeda Trading Platform
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Connection Status */}
            <Tooltip title={`WebSocket: ${connectionStatus}`}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: connectionStatus === 'connected' ? 'success.main' : 'error.main',
                  }}
                />
                <Typography variant="body2" color="textSecondary">
                  {connectionStatus}
                </Typography>
              </Box>
            </Tooltip>

            {/* Alerts */}
            <Badge badgeContent={alerts.filter(a => !a.acknowledged).length} color="error">
              <IconButton>
                <Notifications />
              </IconButton>
            </Badge>

            {/* Settings */}
            <IconButton>
              <Settings />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Alert Snackbars */}
      {alerts.filter(a => !a.acknowledged).slice(0, 3).map((alert) => (
        <Snackbar
          key={alert.id}
          open={!alert.acknowledged}
          autoHideDuration={6000}
          onClose={() => handleAlertAcknowledge(alert.id)}
        >
          <Alert 
            severity={alert.type as any} 
            onClose={() => handleAlertAcknowledge(alert.id)}
          >
            {alert.message}
          </Alert>
        </Snackbar>
      ))}

      {/* Main Content */}
      <Box sx={{ p: 3 }}>
        {/* Portfolio Overview */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            Portfolio Overview
          </Typography>
          <PortfolioOverview summary={portfolioSummary} />
        </Box>

        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Positions" />
            <Tab label="Orders" />
            <Tab label="Analytics" />
            <Tab label="Risk" />
            <Tab label="Strategies" />
          </Tabs>

          <Box sx={{ p: 3 }}>
            {/* Positions Tab */}
            {activeTab === 0 && (
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Current Positions</Typography>
                  <Box>
                    <IconButton>
                      <FilterList />
                    </IconButton>
                    <IconButton>
                      <Download />
                    </IconButton>
                    <IconButton>
                      <Refresh />
                    </IconButton>
                  </Box>
                </Box>
                <PositionsTable positions={positions} />
              </Box>
            )}

            {/* Orders Tab */}
            {activeTab === 1 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Order Management
                </Typography>
                <OrdersTable orders={orders} />
              </Box>
            )}

            {/* Analytics Tab */}
            {activeTab === 2 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Portfolio Analytics
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                      <PnLChart data={pnlData} />
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            )}

            {/* Risk Tab */}
            {activeTab === 3 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Risk Management
                </Typography>
                <RiskMetrics metrics={riskMetrics} />
              </Box>
            )}

            {/* Strategies Tab */}
            {activeTab === 4 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Trading Strategies
                </Typography>
                <StrategyPerformance strategies={strategies} />
              </Box>
            )}
          </Box>
        </Paper>
      </Box>
    </Box>
  );
};

export default TradingDashboard;
