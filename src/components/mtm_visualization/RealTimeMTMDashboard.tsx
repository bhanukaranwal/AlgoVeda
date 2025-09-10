/*!
 * Real-Time Mark-to-Market Dashboard
 * Live portfolio P&L visualization with streaming updates
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  Box, 
  Card, 
  CardContent,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  Timeline,
  AttachMoney,
  AccountBalance
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
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { styled } from '@mui/material/styles';

// Styled components
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
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 25px rgba(0, 0, 0, 0.3)',
  },
}));

interface MTMDashboardProps {
  marketData: any;
  portfolioData: any;
  orders: any[];
  positions: any[];
}

interface PnLDataPoint {
  timestamp: string;
  totalPnL: number;
  realizedPnL: number;
  unrealizedPnL: number;
  portfolioValue: number;
}

export const RealTimeMTMDashboard: React.FC<MTMDashboardProps> = ({
  marketData,
  portfolioData,
  orders,
  positions
}) => {
  const [pnlHistory, setPnlHistory] = useState<PnLDataPoint[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const wsRef = useRef<WebSocket | null>(null);

  // Real-time data subscription
  useEffect(() => {
    const connectWebSocket = () => {
      wsRef.current = new WebSocket('ws://localhost:8081/ws');
      
      wsRef.current.onopen = () => {
        console.log('MTM WebSocket connected');
        // Subscribe to portfolio updates
        wsRef.current?.send(JSON.stringify({
          type: 'SUBSCRIBE_PORTFOLIO',
          account_id: 'main_account'
        }));
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'PORTFOLIO_UPDATE') {
          updatePnLHistory(data.portfolio);
        }
      };

      wsRef.current.onclose = () => {
        console.log('MTM WebSocket disconnected, attempting to reconnect...');
        setTimeout(connectWebSocket, 3000);
      };
    };

    connectWebSocket();

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const updatePnLHistory = (portfolioUpdate: any) => {
    const newDataPoint: PnLDataPoint = {
      timestamp: new Date().toISOString(),
      totalPnL: portfolioUpdate.total_pnl || 0,
      realizedPnL: portfolioUpdate.realized_pnl || 0,
      unrealizedPnL: portfolioUpdate.unrealized_pnl || 0,
      portfolioValue: portfolioUpdate.total_value || 0,
    };

    setPnlHistory(prev => {
      const updated = [...prev, newDataPoint].slice(-1000); // Keep last 1000 points
      return updated;
    });
  };

  // Calculate portfolio metrics
  const portfolioMetrics = useMemo(() => {
    if (!portfolioData) return null;

    const totalValue = positions.reduce((sum, pos) => sum + (pos.quantity * pos.currentPrice), 0);
    const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
    const dayPnL = positions.reduce((sum, pos) => sum + pos.dayPnL, 0);
    const realizedPnL = positions.reduce((sum, pos) => sum + (pos.realizedPnL || 0), 0);

    return {
      totalValue,
      totalPnL,
      dayPnL,
      realizedPnL,
      pnlPercentage: totalValue > 0 ? (totalPnL / totalValue) * 100 : 0,
      dayPnlPercentage: totalValue > 0 ? (dayPnL / totalValue) * 100 : 0,
      cashBalance: portfolioData.cash || 0,
      marginUsed: portfolioData.marginUsed || 0,
      buyingPower: portfolioData.buyingPower || 0,
    };
  }, [portfolioData, positions]);

  // Position breakdown for pie chart
  const positionBreakdown = useMemo(() => {
    return positions
      .filter(pos => Math.abs(pos.quantity) > 0.01)
      .map(pos => ({
        symbol: pos.symbol,
        value: Math.abs(pos.quantity * pos.currentPrice),
        pnl: pos.unrealizedPnL,
        color: pos.unrealizedPnL >= 0 ? '#4caf50' : '#f44336'
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10); // Top 10 positions
  }, [positions]);

  // Color schemes
  const colors = {
    profit: '#4caf50',
    loss: '#f44336',
    neutral: '#9e9e9e',
    primary: '#2196f3',
    secondary: '#ff9800'
  };

  const getPnLColor = (value: number) => value >= 0 ? colors.profit : colors.loss;

  return (
    <Grid container spacing={3}>
      {/* Key Metrics Row */}
      <Grid item xs={12}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <AccountBalance color="primary" />
                  <Typography variant="h6" ml={1}>Portfolio Value</Typography>
                </Box>
                <Typography variant="h4" color="primary.main">
                  ${portfolioMetrics?.totalValue.toLocaleString() || '0'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Cash: ${portfolioMetrics?.cashBalance.toLocaleString() || '0'}
                </Typography>
              </CardContent>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <TrendingUp color={portfolioMetrics?.totalPnL >= 0 ? "success" : "error"} />
                  <Typography variant="h6" ml={1}>Total P&L</Typography>
                </Box>
                <Typography 
                  variant="h4" 
                  color={getPnLColor(portfolioMetrics?.totalPnL || 0)}
                >
                  {portfolioMetrics?.totalPnL >= 0 ? '+' : ''}
                  ${portfolioMetrics?.totalPnL.toFixed(2) || '0.00'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  ({portfolioMetrics?.pnlPercentage.toFixed(2) || '0.00'}%)
                </Typography>
              </CardContent>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <Timeline color={portfolioMetrics?.dayPnL >= 0 ? "success" : "error"} />
                  <Typography variant="h6" ml={1}>Day P&L</Typography>
                </Box>
                <Typography 
                  variant="h4" 
                  color={getPnLColor(portfolioMetrics?.dayPnL || 0)}
                >
                  {portfolioMetrics?.dayPnL >= 0 ? '+' : ''}
                  ${portfolioMetrics?.dayPnL.toFixed(2) || '0.00'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  ({portfolioMetrics?.dayPnlPercentage.toFixed(2) || '0.00'}%)
                </Typography>
              </CardContent>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <AttachMoney color="primary" />
                  <Typography variant="h6" ml={1}>Buying Power</Typography>
                </Box>
                <Typography variant="h4" color="primary.main">
                  ${portfolioMetrics?.buyingPower.toLocaleString() || '0'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Margin Used: ${portfolioMetrics?.marginUsed.toLocaleString() || '0'}
                </Typography>
              </CardContent>
            </MetricCard>
          </Grid>
        </Grid>
      </Grid>

      {/* P&L Chart */}
      <Grid item xs={12} md={8}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Real-Time P&L Chart
          </Typography>
          <Box height={400}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={pnlHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#ffffff"
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis stroke="#ffffff" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#2d2d2d', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff'
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value, name) => [
                    `$${Number(value).toFixed(2)}`, 
                    name === 'totalPnL' ? 'Total P&L' : 
                    name === 'realizedPnL' ? 'Realized P&L' : 'Unrealized P&L'
                  ]}
                />
                <Area 
                  type="monotone" 
                  dataKey="totalPnL" 
                  stroke={colors.primary}
                  fill={colors.primary}
                  fillOpacity={0.3}
                />
                <Area 
                  type="monotone" 
                  dataKey="realizedPnL" 
                  stroke={colors.profit}
                  fill="transparent"
                />
                <Area 
                  type="monotone" 
                  dataKey="unrealizedPnL" 
                  stroke={colors.secondary}
                  fill="transparent"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </Grid>

      {/* Position Breakdown */}
      <Grid item xs={12} md={4}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Position Breakdown
          </Typography>
          <Box height={400}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={positionBreakdown}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {positionBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#2d2d2d', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff'
                  }}
                  formatter={(value, name, props) => [
                    `$${Number(value).toLocaleString()}`,
                    props.payload.symbol
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </Grid>

      {/* Positions Table */}
      <Grid item xs={12}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Current Positions
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell align="right">Quantity</TableCell>
                  <TableCell align="right">Current Price</TableCell>
                  <TableCell align="right">Market Value</TableCell>
                  <TableCell align="right">Unrealized P&L</TableCell>
                  <TableCell align="right">Day P&L</TableCell>
                  <TableCell align="right">% Change</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((position, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Typography variant="body2" fontWeight="bold">
                          {position.symbol}
                        </Typography>
                        <Chip 
                          size="small" 
                          label={position.quantity > 0 ? 'LONG' : 'SHORT'}
                          color={position.quantity > 0 ? 'success' : 'error'}
                          sx={{ ml: 1 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell align="right">{position.quantity.toFixed(2)}</TableCell>
                    <TableCell align="right">${position.currentPrice.toFixed(2)}</TableCell>
                    <TableCell align="right">
                      ${(position.quantity * position.currentPrice).toLocaleString()}
                    </TableCell>
                    <TableCell 
                      align="right" 
                      sx={{ color: getPnLColor(position.unrealizedPnL) }}
                    >
                      {position.unrealizedPnL >= 0 ? '+' : ''}${position.unrealizedPnL.toFixed(2)}
                    </TableCell>
                    <TableCell 
                      align="right" 
                      sx={{ color: getPnLColor(position.dayPnL) }}
                    >
                      {position.dayPnL >= 0 ? '+' : ''}${position.dayPnL.toFixed(2)}
                    </TableCell>
                    <TableCell 
                      align="right" 
                      sx={{ color: getPnLColor(position.dayPnL) }}
                    >
                      {position.dayChangePercent >= 0 ? '+' : ''}{position.dayChangePercent.toFixed(2)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </StyledPaper>
      </Grid>
    </Grid>
  );
};

export default RealTimeMTMDashboard;
