/*!
 * Risk Visualization Dashboard
 * Comprehensive risk metrics visualization with VaR, CVaR, and exposures
 */

import React, { useState, useEffect, useMemo } from 'react';
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
  TableRow,
  Alert
} from '@mui/material';
import {
  Warning,
  Security,
  TrendingDown,
  Assessment,
  PieChart as PieChartIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  background: 'linear-gradient(145deg, #1e1e1e 0%, #2d2d2d 100%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 12,
  color: '#ffffff',
}));

const RiskCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 8,
  color: '#ffffff',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
  },
}));

interface RiskDashboardProps {
  portfolioData: any;
  positions: any[];
}

interface RiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  max_drawdown: number;
  beta: number;
  alpha: number;
  sharpe_ratio: number;
  volatility: number;
  correlation_to_market: number;
}

export const RiskDashboard: React.FC<RiskDashboardProps> = ({
  portfolioData,
  positions
}) => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [riskAlerts, setRiskAlerts] = useState<any[]>([]);

  // Calculate risk metrics
  const calculatedMetrics = useMemo(() => {
    if (!portfolioData || !positions) return null;

    const totalValue = positions.reduce((sum, pos) => sum + Math.abs(pos.quantity * pos.currentPrice), 0);
    
    // Sector concentration
    const sectorExposure = positions.reduce((sectors, pos) => {
      const sector = pos.sector || 'Unknown';
      sectors[sector] = (sectors[sector] || 0) + Math.abs(pos.quantity * pos.currentPrice);
      return sectors;
    }, {} as Record<string, number>);

    const sectorConcentration = Object.entries(sectorExposure).map(([sector, exposure]) => ({
      sector,
      exposure,
      percentage: (exposure / totalValue) * 100,
      color: getSectorColor(sector)
    }));

    // Geographic exposure
    const geoExposure = positions.reduce((geos, pos) => {
      const country = pos.country || 'Unknown';
      geos[country] = (geos[country] || 0) + Math.abs(pos.quantity * pos.currentPrice);
      return geos;
    }, {} as Record<string, number>);

    // Asset class breakdown
    const assetClasses = positions.reduce((classes, pos) => {
      const assetClass = pos.assetClass || 'Equity';
      classes[assetClass] = (classes[assetClass] || 0) + Math.abs(pos.quantity * pos.currentPrice);
      return classes;
    }, {} as Record<string, number>);

    return {
      sectorConcentration,
      geoExposure,
      assetClasses,
      totalValue,
      concentrationRisk: Math.max(...Object.values(sectorExposure)) / totalValue,
    };
  }, [portfolioData, positions]);

  // Risk metrics calculation (simplified)
  useEffect(() => {
    if (calculatedMetrics) {
      // In practice, this would call the backend risk calculation service
      setRiskMetrics({
        var_95: calculatedMetrics.totalValue * 0.025, // 2.5% of portfolio
        var_99: calculatedMetrics.totalValue * 0.045, // 4.5% of portfolio
        cvar_95: calculatedMetrics.totalValue * 0.035, // 3.5% of portfolio
        cvar_99: calculatedMetrics.totalValue * 0.065, // 6.5% of portfolio
        max_drawdown: 0.15, // 15%
        beta: 1.2,
        alpha: 0.03, // 3% annual alpha
        sharpe_ratio: 1.8,
        volatility: 0.18, // 18% annual volatility
        correlation_to_market: 0.85,
      });

      // Generate risk alerts
      const alerts = [];
      if (calculatedMetrics.concentrationRisk > 0.3) {
        alerts.push({
          type: 'warning',
          message: 'High sector concentration detected',
          value: `${(calculatedMetrics.concentrationRisk * 100).toFixed(1)}%`,
        });
      }
      
      setRiskAlerts(alerts);
    }
  }, [calculatedMetrics]);

  const colors = {
    danger: '#f44336',
    warning: '#ff9800',
    success: '#4caf50',
    info: '#2196f3',
    neutral: '#9e9e9e',
  };

  const getRiskColor = (value: number, thresholds: number[]) => {
    if (value >= thresholds[1]) return colors.danger;
    if (value >= thresholds[0]) return colors.warning;
    return colors.success;
  };

  const getSectorColor = (sector: string): string => {
    const sectorColors: Record<string, string> = {
      'Technology': '#2196f3',
      'Healthcare': '#4caf50',
      'Financials': '#ff9800',
      'Consumer': '#9c27b0',
      'Energy': '#f44336',
      'Industrials': '#795548',
      'Unknown': '#9e9e9e',
    };
    return sectorColors[sector] || colors.neutral;
  };

  return (
    <Grid container spacing={3}>
      {/* Risk Alerts */}
      {riskAlerts.length > 0 && (
        <Grid item xs={12}>
          <Box display="flex" flexDirection="column" gap={1}>
            {riskAlerts.map((alert, index) => (
              <Alert key={index} severity={alert.type} variant="filled">
                <Box display="flex" justifyContent="space-between" alignItems="center" width="100%">
                  <Typography>{alert.message}</Typography>
                  <Chip label={alert.value} color="inherit" size="small" />
                </Box>
              </Alert>
            ))}
          </Box>
        </Grid>
      )}

      {/* Risk Metrics Cards */}
      <Grid item xs={12}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <RiskCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <TrendingDown color="error" />
                  <Typography variant="h6" ml={1}>VaR (95%)</Typography>
                </Box>
                <Typography variant="h4" color="error.main">
                  ${riskMetrics?.var_95.toLocaleString() || '0'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Daily Value at Risk
                </Typography>
              </CardContent>
            </RiskCard>
          </Grid>

          <Grid item xs={12} md={3}>
            <RiskCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <Warning color="error" />
                  <Typography variant="h6" ml={1}>CVaR (95%)</Typography>
                </Box>
                <Typography variant="h4" color="error.main">
                  ${riskMetrics?.cvar_95.toLocaleString() || '0'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Expected Shortfall
                </Typography>
              </CardContent>
            </RiskCard>
          </Grid>

          <Grid item xs={12} md={3}>
            <RiskCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <Security color="primary" />
                  <Typography variant="h6" ml={1}>Beta</Typography>
                </Box>
                <Typography variant="h4" color="primary.main">
                  {riskMetrics?.beta.toFixed(2) || '0.00'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Market Sensitivity
                </Typography>
              </CardContent>
            </RiskCard>
          </Grid>

          <Grid item xs={12} md={3}>
            <RiskCard>
              <CardContent>
                <Box display="flex" alignItems="center" mb={1}>
                  <Assessment color="success" />
                  <Typography variant="h6" ml={1}>Sharpe Ratio</Typography>
                </Box>
                <Typography variant="h4" color="success.main">
                  {riskMetrics?.sharpe_ratio.toFixed(2) || '0.00'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Risk-Adjusted Return
                </Typography>
              </CardContent>
            </RiskCard>
          </Grid>
        </Grid>
      </Grid>

      {/* VaR Chart */}
      <Grid item xs={12} md={8}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Value at Risk Distribution
          </Typography>
          <Box height={350}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                { metric: 'VaR 95%', value: riskMetrics?.var_95 || 0 },
                { metric: 'VaR 99%', value: riskMetrics?.var_99 || 0 },
                { metric: 'CVaR 95%', value: riskMetrics?.cvar_95 || 0 },
                { metric: 'CVaR 99%', value: riskMetrics?.cvar_99 || 0 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="metric" stroke="#ffffff" />
                <YAxis stroke="#ffffff" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d2d',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff',
                  }}
                  formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Risk Amount']}
                />
                <Bar dataKey="value" fill={colors.danger} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </Grid>

      {/* Sector Concentration */}
      <Grid item xs={12} md={4}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Sector Concentration
          </Typography>
          <Box height={350}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={calculatedMetrics?.sectorConcentration || []}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={2}
                  dataKey="percentage"
                >
                  {calculatedMetrics?.sectorConcentration.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d2d',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff',
                  }}
                  formatter={(value, name, props) => [
                    `${Number(value).toFixed(1)}%`,
                    props.payload.sector
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </Grid>

      {/* Risk Factor Radar */}
      <Grid item xs={12} md={6}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Risk Factor Analysis
          </Typography>
          <Box height={400}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={[
                { factor: 'Market Risk', value: riskMetrics?.beta ? riskMetrics.beta * 50 : 0 },
                { factor: 'Volatility', value: riskMetrics?.volatility ? riskMetrics.volatility * 100 : 0 },
                { factor: 'Concentration', value: calculatedMetrics?.concentrationRisk ? calculatedMetrics.concentrationRisk * 100 : 0 },
                { factor: 'Liquidity', value: 25 }, // Placeholder
                { factor: 'Credit', value: 15 }, // Placeholder
              ]}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="factor" stroke="#ffffff" />
                <PolarRadiusAxis stroke="#ffffff" />
                <Radar
                  dataKey="value"
                  stroke={colors.info}
                  fill={colors.info}
                  fillOpacity={0.3}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#2d2d2d',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: '#ffffff',
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </Box>
        </StyledPaper>
      </Grid>

      {/* Position Risk Table */}
      <Grid item xs={12} md={6}>
        <StyledPaper>
          <Typography variant="h6" gutterBottom>
            Position Risk Breakdown
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell align="right">Exposure</TableCell>
                  <TableCell align="right">% of Portfolio</TableCell>
                  <TableCell align="right">Beta</TableCell>
                  <TableCell align="right">Risk Level</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.slice(0, 10).map((position, index) => {
                  const exposure = Math.abs(position.quantity * position.currentPrice);
                  const portfolioPercent = calculatedMetrics ? (exposure / calculatedMetrics.totalValue) * 100 : 0;
                  const beta = position.beta || 1.0;
                  const riskLevel = portfolioPercent > 10 ? 'High' : portfolioPercent > 5 ? 'Medium' : 'Low';
                  
                  return (
                    <TableRow key={index}>
                      <TableCell>{position.symbol}</TableCell>
                      <TableCell align="right">${exposure.toLocaleString()}</TableCell>
                      <TableCell align="right">{portfolioPercent.toFixed(1)}%</TableCell>
                      <TableCell align="right">{beta.toFixed(2)}</TableCell>
                      <TableCell align="right">
                        <Chip
                          label={riskLevel}
                          color={riskLevel === 'High' ? 'error' : riskLevel === 'Medium' ? 'warning' : 'success'}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </StyledPaper>
      </Grid>
    </Grid>
  );
};

export default RiskDashboard;
