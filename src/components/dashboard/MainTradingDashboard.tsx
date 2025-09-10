/*!
 * Main Trading Dashboard - Central Hub for AlgoVeda
 * Real-time trading interface with comprehensive market data
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Grid, Box, Paper, Typography, Tabs, Tab, IconButton } from '@mui/material';
import { 
  Timeline, 
  TrendingUp, 
  Assessment, 
  Security, 
  Settings,
  FullscreenIcon,
  RefreshIcon 
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

import { RealTimeMTMDashboard } from '../mtm_visualization/RealTimeMTMDashboard';
import { OptionsAnalyticsDashboard } from '../options_analytics/OptionsAnalyticsDashboard';
import { RiskDashboard } from '../risk_visualization/RiskDashboard';
import { ComprehensiveBacktestDashboard } from '../backtesting_visualization/ComprehensiveBacktestDashboard';
import { useRealtimeData } from '../../hooks/useRealtimeVisualization';
import { useOrderManagement } from '../../hooks/useOrderManagement';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  background: 'linear-gradient(145deg, #1e1e1e 0%, #2d2d2d 100%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 12,
  minHeight: 400,
}));

const DashboardContainer = styled(Box)(({ theme }) => ({
  background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
  minHeight: '100vh',
  padding: theme.spacing(2),
  color: '#ffffff',
}));

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const MainTradingDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  
  // Custom hooks for real-time data
  const { 
    marketData, 
    portfolioData, 
    isConnected, 
    connectionStatus 
  } = useRealtimeData();
  
  const { 
    orders, 
    positions, 
    submitOrder, 
    cancelOrder 
  } = useOrderManagement();

  // Tab change handler
  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  }, []);

  // Refresh handler
  const handleRefresh = useCallback(() => {
    setRefreshKey(prev => prev + 1);
  }, []);

  // Fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // Performance metrics calculation
  const performanceMetrics = useMemo(() => {
    if (!portfolioData) return null;
    
    const totalValue = portfolioData.positions?.reduce((sum, pos) => 
      sum + (pos.quantity * pos.currentPrice), 0) || 0;
    
    const totalPnL = portfolioData.positions?.reduce((sum, pos) => 
      sum + pos.unrealizedPnL, 0) || 0;
    
    const dayPnL = portfolioData.positions?.reduce((sum, pos) => 
      sum + pos.dayPnL, 0) || 0;
    
    return {
      totalValue,
      totalPnL,
      dayPnL,
      pnlPercentage: totalValue > 0 ? (totalPnL / totalValue) * 100 : 0,
    };
  }, [portfolioData]);

  // Connection status indicator
  const ConnectionStatus = () => (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Box
        sx={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          backgroundColor: isConnected ? '#4caf50' : '#f44336',
        }}
      />
      <Typography variant="caption" color={isConnected ? 'success.main' : 'error.main'}>
        {connectionStatus}
      </Typography>
    </Box>
  );

  return (
    <DashboardContainer>
      {/* Header */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 3,
        p: 2,
        background: 'rgba(255, 255, 255, 0.05)',
        borderRadius: 2,
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <Box>
          <Typography variant="h4" component="h1" sx={{ 
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #ff6b6b, #4ecdc4)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            color: 'transparent',
          }}>
            AlgoVeda Trading Platform
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            Professional Algorithmic Trading Dashboard
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <ConnectionStatus />
          
          {/* Performance Summary */}
          {performanceMetrics && (
            <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="textSecondary">Portfolio Value</Typography>
                <Typography variant="h6" color="primary.main">
                  ${performanceMetrics.totalValue.toLocaleString()}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="textSecondary">Total P&L</Typography>
                <Typography 
                  variant="h6" 
                  color={performanceMetrics.totalPnL >= 0 ? 'success.main' : 'error.main'}
                >
                  {performanceMetrics.totalPnL >= 0 ? '+' : ''}${performanceMetrics.totalPnL.toFixed(2)}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="caption" color="textSecondary">Day P&L</Typography>
                <Typography 
                  variant="h6" 
                  color={performanceMetrics.dayPnL >= 0 ? 'success.main' : 'error.main'}
                >
                  {performanceMetrics.dayPnL >= 0 ? '+' : ''}${performanceMetrics.dayPnL.toFixed(2)}
                </Typography>
              </Box>
            </Box>
          )}
          
          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <IconButton onClick={handleRefresh} color="primary">
              <RefreshIcon />
            </IconButton>
            <IconButton onClick={toggleFullscreen} color="primary">
              <FullscreenIcon />
            </IconButton>
            <IconButton color="primary">
              <Settings />
            </IconButton>
          </Box>
        </Box>
      </Box>

      {/* Navigation Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'rgba(255, 255, 255, 0.1)', mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              color: 'rgba(255, 255, 255, 0.7)',
              '&.Mui-selected': {
                color: '#4ecdc4',
              },
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#4ecdc4',
            },
          }}
        >
          <Tab 
            icon={<Timeline />} 
            label="Portfolio & P&L" 
            id="dashboard-tab-0"
            aria-controls="dashboard-tabpanel-0"
          />
          <Tab 
            icon={<TrendingUp />} 
            label="Options Analytics" 
            id="dashboard-tab-1"
            aria-controls="dashboard-tabpanel-1"
          />
          <Tab 
            icon={<Security />} 
            label="Risk Management" 
            id="dashboard-tab-2"
            aria-controls="dashboard-tabpanel-2"
          />
          <Tab 
            icon={<Assessment />} 
            label="Backtesting" 
            id="dashboard-tab-3"
            aria-controls="dashboard-tabpanel-3"
          />
        </Tabs>
      </Box>

      {/* Tab Content */}
      <TabPanel value={activeTab} index={0}>
        <RealTimeMTMDashboard 
          key={`mtm-${refreshKey}`}
          marketData={marketData}
          portfolioData={portfolioData}
          orders={orders}
          positions={positions}
        />
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <OptionsAnalyticsDashboard 
          key={`options-${refreshKey}`}
          marketData={marketData}
          portfolioData={portfolioData}
        />
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        <RiskDashboard 
          key={`risk-${refreshKey}`}
          portfolioData={portfolioData}
          positions={positions}
        />
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <ComprehensiveBacktestDashboard 
          key={`backtest-${refreshKey}`}
        />
      </TabPanel>
    </DashboardContainer>
  );
};

export default MainTradingDashboard;
