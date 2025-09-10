// Real-time trading dashboard with sub-second updates
import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { OrderBook } from './OrderBook';
import { PortfolioView } from './PortfolioView';
import { RiskMonitor } from './RiskMonitor';

export const TradingDashboard: React.FC = () => {
  const { marketData, orders, positions, connected } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('NIFTY');
  
  return (
    <div className="trading-dashboard">
      <Header connected={connected} />
      <div className="dashboard-grid">
        <OrderBook symbol={selectedSymbol} data={marketData[selectedSymbol]} />
        <OrderEntry symbol={selectedSymbol} />
        <PortfolioView positions={positions} />
        <RiskMonitor realTimeMetrics={true} />
        <ChartView symbol={selectedSymbol} />
        <OrderHistory orders={orders} />
      </div>
    </div>
  );
};
