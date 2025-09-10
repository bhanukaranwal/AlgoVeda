/*!
 * Real-time Visualization Hook
 * WebSocket-based streaming data management for charts and dashboards
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface MarketDataPoint {
  symbol: string;
  timestamp: number;
  price: number;
  volume: number;
  bid: number;
  ask: number;
  change: number;
  changePercent: number;
}

interface PortfolioDataPoint {
  timestamp: number;
  totalValue: number;
  cashBalance: number;
  totalPnL: number;
  dayPnL: number;
  realizedPnL: number;
  unrealizedPnL: number;
  positions: Array<{
    symbol: string;
    quantity: number;
    currentPrice: number;
    unrealizedPnL: number;
    dayPnL: number;
    dayChangePercent: number;
  }>;
}

interface UseRealtimeVisualizationReturn {
  marketData: Map<string, MarketDataPoint>;
  portfolioData: PortfolioDataPoint | null;
  isConnected: boolean;
  connectionStatus: string;
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
  reconnect: () => void;
  clearData: () => void;
}

export const useRealtimeVisualization = (
  websocketUrl: string = 'ws://localhost:8081/ws'
): UseRealtimeVisualizationReturn => {
  const [marketData, setMarketData] = useState<Map<string, MarketDataPoint>>(new Map());
  const [portfolioData, setPortfolioData] = useState<PortfolioDataPoint | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const subscribedSymbolsRef = useRef<Set<string>>(new Set());
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Connection management
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('Connecting...');
    
    try {
      wsRef.current = new WebSocket(websocketUrl);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('Connected');
        
        // Start heartbeat
        heartbeatIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'PING' }));
          }
        }, 30000);
        
        // Resubscribe to symbols
        if (subscribedSymbolsRef.current.size > 0) {
          const symbols = Array.from(subscribedSymbolsRef.current);
          wsRef.current.send(JSON.stringify({
            type: 'SUBSCRIBE_MARKET_DATA',
            symbols
          }));
        }
        
        // Subscribe to portfolio updates
        wsRef.current.send(JSON.stringify({
          type: 'SUBSCRIBE_PORTFOLIO',
          account_id: 'main_account'
        }));
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('Disconnected');
        
        // Clear heartbeat
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
          heartbeatIntervalRef.current = null;
        }
        
        // Attempt reconnection
        if (event.code !== 1000) { // Not a normal closure
          scheduleReconnect();
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('Error');
      };
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('Failed to connect');
      scheduleReconnect();
    }
  }, [websocketUrl]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    setConnectionStatus('Reconnecting...');
    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, 3000);
  }, [connect]);

  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'MARKET_DATA_UPDATE':
        setMarketData(prev => {
          const updated = new Map(prev);
          updated.set(data.symbol, {
            symbol: data.symbol,
            timestamp: data.timestamp,
            price: data.price,
            volume: data.volume,
            bid: data.bid,
            ask: data.ask,
            change: data.change,
            changePercent: data.changePercent
          });
          return updated;
        });
        break;
        
      case 'PORTFOLIO_UPDATE':
        setPortfolioData({
          timestamp: data.timestamp,
          totalValue: data.total_value,
          cashBalance: data.cash_balance,
          totalPnL: data.total_pnl,
          dayPnL: data.day_pnl,
          realizedPnL: data.realized_pnl,
          unrealizedPnL: data.unrealized_pnl,
          positions: data.positions.map((pos: any) => ({
            symbol: pos.symbol,
            quantity: pos.quantity,
            currentPrice: pos.current_price,
            unrealizedPnL: pos.unrealized_pnl,
            dayPnL: pos.day_pnl,
            dayChangePercent: pos.day_change_percent
          }))
        });
        break;
        
      case 'PONG':
        // Heartbeat response
        break;
        
      case 'ERROR':
        console.error('WebSocket error:', data.message);
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  }, []);

  const subscribe = useCallback((symbols: string[]) => {
    symbols.forEach(symbol => subscribedSymbolsRef.current.add(symbol));
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'SUBSCRIBE_MARKET_DATA',
        symbols
      }));
    }
  }, []);

  const unsubscribe = useCallback((symbols: string[]) => {
    symbols.forEach(symbol => {
      subscribedSymbolsRef.current.delete(symbol);
      setMarketData(prev => {
        const updated = new Map(prev);
        updated.delete(symbol);
        return updated;
      });
    });
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'UNSUBSCRIBE_MARKET_DATA',
        symbols
      }));
    }
  }, []);

  const reconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    connect();
  }, [connect]);

  const clearData = useCallback(() => {
    setMarketData(new Map());
    setPortfolioData(null);
  }, []);

  // Initialize connection
  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, [connect]);

  return {
    marketData,
    portfolioData,
    isConnected,
    connectionStatus,
    subscribe,
    unsubscribe,
    reconnect,
    clearData
  };
};

export default useRealtimeVisualization;
