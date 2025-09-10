/*!
 * Advanced Chart Component with TradingView Integration
 * High-performance real-time charting with technical indicators
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData, CandlestickData } from 'lightweight-charts';

interface AdvancedChartProps {
  symbol: string;
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  height?: number;
  showVolume?: boolean;
  indicators?: string[];
  realTimeUpdates?: boolean;
}

export const AdvancedChart: React.FC<AdvancedChartProps> = ({
  symbol,
  timeframe,
  height = 600,
  showVolume = true,
  indicators = ['EMA20', 'EMA50', 'RSI'],
  realTimeUpdates = true
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const indicatorSeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { color: '#1a1a1a' },
        textColor: '#ffffff',
      },
      grid: {
        vertLines: { color: '#2a2a2a' },
        horzLines: { color: '#2a2a2a' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#758696',
          width: 1,
          style: 2,
          visible: true,
          labelVisible: true,
        },
        horzLine: {
          color: '#758696',
          width: 1,
          style: 2,
          visible: true,
          labelVisible: true,
        },
      },
      rightPriceScale: {
        borderColor: '#2a2a2a',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: '#2a2a2a',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create volume series if enabled
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
      });
      volumeSeriesRef.current = volumeSeries;
      
      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
    }

    // Add technical indicators
    indicators.forEach(indicator => {
      const series = chart.addLineSeries({
        color: getIndicatorColor(indicator),
        lineWidth: 2,
      });
      indicatorSeriesRef.current.set(indicator, series);
    });

    return () => {
      chart.remove();
      chartRef.current = null;
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      indicatorSeriesRef.current.clear();
    };
  }, [height, showVolume, indicators]);

  // Load historical data
  useEffect(() => {
    const loadHistoricalData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/market-data/history/${symbol}/${timeframe}?limit=1000`);
        if (!response.ok) throw new Error('Failed to fetch data');
        
        const data = await response.json();
        
        if (candlestickSeriesRef.current) {
          candlestickSeriesRef.current.setData(data.candles.map((candle: any) => ({
            time: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
          })));
        }

        if (volumeSeriesRef.current) {
          volumeSeriesRef.current.setData(data.candles.map((candle: any) => ({
            time: candle.timestamp,
            value: candle.volume,
            color: candle.close >= candle.open ? '#26a69a80' : '#ef535080',
          })));
        }

        // Load indicator data
        for (const [indicator, series] of indicatorSeriesRef.current) {
          if (data.indicators && data.indicators[indicator]) {
            series.setData(data.indicators[indicator]);
          }
        }

        setIsLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setIsLoading(false);
      }
    };

    loadHistoricalData();
  }, [symbol, timeframe]);

  // Handle real-time updates
  useEffect(() => {
    if (!realTimeUpdates) return;

    const ws = new WebSocket(`wss://localhost:8080/ws`);
    
    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'SUBSCRIBE_CANDLES',
        symbol,
        timeframe
      }));
    };

    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      
      if (update.type === 'CANDLE_UPDATE' && update.symbol === symbol) {
        const candle = {
          time: update.timestamp,
          open: update.open,
          high: update.high,
          low: update.low,
          close: update.close,
        };

        candlestickSeriesRef.current?.update(candle);
        
        if (volumeSeriesRef.current) {
          volumeSeriesRef.current.update({
            time: update.timestamp,
            value: update.volume,
            color: update.close >= update.open ? '#26a69a80' : '#ef535080',
          });
        }
      }
    };

    return () => {
      ws.close();
    };
  }, [symbol, timeframe, realTimeUpdates]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getIndicatorColor = (indicator: string): string => {
    const colors: Record<string, string> = {
      'EMA20': '#ff6b6b',
      'EMA50': '#4ecdc4',
      'SMA200': '#ffe66d',
      'RSI': '#a8e6cf',
      'MACD': '#ff8b94',
      'BB_UPPER': '#c7ceea',
      'BB_LOWER': '#c7ceea',
    };
    return colors[indicator] || '#ffffff';
  };

  if (error) {
    return (
      <div className="chart-error" style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1a1a1a', color: '#ff6b6b' }}>
        Error loading chart: {error}
      </div>
    );
  }

  return (
    <div className="advanced-chart" style={{ position: 'relative', height }}>
      {isLoading && (
        <div style={{ 
          position: 'absolute', 
          top: 0, 
          left: 0, 
          right: 0, 
          bottom: 0, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          background: '#1a1a1a80', 
          color: '#ffffff',
          zIndex: 10 
        }}>
          Loading chart data...
        </div>
      )}
      <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
};
