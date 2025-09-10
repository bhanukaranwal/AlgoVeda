/*!
 * Ultra-Fast Trading Dashboard with Sub-Second Updates
 * Optimized for 100+ symbols with real-time streaming data
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { FixedSizeList as List } from 'react-window';
import { useVirtual } from 'react-virtual';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid: number;
  ask: number;
  timestamp: number;
}

interface DashboardProps {
  symbols: string[];
  maxUpdatesPerSecond?: number;
  enableVirtualization?: boolean;
  enable3DVisualization?: boolean;
}

// High-performance market data store with temporal compression
class MarketDataStore {
  private data = new Map<string, MarketData>();
  private updateQueue = new Map<string, MarketData>();
  private lastFlush = Date.now();
  private flushInterval = 16; // ~60 FPS
  
  update(symbol: string, data: Partial<MarketData>) {
    const existing = this.data.get(symbol) || this.createDefaultData(symbol);
    const updated = { ...existing, ...data, timestamp: Date.now() };
    this.updateQueue.set(symbol, updated);
  }
  
  flush(): Map<string, MarketData> {
    const now = Date.now();
    if (now - this.lastFlush < this.flushInterval) {
      return new Map();
    }
    
    // Apply all queued updates atomically
    for (const [symbol, data] of this.updateQueue) {
      this.data.set(symbol, data);
    }
    
    const updates = new Map(this.updateQueue);
    this.updateQueue.clear();
    this.lastFlush = now;
    return updates;
  }
  
  get(symbol: string): MarketData | undefined {
    return this.data.get(symbol);
  }
  
  getAll(): Map<string, MarketData> {
    return new Map(this.data);
  }
  
  private createDefaultData(symbol: string): MarketData {
    return {
      symbol,
      price: 0,
      change: 0,
      changePercent: 0,
      volume: 0,
      bid: 0,
      ask: 0,
      timestamp: Date.now()
    };
  }
}

// Ultra-fast WebSocket hook with batched updates
function useUltraFastWebSocket(symbols: string[], maxUpdatesPerSecond = 60) {
  const [marketData, setMarketData] = useState<Map<string, MarketData>>(new Map());
  const storeRef = useRef(new MarketDataStore());
  const wsRef = useRef<WebSocket | null>(null);
  const frameRef = useRef<number>();
  
  // Batched state updates using RAF for smooth performance
  const scheduleUpdate = useCallback(() => {
    if (frameRef.current) return;
    
    frameRef.current = requestAnimationFrame(() => {
      const updates = storeRef.current.flush();
      if (updates.size > 0) {
        setMarketData(storeRef.current.getAll());
      }
      frameRef.current = undefined;
    });
  }, []);
  
  useEffect(() => {
    const ws = new WebSocket('wss://localhost:8080/ws');
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    
    ws.onopen = () => {
      // Subscribe to symbols with batching preferences
      ws.send(JSON.stringify({
        type: 'SUBSCRIBE',
        symbols,
        batchSize: 50,
        maxLatency: 16 // 16ms max batching delay
      }));
    };
    
    ws.onmessage = (event) => {
      try {
        let data;
        
        // Handle binary messages for maximum performance
        if (event.data instanceof ArrayBuffer) {
          const view = new DataView(event.data);
          data = this.deserializeBinaryMessage(view);
        } else {
          data = JSON.parse(event.data);
        }
        
        // Process batch updates
        if (data.type === 'MARKET_DATA_BATCH') {
          for (const update of data.updates) {
            storeRef.current.update(update.symbol, update);
          }
          scheduleUpdate();
        }
      } catch (error) {
        console.error('WebSocket message processing error:', error);
      }
    };
    
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      ws.close();
    };
  }, [symbols, scheduleUpdate]);
  
  return marketData;
}

// Virtualized symbol list for handling 1000+ symbols
const VirtualizedSymbolList: React.FC<{
  symbols: string[];
  marketData: Map<string, MarketData>;
  height: number;
}> = ({ symbols, marketData, height }) => {
  
  const Row = useCallback(({ index, style }: any) => {
    const symbol = symbols[index];
    const data = marketData.get(symbol);
    
    if (!data) return <div style={style}>Loading...</div>;
    
    const changeColor = data.change >= 0 ? '#00ff88' : '#ff4444';
    
    return (
      <div
        style={{
          ...style,
          display: 'flex',
          alignItems: 'center',
          padding: '4px 8px',
          borderBottom: '1px solid #333',
          fontFamily: 'monospace',
          fontSize: '12px'
        }}
      >
        <span style={{ width: '80px', fontWeight: 'bold' }}>{symbol}</span>
        <span style={{ width: '80px', textAlign: 'right' }}>
          ${data.price.toFixed(2)}
        </span>
        <span style={{ width: '80px', textAlign: 'right', color: changeColor }}>
          {data.change > 0 ? '+' : ''}{data.change.toFixed(2)}
        </span>
        <span style={{ width: '60px', textAlign: 'right', color: changeColor }}>
          {data.changePercent.toFixed(2)}%
        </span>
        <span style={{ width: '100px', textAlign: 'right', fontSize: '10px' }}>
          Vol: {(data.volume / 1000).toFixed(0)}K
        </span>
      </div>
    );
  }, [marketData, symbols]);
  
  return (
    <List
      height={height}
      itemCount={symbols.length}
      itemSize={32}
      overscanCount={5}
    >
      {Row}
    </List>
  );
};

// 3D market visualization using Three.js for advanced users
const Market3DVisualization: React.FC<{
  marketData: Map<string, MarketData>;
}> = ({ marketData }) => {
  
  const Spheres = () => {
    const meshRef = useRef<THREE.InstancedMesh>();
    const symbolsArray = Array.from(marketData.keys()).slice(0, 100); // Limit for performance
    
    useFrame(() => {
      if (!meshRef.current) return;
      
      const dummy = new THREE.Object3D();
      symbolsArray.forEach((symbol, i) => {
        const data = marketData.get(symbol);
        if (!data) return;
        
        // Position based on price and volume
        const x = (i % 10 - 5) * 2;
        const y = (Math.floor(i / 10) - 5) * 2;
        const z = (data.changePercent || 0) * 10;
        
        // Scale based on volume
        const scale = Math.max(0.1, Math.min(2, (data.volume || 1) / 1000000));
        
        dummy.position.set(x, y, z);
        dummy.scale.setScalar(scale);
        dummy.updateMatrix();
        
        meshRef.current!.setMatrixAt(i, dummy.matrix);
        
        // Color based on change
        const color = new THREE.Color(data.change >= 0 ? 0x00ff88 : 0xff4444);
        meshRef.current!.setColorAt(i, color);
      });
      
      meshRef.current.instanceMatrix.needsUpdate = true;
      if (meshRef.current.instanceColor) {
        meshRef.current.instanceColor.needsUpdate = true;
      }
    });
    
    return (
      <instancedMesh ref={meshRef} args={[undefined, undefined, symbolsArray.length]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial />
      </instancedMesh>
    );
  };
  
  return (
    <Canvas camera={{ position: [0, 0, 10] }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Spheres />
    </Canvas>
  );
};

// Main ultra-fast dashboard component
export const UltraFastDashboard: React.FC<DashboardProps> = ({
  symbols,
  maxUpdatesPerSecond = 60,
  enableVirtualization = true,
  enable3DVisualization = false
}) => {
  const marketData = useUltraFastWebSocket(symbols, maxUpdatesPerSecond);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbols);
  const [viewMode, setViewMode] = useState<'list' | '3d' | 'grid'>('list');
  
  // Performance monitoring
  const renderCountRef = useRef(0);
  const [fps, setFps] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setFps(renderCountRef.current);
      renderCountRef.current = 0;
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Memoized sorted symbols for performance
  const sortedSymbols = useMemo(() => {
    return symbols.sort((a, b) => {
      const dataA = marketData.get(a);
      const dataB = marketData.get(b);
      if (!dataA || !dataB) return 0;
      return Math.abs(dataB.changePercent) - Math.abs(dataA.changePercent);
    });
  }, [symbols, marketData]);
  
  renderCountRef.current++;
  
  return (
    <div className="ultra-fast-dashboard" style={{ 
      height: '100vh', 
      background: '#111', 
      color: '#fff',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Header with performance metrics */}
      <div style={{ 
        padding: '8px 16px', 
        borderBottom: '1px solid #333',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, fontSize: '18px' }}>AlgoVeda Live Market Data</h1>
        <div style={{ fontSize: '12px', color: '#888' }}>
          Symbols: {symbols.length} | Updates: {marketData.size} | FPS: {fps}
        </div>
      </div>
      
      {/* View mode selector */}
      <div style={{ padding: '8px 16px', borderBottom: '1px solid #333' }}>
        {(['list', '3d', 'grid'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            style={{
              marginRight: '8px',
              padding: '4px 12px',
              background: viewMode === mode ? '#007acc' : '#333',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>
      
      {/* Main content area */}
      <div style={{ height: 'calc(100vh - 120px)' }}>
        {viewMode === 'list' && enableVirtualization && (
          <VirtualizedSymbolList
            symbols={sortedSymbols}
            marketData={marketData}
            height={window.innerHeight - 120}
          />
        )}
        
        {viewMode === '3d' && enable3DVisualization && (
          <Market3DVisualization marketData={marketData} />
        )}
        
        {viewMode === 'grid' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '8px',
            padding: '16px',
            height: '100%',
            overflow: 'auto'
          }}>
            {sortedSymbols.slice(0, 50).map(symbol => {
              const data = marketData.get(symbol);
              if (!data) return null;
              
              return (
                <div
                  key={symbol}
                  style={{
                    background: '#222',
                    padding: '12px',
                    borderRadius: '4px',
                    border: `2px solid ${data.change >= 0 ? '#00ff88' : '#ff4444'}`,
                    cursor: 'pointer'
                  }}
                  onClick={() => setSelectedSymbol(symbol)}
                >
                  <div style={{ fontSize: '14px', fontWeight: 'bold' }}>{symbol}</div>
                  <div style={{ fontSize: '18px' }}>${data.price.toFixed(2)}</div>
                  <div style={{
                    color: data.change >= 0 ? '#00ff88' : '#ff4444',
                    fontSize: '12px'
                  }}>
                    {data.change > 0 ? '+' : ''}{data.change.toFixed(2)} ({data.changePercent.toFixed(2)}%)
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default UltraFastDashboard;
