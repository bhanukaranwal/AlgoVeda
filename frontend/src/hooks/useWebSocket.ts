// Advanced WebSocket hook with automatic reconnection and compression
export const useWebSocket = () => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [marketData, setMarketData] = useState<MarketDataMap>({});
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket('ws://localhost:8080/ws');
      ws.binaryType = 'arraybuffer'; // For compressed data
      
      ws.onopen = () => setConnected(true);
      ws.onmessage = handleMessage;
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 1000); // Auto-reconnect
      };
      
      setSocket(ws);
    };
    
    connect();
  }, []);

  // Handle high-frequency market data updates
  const handleMessage = useCallback((event: MessageEvent) => {
    const data = JSON.parse(event.data);
    switch (data.type) {
      case 'MARKET_DATA':
        setMarketData(prev => ({
          ...prev,
          [data.symbol]: data.data
        }));
        break;
      // Handle other message types...
    }
  }, []);
};
