/*!
 * Ultra Fast Feed Handler
 * Ultra-low latency market data processing with kernel bypass
 */

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
};
use tokio::{
    net::UdpSocket,
    sync::{broadcast, mpsc, RwLock},
    time::interval,
};
use bytes::{Bytes, BytesMut, Buf, BufMut};
use tracing::{info, debug, warn, error, instrument};
use serde::{Serialize, Deserialize};

use crate::{
    config::MarketDataConfig,
    market_data::{MarketData, Tick, Quote, Trade, Level2Book, BookLevel},
    networking::NetworkManager,
    storage::StorageManager,
    monitoring::MetricsCollector,
    error::{Result, AlgoVedaError},
    utils::{
        latency::LatencyTracker,
        performance::PerformanceOptimizer,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedSubscription {
    pub symbol: String,
    pub feed_type: FeedType,
    pub venue: String,
    pub subscription_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedType {
    Level1,
    Level2,
    TimeAndSales,
    ImpliedVolatility,
    OptionsChain,
    Index,
}

#[derive(Debug, Clone)]
pub struct FeedMetrics {
    pub messages_processed: AtomicU64,
    pub messages_per_second: AtomicU64,
    pub avg_latency_nanos: AtomicU64,
    pub max_latency_nanos: AtomicU64,
    pub packet_drops: AtomicU64,
    pub parse_errors: AtomicU64,
    pub last_update: std::sync::Mutex<Instant>,
}

pub struct UltraFastFeedHandler {
    config: MarketDataConfig,
    subscriptions: Arc<RwLock<HashMap<u64, FeedSubscription>>>,
    active_symbols: Arc<RwLock<HashMap<String, Instant>>>,
    
    // High-performance data structures
    level1_cache: Arc<RwLock<HashMap<String, Quote>>>,
    level2_books: Arc<RwLock<HashMap<String, Level2Book>>>,
    recent_trades: Arc<RwLock<HashMap<String, Vec<Trade>>>>,
    
    // Communication channels
    tick_sender: broadcast::Sender<Tick>,
    quote_sender: broadcast::Sender<Quote>,
    trade_sender: broadcast::Sender<Trade>,
    book_sender: broadcast::Sender<(String, Level2Book)>,
    
    // Network components
    network_manager: Arc<NetworkManager>,
    storage_manager: Arc<StorageManager>,
    
    // Performance monitoring
    metrics_collector: Arc<MetricsCollector>,
    latency_tracker: Arc<LatencyTracker>,
    feed_metrics: Arc<FeedMetrics>,
    performance_optimizer: Arc<PerformanceOptimizer>,
    
    // State
    is_running: Arc<std::sync::atomic::AtomicBool>,
    next_subscription_id: AtomicU64,
}

impl UltraFastFeedHandler {
    pub async fn new(
        config: &MarketDataConfig,
        network_manager: Arc<NetworkManager>,
        storage_manager: Arc<StorageManager>,
        metrics_collector: Arc<MetricsCollector>,
    ) -> Result<Self> {
        // Create broadcast channels with large buffers for high throughput
        let (tick_sender, _) = broadcast::channel(config.buffer_size);
        let (quote_sender, _) = broadcast::channel(config.buffer_size);
        let (trade_sender, _) = broadcast::channel(config.buffer_size);
        let (book_sender, _) = broadcast::channel(config.buffer_size / 10); // Books are larger
        
        let feed_metrics = Arc::new(FeedMetrics {
            messages_processed: AtomicU64::new(0),
            messages_per_second: AtomicU64::new(0),
            avg_latency_nanos: AtomicU64::new(0),
            max_latency_nanos: AtomicU64::new(0),
            packet_drops: AtomicU64::new(0),
            parse_errors: AtomicU64::new(0),
            last_update: std::sync::Mutex::new(Instant::now()),
        });
        
        Ok(Self {
            config: config.clone(),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            active_symbols: Arc::new(RwLock::new(HashMap::new())),
            level1_cache: Arc::new(RwLock::new(HashMap::with_capacity(config.max_symbols))),
            level2_books: Arc::new(RwLock::new(HashMap::with_capacity(config.max_symbols))),
            recent_trades: Arc::new(RwLock::new(HashMap::with_capacity(config.max_symbols))),
            tick_sender,
            quote_sender,
            trade_sender,
            book_sender,
            network_manager,
            storage_manager,
            metrics_collector,
            latency_tracker: Arc::new(LatencyTracker::new()),
            feed_metrics,
            performance_optimizer: Arc::new(PerformanceOptimizer::new()),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            next_subscription_id: AtomicU64::new(1),
        })
    }

    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting ultra-fast feed handler");
        
        self.is_running.store(true, Ordering::Relaxed);
        
        // Apply performance optimizations
        self.performance_optimizer.optimize_for_latency().await?;
        
        // Start multicast listeners for each configured group
        let mut handles = Vec::new();
        
        for multicast_group in &self.config.multicast_groups {
            let handle = self.start_multicast_listener(multicast_group.clone()).await?;
            handles.push(handle);
        }
        
        // Start metrics collection
        let metrics_handle = self.start_metrics_collection().await;
        handles.push(metrics_handle);
        
        // Start data persistence
        let persistence_handle = self.start_data_persistence().await;
        handles.push(persistence_handle);
        
        info!("Ultra-fast feed handler started successfully");
        Ok(())
    }

    async fn start_multicast_listener(&self, multicast_addr: String) -> Result<tokio::task::JoinHandle<()>> {
        let socket_addr: SocketAddr = multicast_addr.parse()
            .map_err(|e| AlgoVedaError::MarketData(format!("Invalid multicast address: {}", e)))?;
        
        let socket = self.network_manager.create_multicast_socket(&socket_addr).await?;
        
        // Clone necessary data for the task
        let feed_metrics = self.feed_metrics.clone();
        let latency_tracker = self.latency_tracker.clone();
        let tick_sender = self.tick_sender.clone();
        let quote_sender = self.quote_sender.clone();
        let trade_sender = self.trade_sender.clone();
        let book_sender = self.book_sender.clone();
        let level1_cache = self.level1_cache.clone();
        let level2_books = self.level2_books.clone();
        let recent_trades = self.recent_trades.clone();
        let is_running = self.is_running.clone();
        
        let handle = tokio::spawn(async move {
            let mut buffer = BytesMut::with_capacity(65536); // 64KB buffer
            
            while is_running.load(Ordering::Relaxed) {
                buffer.clear();
                
                match socket.recv_buf(&mut buffer).await {
                    Ok(n) => {
                        if n > 0 {
                            let receive_time = Instant::now();
                            
                            // Process the packet
                            if let Err(e) = Self::process_packet(
                                &buffer[..n],
                                receive_time,
                                &feed_metrics,
                                &latency_tracker,
                                &tick_sender,
                                &quote_sender,
                                &trade_sender,
                                &book_sender,
                                &level1_cache,
                                &level2_books,
                                &recent_trades,
                            ).await {
                                feed_metrics.parse_errors.fetch_add(1, Ordering::Relaxed);
                                debug!("Packet processing error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        feed_metrics.packet_drops.fetch_add(1, Ordering::Relaxed);
                        warn!("UDP receive error: {}", e);
                    }
                }
            }
        });
        
        Ok(handle)
    }

    #[instrument(skip_all)]
    async fn process_packet(
        data: &[u8],
        receive_time: Instant,
        feed_metrics: &Arc<FeedMetrics>,
        latency_tracker: &Arc<LatencyTracker>,
        tick_sender: &broadcast::Sender<Tick>,
        quote_sender: &broadcast::Sender<Quote>,
        trade_sender: &broadcast::Sender<Trade>,
        book_sender: &broadcast::Sender<(String, Level2Book)>,
        level1_cache: &Arc<RwLock<HashMap<String, Quote>>>,
        level2_books: &Arc<RwLock<HashMap<String, Level2Book>>>,
        recent_trades: &Arc<RwLock<HashMap<String, Vec<Trade>>>>,
    ) -> Result<()> {
        let process_start = Instant::now();
        
        // Increment message counter
        feed_metrics.messages_processed.fetch_add(1, Ordering::Relaxed);
        
        // Parse the packet based on protocol
        let message = Self::parse_message(data)?;
        
        match message {
            MarketMessage::Quote(quote) => {
                // Update Level 1 cache
                {
                    let mut cache = level1_cache.write().await;
                    cache.insert(quote.symbol.clone(), quote.clone());
                }
                
                // Broadcast quote
                let _ = quote_sender.send(quote.clone());
                
                // Create tick from quote
                let tick = Tick {
                    symbol: quote.symbol.clone(),
                    timestamp: quote.timestamp,
                    price: quote.bid_price,
                    size: quote.bid_size,
                    side: crate::market_data::TickSide::Bid,
                };
                
                let _ = tick_sender.send(tick);
            }
            
            MarketMessage::Trade(trade) => {
                // Update recent trades
                {
                    let mut trades = recent_trades.write().await;
                    let symbol_trades = trades.entry(trade.symbol.clone()).or_insert_with(Vec::new);
                    symbol_trades.push(trade.clone());
                    
                    // Keep only recent trades (last 1000)
                    if symbol_trades.len() > 1000 {
                        symbol_trades.drain(0..500);
                    }
                }
                
                // Broadcast trade
                let _ = trade_sender.send(trade.clone());
                
                // Create tick from trade
                let tick = Tick {
                    symbol: trade.symbol.clone(),
                    timestamp: trade.timestamp,
                    price: trade.price,
                    size: trade.quantity,
                    side: crate::market_data::TickSide::Trade,
                };
                
                let _ = tick_sender.send(tick);
            }
            
            MarketMessage::Level2Update(update) => {
                // Update Level 2 book
                {
                    let mut books = level2_books.write().await;
                    let book = books.entry(update.symbol.clone())
                        .or_insert_with(|| Level2Book::new(update.symbol.clone()));
                    
                    // Apply the update
                    book.apply_update(update);
                    
                    // Broadcast updated book
                    let _ = book_sender.send((book.symbol.clone(), book.clone()));
                }
            }
        }
        
        // Calculate and track latency
        let processing_latency = process_start.elapsed();
        let total_latency = receive_time.elapsed();
        
        latency_tracker.record_latency("packet_processing", processing_latency);
        latency_tracker.record_latency("total_market_data", total_latency);
        
        // Update feed metrics
        let latency_nanos = total_latency.as_nanos() as u64;
        feed_metrics.avg_latency_nanos.store(latency_nanos, Ordering::Relaxed);
        
        let max_latency = feed_metrics.max_latency_nanos.load(Ordering::Relaxed);
        if latency_nanos > max_latency {
            feed_metrics.max_latency_nanos.store(latency_nanos, Ordering::Relaxed);
        }
        
        Ok(())
    }

    fn parse_message(data: &[u8]) -> Result<MarketMessage> {
        if data.len() < 4 {
            return Err(AlgoVedaError::MarketData("Message too short".to_string()));
        }
        
        let mut cursor = std::io::Cursor::new(data);
        
        // Read message type (first 2 bytes)
        let msg_type = cursor.get_u16();
        let msg_length = cursor.get_u16();
        
        if data.len() < msg_length as usize {
            return Err(AlgoVedaError::MarketData("Incomplete message".to_string()));
        }
        
        match msg_type {
            1 => Self::parse_quote_message(&mut cursor),
            2 => Self::parse_trade_message(&mut cursor),
            3 => Self::parse_level2_update(&mut cursor),
            _ => Err(AlgoVedaError::MarketData(format!("Unknown message type: {}", msg_type))),
        }
    }

    fn parse_quote_message(cursor: &mut std::io::Cursor<&[u8]>) -> Result<MarketMessage> {
        // Parse binary quote message
        let symbol_len = cursor.get_u8();
        let mut symbol_bytes = vec![0u8; symbol_len as usize];
        cursor.copy_to_slice(&mut symbol_bytes);
        let symbol = String::from_utf8(symbol_bytes)
            .map_err(|e| AlgoVedaError::MarketData(format!("Invalid symbol: {}", e)))?;
        
        let timestamp = cursor.get_u64();
        let bid_price = cursor.get_f64();
        let ask_price = cursor.get_f64();
        let bid_size = cursor.get_f64();
        let ask_size = cursor.get_f64();
        
        Ok(MarketMessage::Quote(Quote {
            symbol,
            timestamp: std::time::UNIX_EPOCH + std::time::Duration::from_nanos(timestamp),
            bid_price: rust_decimal::Decimal::from_f64(bid_price).unwrap(),
            ask_price: rust_decimal::Decimal::from_f64(ask_price).unwrap(),
            bid_size: rust_decimal::Decimal::from_f64(bid_size).unwrap(),
            ask_size: rust_decimal::Decimal::from_f64(ask_size).unwrap(),
        }))
    }

    fn parse_trade_message(cursor: &mut std::io::Cursor<&[u8]>) -> Result<MarketMessage> {
        // Parse binary trade message
        let symbol_len = cursor.get_u8();
        let mut symbol_bytes = vec![0u8; symbol_len as usize];
        cursor.copy_to_slice(&mut symbol_bytes);
        let symbol = String::from_utf8(symbol_bytes)
            .map_err(|e| AlgoVedaError::MarketData(format!("Invalid symbol: {}", e)))?;
        
        let timestamp = cursor.get_u64();
        let price = cursor.get_f64();
        let quantity = cursor.get_f64();
        let side = cursor.get_u8();
        
        Ok(MarketMessage::Trade(Trade {
            symbol,
            timestamp: std::time::UNIX_EPOCH + std::time::Duration::from_nanos(timestamp),
            price: rust_decimal::Decimal::from_f64(price).unwrap(),
            quantity: rust_decimal::Decimal::from_f64(quantity).unwrap(),
            side: if side == 1 { 
                crate::market_data::TradeSide::Buy 
            } else { 
                crate::market_data::TradeSide::Sell 
            },
            trade_id: format!("trade_{}", timestamp),
        }))
    }

    fn parse_level2_update(cursor: &mut std::io::Cursor<&[u8]>) -> Result<MarketMessage> {
        // Parse Level 2 book update
        let symbol_len = cursor.get_u8();
        let mut symbol_bytes = vec![0u8; symbol_len as usize];
        cursor.copy_to_slice(&mut symbol_bytes);
        let symbol = String::from_utf8(symbol_bytes)
            .map_err(|e| AlgoVedaError::MarketData(format!("Invalid symbol: {}", e)))?;
        
        let timestamp = cursor.get_u64();
        let num_levels = cursor.get_u16();
        
        let mut updates = Vec::new();
        for _ in 0..num_levels {
            let side = cursor.get_u8();
            let price = cursor.get_f64();
            let size = cursor.get_f64();
            let action = cursor.get_u8(); // 0=Add, 1=Update, 2=Delete
            
            updates.push(Level2Update {
                side: if side == 0 { 
                    crate::market_data::BookSide::Bid 
                } else { 
                    crate::market_data::BookSide::Ask 
                },
                price: rust_decimal::Decimal::from_f64(price).unwrap(),
                size: rust_decimal::Decimal::from_f64(size).unwrap(),
                action: match action {
                    0 => Level2Action::Add,
                    1 => Level2Action::Update,
                    2 => Level2Action::Delete,
                    _ => Level2Action::Update,
                },
            });
        }
        
        Ok(MarketMessage::Level2Update(Level2BookUpdate {
            symbol,
            timestamp: std::time::UNIX_EPOCH + std::time::Duration::from_nanos(timestamp),
            updates,
        }))
    }

    pub async fn subscribe(&self, symbol: String, feed_type: FeedType, venue: String) -> u64 {
        let subscription_id = self.next_subscription_id.fetch_add(1, Ordering::Relaxed);
        
        let subscription = FeedSubscription {
            symbol: symbol.clone(),
            feed_type,
            venue,
            subscription_id,
        };
        
        {
            let mut subscriptions = self.subscriptions.write().await;
            subscriptions.insert(subscription_id, subscription);
        }
        
        {
            let mut active_symbols = self.active_symbols.write().await;
            active_symbols.insert(symbol, Instant::now());
        }
        
        info!("Subscribed to {} with ID {}", symbol, subscription_id);
        subscription_id
    }

    pub async fn unsubscribe(&self, subscription_id: u64) -> Result<()> {
        let mut subscriptions = self.subscriptions.write().await;
        
        if let Some(subscription) = subscriptions.remove(&subscription_id) {
            info!("Unsubscribed from {} (ID: {})", subscription.symbol, subscription_id);
            Ok(())
        } else {
            Err(AlgoVedaError::MarketData(
                format!("Subscription not found: {}", subscription_id)
            ))
        }
    }

    pub fn get_tick_receiver(&self) -> broadcast::Receiver<Tick> {
        self.tick_sender.subscribe()
    }

    pub fn get_quote_receiver(&self) -> broadcast::Receiver<Quote> {
        self.quote_sender.subscribe()
    }

    pub fn get_trade_receiver(&self) -> broadcast::Receiver<Trade> {
        self.trade_sender.subscribe()
    }

    pub fn get_book_receiver(&self) -> broadcast::Receiver<(String, Level2Book)> {
        self.book_sender.subscribe()
    }

    async fn start_metrics_collection(&self) -> tokio::task::JoinHandle<()> {
        let metrics_collector = self.metrics_collector.clone();
        let feed_metrics = self.feed_metrics.clone();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            let mut last_message_count = 0u64;
            
            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                let current_messages = feed_metrics.messages_processed.load(Ordering::Relaxed);
                let messages_this_second = current_messages.saturating_sub(last_message_count);
                last_message_count = current_messages;
                
                feed_metrics.messages_per_second.store(messages_this_second, Ordering::Relaxed);
                
                // Update Prometheus metrics
                metrics_collector.set_gauge(
                    "market_data_messages_per_second",
                    messages_this_second as f64,
                    &[],
                );
                
                metrics_collector.set_gauge(
                    "market_data_avg_latency_nanos",
                    feed_metrics.avg_latency_nanos.load(Ordering::Relaxed) as f64,
                    &[],
                );
                
                metrics_collector.set_gauge(
                    "market_data_max_latency_nanos",
                    feed_metrics.max_latency_nanos.load(Ordering::Relaxed) as f64,
                    &[],
                );
            }
        })
    }

    async fn start_data_persistence(&self) -> tokio::task::JoinHandle<()> {
        let storage_manager = self.storage_manager.clone();
        let mut trade_receiver = self.trade_sender.subscribe();
        let mut quote_receiver = self.quote_sender.subscribe();
        let is_running = self.is_running.clone();
        
        tokio::spawn(async move {
            while is_running.load(Ordering::Relaxed) {
                tokio::select! {
                    Ok(trade) = trade_receiver.recv() => {
                        if let Err(e) = storage_manager.store_trade(&trade).await {
                            debug!("Failed to store trade: {}", e);
                        }
                    }
                    Ok(quote) = quote_receiver.recv() => {
                        if let Err(e) = storage_manager.store_quote(&quote).await {
                            debug!("Failed to store quote: {}", e);
                        }
                    }
                }
            }
        })
    }

    pub async fn get_feed_statistics(&self) -> FeedStatistics {
        let messages_processed = self.feed_metrics.messages_processed.load(Ordering::Relaxed);
        let messages_per_second = self.feed_metrics.messages_per_second.load(Ordering::Relaxed);
        let avg_latency_nanos = self.feed_metrics.avg_latency_nanos.load(Ordering::Relaxed);
        let max_latency_nanos = self.feed_metrics.max_latency_nanos.load(Ordering::Relaxed);
        let packet_drops = self.feed_metrics.packet_drops.load(Ordering::Relaxed);
        let parse_errors = self.feed_metrics.parse_errors.load(Ordering::Relaxed);
        
        let active_subscriptions = self.subscriptions.read().await.len();
        let active_symbols = self.active_symbols.read().await.len();
        
        FeedStatistics {
            messages_processed,
            messages_per_second,
            avg_latency_nanos,
            max_latency_nanos,
            packet_drops,
            parse_errors,
            active_subscriptions,
            active_symbols,
        }
    }
}

#[derive(Debug, Clone)]
enum MarketMessage {
    Quote(Quote),
    Trade(Trade),
    Level2Update(Level2BookUpdate),
}

#[derive(Debug, Clone)]
struct Level2BookUpdate {
    symbol: String,
    timestamp: std::time::SystemTime,
    updates: Vec<Level2Update>,
}

#[derive(Debug, Clone)]
struct Level2Update {
    side: crate::market_data::BookSide,
    price: rust_decimal::Decimal,
    size: rust_decimal::Decimal,
    action: Level2Action,
}

#[derive(Debug, Clone)]
enum Level2Action {
    Add,
    Update,
    Delete,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeedStatistics {
    pub messages_processed: u64,
    pub messages_per_second: u64,
    pub avg_latency_nanos: u64,
    pub max_latency_nanos: u64,
    pub packet_drops: u64,
    pub parse_errors: u64,
    pub active_subscriptions: usize,
    pub active_symbols: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feed_handler_creation() {
        let config = MarketDataConfig {
            providers: vec![],
            buffer_size: 1024,
            max_symbols: 1000,
            enable_level2: true,
            enable_time_and_sales: true,
            multicast_groups: vec!["239.1.1.1:9001".to_string()],
            failover_enabled: true,
            latency_monitoring: true,
        };
        
        let network_manager = Arc::new(NetworkManager::new_test());
        let storage_manager = Arc::new(StorageManager::new_test());
        let metrics_collector = Arc::new(MetricsCollector::new_test());
        
        let feed_handler = UltraFastFeedHandler::new(
            &config,
            network_manager,
            storage_manager,
            metrics_collector,
        ).await;
        
        assert!(feed_handler.is_ok());
    }

    #[tokio::test]
    async fn test_subscription_management() {
        let config = MarketDataConfig::default();
        let network_manager = Arc::new(NetworkManager::new_test());
        let storage_manager = Arc::new(StorageManager::new_test());
        let metrics_collector = Arc::new(MetricsCollector::new_test());
        
        let feed_handler = UltraFastFeedHandler::new(
            &config,
            network_manager,
            storage_manager,
            metrics_collector,
        ).await.unwrap();
        
        // Test subscription
        let sub_id = feed_handler.subscribe(
            "AAPL".to_string(),
            FeedType::Level1,
            "NASDAQ".to_string(),
        ).await;
        
        assert!(sub_id > 0);
        
        // Test unsubscription
        let result = feed_handler.unsubscribe(sub_id).await;
        assert!(result.is_ok());
    }
}
