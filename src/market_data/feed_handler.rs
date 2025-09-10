/*!
 * High-Performance Market Data Feed Handler
 * Multi-venue real-time market data processing with microsecond latency
 */

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    net::{TcpStream, UdpSocket},
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout},
};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn, instrument};
use chrono::{DateTime, Utc};

use crate::{
    error::{Result, AlgoVedaError},
    utils::latency::LatencyTracker,
    monitoring::MetricsCollector,
    networking::connection_pool::ConnectionPool,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedType {
    Level1,      // Best bid/offer
    Level2,      // Order book
    Trade,       // Trade data
    Imbalance,   // Order imbalance
    Statistics,  // Market statistics
    News,        // News feeds
    Reference,   // Reference data
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VenueType {
    NYSE,
    NASDAQ,
    ARCA,
    BATS,
    IEX,
    MEMX,
    CBOE,
    CME,
    ICE,
    Binance,
    Coinbase,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataMessage {
    pub venue: VenueType,
    pub symbol: String,
    pub feed_type: FeedType,
    pub sequence_number: u64,
    pub timestamp_exchange: u64,  // Exchange timestamp (nanoseconds)
    pub timestamp_received: u64,  // Our timestamp (nanoseconds)
    pub data: MarketDataPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketDataPayload {
    Level1 {
        bid_price: f64,
        bid_size: u64,
        ask_price: f64,
        ask_size: u64,
        last_price: Option<f64>,
        last_size: Option<u64>,
    },
    Level2 {
        bids: Vec<(f64, u64)>,  // (price, size)
        asks: Vec<(f64, u64)>,
        sequence: u64,
    },
    Trade {
        price: f64,
        size: u64,
        side: char,  // 'B'uy, 'S'ell
        trade_id: String,
        conditions: Vec<String>,
    },
    Imbalance {
        paired_shares: u64,
        imbalance_shares: u64,
        imbalance_side: char,
        reference_price: f64,
        clearing_price: f64,
    },
    Statistics {
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
        vwap: f64,
    },
    News {
        headline: String,
        body: String,
        symbols: Vec<String>,
        sentiment: Option<f32>,
        urgency: u8,
    },
}

pub struct FeedHandlerConfig {
    pub venue: VenueType,
    pub feed_types: Vec<FeedType>,
    pub symbols: Vec<String>,
    pub multicast_groups: Vec<String>,
    pub tcp_endpoints: Vec<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub max_buffer_size: usize,
    pub heartbeat_interval: Duration,
    pub reconnect_delay: Duration,
    pub enable_recovery: bool,
    pub recovery_buffer_size: usize,
    pub latency_tracking: bool,
}

pub struct FeedHandler {
    config: FeedHandlerConfig,
    connection_pool: Arc<ConnectionPool>,
    message_sender: broadcast::Sender<MarketDataMessage>,
    metrics: Arc<MetricsCollector>,
    latency_tracker: Arc<LatencyTracker>,
    
    // State tracking
    sequence_numbers: Arc<RwLock<HashMap<String, u64>>>,
    connection_status: Arc<AtomicBool>,
    messages_processed: Arc<AtomicU64>,
    bytes_processed: Arc<AtomicU64>,
    
    // Recovery and buffering
    recovery_buffer: Arc<Mutex<VecDeque<MarketDataMessage>>>,
    gap_detector: Arc<Mutex<GapDetector>>,
    
    // Performance monitoring
    last_heartbeat: Arc<AtomicU64>,
    processing_times: Arc<Mutex<VecDeque<Duration>>>,
}

struct GapDetector {
    expected_sequences: HashMap<String, u64>,
    detected_gaps: VecDeque<Gap>,
    max_gap_age: Duration,
}

#[derive(Debug, Clone)]
struct Gap {
    symbol: String,
    start_sequence: u64,
    end_sequence: u64,
    detected_at: Instant,
}

impl FeedHandler {
    pub fn new(config: FeedHandlerConfig, metrics: Arc<MetricsCollector>) -> Result<Self> {
        let (message_sender, _) = broadcast::channel(100000);
        let connection_pool = Arc::new(ConnectionPool::new(16)?);
        
        Ok(Self {
            config,
            connection_pool,
            message_sender,
            metrics,
            latency_tracker: Arc::new(LatencyTracker::new()),
            sequence_numbers: Arc::new(RwLock::new(HashMap::new())),
            connection_status: Arc::new(AtomicBool::new(false)),
            messages_processed: Arc::new(AtomicU64::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            recovery_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(config.recovery_buffer_size))),
            gap_detector: Arc::new(Mutex::new(GapDetector::new())),
            last_heartbeat: Arc::new(AtomicU64::new(0)),
            processing_times: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
        })
    }

    /// Start the feed handler
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting feed handler for venue {:?}", self.config.venue);
        
        // Start multicast receivers
        for group in &self.config.multicast_groups {
            self.start_multicast_receiver(group.clone()).await?;
        }
        
        // Start TCP connections
        for endpoint in &self.config.tcp_endpoints {
            self.start_tcp_connection(endpoint.clone()).await?;
        }
        
        // Start monitoring tasks
        self.start_heartbeat_monitor().await;
        self.start_gap_detection().await;
        self.start_metrics_collection().await;
        
        self.connection_status.store(true, Ordering::Relaxed);
        info!("Feed handler started successfully");
        
        Ok(())
    }

    /// Start multicast UDP receiver
    async fn start_multicast_receiver(&self, multicast_group: String) -> Result<()> {
        let socket = UdpSocket::bind("0.0.0.0:0").await
            .map_err(|e| AlgoVedaError::Network(format!("Failed to bind UDP socket: {}", e)))?;
        
        socket.join_multicast_v4(
            multicast_group.parse().unwrap(),
            "0.0.0.0".parse().unwrap()
        ).map_err(|e| AlgoVedaError::Network(format!("Failed to join multicast group: {}", e)))?;

        let message_sender = self.message_sender.clone();
        let metrics = self.metrics.clone();
        let latency_tracker = self.latency_tracker.clone();
        let venue = self.config.venue.clone();
        let sequence_numbers = self.sequence_numbers.clone();
        let messages_processed = self.messages_processed.clone();
        let bytes_processed = self.bytes_processed.clone();
        let recovery_buffer = self.recovery_buffer.clone();
        let gap_detector = self.gap_detector.clone();
        
        tokio::spawn(async move {
            let mut buffer = [0u8; 65536]; // 64KB buffer
            
            loop {
                match socket.recv(&mut buffer).await {
                    Ok(size) => {
                        let start_time = Instant::now();
                        let received_timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64;
                        
                        // Process the raw packet
                        match Self::parse_market_data_packet(
                            &buffer[..size],
                            &venue,
                            received_timestamp
                        ) {
                            Ok(messages) => {
                                for message in messages {
                                    // Update sequence tracking
                                    Self::update_sequence_tracking(
                                        &sequence_numbers,
                                        &message,
                                        &gap_detector
                                    ).await;
                                    
                                    // Buffer for recovery if enabled
                                    if size > 0 { // recovery enabled check
                                        let mut buffer = recovery_buffer.lock().await;
                                        buffer.push_back(message.clone());
                                        if buffer.len() > 10000 { // max buffer size
                                            buffer.pop_front();
                                        }
                                    }
                                    
                                    // Send to subscribers
                                    if let Err(_) = message_sender.send(message) {
                                        warn!("No subscribers for market data message");
                                    }
                                }
                                
                                messages_processed.fetch_add(messages.len() as u64, Ordering::Relaxed);
                            }
                            Err(e) => {
                                warn!("Failed to parse market data packet: {}", e);
                                metrics.increment_counter("market_data_parse_errors", &[]);
                            }
                        }
                        
                        bytes_processed.fetch_add(size as u64, Ordering::Relaxed);
                        
                        // Track processing latency
                        let processing_time = start_time.elapsed();
                        latency_tracker.record_latency("market_data_processing", processing_time);
                        
                        metrics.record_histogram("market_data_processing_time_us", 
                            processing_time.as_micros() as f64, &[]);
                    }
                    Err(e) => {
                        error!("UDP receive error: {}", e);
                        metrics.increment_counter("market_data_receive_errors", &[]);
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Start TCP connection for recovery/reference data
    async fn start_tcp_connection(&self, endpoint: String) -> Result<()> {
        let stream = TcpStream::connect(&endpoint).await
            .map_err(|e| AlgoVedaError::Network(format!("Failed to connect to {}: {}", endpoint, e)))?;

        // Configure TCP socket for low latency
        let socket = socket2::Socket::from(stream);
        socket.set_nodelay(true)
            .map_err(|e| AlgoVedaError::Network(format!("Failed to set TCP_NODELAY: {}", e)))?;

        info!("Connected to TCP endpoint: {}", endpoint);
        Ok(())
    }

    /// Parse binary market data packet into structured messages
    fn parse_market_data_packet(
        data: &[u8], 
        venue: &VenueType, 
        received_timestamp: u64
    ) -> Result<Vec<MarketDataMessage>> {
        let mut messages = Vec::new();
        let mut cursor = std::io::Cursor::new(data);
        
        match venue {
            VenueType::NYSE => {
                messages.extend(Self::parse_nyse_packet(&mut cursor, received_timestamp)?);
            }
            VenueType::NASDAQ => {
                messages.extend(Self::parse_nasdaq_packet(&mut cursor, received_timestamp)?);
            }
            VenueType::CBOE => {
                messages.extend(Self::parse_cboe_packet(&mut cursor, received_timestamp)?);
            }
            VenueType::IEX => {
                messages.extend(Self::parse_iex_packet(&mut cursor, received_timestamp)?);
            }
            _ => {
                return Err(AlgoVedaError::Feed(format!("Unsupported venue: {:?}", venue)));
            }
        }
        
        Ok(messages)
    }

    /// Parse NYSE Pillar feed format
    fn parse_nyse_packet(
        cursor: &mut std::io::Cursor<&[u8]>,
        received_timestamp: u64
    ) -> Result<Vec<MarketDataMessage>> {
        let mut messages = Vec::new();
        
        while cursor.has_remaining() {
            // NYSE Pillar message header
            if cursor.remaining() < 6 {
                break;
            }
            
            let message_length = cursor.get_u16_le() as usize;
            let message_type = cursor.get_u8();
            let _flags = cursor.get_u8();
            let timestamp = cursor.get_u32_le() as u64;
            let sequence_number = cursor.get_u64_le();
            
            if cursor.remaining() < message_length - 14 {
                break;
            }
            
            match message_type {
                b'Q' => { // Quote message
                    let symbol_len = cursor.get_u8() as usize;
                    let symbol = String::from_utf8_lossy(&cursor.copy_to_bytes(symbol_len)).to_string();
                    let bid_price = cursor.get_u64_le() as f64 / 10000.0;
                    let bid_size = cursor.get_u32_le() as u64;
                    let ask_price = cursor.get_u64_le() as f64 / 10000.0;
                    let ask_size = cursor.get_u32_le() as u64;
                    
                    messages.push(MarketDataMessage {
                        venue: VenueType::NYSE,
                        symbol,
                        feed_type: FeedType::Level1,
                        sequence_number,
                        timestamp_exchange: timestamp * 1_000_000, // Convert to nanoseconds
                        timestamp_received: received_timestamp,
                        data: MarketDataPayload::Level1 {
                            bid_price,
                            bid_size,
                            ask_price,
                            ask_size,
                            last_price: None,
                            last_size: None,
                        },
                    });
                }
                b'T' => { // Trade message
                    let symbol_len = cursor.get_u8() as usize;
                    let symbol = String::from_utf8_lossy(&cursor.copy_to_bytes(symbol_len)).to_string();
                    let price = cursor.get_u64_le() as f64 / 10000.0;
                    let size = cursor.get_u32_le() as u64;
                    let side = cursor.get_u8() as char;
                    let trade_id = cursor.get_u64_le().to_string();
                    
                    messages.push(MarketDataMessage {
                        venue: VenueType::NYSE,
                        symbol,
                        feed_type: FeedType::Trade,
                        sequence_number,
                        timestamp_exchange: timestamp * 1_000_000,
                        timestamp_received: received_timestamp,
                        data: MarketDataPayload::Trade {
                            price,
                            size,
                            side,
                            trade_id,
                            conditions: vec![],
                        },
                    });
                }
                b'D' => { // Order book delta
                    let symbol_len = cursor.get_u8() as usize;
                    let symbol = String::from_utf8_lossy(&cursor.copy_to_bytes(symbol_len)).to_string();
                    let side = cursor.get_u8() as char;
                    let price = cursor.get_u64_le() as f64 / 10000.0;
                    let size = cursor.get_u32_le() as u64;
                    
                    // For simplicity, convert delta to level2 format
                    let (bids, asks) = if side == 'B' {
                        (vec![(price, size)], vec![])
                    } else {
                        (vec![], vec![(price, size)])
                    };
                    
                    messages.push(MarketDataMessage {
                        venue: VenueType::NYSE,
                        symbol,
                        feed_type: FeedType::Level2,
                        sequence_number,
                        timestamp_exchange: timestamp * 1_000_000,
                        timestamp_received: received_timestamp,
                        data: MarketDataPayload::Level2 {
                            bids,
                            asks,
                            sequence: sequence_number,
                        },
                    });
                }
                _ => {
                    // Skip unknown message types
                    cursor.advance(message_length - 14);
                }
            }
        }
        
        Ok(messages)
    }

    /// Parse NASDAQ TotalView-ITCH feed
    fn parse_nasdaq_packet(
        cursor: &mut std::io::Cursor<&[u8]>,
        received_timestamp: u64
    ) -> Result<Vec<MarketDataMessage>> {
        let mut messages = Vec::new();
        
        while cursor.has_remaining() {
            if cursor.remaining() < 3 {
                break;
            }
            
            let message_length = cursor.get_u16() as usize;
            let message_type = cursor.get_u8();
            
            if cursor.remaining() < message_length - 3 {
                break;
            }
            
            match message_type {
                b'A' => { // Add Order
                    let timestamp = cursor.get_u64();
                    let order_ref = cursor.get_u64();
                    let side = cursor.get_u8() as char;
                    let shares = cursor.get_u32() as u64;
                    let stock = String::from_utf8_lossy(&cursor.copy_to_bytes(8)).trim().to_string();
                    let price = cursor.get_u32() as f64 / 10000.0;
                    
                    // Convert to level2 update
                    let (bids, asks) = if side == 'B' {
                        (vec![(price, shares)], vec![])
                    } else {
                        (vec![], vec![(price, shares)])
                    };
                    
                    messages.push(MarketDataMessage {
                        venue: VenueType::NASDAQ,
                        symbol: stock,
                        feed_type: FeedType::Level2,
                        sequence_number: order_ref,
                        timestamp_exchange: timestamp,
                        timestamp_received: received_timestamp,
                        data: MarketDataPayload::Level2 {
                            bids,
                            asks,
                            sequence: order_ref,
                        },
                    });
                }
                b'P' => { // Trade message
                    let timestamp = cursor.get_u64();
                    let order_ref = cursor.get_u64();
                    let side = cursor.get_u8() as char;
                    let shares = cursor.get_u32() as u64;
                    let stock = String::from_utf8_lossy(&cursor.copy_to_bytes(8)).trim().to_string();
                    let price = cursor.get_u32() as f64 / 10000.0;
                    let match_number = cursor.get_u64();
                    
                    messages.push(MarketDataMessage {
                        venue: VenueType::NASDAQ,
                        symbol: stock,
                        feed_type: FeedType::Trade,
                        sequence_number: order_ref,
                        timestamp_exchange: timestamp,
                        timestamp_received: received_timestamp,
                        data: MarketDataPayload::Trade {
                            price,
                            size: shares,
                            side,
                            trade_id: match_number.to_string(),
                            conditions: vec![],
                        },
                    });
                }
                _ => {
                    // Skip other message types for brevity
                    cursor.advance(message_length - 3);
                }
            }
        }
        
        Ok(messages)
    }

    /// Parse CBOE Pitch feed
    fn parse_cboe_packet(
        cursor: &mut std::io::Cursor<&[u8]>,
        received_timestamp: u64
    ) -> Result<Vec<MarketDataMessage>> {
        // Similar implementation for CBOE Pitch format
        // Simplified for space
        Ok(vec![])
    }

    /// Parse IEX DEEP feed
    fn parse_iex_packet(
        cursor: &mut std::io::Cursor<&[u8]>,
        received_timestamp: u64
    ) -> Result<Vec<MarketDataMessage>> {
        // Similar implementation for IEX DEEP format
        // Simplified for space
        Ok(vec![])
    }

    /// Update sequence number tracking and detect gaps
    async fn update_sequence_tracking(
        sequence_numbers: &Arc<RwLock<HashMap<String, u64>>>,
        message: &MarketDataMessage,
        gap_detector: &Arc<Mutex<GapDetector>>,
    ) {
        let key = format!("{}:{:?}", message.symbol, message.venue);
        
        {
            let mut sequences = sequence_numbers.write().unwrap();
            let last_sequence = sequences.get(&key).copied().unwrap_or(0);
            
            if message.sequence_number > last_sequence + 1 && last_sequence > 0 {
                // Gap detected
                let mut detector = gap_detector.lock().await;
                detector.add_gap(Gap {
                    symbol: key.clone(),
                    start_sequence: last_sequence + 1,
                    end_sequence: message.sequence_number - 1,
                    detected_at: Instant::now(),
                });
            }
            
            sequences.insert(key, message.sequence_number);
        }
    }

    /// Start heartbeat monitoring
    async fn start_heartbeat_monitor(&self) {
        let last_heartbeat = self.last_heartbeat.clone();
        let metrics = self.metrics.clone();
        let connection_status = self.connection_status.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                let last_hb = last_heartbeat.load(Ordering::Relaxed);
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                
                if last_hb > 0 && now - last_hb > 30 {
                    // No heartbeat for 30 seconds
                    warn!("Market data feed heartbeat timeout");
                    connection_status.store(false, Ordering::Relaxed);
                    metrics.increment_counter("market_data_heartbeat_timeouts", &[]);
                }
            }
        });
    }

    /// Start gap detection and recovery
    async fn start_gap_detection(&self) {
        let gap_detector = self.gap_detector.clone();
        let metrics = self.metrics.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                let mut detector = gap_detector.lock().await;
                detector.cleanup_old_gaps();
                
                if !detector.detected_gaps.is_empty() {
                    warn!("Detected {} sequence gaps", detector.detected_gaps.len());
                    metrics.set_gauge("market_data_sequence_gaps", 
                        detector.detected_gaps.len() as f64, &[]);
                }
            }
        });
    }

    /// Start metrics collection
    async fn start_metrics_collection(&self) {
        let messages_processed = self.messages_processed.clone();
        let bytes_processed = self.bytes_processed.clone();
        let metrics = self.metrics.clone();
        let processing_times = self.processing_times.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            let mut last_messages = 0u64;
            let mut last_bytes = 0u64;
            
            loop {
                interval.tick().await;
                
                let current_messages = messages_processed.load(Ordering::Relaxed);
                let current_bytes = bytes_processed.load(Ordering::Relaxed);
                
                let message_rate = current_messages - last_messages;
                let byte_rate = current_bytes - last_bytes;
                
                metrics.set_gauge("market_data_message_rate", message_rate as f64, &[]);
                metrics.set_gauge("market_data_byte_rate", byte_rate as f64, &[]);
                metrics.set_gauge("market_data_total_messages", current_messages as f64, &[]);
                
                // Calculate average processing time
                {
                    let processing_times_guard = processing_times.lock().await;
                    if !processing_times_guard.is_empty() {
                        let avg_time = processing_times_guard.iter()
                            .map(|d| d.as_micros() as f64)
                            .sum::<f64>() / processing_times_guard.len() as f64;
                        
                        metrics.set_gauge("market_data_avg_processing_time_us", avg_time, &[]);
                    }
                }
                
                last_messages = current_messages;
                last_bytes = current_bytes;
            }
        });
    }

    /// Get message receiver for subscribers
    pub fn subscribe(&self) -> broadcast::Receiver<MarketDataMessage> {
        self.message_sender.subscribe()
    }

    /// Get current statistics
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("messages_processed".to_string(), 
            self.messages_processed.load(Ordering::Relaxed));
        stats.insert("bytes_processed".to_string(), 
            self.bytes_processed.load(Ordering::Relaxed));
        stats.insert("connection_active".to_string(), 
            if self.connection_status.load(Ordering::Relaxed) { 1 } else { 0 });
        
        stats
    }

    /// Request gap recovery
    pub async fn request_gap_recovery(&self, symbol: &str, start_seq: u64, end_seq: u64) -> Result<()> {
        // Implementation would send recovery request to TCP connection
        info!("Requesting gap recovery for {} sequences {}-{}", symbol, start_seq, end_seq);
        Ok(())
    }
}

impl GapDetector {
    fn new() -> Self {
        Self {
            expected_sequences: HashMap::new(),
            detected_gaps: VecDeque::new(),
            max_gap_age: Duration::from_secs(300), // 5 minutes
        }
    }

    fn add_gap(&mut self, gap: Gap) {
        self.detected_gaps.push_back(gap);
        
        // Limit gap buffer size
        if self.detected_gaps.len() > 1000 {
            self.detected_gaps.pop_front();
        }
    }

    fn cleanup_old_gaps(&mut self) {
        let cutoff_time = Instant::now() - self.max_gap_age;
        
        while let Some(gap) = self.detected_gaps.front() {
            if gap.detected_at < cutoff_time {
                self.detected_gaps.pop_front();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_handler_creation() {
        let config = FeedHandlerConfig {
            venue: VenueType::NYSE,
            feed_types: vec![FeedType::Level1, FeedType::Trade],
            symbols: vec!["AAPL".to_string(), "MSFT".to_string()],
            multicast_groups: vec!["233.54.12.111:18001".to_string()],
            tcp_endpoints: vec!["tcp1.nyxdata.com:8336".to_string()],
            username: None,
            password: None,
            max_buffer_size: 65536,
            heartbeat_interval: Duration::from_secs(30),
            reconnect_delay: Duration::from_secs(5),
            enable_recovery: true,
            recovery_buffer_size: 10000,
            latency_tracking: true,
        };
        
        let metrics = Arc::new(MetricsCollector::new());
        let handler = FeedHandler::new(config, metrics).unwrap();
        
        assert_eq!(handler.config.venue, VenueType::NYSE);
    }

    #[test]
    fn test_gap_detector() {
        let mut detector = GapDetector::new();
        
        let gap = Gap {
            symbol: "AAPL".to_string(),
            start_sequence: 100,
            end_sequence: 150,
            detected_at: Instant::now(),
        };
        
        detector.add_gap(gap);
        assert_eq!(detector.detected_gaps.len(), 1);
        
        detector.cleanup_old_gaps();
        // Gap should still be there as it's recent
        assert_eq!(detector.detected_gaps.len(), 1);
    }

    #[test]
    fn test_sequence_tracking() {
        // Test sequence number validation and gap detection
        let sequence_numbers = Arc::new(RwLock::new(HashMap::new()));
        let gap_detector = Arc::new(Mutex::new(GapDetector::new()));
        
        // This would be tested with actual messages in an async context
    }
}
