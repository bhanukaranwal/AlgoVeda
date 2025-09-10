/*!
 * Ultra-Fast Feed Handler for AlgoVeda Trading Platform
 * 
 * High-throughput, low-latency market data processing with advanced
 * feed management, conflation, and real-time analytics.
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

use crate::trading::{MarketData, TradingResult, Venue};
use crate::config::MarketDataConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: u64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub bid_size: Option<u64>,
    pub ask_size: Option<u64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub open: Option<f64>,
    pub close: Option<f64>,
    pub vwap: Option<f64>,
    pub timestamp: DateTime<Utc>,
    pub sequence_number: u64,
    pub venue: Option<Venue>,
    pub trade_count: Option<u32>,
}

pub trait MarketDataSource: std::fmt::Debug + Send + Sync {
    fn get_latest_data(&self) -> Option<MarketData>;
    fn subscribe_symbol(&self, symbol: String) -> TradingResult<()>;
    fn unsubscribe_symbol(&self, symbol: String) -> TradingResult<()>;
    fn get_subscribed_symbols(&self) -> Vec<String>;
    fn is_connected(&self) -> bool;
    fn get_latency_stats(&self) -> LatencyStats;
}

#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub avg_latency_micros: u64,
    pub min_latency_micros: u64,
    pub max_latency_micros: u64,
    pub p95_latency_micros: u64,
    pub p99_latency_micros: u64,
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    Quote(MarketData),
    Trade(MarketData),
    OrderBookUpdate(MarketData),
    ConnectionStatus(String, bool),
    LatencyAlert(String, u64),
    DataQualityAlert(String, String),
}

/// Ultra-Fast Feed Handler with advanced features
#[derive(Debug)]
pub struct UltraFastFeedHandler {
    config: Arc<MarketDataConfig>,
    sources: Arc<RwLock<HashMap<String, Arc<dyn MarketDataSource>>>>,
    latest_data: Arc<RwLock<HashMap<String, MarketData>>>,
    conflated_data: Arc<RwLock<HashMap<String, VecDeque<MarketData>>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<String>>>>, // source -> symbols
    latency_stats: Arc<RwLock<HashMap<String, LatencyStats>>>,
    data_events_tx: mpsc::UnboundedSender<MarketDataEvent>,
    sequence_numbers: Arc<RwLock<HashMap<String, u64>>>,
    is_running: Arc<RwLock<bool>>,
    messages_processed: Arc<RwLock<u64>>,
}

impl UltraFastFeedHandler {
    pub fn new(
        config: Arc<MarketDataConfig>,
        data_events_tx: mpsc::UnboundedSender<MarketDataEvent>,
    ) -> Self {
        Self {
            config,
            sources: Arc::new(RwLock::new(HashMap::new())),
            latest_data: Arc::new(RwLock::new(HashMap::new())),
            conflated_data: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            latency_stats: Arc::new(RwLock::new(HashMap::new())),
            data_events_tx,
            sequence_numbers: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
            messages_processed: Arc::new(RwLock::new(0)),
        }
    }

    pub fn add_source(&self, name: String, source: Arc<dyn MarketDataSource>) {
        let mut sources = self.sources.write();
        sources.insert(name.clone(), source);
        
        let mut subscriptions = self.subscriptions.write();
        subscriptions.insert(name, Vec::new());
        
        info!("Added market data source: {}", name);
    }

    pub async fn start_feeds(&self) -> TradingResult<()> {
        {
            let mut running = self.is_running.write();
            *running = true;
        }

        info!("Starting ultra-fast feed handler");

        // Start data collection loop
        let handler = self.clone();
        tokio::spawn(async move {
            handler.data_collection_loop().await;
        });

        // Start conflation loop
        let handler = self.clone();
        tokio::spawn(async move {
            handler.conflation_loop().await;
        });

        // Start monitoring loop
        let handler = self.clone();
        tokio::spawn(async move {
            handler.monitoring_loop().await;
        });

        Ok(())
    }

    pub async fn stop_feeds(&self) -> TradingResult<()> {
        {
            let mut running = self.is_running.write();
            *running = false;
        }

        info!("Stopped ultra-fast feed handler");
        Ok(())
    }

    async fn data_collection_loop(&self) {
        let collection_interval = Duration::from_millis(self.config.update_frequency);
        let mut interval = tokio::time::interval(collection_interval);

        while *self.is_running.read() {
            interval.tick().await;
            
            if let Err(e) = self.collect_data_from_sources().await {
                error!("Data collection error: {}", e);
            }
        }
    }

    async fn conflation_loop(&self) {
        let conflation_interval = Duration::from_millis(self.config.update_frequency / 2);
        let mut interval = tokio::time::interval(conflation_interval);

        while *self.is_running.read() {
            interval.tick().await;
            
            if let Err(e) = self.conflate_data().await {
                error!("Data conflation error: {}", e);
            }
        }
    }

    async fn monitoring_loop(&self) {
        let monitoring_interval = Duration::from_secs(1);
        let mut interval = tokio::time::interval(monitoring_interval);

        while *self.is_running.read() {
            interval.tick().await;
            
            self.update_latency_statistics().await;
            self.check_data_quality().await;
            self.monitor_source_health().await;
        }
    }

    async fn collect_data_from_sources(&self) -> TradingResult<()> {
        let sources = self.sources.read();
        let start_time = Instant::now();

        for (source_name, source) in sources.iter() {
            if !source.is_connected() {
                warn!("Source {} is disconnected", source_name);
                continue;
            }

            if let Some(data) = source.get_latest_data() {
                let latency = start_time.elapsed().as_micros() as u64;
                self.process_market_data(source_name, data, latency).await?;
            }
        }

        Ok(())
    }

    #[instrument(skip(self, data))]
    async fn process_market_data(
        &self, 
        source_name: &str, 
        mut data: MarketData, 
        latency_micros: u64
    ) -> TradingResult<()> {
        // Assign sequence number
        {
            let mut seq_numbers = self.sequence_numbers.write();
            let seq = seq_numbers.entry(data.symbol.clone()).or_insert(0);
            *seq += 1;
            data.sequence_number = *seq;
        }

        // Update latest data
        {
            let mut latest = self.latest_data.write();
            latest.insert(data.symbol.clone(), data.clone());
        }

        // Add to conflation buffer
        {
            let mut conflated = self.conflated_data.write();
            let buffer = conflated.entry(data.symbol.clone()).or_insert_with(VecDeque::new);
            buffer.push_back(data.clone());
            
            // Keep buffer size manageable
            while buffer.len() > 100 {
                buffer.pop_front();
            }
        }

        // Update message counter
        {
            let mut messages = self.messages_processed.write();
            *messages += 1;
        }

        // Record latency
        self.record_latency(source_name, latency_micros).await;

        // Send event
        let _ = self.data_events_tx.send(MarketDataEvent::Quote(data));

        Ok(())
    }

    async fn conflate_data(&self) -> TradingResult<()> {
        if !self.config.compression_enabled {
            return Ok(());
        }

        let mut conflated = self.conflated_data.write();
        
        for (symbol, buffer) in conflated.iter_mut() {
            if buffer.len() > 10 {
                // Conflate to most recent data point
                let latest = buffer.back().cloned();
                buffer.clear();
                if let Some(data) = latest {
                    buffer.push_back(data);
                }
            }
        }

        Ok(())
    }

    async fn record_latency(&self, source_name: &str, latency_micros: u64) {
        let mut latency_stats = self.latency_stats.write();
        
        let stats = latency_stats.entry(source_name.to_string())
            .or_insert_with(|| LatencyStats {
                avg_latency_micros: 0,
                min_latency_micros: u64::MAX,
                max_latency_micros: 0,
                p95_latency_micros: 0,
                p99_latency_micros: 0,
                sample_count: 0,
                last_updated: Utc::now(),
            });

        // Update statistics
        stats.min_latency_micros = stats.min_latency_micros.min(latency_micros);
        stats.max_latency_micros = stats.max_latency_micros.max(latency_micros);
        
        // Simple moving average for now - in production would use more sophisticated methods
        stats.avg_latency_micros = if stats.sample_count == 0 {
            latency_micros
        } else {
            (stats.avg_latency_micros * stats.sample_count as u64 + latency_micros) / (stats.sample_count as u64 + 1)
        };
        
        stats.sample_count += 1;
        stats.last_updated = Utc::now();

        // Alert on high latency
        if latency_micros > 1000 { // 1ms threshold
            let _ = self.data_events_tx.send(MarketDataEvent::LatencyAlert(
                source_name.to_string(),
                latency_micros,
            ));
        }
    }

    async fn update_latency_statistics(&self) {
        // Implementation would calculate percentiles and other advanced statistics
    }

    async fn check_data_quality(&self) {
        let latest_data = self.latest_data.read();
        let now = Utc::now();

        for (symbol, data) in latest_data.iter() {
            let age_ms = (now - data.timestamp).num_milliseconds() as u64;
            
            // Alert on stale data
            if age_ms > 5000 { // 5 second threshold
                let _ = self.data_events_tx.send(MarketDataEvent::DataQualityAlert(
                    symbol.clone(),
                    format!("Stale data: {} ms old", age_ms),
                ));
            }

            // Check for reasonable price ranges
            if data.price <= 0.0 || data.price > 1_000_000.0 {
                let _ = self.data_events_tx.send(MarketDataEvent::DataQualityAlert(
                    symbol.clone(),
                    format!("Unreasonable price: {}", data.price),
                ));
            }
        }
    }

    async fn monitor_source_health(&self) {
        let sources = self.sources.read();
        
        for (source_name, source) in sources.iter() {
            let is_connected = source.is_connected();
            let _ = self.data_events_tx.send(MarketDataEvent::ConnectionStatus(
                source_name.clone(),
                is_connected,
            ));
            
            if !is_connected {
                warn!("Market data source {} is disconnected", source_name);
            }
        }
    }

    pub fn subscribe_symbol(&self, source_name: String, symbol: String) -> TradingResult<()> {
        let sources = self.sources.read();
        if let Some(source) = sources.get(&source_name) {
            source.subscribe_symbol(symbol.clone())?;
            
            let mut subscriptions = self.subscriptions.write();
            if let Some(symbols) = subscriptions.get_mut(&source_name) {
                symbols.push(symbol);
            }
            
            info!("Subscribed to {} on source {}", symbol, source_name);
        }
        
        Ok(())
    }

    pub fn get_latest_data(&self, symbol: &str) -> Option<MarketData> {
        let latest_data = self.latest_data.read();
        latest_data.get(symbol).cloned()
    }

    pub fn get_all_latest_data(&self) -> HashMap<String, MarketData> {
        let latest_data = self.latest_data.read();
        latest_data.clone()
    }

    pub fn get_source_latency_stats(&self, source_name: &str) -> Option<LatencyStats> {
        let latency_stats = self.latency_stats.read();
        latency_stats.get(source_name).cloned()
    }

    pub async fn get_messages_processed(&self) -> u64 {
        let messages = self.messages_processed.read();
        *messages
    }

    pub async fn health_check(&self) -> bool {
        let sources = self.sources.read();
        let connected_sources = sources.values()
            .filter(|source| source.is_connected())
            .count();
        
        connected_sources > 0
    }

    pub async fn subscribe_risk_updates(&self, _risk_engine: Arc<dyn Send + Sync>) -> TradingResult<()> {
        // Implementation would connect to risk engine for updates
        Ok(())
    }
}

// Implement Clone to allow spawning async tasks
impl Clone for UltraFastFeedHandler {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            sources: Arc::clone(&self.sources),
            latest_data: Arc::clone(&self.latest_data),
            conflated_data: Arc::clone(&self.conflated_data),
            subscriptions: Arc::clone(&self.subscriptions),
            latency_stats: Arc::clone(&self.latency_stats),
            data_events_tx: self.data_events_tx.clone(),
            sequence_numbers: Arc::clone(&self.sequence_numbers),
            is_running: Arc::clone(&self.is_running),
            messages_processed: Arc::clone(&self.messages_processed),
        }
    }
}

// Mock market data source for testing
#[derive(Debug)]
pub struct MockMarketDataSource {
    symbols: RwLock<Vec<String>>,
    connected: RwLock<bool>,
}

impl MockMarketDataSource {
    pub fn new() -> Self {
        Self {
            symbols: RwLock::new(Vec::new()),
            connected: RwLock::new(true),
        }
    }
}

impl MarketDataSource for MockMarketDataSource {
    fn get_latest_data(&self) -> Option<MarketData> {
        let symbols = self.symbols.read();
        if symbols.is_empty() {
            return None;
        }

        // Generate mock data
        Some(MarketData {
            symbol: symbols[0].clone(),
            price: 100.0 + (rand::random::<f64>() - 0.5) * 2.0,
            volume: 1000,
            bid: Some(99.95),
            ask: Some(100.05),
            bid_size: Some(500),
            ask_size: Some(500),
            high: Some(101.0),
            low: Some(99.0),
            open: Some(100.0),
            close: Some(100.0),
            vwap: Some(100.0),
            timestamp: Utc::now(),
            sequence_number: 0,
            venue: None,
            trade_count: Some(50),
        })
    }

    fn subscribe_symbol(&self, symbol: String) -> TradingResult<()> {
        let mut symbols = self.symbols.write();
        symbols.push(symbol);
        Ok(())
    }

    fn unsubscribe_symbol(&self, symbol: String) -> TradingResult<()> {
        let mut symbols = self.symbols.write();
        symbols.retain(|s| s != &symbol);
        Ok(())
    }

    fn get_subscribed_symbols(&self) -> Vec<String> {
        let symbols = self.symbols.read();
        symbols.clone()
    }

    fn is_connected(&self) -> bool {
        *self.connected.read()
    }

    fn get_latency_stats(&self) -> LatencyStats {
        LatencyStats {
            avg_latency_micros: 50,
            min_latency_micros: 10,
            max_latency_micros: 200,
            p95_latency_micros: 150,
            p99_latency_micros: 180,
            sample_count: 1000,
            last_updated: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_feed_handler_creation() {
        let config = Arc::new(MarketDataConfig {
            primary_provider: "mock".to_string(),
            backup_providers: vec![],
            update_frequency: 1000,
            history_retention: 30,
            enable_level2: false,
            enable_level3: false,
            compression_enabled: true,
        });
        
        let (tx, _rx) = mpsc::unbounded_channel();
        let handler = UltraFastFeedHandler::new(config, tx);
        
        assert!(!*handler.is_running.read());
    }

    #[tokio::test]
    async fn test_mock_data_source() {
        let source = MockMarketDataSource::new();
        assert!(source.is_connected());
        
        let result = source.subscribe_symbol("AAPL".to_string());
        assert!(result.is_ok());
        
        let data = source.get_latest_data();
        assert!(data.is_some());
        
        let market_data = data.unwrap();
        assert_eq!(market_data.symbol, "AAPL");
        assert!(market_data.price > 0.0);
    }
}
