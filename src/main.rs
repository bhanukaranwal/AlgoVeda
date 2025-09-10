/*!
 * AlgoVeda Core Engine - Main Application Entry Point
 * Ultra-low latency algorithmic trading platform
 */

use algoveda_core::{
    config::{AppConfig, TradingConfig, RiskConfig},
    trading::{OrderManager, ExecutionEngine, SmartOrderRouter},
    market_data::UltraFastFeedHandler,
    risk_engine::RiskEngine,
    networking::NetworkManager,
    storage::StorageManager,
    monitoring::MetricsCollector,
    security::SecurityManager,
};

use tokio::{
    signal,
    sync::{broadcast, mpsc},
    time::{Duration, interval},
};
use tracing::{info, error, warn};
use std::{
    sync::Arc,
    process,
    time::Instant,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🚀 AlgoVeda Core Engine Starting...");

    // Load configuration
    let app_config = AppConfig::load().await?;
    let trading_config = TradingConfig::load().await?;
    let risk_config = RiskConfig::load().await?;

    info!("✅ Configuration loaded successfully");

    // Create shutdown channel
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel(1);

    // Initialize core components
    let storage_manager = Arc::new(StorageManager::new(&app_config.storage).await?);
    let network_manager = Arc::new(NetworkManager::new(&app_config.network).await?);
    let security_manager = Arc::new(SecurityManager::new(&app_config.security).await?);
    let metrics_collector = Arc::new(MetricsCollector::new(&app_config.monitoring).await?);

    info!("✅ Core infrastructure initialized");

    // Initialize trading components
    let order_manager = Arc::new(OrderManager::new(
        &trading_config,
        storage_manager.clone(),
        metrics_collector.clone(),
    ).await?);

    let execution_engine = Arc::new(ExecutionEngine::new(
        &trading_config,
        order_manager.clone(),
        network_manager.clone(),
        metrics_collector.clone(),
    ).await?);

    let smart_router = Arc::new(SmartOrderRouter::new(
        &trading_config,
        execution_engine.clone(),
        metrics_collector.clone(),
    ).await?);

    let risk_engine = Arc::new(RiskEngine::new(
        &risk_config,
        order_manager.clone(),
        storage_manager.clone(),
        metrics_collector.clone(),
    ).await?);

    info!("✅ Trading engines initialized");

    // Initialize market data handler
    let feed_handler = Arc::new(UltraFastFeedHandler::new(
        &app_config.market_data,
        network_manager.clone(),
        storage_manager.clone(),
        metrics_collector.clone(),
    ).await?);

    info!("✅ Market data handler initialized");

    // Start all services
    let mut handles = vec![];

    // Start order manager
    let om_handle = tokio::spawn({
        let order_manager = order_manager.clone();
        let mut shutdown_rx = shutdown_tx.subscribe();
        async move {
            tokio::select! {
                result = order_manager.run() => {
                    if let Err(e) = result {
                        error!("Order manager error: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Order manager shutting down...");
                }
            }
        }
    });
    handles.push(om_handle);

    // Start execution engine
    let ee_handle = tokio::spawn({
        let execution_engine = execution_engine.clone();
        let mut shutdown_rx = shutdown_tx.subscribe();
        async move {
            tokio::select! {
                result = execution_engine.run() => {
                    if let Err(e) = result {
                        error!("Execution engine error: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Execution engine shutting down...");
                }
            }
        }
    });
    handles.push(ee_handle);

    // Start risk engine
    let re_handle = tokio::spawn({
        let risk_engine = risk_engine.clone();
        let mut shutdown_rx = shutdown_tx.subscribe();
        async move {
            tokio::select! {
                result = risk_engine.run() => {
                    if let Err(e) = result {
                        error!("Risk engine error: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Risk engine shutting down...");
                }
            }
        }
    });
    handles.push(re_handle);

    // Start market data feed
    let md_handle = tokio::spawn({
        let feed_handler = feed_handler.clone();
        let mut shutdown_rx = shutdown_tx.subscribe();
        async move {
            tokio::select! {
                result = feed_handler.run() => {
                    if let Err(e) = result {
                        error!("Feed handler error: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Feed handler shutting down...");
                }
            }
        }
    });
    handles.push(md_handle);

    // Start metrics collection
    let metrics_handle = tokio::spawn({
        let metrics_collector = metrics_collector.clone();
        let mut shutdown_rx = shutdown_tx.subscribe();
        async move {
            let mut interval = interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = metrics_collector.collect().await {
                            error!("Metrics collection error: {}", e);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Metrics collector shutting down...");
                        break;
                    }
                }
            }
        }
    });
    handles.push(metrics_handle);

    info!("🎯 AlgoVeda Core Engine fully operational");

    // Wait for shutdown signal
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("Received Ctrl+C, initiating graceful shutdown...");
        }
        _ = shutdown_rx.recv() => {
            info!("Received shutdown signal");
        }
    }

    // Initiate graceful shutdown
    info!("🛑 Shutting down AlgoVeda Core Engine...");
    
    if let Err(e) = shutdown_tx.send(()) {
        warn!("Error sending shutdown signal: {}", e);
    }

    // Wait for all services to shut down
    for handle in handles {
        if let Err(e) = handle.await {
            error!("Error waiting for service shutdown: {}", e);
        }
    }

    info!("✅ AlgoVeda Core Engine shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_main_initialization() {
        // Test configuration loading
        assert!(AppConfig::load().await.is_ok());
    }
}
