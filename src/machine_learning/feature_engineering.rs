/*!
 * Feature Engineering for AlgoVeda ML Pipeline
 * Advanced feature extraction and preprocessing for trading signals
 */

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use tokio::sync::RwLock;
use ndarray::{Array1, Array2, Axis};
use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};
use tracing::{debug, instrument};

use crate::{
    market_data::{MarketData, Trade, Quote, Level2Book},
    calculations::technical_indicators::TechnicalIndicators,
    utils::statistics::StatisticalFunctions,
    error::{Result, AlgoVedaError},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub lookback_periods: usize,
    pub forward_periods: usize,
    pub min_observations: usize,
    pub normalize_features: bool,
    pub include_technical: bool,
    pub include_microstructure: bool,
    pub include_alternative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub timestamp: u64,
    pub symbol: String,
    pub features: HashMap<String, f64>,
    pub target: Option<f64>,
    pub metadata: FeatureMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub market_regime: String,
    pub volatility_regime: String,
    pub liquidity_score: f64,
    pub news_sentiment: Option<f64>,
}

pub struct FeatureEngineering {
    config: FeatureConfig,
    technical_indicators: TechnicalIndicators,
    price_history: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    volume_history: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    return_history: Arc<RwLock<HashMap<String, VecDeque<f64>>>>,
    microstructure_features: Arc<RwLock<HashMap<String, MicrostructureFeatures>>>,
    feature_cache: Arc<RwLock<HashMap<String, Vec<FeatureVector>>>>,
}

#[derive(Debug, Clone)]
struct MicrostructureFeatures {
    bid_ask_spread: VecDeque<f64>,
    order_flow_imbalance: VecDeque<f64>,
    trade_intensity: VecDeque<f64>,
    price_impact: VecDeque<f64>,
    effective_spread: VecDeque<f64>,
}

impl FeatureEngineering {
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
            technical_indicators: TechnicalIndicators::new(),
            price_history: Arc::new(RwLock::new(HashMap::new())),
            volume_history: Arc::new(RwLock::new(HashMap::new())),
            return_history: Arc::new(RwLock::new(HashMap::new())),
            microstructure_features: Arc::new(RwLock::new(HashMap::new())),
            feature_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[instrument(skip(self, market_data))]
    pub async fn extract_features(
        &self,
        symbol: &str,
        market_data: &MarketData,
    ) -> Result<FeatureVector> {
        let mut features = HashMap::new();

        // Update price and volume history
        self.update_histories(symbol, market_data).await;

        // Extract price-based features
        let price_features = self.extract_price_features(symbol).await?;
        features.extend(price_features);

        // Extract volume-based features
        let volume_features = self.extract_volume_features(symbol).await?;
        features.extend(volume_features);

        // Extract return-based features
        let return_features = self.extract_return_features(symbol).await?;
        features.extend(return_features);

        // Extract technical indicators
        if self.config.include_technical {
            let technical_features = self.extract_technical_features(symbol).await?;
            features.extend(technical_features);
        }

        // Extract microstructure features
        if self.config.include_microstructure {
            let micro_features = self.extract_microstructure_features(symbol).await?;
            features.extend(micro_features);
        }

        // Extract cross-asset features
        let cross_features = self.extract_cross_asset_features(symbol).await?;
        features.extend(cross_features);

        // Extract time-based features
        let time_features = self.extract_time_features(market_data.timestamp);
        features.extend(time_features);

        // Normalize features if configured
        if self.config.normalize_features {
            self.normalize_feature_vector(&mut features, symbol).await;
        }

        // Create feature metadata
        let metadata = self.create_feature_metadata(symbol, market_data).await?;

        let feature_vector = FeatureVector {
            timestamp: market_data.timestamp,
            symbol: symbol.to_string(),
            features,
            target: None, // Will be set during training
            metadata,
        };

        // Cache the feature vector
        self.cache_feature_vector(symbol, feature_vector.clone()).await;

        Ok(feature_vector)
    }

    async fn update_histories(&self, symbol: &str, market_data: &MarketData) {
        let price = market_data.close.to_f64().unwrap_or(0.0);
        let volume = market_data.volume.to_f64().unwrap_or(0.0);

        // Update price history
        {
            let mut price_history = self.price_history.write().await;
            let symbol_prices = price_history.entry(symbol.to_string())
                .or_insert_with(VecDeque::new);
            
            symbol_prices.push_back(price);
            if symbol_prices.len() > self.config.lookback_periods * 2 {
                symbol_prices.pop_front();
            }
        }

        // Update volume history
        {
            let mut volume_history = self.volume_history.write().await;
            let symbol_volumes = volume_history.entry(symbol.to_string())
                .or_insert_with(VecDeque::new);
            
            symbol_volumes.push_back(volume);
            if symbol_volumes.len() > self.config.lookback_periods * 2 {
                symbol_volumes.pop_front();
            }
        }

        // Calculate and update returns
        if let Some(prev_price) = self.get_previous_price(symbol).await {
            if prev_price > 0.0 {
                let return_val = (price - prev_price) / prev_price;
                
                let mut return_history = self.return_history.write().await;
                let symbol_returns = return_history.entry(symbol.to_string())
                    .or_insert_with(VecDeque::new);
                
                symbol_returns.push_back(return_val);
                if symbol_returns.len() > self.config.lookback_periods * 2 {
                    symbol_returns.pop_front();
                }
            }
        }
    }

    async fn extract_price_features(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        let price_history = self.price_history.read().await;
        if let Some(prices) = price_history.get(symbol) {
            if prices.len() < self.config.min_observations {
                return Ok(features);
            }

            let prices_vec: Vec<f64> = prices.iter().cloned().collect();
            let current_price = *prices.back().unwrap();

            // Basic price features
            features.insert("price_current".to_string(), current_price);
            
            // Moving averages
            if prices_vec.len() >= 5 {
                let ma5 = prices_vec.iter().rev().take(5).sum::<f64>() / 5.0;
                features.insert("price_ma5".to_string(), ma5);
                features.insert("price_ma5_ratio".to_string(), current_price / ma5);
            }

            if prices_vec.len() >= 20 {
                let ma20 = prices_vec.iter().rev().take(20).sum::<f64>() / 20.0;
                features.insert("price_ma20".to_string(), ma20);
                features.insert("price_ma20_ratio".to_string(), current_price / ma20);
            }

            // Price percentiles
            let mut sorted_prices = prices_vec.clone();
            sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let p25 = sorted_prices[sorted_prices.len() / 4];
            let p50 = sorted_prices[sorted_prices.len() / 2];
            let p75 = sorted_prices[sorted_prices.len() * 3 / 4];
            
            features.insert("price_percentile_rank".to_string(), 
                self.percentile_rank(&sorted_prices, current_price));
            features.insert("price_z_score".to_string(), 
                self.z_score(&prices_vec, current_price));

            // Price momentum features
            if prices_vec.len() >= 2 {
                let momentum_1 = (current_price - prices_vec[prices_vec.len() - 2]) / prices_vec[prices_vec.len() - 2];
                features.insert("price_momentum_1".to_string(), momentum_1);
            }

            if prices_vec.len() >= 5 {
                let momentum_5 = (current_price - prices_vec[prices_vec.len() - 5]) / prices_vec[prices_vec.len() - 5];
                features.insert("price_momentum_5".to_string(), momentum_5);
            }

            // Price volatility features
            let volatility = self.calculate_volatility(&prices_vec, 20);
            features.insert("price_volatility_20".to_string(), volatility);
        }

        Ok(features)
    }

    async fn extract_volume_features(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        let volume_history = self.volume_history.read().await;
        if let Some(volumes) = volume_history.get(symbol) {
            if volumes.len() < self.config.min_observations {
                return Ok(features);
            }

            let volumes_vec: Vec<f64> = volumes.iter().cloned().collect();
            let current_volume = *volumes.back().unwrap();

            // Basic volume features
            features.insert("volume_current".to_string(), current_volume);

            // Volume moving averages
            if volumes_vec.len() >= 20 {
                let vol_ma20 = volumes_vec.iter().rev().take(20).sum::<f64>() / 20.0;
                features.insert("volume_ma20".to_string(), vol_ma20);
                if vol_ma20 > 0.0 {
                    features.insert("volume_ratio_ma20".to_string(), current_volume / vol_ma20);
                }
            }

            // Volume percentile rank
            let mut sorted_volumes = volumes_vec.clone();
            sorted_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            features.insert("volume_percentile_rank".to_string(), 
                self.percentile_rank(&sorted_volumes, current_volume));

            // Volume momentum
            if volumes_vec.len() >= 2 {
                let vol_momentum = (current_volume - volumes_vec[volumes_vec.len() - 2]) / 
                    (volumes_vec[volumes_vec.len() - 2] + 1.0);
                features.insert("volume_momentum_1".to_string(), vol_momentum);
            }
        }

        Ok(features)
    }

    async fn extract_return_features(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        let return_history = self.return_history.read().await;
        if let Some(returns) = return_history.get(symbol) {
            if returns.len() < self.config.min_observations {
                return Ok(features);
            }

            let returns_vec: Vec<f64> = returns.iter().cloned().collect();

            // Basic return statistics
            let mean_return = returns_vec.iter().sum::<f64>() / returns_vec.len() as f64;
            let return_volatility = self.calculate_volatility(&returns_vec, returns_vec.len());
            
            features.insert("return_mean".to_string(), mean_return);
            features.insert("return_volatility".to_string(), return_volatility);

            if return_volatility > 0.0 {
                features.insert("return_sharpe".to_string(), mean_return / return_volatility);
            }

            // Return skewness and kurtosis
            features.insert("return_skewness".to_string(), self.calculate_skewness(&returns_vec));
            features.insert("return_kurtosis".to_string(), self.calculate_kurtosis(&returns_vec));

            // Recent return features
            if !returns_vec.is_empty() {
                features.insert("return_1".to_string(), *returns_vec.last().unwrap());
            }

            if returns_vec.len() >= 5 {
                let return_5 = returns_vec.iter().rev().take(5).sum::<f64>();
                features.insert("return_5".to_string(), return_5);
            }

            // Autocorrelation features
            if returns_vec.len() >= 10 {
                features.insert("return_autocorr_1".to_string(), 
                    self.calculate_autocorrelation(&returns_vec, 1));
                features.insert("return_autocorr_5".to_string(), 
                    self.calculate_autocorrelation(&returns_vec, 5));
            }
        }

        Ok(features)
    }

    async fn extract_technical_features(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        let price_history = self.price_history.read().await;
        if let Some(prices) = price_history.get(symbol) {
            if prices.len() < 20 {
                return Ok(features);
            }

            let prices_array = Array1::from_vec(prices.iter().cloned().collect());

            // RSI
            if let Ok(rsi) = self.technical_indicators.rsi(&prices_array, 14) {
                if let Some(current_rsi) = rsi.last() {
                    features.insert("rsi_14".to_string(), *current_rsi);
                }
            }

            // MACD
            if let Ok((macd, signal, histogram)) = self.technical_indicators.macd(&prices_array, 12, 26, 9) {
                if let Some(current_macd) = macd.last() {
                    features.insert("macd".to_string(), *current_macd);
                }
                if let Some(current_signal) = signal.last() {
                    features.insert("macd_signal".to_string(), *current_signal);
                }
                if let Some(current_histogram) = histogram.last() {
                    features.insert("macd_histogram".to_string(), *current_histogram);
                }
            }

            // Bollinger Bands
            if let Ok((upper, lower, middle)) = self.technical_indicators.bollinger_bands(&prices_array, 20, 2.0) {
                if let (Some(u), Some(l), Some(m)) = (upper.last(), lower.last(), middle.last()) {
                    let current_price = *prices.back().unwrap();
                    features.insert("bb_upper".to_string(), *u);
                    features.insert("bb_lower".to_string(), *l);
                    features.insert("bb_middle".to_string(), *m);
                    features.insert("bb_position".to_string(), (current_price - l) / (u - l));
                    features.insert("bb_width".to_string(), (u - l) / m);
                }
            }

            // Stochastic oscillator
            // Would need high/low data for proper implementation
            // Simplified version using close prices
            let stoch_k = self.calculate_stochastic_k(&prices.iter().cloned().collect::<Vec<_>>(), 14);
            features.insert("stoch_k".to_string(), stoch_k);
        }

        Ok(features)
    }

    async fn extract_microstructure_features(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        let micro_features = self.microstructure_features.read().await;
        if let Some(micro) = micro_features.get(symbol) {
            // Bid-ask spread features
            if !micro.bid_ask_spread.is_empty() {
                let current_spread = *micro.bid_ask_spread.back().unwrap();
                features.insert("bid_ask_spread".to_string(), current_spread);
                
                if micro.bid_ask_spread.len() >= 20 {
                    let spread_ma = micro.bid_ask_spread.iter().rev().take(20).sum::<f64>() / 20.0;
                    features.insert("spread_ma20".to_string(), spread_ma);
                    if spread_ma > 0.0 {
                        features.insert("spread_ratio_ma20".to_string(), current_spread / spread_ma);
                    }
                }
            }

            // Order flow imbalance
            if !micro.order_flow_imbalance.is_empty() {
                features.insert("order_flow_imbalance".to_string(), 
                    *micro.order_flow_imbalance.back().unwrap());
            }

            // Trade intensity
            if !micro.trade_intensity.is_empty() {
                features.insert("trade_intensity".to_string(), 
                    *micro.trade_intensity.back().unwrap());
            }
        }

        Ok(features)
    }

    async fn extract_cross_asset_features(&self, symbol: &str) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // Correlation with market indices
        // This would require market index data
        // Simplified implementation
        features.insert("market_correlation".to_string(), 0.5); // Placeholder
        
        // Sector relative performance
        features.insert("sector_relative_performance".to_string(), 0.0); // Placeholder
        
        Ok(features)
    }

    fn extract_time_features(&self, timestamp: u64) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        use chrono::{DateTime, Utc, Datelike, Timelike};
        
        let dt = DateTime::<Utc>::from_timestamp(timestamp as i64, 0).unwrap();
        
        // Time of day features
        features.insert("hour_of_day".to_string(), dt.hour() as f64);
        features.insert("minute_of_hour".to_string(), dt.minute() as f64);
        
        // Day of week (0 = Sunday)
        features.insert("day_of_week".to_string(), dt.weekday().num_days_from_sunday() as f64);
        
        // Day of month
        features.insert("day_of_month".to_string(), dt.day() as f64);
        
        // Month of year
        features.insert("month_of_year".to_string(), dt.month() as f64);
        
        // Market session indicators
        let hour = dt.hour();
        features.insert("is_market_open".to_string(), 
            if hour >= 9 && hour < 16 { 1.0 } else { 0.0 });
        features.insert("is_opening_hour".to_string(), 
            if hour == 9 { 1.0 } else { 0.0 });
        features.insert("is_closing_hour".to_string(), 
            if hour == 15 { 1.0 } else { 0.0 });
        
        features
    }

    async fn normalize_feature_vector(&self, features: &mut HashMap<String, f64>, symbol: &str) {
        // Z-score normalization for each feature
        // This would require maintaining running statistics for each feature
        // Simplified implementation
        for (key, value) in features.iter_mut() {
            if key.contains("ratio") || key.contains("momentum") {
                // Already normalized features
                continue;
            }
            
            // Apply simple scaling
            *value = *value / 100.0; // Simple scaling
        }
    }

    async fn create_feature_metadata(&self, symbol: &str, market_data: &MarketData) -> Result<FeatureMetadata> {
        // Determine market regime
        let market_regime = self.determine_market_regime(symbol).await;
        
        // Determine volatility regime
        let volatility_regime = self.determine_volatility_regime(symbol).await;
        
        // Calculate liquidity score
        let liquidity_score = self.calculate_liquidity_score(symbol).await;
        
        Ok(FeatureMetadata {
            market_regime,
            volatility_regime,
            liquidity_score,
            news_sentiment: None, // Would integrate with news sentiment service
        })
    }

    async fn cache_feature_vector(&self, symbol: &str, feature_vector: FeatureVector) {
        let mut cache = self.feature_cache.write().await;
        let symbol_cache = cache.entry(symbol.to_string()).or_insert_with(Vec::new);
        
        symbol_cache.push(feature_vector);
        
        // Keep only recent features
        if symbol_cache.len() > 1000 {
            symbol_cache.drain(0..500);
        }
    }

    // Helper methods
    async fn get_previous_price(&self, symbol: &str) -> Option<f64> {
        let price_history = self.price_history.read().await;
        price_history.get(symbol)
            .and_then(|prices| prices.iter().rev().nth(1).copied())
    }

    fn percentile_rank(&self, sorted_values: &[f64], value: f64) -> f64 {
        let mut count = 0;
        for &v in sorted_values {
            if v <= value {
                count += 1;
            } else {
                break;
            }
        }
        count as f64 / sorted_values.len() as f64
    }

    fn z_score(&self, values: &[f64], value: f64) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            (value - mean) / std_dev
        } else {
            0.0
        }
    }

    fn calculate_volatility(&self, values: &[f64], period: usize) -> f64 {
        if values.len() < period || period < 2 {
            return 0.0;
        }

        let recent_values: Vec<f64> = values.iter().rev().take(period).cloned().collect();
        let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let variance = recent_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (recent_values.len() - 1) as f64;
        
        variance.sqrt()
    }

    fn calculate_skewness(&self, values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let skewness = values.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / values.len() as f64;

        skewness
    }

    fn calculate_kurtosis(&self, values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let kurtosis = values.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / values.len() as f64;

        kurtosis - 3.0 // Excess kurtosis
    }

    fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> f64 {
        if values.len() <= lag {
            return 0.0;
        }

        let n = values.len() - lag;
        let x1: Vec<f64> = values[..n].to_vec();
        let x2: Vec<f64> = values[lag..].to_vec();

        let mean1 = x1.iter().sum::<f64>() / x1.len() as f64;
        let mean2 = x2.iter().sum::<f64>() / x2.len() as f64;

        let numerator: f64 = x1.iter().zip(x2.iter())
            .map(|(a, b)| (a - mean1) * (b - mean2))
            .sum();

        let var1: f64 = x1.iter().map(|x| (x - mean1).powi(2)).sum();
        let var2: f64 = x2.iter().map(|x| (x - mean2).powi(2)).sum();

        let denominator = (var1 * var2).sqrt();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn calculate_stochastic_k(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return 50.0; // Neutral value
        }

        let recent_prices: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
        let current_price = prices.last().unwrap();
        let lowest = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let highest = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if highest > lowest {
            ((current_price - lowest) / (highest - lowest)) * 100.0
        } else {
            50.0
        }
    }

    async fn determine_market_regime(&self, symbol: &str) -> String {
        // Simplified market regime detection
        let return_history = self.return_history.read().await;
        if let Some(returns) = return_history.get(symbol) {
            if returns.len() >= 20 {
                let recent_returns: Vec<f64> = returns.iter().rev().take(20).cloned().collect();
                let mean_return = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
                let volatility = self.calculate_volatility(&recent_returns, recent_returns.len());

                if mean_return > 0.001 && volatility < 0.02 {
                    "Bull_Low_Vol".to_string()
                } else if mean_return > 0.001 && volatility >= 0.02 {
                    "Bull_High_Vol".to_string()
                } else if mean_return <= -0.001 && volatility < 0.02 {
                    "Bear_Low_Vol".to_string()
                } else if mean_return <= -0.001 && volatility >= 0.02 {
                    "Bear_High_Vol".to_string()
                } else {
                    "Sideways".to_string()
                }
            } else {
                "Unknown".to_string()
            }
        } else {
            "Unknown".to_string()
        }
    }

    async fn determine_volatility_regime(&self, symbol: &str) -> String {
        let return_history = self.return_history.read().await;
        if let Some(returns) = return_history.get(symbol) {
            if returns.len() >= 20 {
                let volatility = self.calculate_volatility(
                    &returns.iter().cloned().collect::<Vec<_>>(), 
                    20
                );

                if volatility < 0.01 {
                    "Low".to_string()
                } else if volatility < 0.03 {
                    "Medium".to_string()
                } else {
                    "High".to_string()
                }
            } else {
                "Unknown".to_string()
            }
        } else {
            "Unknown".to_string()
        }
    }

    async fn calculate_liquidity_score(&self, symbol: &str) -> f64 {
        let volume_history = self.volume_history.read().await;
        if let Some(volumes) = volume_history.get(symbol) {
            if volumes.len() >= 20 {
                let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
                let current_volume = *volumes.back().unwrap();
                
                // Simple liquidity score based on volume
                if avg_volume > 0.0 {
                    (current_volume / avg_volume).min(2.0) / 2.0
                } else {
                    0.5
                }
            } else {
                0.5
            }
        } else {
            0.5
        }
    }

    pub async fn get_feature_importance(&self) -> HashMap<String, f64> {
        // Return feature importance scores
        // This would be calculated based on model training results
        let mut importance = HashMap::new();
        
        importance.insert("return_1".to_string(), 0.15);
        importance.insert("rsi_14".to_string(), 0.12);
        importance.insert("volume_ratio_ma20".to_string(), 0.10);
        importance.insert("price_momentum_5".to_string(), 0.09);
        importance.insert("macd_histogram".to_string(), 0.08);
        importance.insert("bb_position".to_string(), 0.07);
        importance.insert("return_volatility".to_string(), 0.06);
        importance.insert("bid_ask_spread".to_string(), 0.05);
        
        importance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feature_engineering_creation() {
        let config = FeatureConfig {
            lookback_periods: 100,
            forward_periods: 5,
            min_observations: 20,
            normalize_features: true,
            include_technical: true,
            include_microstructure: true,
            include_alternative: false,
        };

        let fe = FeatureEngineering::new(config);
        assert!(fe.price_history.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_time_features() {
        let config = FeatureConfig {
            lookback_periods: 100,
            forward_periods: 5,
            min_observations: 20,
            normalize_features: false,
            include_technical: false,
            include_microstructure: false,
            include_alternative: false,
        };

        let fe = FeatureEngineering::new(config);
        let timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
        let time_features = fe.extract_time_features(timestamp);
        
        assert!(time_features.contains_key("hour_of_day"));
        assert!(time_features.contains_key("day_of_week"));
        assert!(time_features.contains_key("month_of_year"));
    }
}
