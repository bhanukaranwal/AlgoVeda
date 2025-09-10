/*!
 * Technical Feature Engineering for AlgoVeda ML Pipeline
 * 
 * Comprehensive technical indicators and feature extraction
 * for machine learning models in algorithmic trading.
 */

use std::collections::VecDeque;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalFeatures {
    // Price-based features
    pub returns_1d: f64,
    pub returns_5d: f64,
    pub returns_20d: f64,
    pub log_returns: f64,
    
    // Moving averages
    pub sma_5: f64,
    pub sma_20: f64,
    pub sma_50: f64,
    pub ema_12: f64,
    pub ema_26: f64,
    
    // Momentum indicators
    pub rsi_14: f64,
    pub macd: f64,
    pub macd_signal: f64,
    pub macd_histogram: f64,
    pub stochastic_k: f64,
    pub stochastic_d: f64,
    
    // Volatility indicators
    pub bollinger_upper: f64,
    pub bollinger_lower: f64,
    pub bollinger_width: f64,
    pub bollinger_position: f64,
    pub atr_14: f64,
    pub realized_volatility: f64,
    
    // Volume indicators
    pub volume_sma_20: f64,
    pub volume_ratio: f64,
    pub on_balance_volume: f64,
    pub volume_price_trend: f64,
    
    // Support/Resistance
    pub price_to_high_20d: f64,
    pub price_to_low_20d: f64,
    pub resistance_strength: f64,
    pub support_strength: f64,
    
    // Market microstructure
    pub bid_ask_spread: f64,
    pub bid_ask_spread_normalized: f64,
    pub order_imbalance: f64,
    pub trade_intensity: f64,
    
    // Cross-sectional features
    pub relative_strength_vs_market: f64,
    pub correlation_with_market: f64,
    pub beta: f64,
}

pub struct TechnicalFeatureExtractor {
    price_history: VecDeque<PricePoint>,
    volume_history: VecDeque<VolumePoint>,
    lookback_period: usize,
}

#[derive(Debug, Clone)]
struct PricePoint {
    timestamp: DateTime<Utc>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
}

#[derive(Debug, Clone)]
struct VolumePoint {
    timestamp: DateTime<Utc>,
    volume: u64,
    vwap: f64,
}

impl TechnicalFeatureExtractor {
    pub fn new(lookback_period: usize) -> Self {
        Self {
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            lookback_period,
        }
    }

    pub fn add_price_data(&mut self, 
        timestamp: DateTime<Utc>,
        open: f64, high: f64, low: f64, close: f64, volume: u64
    ) {
        let price_point = PricePoint { timestamp, open, high, low, close, volume };
        
        self.price_history.push_back(price_point);
        if self.price_history.len() > self.lookback_period {
            self.price_history.pop_front();
        }
        
        let volume_point = VolumePoint {
            timestamp,
            volume,
            vwap: close, // Simplified - would calculate actual VWAP
        };
        
        self.volume_history.push_back(volume_point);
        if self.volume_history.len() > self.lookback_period {
            self.volume_history.pop_front();
        }
    }

    pub fn extract_features(&self) -> Option<TechnicalFeatures> {
        if self.price_history.len() < 50 {
            return None; // Need sufficient history
        }

        let prices: Vec<f64> = self.price_history.iter().map(|p| p.close).collect();
        let volumes: Vec<u64> = self.price_history.iter().map(|p| p.volume).collect();
        let highs: Vec<f64> = self.price_history.iter().map(|p| p.high).collect();
        let lows: Vec<f64> = self.price_history.iter().map(|p| p.low).collect();

        Some(TechnicalFeatures {
            // Returns
            returns_1d: self.calculate_return(&prices, 1),
            returns_5d: self.calculate_return(&prices, 5),
            returns_20d: self.calculate_return(&prices, 20),
            log_returns: (prices[prices.len()-1] / prices[prices.len()-2]).ln(),
            
            // Moving averages
            sma_5: self.simple_moving_average(&prices, 5),
            sma_20: self.simple_moving_average(&prices, 20),
            sma_50: self.simple_moving_average(&prices, 50),
            ema_12: self.exponential_moving_average(&prices, 12),
            ema_26: self.exponential_moving_average(&prices, 26),
            
            // Momentum
            rsi_14: self.relative_strength_index(&prices, 14),
            macd: self.calculate_macd(&prices).0,
            macd_signal: self.calculate_macd(&prices).1,
            macd_histogram: self.calculate_macd(&prices).2,
            stochastic_k: self.stochastic_oscillator(&prices, &highs, &lows, 14).0,
            stochastic_d: self.stochastic_oscillator(&prices, &highs, &lows, 14).1,
            
            // Volatility
            bollinger_upper: self.bollinger_bands(&prices, 20, 2.0).0,
            bollinger_lower: self.bollinger_bands(&prices, 20, 2.0).1,
            bollinger_width: self.bollinger_bands(&prices, 20, 2.0).2,
            bollinger_position: self.bollinger_position(&prices, 20, 2.0),
            atr_14: self.average_true_range(&prices, &highs, &lows, 14),
            realized_volatility: self.realized_volatility(&prices, 20),
            
            // Volume
            volume_sma_20: self.volume_moving_average(&volumes, 20),
            volume_ratio: volumes[volumes.len()-1] as f64 / self.volume_moving_average(&volumes, 20),
            on_balance_volume: self.on_balance_volume(&prices, &volumes),
            volume_price_trend: self.volume_price_trend(&prices, &volumes),
            
            // Support/Resistance
            price_to_high_20d: prices[prices.len()-1] / self.highest_high(&highs, 20),
            price_to_low_20d: prices[prices.len()-1] / self.lowest_low(&lows, 20),
            resistance_strength: self.calculate_resistance_strength(&prices, &highs),
            support_strength: self.calculate_support_strength(&prices, &lows),
            
            // Microstructure (simplified)
            bid_ask_spread: 0.01, // Would calculate from actual bid/ask data
            bid_ask_spread_normalized: 0.01 / prices[prices.len()-1],
            order_imbalance: 0.0, // Would calculate from order book data
            trade_intensity: volumes[volumes.len()-1] as f64 / 1000.0,
            
            // Cross-sectional (simplified)
            relative_strength_vs_market: 1.0, // Would compare vs market index
            correlation_with_market: 0.7, // Would calculate rolling correlation
            beta: 1.2, // Would calculate from regression
        })
    }

    fn calculate_return(&self, prices: &[f64], periods: usize) -> f64 {
        if prices.len() <= periods {
            return 0.0;
        }
        let current = prices[prices.len() - 1];
        let previous = prices[prices.len() - 1 - periods];
        (current - previous) / previous
    }

    fn simple_moving_average(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }
        let sum: f64 = prices[prices.len()-period..].iter().sum();
        sum / period as f64
    }

    fn exponential_moving_average(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return self.simple_moving_average(prices, prices.len());
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = self.simple_moving_average(&prices[..period], period);
        
        for &price in &prices[period..] {
            ema = (price * multiplier) + (ema * (1.0 - multiplier));
        }
        ema
    }

    fn relative_strength_index(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI
        }

        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..prices.len() {
            let change = prices[i] - prices[i-1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        if gains.len() < period {
            return 50.0;
        }

        let avg_gain: f64 = gains[gains.len()-period..].iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = losses[losses.len()-period..].iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64, f64) {
        let ema12 = self.exponential_moving_average(prices, 12);
        let ema26 = self.exponential_moving_average(prices, 26);
        let macd_line = ema12 - ema26;
        
        // Calculate signal line (9-period EMA of MACD)
        let macd_signal = macd_line; // Simplified - would need MACD history for proper signal
        let histogram = macd_line - macd_signal;
        
        (macd_line, macd_signal, histogram)
    }

    fn stochastic_oscillator(&self, prices: &[f64], highs: &[f64], lows: &[f64], period: usize) -> (f64, f64) {
        if prices.len() < period {
            return (50.0, 50.0);
        }

        let current_price = prices[prices.len()-1];
        let highest_high = self.highest_high(&highs[highs.len()-period..], period);
        let lowest_low = self.lowest_low(&lows[lows.len()-period..], period);

        let k_percent = if highest_high != lowest_low {
            ((current_price - lowest_low) / (highest_high - lowest_low)) * 100.0
        } else {
            50.0
        };

        let d_percent = k_percent; // Simplified - would need %K history for proper %D
        (k_percent, d_percent)
    }

    fn bollinger_bands(&self, prices: &[f64], period: usize, std_dev: f64) -> (f64, f64, f64) {
        let sma = self.simple_moving_average(prices, period);
        let variance = self.calculate_variance(prices, period, sma);
        let std_deviation = variance.sqrt();
        
        let upper_band = sma + (std_deviation * std_dev);
        let lower_band = sma - (std_deviation * std_dev);
        let width = (upper_band - lower_band) / sma;
        
        (upper_band, lower_band, width)
    }

    fn bollinger_position(&self, prices: &[f64], period: usize, std_dev: f64) -> f64 {
        let current_price = prices[prices.len()-1];
        let (upper, lower, _) = self.bollinger_bands(prices, period, std_dev);
        
        if upper != lower {
            (current_price - lower) / (upper - lower)
        } else {
            0.5
        }
    }

    fn calculate_variance(&self, prices: &[f64], period: usize, mean: f64) -> f64 {
        if prices.len() < period {
            return 0.0;
        }
        
        let sum_squared_diff: f64 = prices[prices.len()-period..]
            .iter()
            .map(|&price| (price - mean).powi(2))
            .sum();
        
        sum_squared_diff / period as f64
    }

    fn average_true_range(&self, prices: &[f64], highs: &[f64], lows: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.0;
        }

        let mut true_ranges = Vec::new();
        
        for i in 1..prices.len() {
            let high_low = highs[i] - lows[i];
            let high_prev_close = (highs[i] - prices[i-1]).abs();
            let low_prev_close = (lows[i] - prices[i-1]).abs();
            
            let true_range = high_low.max(high_prev_close).max(low_prev_close);
            true_ranges.push(true_range);
        }

        if true_ranges.len() >= period {
            true_ranges[true_ranges.len()-period..].iter().sum::<f64>() / period as f64
        } else {
            0.0
        }
    }

    fn realized_volatility(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.0;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|window| (window[1] / window[0]).ln())
            .collect();

        if returns.len() >= period {
            let recent_returns = &returns[returns.len()-period..];
            let mean_return = recent_returns.iter().sum::<f64>() / period as f64;
            let variance = recent_returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f64>() / period as f64;
            
            variance.sqrt() * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        }
    }

    fn volume_moving_average(&self, volumes: &[u64], period: usize) -> f64 {
        if volumes.len() < period {
            return volumes.iter().sum::<u64>() as f64 / volumes.len() as f64;
        }
        let sum: u64 = volumes[volumes.len()-period..].iter().sum();
        sum as f64 / period as f64
    }

    fn on_balance_volume(&self, prices: &[f64], volumes: &[u64]) -> f64 {
        let mut obv = 0.0;
        
        for i in 1..prices.len().min(volumes.len()) {
            if prices[i] > prices[i-1] {
                obv += volumes[i] as f64;
            } else if prices[i] < prices[i-1] {
                obv -= volumes[i] as f64;
            }
        }
        
        obv
    }

    fn volume_price_trend(&self, prices: &[f64], volumes: &[u64]) -> f64 {
        let mut vpt = 0.0;
        
        for i in 1..prices.len().min(volumes.len()) {
            if prices[i-1] != 0.0 {
                let price_change_ratio = (prices[i] - prices[i-1]) / prices[i-1];
                vpt += price_change_ratio * volumes[i] as f64;
            }
        }
        
        vpt
    }

    fn highest_high(&self, highs: &[f64], period: usize) -> f64 {
        if highs.is_empty() {
            return 0.0;
        }
        let start = highs.len().saturating_sub(period);
        highs[start..].iter().fold(f64::NEG_INFINITY, |max, &val| max.max(val))
    }

    fn lowest_low(&self, lows: &[f64], period: usize) -> f64 {
        if lows.is_empty() {
            return 0.0;
        }
        let start = lows.len().saturating_sub(period);
        lows[start..].iter().fold(f64::INFINITY, |min, &val| min.min(val))
    }

    fn calculate_resistance_strength(&self, prices: &[f64], highs: &[f64]) -> f64 {
        // Simplified resistance calculation
        // In practice, would identify actual resistance levels
        let recent_high = self.highest_high(highs, 20);
        let current_price = prices[prices.len()-1];
        (recent_high - current_price) / current_price
    }

    fn calculate_support_strength(&self, prices: &[f64], lows: &[f64]) -> f64 {
        // Simplified support calculation
        let recent_low = self.lowest_low(lows, 20);
        let current_price = prices[prices.len()-1];
        (current_price - recent_low) / current_price
    }
}

impl Default for TechnicalFeatures {
    fn default() -> Self {
        Self {
            returns_1d: 0.0,
            returns_5d: 0.0,
            returns_20d: 0.0,
            log_returns: 0.0,
            sma_5: 0.0,
            sma_20: 0.0,
            sma_50: 0.0,
            ema_12: 0.0,
            ema_26: 0.0,
            rsi_14: 50.0,
            macd: 0.0,
            macd_signal: 0.0,
            macd_histogram: 0.0,
            stochastic_k: 50.0,
            stochastic_d: 50.0,
            bollinger_upper: 0.0,
            bollinger_lower: 0.0,
            bollinger_width: 0.0,
            bollinger_position: 0.5,
            atr_14: 0.0,
            realized_volatility: 0.0,
            volume_sma_20: 0.0,
            volume_ratio: 1.0,
            on_balance_volume: 0.0,
            volume_price_trend: 0.0,
            price_to_high_20d: 1.0,
            price_to_low_20d: 1.0,
            resistance_strength: 0.0,
            support_strength: 0.0,
            bid_ask_spread: 0.0,
            bid_ask_spread_normalized: 0.0,
            order_imbalance: 0.0,
            trade_intensity: 0.0,
            relative_strength_vs_market: 1.0,
            correlation_with_market: 0.0,
            beta: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_technical_feature_extractor() {
        let mut extractor = TechnicalFeatureExtractor::new(100);
        
        // Add sample data
        for i in 0..60 {
            let price = 100.0 + (i as f64 * 0.1);
            extractor.add_price_data(
                Utc::now(), 
                price - 0.5, price + 0.5, price - 1.0, price, 
                1000 + i * 10
            );
        }
        
        let features = extractor.extract_features();
        assert!(features.is_some());
        
        let features = features.unwrap();
        assert!(features.sma_20 > 0.0);
        assert!(features.rsi_14 >= 0.0 && features.rsi_14 <= 100.0);
    }

    #[test]
    fn test_moving_averages() {
        let extractor = TechnicalFeatureExtractor::new(100);
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        
        let sma = extractor.simple_moving_average(&prices, 5);
        assert!((sma - 102.0).abs() < 0.001);
        
        let ema = extractor.exponential_moving_average(&prices, 3);
        assert!(ema > 0.0);
    }

    #[test]
    fn test_rsi_calculation() {
        let extractor = TechnicalFeatureExtractor::new(100);
        let prices = vec![100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0];
        
        let rsi = extractor.relative_strength_index(&prices, 6);
        assert!(rsi >= 0.0 && rsi <= 100.0);
    }
}
