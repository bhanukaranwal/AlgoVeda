/*!
 * Distributed Cache System for AlgoVeda Trading Platform
 * Redis cluster integration with failover and sub-millisecond access
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use redis::{Commands, Connection, RedisResult, Client, cluster::ClusterClient};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{info, warn, error};

pub struct DistributedCache {
    cluster_client: ClusterClient,
    local_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    config: CacheConfig,
    metrics: CacheMetrics,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub cluster_urls: Vec<String>,
    pub local_cache_size: usize,
    pub default_ttl: Duration,
    pub max_retries: u32,
    pub connection_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
}

#[derive(Debug)]
struct CacheEntry {
    data: Vec<u8>,
    expires_at: Instant,
    access_count: u64,
}

impl DistributedCache {
    pub async fn new(config: CacheConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let cluster_client = ClusterClient::new(config.cluster_urls.clone())?;
        
        Ok(Self {
            cluster_client,
            local_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: CacheMetrics::new(),
        })
    }

    pub async fn get<T>(&self, key: &str) -> Result<Option<T>, Box<dyn std::error::Error>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let start = Instant::now();
        
        // Try local cache first (L1)
        if let Some(value) = self.get_from_local_cache(key) {
            self.metrics.record_hit(CacheLevel::Local, start.elapsed());
            return Ok(Some(bincode::deserialize(&value)?));
        }
        
        // Try Redis cluster (L2)
        match self.get_from_redis(key).await {
            Ok(Some(data)) => {
                // Store in local cache for next time
                self.store_in_local_cache(key, &data);
                self.metrics.record_hit(CacheLevel::Distributed, start.elapsed());
                Ok(Some(bincode::deserialize(&data)?))
            }
            Ok(None) => {
                self.metrics.record_miss(start.elapsed());
                Ok(None)
            }
            Err(e) => {
                self.metrics.record_error(start.elapsed());
                Err(e)
            }
        }
    }

    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<Duration>) -> Result<(), Box<dyn std::error::Error>>
    where
        T: Serialize,
    {
        let data = bincode::serialize(value)?;
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        
        // Store in both caches
        self.store_in_local_cache(key, &data);
        self.store_in_redis(key, &data, ttl).await?;
        
        Ok(())
    }

    fn get_from_local_cache(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.local_cache.write();
        
        if let Some(entry) = cache.get_mut(key) {
            if entry.expires_at > Instant::now() {
                entry.access_count += 1;
                return Some(entry.data.clone());
            } else {
                cache.remove(key);
            }
        }
        None
    }
}
