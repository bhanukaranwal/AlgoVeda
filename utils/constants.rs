/*!
 * Constants and Configuration Values for AlgoVeda Trading Platform
 */

// Trading Constants
pub const DEFAULT_COMMISSION: f64 = 0.001; // 0.1%
pub const DEFAULT_SLIPPAGE: f64 = 0.0005; // 0.05%
pub const MIN_ORDER_SIZE: f64 = 1.0;
pub const MAX_ORDER_SIZE: f64 = 1_000_000.0;
pub const MAX_POSITION_SIZE: f64 = 5_000_000.0;

// Market Data Constants
pub const MARKET_DATA_UPDATE_FREQUENCY_MS: u64 = 1000;
pub const MAX_MARKET_DATA_AGE_MS: u64 = 5000;
pub const PRICE_PRECISION: i32 = 4;
pub const QUANTITY_PRECISION: i32 = 2;

// Risk Management Constants
pub const DEFAULT_VAR_CONFIDENCE: f64 = 0.95;
pub const DEFAULT_VAR_HORIZON_DAYS: u32 = 1;
pub const MAX_DAILY_LOSS: f64 = 50_000.0;
pub const MAX_DRAWDOWN: f64 = 0.15; // 15%
pub const RISK_CHECK_INTERVAL_MS: u64 = 1000;

// Options Constants
pub const MIN_TIME_TO_EXPIRY: f64 = 0.0027; // 1 day in years
pub const MAX_TIME_TO_EXPIRY: f64 = 2.0; // 2 years
pub const MIN_VOLATILITY: f64 = 0.01; // 1%
pub const MAX_VOLATILITY: f64 = 3.0; // 300%
pub const RISK_FREE_RATE: f64 = 0.05; // 5%

// System Constants
pub const MAX_CONCURRENT_ORDERS: u32 = 10_000;
pub const MAX_SYMBOLS: u32 = 1_000;
pub const MAX_STRATEGIES: u32 = 100;
pub const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 30_000;

// Venue Constants
pub const VENUE_NSE: &str = "NSE";
pub const VENUE_BSE: &str = "BSE";
pub const VENUE_MCX: &str = "MCX";
pub const VENUE_NCDEX: &str = "NCDEX";

// Currency Constants
pub const CURRENCY_INR: &str = "INR";
pub const CURRENCY_USD: &str = "USD";
pub const CURRENCY_EUR: &str = "EUR";

// Time Constants
pub const SECONDS_IN_DAY: u64 = 86_400;
pub const MILLISECONDS_IN_SECOND: u64 = 1_000;
pub const MICROSECONDS_IN_MILLISECOND: u64 = 1_000;
pub const NANOSECONDS_IN_MICROSECOND: u64 = 1_000;
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;
