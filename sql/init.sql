-- AlgoVeda Trading Platform Database Schema
-- PostgreSQL initialization script

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Orders table
CREATE TABLE trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(18,8),
    stop_price DECIMAL(18,8),
    time_in_force VARCHAR(10) NOT NULL DEFAULT 'DAY',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    filled_quantity DECIMAL(18,8) DEFAULT 0,
    average_fill_price DECIMAL(18,8) DEFAULT 0,
    remaining_quantity DECIMAL(18,8) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    account_id VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(50),
    venue VARCHAR(20),
    execution_instructions JSONB,
    tags JSONB
);

-- Create hypertable for orders
SELECT create_hypertable('trading.orders', 'created_at');

-- Trades table
CREATE TABLE trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(18,8) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    venue VARCHAR(20) NOT NULL,
    execution_id VARCHAR(50) NOT NULL,
    commission DECIMAL(18,8) DEFAULT 0,
    fees DECIMAL(18,8) DEFAULT 0,
    counterparty VARCHAR(50),
    settlement_date DATE
);

SELECT create_hypertable('trading.trades', 'executed_at');

-- Positions table
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    average_cost DECIMAL(18,8) NOT NULL,
    market_value DECIMAL(18,8) NOT NULL,
    unrealized_pnl DECIMAL(18,8) NOT NULL,
    realized_pnl DECIMAL(18,8) NOT NULL DEFAULT 0,
    first_trade_date TIMESTAMPTZ,
    last_trade_date TIMESTAMPTZ,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, symbol)
);

-- Market data tables
CREATE TABLE market_data.quotes (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(18,8),
    ask DECIMAL(18,8),
    bid_size BIGINT,
    ask_size BIGINT,
    last_price DECIMAL(18,8),
    volume BIGINT,
    vwap DECIMAL(18,8),
    venue VARCHAR(20)
);

SELECT create_hypertable('market_data.quotes', 'timestamp');

-- Risk metrics table
CREATE TABLE risk.portfolio_metrics (
    account_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(18,8) NOT NULL,
    total_pnl DECIMAL(18,8) NOT NULL,
    daily_pnl DECIMAL(18,8) NOT NULL,
    var_95 DECIMAL(18,8),
    max_drawdown DECIMAL(18,8),
    leverage DECIMAL(10,4),
    gross_exposure DECIMAL(18,8),
    net_exposure DECIMAL(18,8),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6)
);

SELECT create_hypertable('risk.portfolio_metrics', 'timestamp');

-- Performance analytics
CREATE TABLE analytics.daily_performance (
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    starting_value DECIMAL(18,8) NOT NULL,
    ending_value DECIMAL(18,8) NOT NULL,
    pnl DECIMAL(18,8) NOT NULL,
    return_pct DECIMAL(10,6) NOT NULL,
    trades_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    PRIMARY KEY (account_id, date)
);

-- Audit logging
CREATE TABLE audit.system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    action VARCHAR(50) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT
);

SELECT create_hypertable('audit.system_events', 'timestamp');

-- Indexes for performance
CREATE INDEX idx_orders_symbol_status ON trading.orders(symbol, status);
CREATE INDEX idx_orders_account_created ON trading.orders(account_id, created_at DESC);
CREATE INDEX idx_trades_symbol_time ON trading.trades(symbol, executed_at DESC);
CREATE INDEX idx_positions_account ON trading.positions(account_id);
CREATE INDEX idx_quotes_symbol_time ON market_data.quotes(symbol, timestamp DESC);

-- Views for common queries
CREATE VIEW trading.open_orders AS
SELECT * FROM trading.orders 
WHERE status IN ('NEW', 'PARTIALLY_FILLED', 'PENDING');

CREATE VIEW analytics.position_summary AS
SELECT 
    account_id,
    COUNT(*) as position_count,
    SUM(market_value) as total_market_value,
    SUM(unrealized_pnl) as total_unrealized_pnl,
    SUM(realized_pnl) as total_realized_pnl
FROM trading.positions 
WHERE quantity != 0 
GROUP BY account_id;

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_orders_updated_at 
BEFORE UPDATE ON trading.orders 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at 
BEFORE UPDATE ON trading.positions 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Data retention policies
SELECT add_retention_policy('trading.orders', INTERVAL '2 years');
SELECT add_retention_policy('trading.trades', INTERVAL '7 years');
SELECT add_retention_policy('market_data.quotes', INTERVAL '1 year');
SELECT add_retention_policy('risk.portfolio_metrics', INTERVAL '3 years');
SELECT add_retention_policy('audit.system_events', INTERVAL '5 years');
