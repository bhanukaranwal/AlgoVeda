-- AlgoVeda Trading Platform Database Schema
-- PostgreSQL 15+ with optimizations for high-frequency trading

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO trading, market_data, analytics, audit, public;

-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- Trading Accounts
CREATE TABLE trading.accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    account_name VARCHAR(100) NOT NULL,
    account_type VARCHAR(50) NOT NULL, -- 'paper', 'live', 'demo'
    broker VARCHAR(50) NOT NULL DEFAULT 'dhan',
    broker_account_id VARCHAR(100),
    initial_capital DECIMAL(20,4) NOT NULL,
    current_equity DECIMAL(20,4) NOT NULL DEFAULT 0,
    available_cash DECIMAL(20,4) NOT NULL DEFAULT 0,
    buying_power DECIMAL(20,4) NOT NULL DEFAULT 0,
    margin_used DECIMAL(20,4) NOT NULL DEFAULT 0,
    day_trading_buying_power DECIMAL(20,4) NOT NULL DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    risk_profile JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Strategies
CREATE TABLE trading.strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(100) NOT NULL,
    strategy_class VARCHAR(200) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    risk_limits JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'inactive', -- 'active', 'inactive', 'paused', 'error'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE,
    performance_summary JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Instruments
CREATE TABLE market_data.instruments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    instrument_type VARCHAR(50) NOT NULL, -- 'stock', 'option', 'future', 'forex'
    underlying_symbol VARCHAR(50),
    expiry_date DATE,
    strike_price DECIMAL(20,4),
    option_type VARCHAR(10), -- 'call', 'put'
    contract_size INTEGER DEFAULT 1,
    tick_size DECIMAL(10,8),
    multiplier DECIMAL(10,4) DEFAULT 1,
    currency VARCHAR(10) DEFAULT 'USD',
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE(symbol, exchange)
);

-- Orders
CREATE TABLE trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading.accounts(id),
    strategy_id UUID REFERENCES trading.strategies(id),
    instrument_id UUID NOT NULL REFERENCES market_data.instruments(id),
    client_order_id VARCHAR(100),
    broker_order_id VARCHAR(100),
    parent_order_id UUID REFERENCES trading.orders(id),
    order_type VARCHAR(50) NOT NULL, -- 'market', 'limit', 'stop', etc.
    side VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    quantity DECIMAL(20,4) NOT NULL,
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    time_in_force VARCHAR(20) DEFAULT 'day',
    status VARCHAR(50) NOT NULL DEFAULT 'pending_new',
    filled_quantity DECIMAL(20,4) DEFAULT 0,
    remaining_quantity DECIMAL(20,4),
    average_fill_price DECIMAL(20,8),
    commission DECIMAL(20,4) DEFAULT 0,
    fees DECIMAL(20,4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    rejected_at TIMESTAMP WITH TIME ZONE,
    rejection_reason TEXT,
    execution_instructions JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Executions/Trades
CREATE TABLE trading.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    account_id UUID NOT NULL REFERENCES trading.accounts(id),
    instrument_id UUID NOT NULL REFERENCES market_data.instruments(id),
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,4) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    commission DECIMAL(20,4) DEFAULT 0,
    fees DECIMAL(20,4) DEFAULT 0,
    venue VARCHAR(100),
    liquidity_flag VARCHAR(10), -- 'add', 'remove'
    contra_broker VARCHAR(100),
    settlement_date DATE,
    metadata JSONB DEFAULT '{}'
);

-- Positions
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading.accounts(id),
    instrument_id UUID NOT NULL REFERENCES market_data.instruments(id),
    strategy_id UUID REFERENCES trading.strategies(id),
    quantity DECIMAL(20,4) NOT NULL DEFAULT 0,
    average_price DECIMAL(20,8),
    market_value DECIMAL(20,4),
    unrealized_pnl DECIMAL(20,4) DEFAULT 0,
    realized_pnl DECIMAL(20,4) DEFAULT 0,
    day_pnl DECIMAL(20,4) DEFAULT 0,
    cost_basis DECIMAL(20,4),
    opened_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    max_position_size DECIMAL(20,4) DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    UNIQUE(account_id, instrument_id, strategy_id)
);

-- Market Data - Ticks
CREATE TABLE market_data.ticks (
    id BIGSERIAL PRIMARY KEY,
    instrument_id UUID NOT NULL REFERENCES market_data.instruments(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    volume BIGINT DEFAULT 0,
    bid_price DECIMAL(20,8),
    ask_price DECIMAL(20,8),
    bid_size BIGINT,
    ask_size BIGINT,
    trade_conditions JSONB DEFAULT '{}',
    sequence_number BIGINT,
    exchange_timestamp TIMESTAMP WITH TIME ZONE
);

-- Market Data - OHLCV Bars
CREATE TABLE market_data.bars (
    id BIGSERIAL PRIMARY KEY,
    instrument_id UUID NOT NULL REFERENCES market_data.instruments(id),
    timeframe VARCHAR(20) NOT NULL, -- '1min', '5min', '1hour', '1day', etc.
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume BIGINT DEFAULT 0,
    vwap DECIMAL(20,8),
    number_of_trades INTEGER DEFAULT 0,
    UNIQUE(instrument_id, timeframe, timestamp)
);

-- Performance Analytics
CREATE TABLE analytics.performance_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID REFERENCES trading.accounts(id),
    strategy_id UUID REFERENCES trading.strategies(id),
    snapshot_date DATE NOT NULL,
    equity DECIMAL(20,4) NOT NULL,
    pnl DECIMAL(20,4) NOT NULL,
    day_pnl DECIMAL(20,4) NOT NULL,
    gross_pnl DECIMAL(20,4) NOT NULL,
    commission DECIMAL(20,4) NOT NULL,
    fees DECIMAL(20,4) NOT NULL,
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    volatility DECIMAL(10,6),
    var_95 DECIMAL(20,4),
    beta DECIMAL(10,6),
    alpha DECIMAL(10,6),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(10,4),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    largest_win DECIMAL(20,4),
    largest_loss DECIMAL(20,4),
    average_win DECIMAL(20,4),
    average_loss DECIMAL(20,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(account_id, strategy_id, snapshot_date)
);

-- Risk Metrics
CREATE TABLE analytics.risk_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES trading.accounts(id),
    strategy_id UUID REFERENCES trading.strategies(id),
    snapshot_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    portfolio_value DECIMAL(20,4) NOT NULL,
    var_95_1day DECIMAL(20,4),
    var_99_1day DECIMAL(20,4),
    expected_shortfall_95 DECIMAL(20,4),
    maximum_drawdown DECIMAL(10,6),
    current_drawdown DECIMAL(10,6),
    leverage DECIMAL(10,4),
    beta DECIMAL(10,6),
    correlation DECIMAL(10,6),
    concentration_ratio DECIMAL(10,6),
    liquidity_ratio DECIMAL(10,6),
    delta_exposure DECIMAL(20,4) DEFAULT 0,
    gamma_exposure DECIMAL(20,4) DEFAULT 0,
    vega_exposure DECIMAL(20,4) DEFAULT 0,
    theta_exposure DECIMAL(20,4) DEFAULT 0,
    rho_exposure DECIMAL(20,4) DEFAULT 0,
    stress_test_results JSONB DEFAULT '{}',
    sector_exposures JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit Trail
CREATE TABLE audit.trade_audit (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    account_id UUID REFERENCES trading.accounts(id),
    order_id UUID REFERENCES trading.orders(id),
    action VARCHAR(50) NOT NULL, -- 'order_placed', 'order_filled', 'order_cancelled', etc.
    details JSONB NOT NULL,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE audit.system_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for Performance
-- Users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_created_at ON users(created_at);

-- Accounts
CREATE INDEX idx_accounts_user_id ON trading.accounts(user_id);
CREATE INDEX idx_accounts_status ON trading.accounts(status);

-- Instruments
CREATE INDEX idx_instruments_symbol ON market_data.instruments(symbol);
CREATE INDEX idx_instruments_exchange ON market_data.instruments(exchange);
CREATE INDEX idx_instruments_type ON market_data.instruments(instrument_type);
CREATE INDEX idx_instruments_underlying ON market_data.instruments(underlying_symbol);
CREATE INDEX idx_instruments_expiry ON market_data.instruments(expiry_date);

-- Orders
CREATE INDEX idx_orders_account_id ON trading.orders(account_id);
CREATE INDEX idx_orders_strategy_id ON trading.orders(strategy_id);
CREATE INDEX idx_orders_instrument_id ON trading.orders(instrument_id);
CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_created_at ON trading.orders(created_at);
CREATE INDEX idx_orders_broker_order_id ON trading.orders(broker_order_id);

-- Executions
CREATE INDEX idx_executions_order_id ON trading.executions(order_id);
CREATE INDEX idx_executions_account_id ON trading.executions(account_id);
CREATE INDEX idx_executions_executed_at ON trading.executions(executed_at);

-- Positions
CREATE INDEX idx_positions_account_id ON trading.positions(account_id);
CREATE INDEX idx_positions_instrument_id ON trading.positions(instrument_id);
CREATE INDEX idx_positions_strategy_id ON trading.positions(strategy_id);

-- Market Data
CREATE INDEX idx_ticks_instrument_timestamp ON market_data.ticks(instrument_id, timestamp DESC);
CREATE INDEX idx_bars_instrument_timeframe_timestamp ON market_data.bars(instrument_id, timeframe, timestamp DESC);

-- Performance Analytics
CREATE INDEX idx_performance_account_date ON analytics.performance_snapshots(account_id, snapshot_date DESC);
CREATE INDEX idx_performance_strategy_date ON analytics.performance_snapshots(strategy_id, snapshot_date DESC);

-- Risk Analytics
CREATE INDEX idx_risk_account_timestamp ON analytics.risk_snapshots(account_id, snapshot_timestamp DESC);
CREATE INDEX idx_risk_strategy_timestamp ON analytics.risk_snapshots(strategy_id, snapshot_timestamp DESC);

-- Audit
CREATE INDEX idx_trade_audit_user_id ON audit.trade_audit(user_id);
CREATE INDEX idx_trade_audit_account_id ON audit.trade_audit(account_id);
CREATE INDEX idx_trade_audit_created_at ON audit.trade_audit(created_at DESC);
CREATE INDEX idx_system_events_created_at ON audit.system_events(created_at DESC);
CREATE INDEX idx_system_events_severity ON audit.system_events(severity);

-- Partitioning for Time-Series Data (PostgreSQL 11+)
-- Partition ticks table by month
CREATE TABLE market_data.ticks_y2025m01 PARTITION OF market_data.ticks
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE market_data.ticks_y2025m02 PARTITION OF market_data.ticks
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
-- Add more partitions as needed

-- Functions and Triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON trading.accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON trading.strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Performance optimization settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO algoveda;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO algoveda;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO algoveda;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO algoveda;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO algoveda;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO algoveda;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO algoveda;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO algoveda;

-- Initial data
INSERT INTO users (username, email, password_hash, first_name, last_name, role)
VALUES ('admin', 'admin@algoveda.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewXtJqJLQC3DQrjK', 'Admin', 'User', 'admin');

INSERT INTO market_data.instruments (symbol, exchange, instrument_type, currency, sector)
VALUES 
    ('AAPL', 'NASDAQ', 'stock', 'USD', 'Technology'),
    ('MSFT', 'NASDAQ', 'stock', 'USD', 'Technology'),
    ('GOOGL', 'NASDAQ', 'stock', 'USD', 'Technology'),
    ('TSLA', 'NASDAQ', 'stock', 'USD', 'Consumer Discretionary'),
    ('SPY', 'NYSE', 'etf', 'USD', 'Broad Market');

COMMIT;
