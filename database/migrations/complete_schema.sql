-- Complete PostgreSQL Schema with TimescaleDB for AlgoVeda Trading Platform
-- Based on TimescaleDB best practices for time-series data[6][10][18]

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS compliance;

-- ====================
-- TRADING SCHEMA TABLES
-- ====================

-- Orders table with comprehensive tracking
CREATE TABLE trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_order_id VARCHAR(100) UNIQUE NOT NULL,
    parent_order_id UUID REFERENCES trading.orders(id),
    strategy_id VARCHAR(100),
    account_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'ICEBERG')),
    quantity DECIMAL(18,8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(18,8),
    stop_price DECIMAL(18,8),
    time_in_force VARCHAR(10) NOT NULL DEFAULT 'DAY' CHECK (time_in_force IN ('DAY', 'GTC', 'IOC', 'FOK')),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED')),
    filled_quantity DECIMAL(18,8) DEFAULT 0 CHECK (filled_quantity >= 0),
    remaining_quantity DECIMAL(18,8) NOT NULL CHECK (remaining_quantity >= 0),
    average_fill_price DECIMAL(18,8) DEFAULT 0,
    commission DECIMAL(18,8) DEFAULT 0,
    venue VARCHAR(50),
    execution_instructions JSONB,
    risk_limits JSONB,
    compliance_flags JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    tags JSONB
);

-- Create hypertable for orders (time-series optimization)
SELECT create_hypertable('trading.orders', 'created_at', 
    chunk_time_interval => INTERVAL '1 day',
    create_default_indexes => TRUE);

-- Trades table for execution records
CREATE TABLE trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    account_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(18,8) NOT NULL CHECK (price > 0),
    gross_amount DECIMAL(18,8) NOT NULL,
    commission DECIMAL(18,8) DEFAULT 0,
    fees DECIMAL(18,8) DEFAULT 0,
    net_amount DECIMAL(18,8) NOT NULL,
    venue VARCHAR(50) NOT NULL,
    execution_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    settlement_date DATE,
    counterparty VARCHAR(100),
    trade_flags JSONB,
    regulatory_info JSONB
);

SELECT create_hypertable('trading.trades', 'execution_time',
    chunk_time_interval => INTERVAL '1 day');

-- Positions table for real-time position tracking
CREATE TABLE trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    position_type VARCHAR(20) NOT NULL DEFAULT 'LONG' CHECK (position_type IN ('LONG', 'SHORT')),
    quantity DECIMAL(18,8) NOT NULL,
    average_cost DECIMAL(18,8) NOT NULL,
    market_value DECIMAL(18,8) NOT NULL,
    unrealized_pnl DECIMAL(18,8) NOT NULL DEFAULT 0,
    realized_pnl DECIMAL(18,8) NOT NULL DEFAULT 0,
    day_pnl DECIMAL(18,8) NOT NULL DEFAULT 0,
    exposure_value DECIMAL(18,8) NOT NULL DEFAULT 0,
    margin_requirement DECIMAL(18,8) DEFAULT 0,
    first_trade_date TIMESTAMPTZ,
    last_trade_date TIMESTAMPTZ,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    position_flags JSONB,
    UNIQUE(account_id, symbol)
);

-- ====================
-- MARKET DATA SCHEMA
-- ====================

-- Real-time quotes with microsecond precision
CREATE TABLE market_data.quotes (
    symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(18,8),
    ask DECIMAL(18,8),
    bid_size BIGINT,
    ask_size BIGINT,
    last_price DECIMAL(18,8),
    last_size BIGINT,
    volume BIGINT DEFAULT 0,
    turnover DECIMAL(18,8) DEFAULT 0,
    vwap DECIMAL(18,8),
    high DECIMAL(18,8),
    low DECIMAL(18,8),
    open DECIMAL(18,8),
    close DECIMAL(18,8),
    venue VARCHAR(50),
    market_status VARCHAR(20) DEFAULT 'OPEN',
    quote_condition VARCHAR(10),
    sequence_number BIGINT
);

SELECT create_hypertable('market_data.quotes', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour');

-- OHLCV bars for different timeframes
CREATE TABLE market_data.bars (
    symbol VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- 1m, 5m, 15m, 1h, 1d, etc.
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    vwap DECIMAL(18,8),
    trade_count INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

SELECT create_hypertable('market_data.bars', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- Options chain data
CREATE TABLE market_data.options_chain (
    symbol VARCHAR(50) NOT NULL, -- Underlying symbol
    option_symbol VARCHAR(100) NOT NULL,
    option_type CHAR(1) NOT NULL CHECK (option_type IN ('C', 'P')),
    strike_price DECIMAL(18,8) NOT NULL,
    expiry_date DATE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    bid DECIMAL(18,8),
    ask DECIMAL(18,8),
    last_price DECIMAL(18,8),
    implied_volatility DECIMAL(10,6),
    delta DECIMAL(10,6),
    gamma DECIMAL(10,6),
    theta DECIMAL(10,6),
    vega DECIMAL(10,6),
    rho DECIMAL(10,6),
    open_interest BIGINT DEFAULT 0,
    volume BIGINT DEFAULT 0
);

SELECT create_hypertable('market_data.options_chain', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour');

-- ====================
-- RISK SCHEMA
-- ====================

-- Portfolio risk metrics with real-time updates
CREATE TABLE risk.portfolio_metrics (
    account_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_equity DECIMAL(18,8) NOT NULL,
    cash_balance DECIMAL(18,8) NOT NULL,
    total_pnl DECIMAL(18,8) NOT NULL,
    day_pnl DECIMAL(18,8) NOT NULL,
    unrealized_pnl DECIMAL(18,8) NOT NULL,
    realized_pnl DECIMAL(18,8) NOT NULL,
    gross_exposure DECIMAL(18,8) NOT NULL DEFAULT 0,
    net_exposure DECIMAL(18,8) NOT NULL DEFAULT 0,
    leverage DECIMAL(10,4) DEFAULT 1.0,
    var_95 DECIMAL(18,8),
    var_99 DECIMAL(18,8),
    expected_shortfall_95 DECIMAL(18,8),
    expected_shortfall_99 DECIMAL(18,8),
    max_drawdown DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    volatility DECIMAL(10,6),
    beta DECIMAL(10,6),
    correlation_to_market DECIMAL(10,6),
    concentration_risk DECIMAL(10,6),
    sector_exposures JSONB,
    currency_exposures JSONB,
    risk_limits JSONB,
    alerts JSONB
);

SELECT create_hypertable('risk.portfolio_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 hour');

-- Risk alerts and violations
CREATE TABLE risk.risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    description TEXT NOT NULL,
    current_value DECIMAL(18,8),
    limit_value DECIMAL(18,8),
    breach_percentage DECIMAL(10,4),
    symbol VARCHAR(50),
    position_id UUID,
    order_id UUID,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    metadata JSONB
);

SELECT create_hypertable('risk.risk_events', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- ====================
-- ANALYTICS SCHEMA
-- ====================

-- Performance analytics aggregated by day
CREATE TABLE analytics.daily_performance (
    account_id VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    starting_equity DECIMAL(18,8) NOT NULL,
    ending_equity DECIMAL(18,8) NOT NULL,
    pnl DECIMAL(18,8) NOT NULL,
    return_pct DECIMAL(10,6) NOT NULL,
    benchmark_return DECIMAL(10,6),
    alpha DECIMAL(10,6),
    beta DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    trades_count INTEGER DEFAULT 0,
    win_trades INTEGER DEFAULT 0,
    loss_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    avg_win DECIMAL(18,8),
    avg_loss DECIMAL(18,8),
    profit_factor DECIMAL(10,4),
    commission_paid DECIMAL(18,8) DEFAULT 0,
    PRIMARY KEY (account_id, date)
);

-- Strategy performance tracking
CREATE TABLE analytics.strategy_performance (
    strategy_id VARCHAR(100) NOT NULL,
    account_id VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_pnl DECIMAL(18,8) NOT NULL DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4) DEFAULT 0,
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    allocated_capital DECIMAL(18,8),
    return_on_capital DECIMAL(10,6),
    metadata JSONB,
    PRIMARY KEY (strategy_id, account_id, date)
);

-- ====================
-- AUDIT SCHEMA  
-- ====================

-- Comprehensive audit trail for compliance[9]
CREATE TABLE audit.system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    result VARCHAR(20) NOT NULL CHECK (result IN ('SUCCESS', 'FAILURE', 'WARNING', 'BLOCKED')),
    risk_level VARCHAR(20) NOT NULL DEFAULT 'LOW' CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    details JSONB,
    compliance_tags TEXT[],
    retention_until DATE
);

SELECT create_hypertable('audit.system_events', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

-- ====================
-- COMPLIANCE SCHEMA
-- ====================

-- Regulatory reporting requirements
CREATE TABLE compliance.regulatory_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_type VARCHAR(50) NOT NULL,
    regulation VARCHAR(50) NOT NULL, -- SOX, MiFID_II, FINRA, etc.
    account_id VARCHAR(100),
    report_date DATE NOT NULL,
    filing_deadline DATE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    report_data JSONB NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    filed_at TIMESTAMPTZ,
    acknowledgment_receipt VARCHAR(200),
    metadata JSONB
);

-- Trade reporting for regulatory compliance
CREATE TABLE compliance.trade_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id UUID NOT NULL REFERENCES trading.trades(id),
    regulation VARCHAR(50) NOT NULL,
    report_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reporting_status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    uti VARCHAR(100), -- Unique Trade Identifier
    lei VARCHAR(20),  -- Legal Entity Identifier
    report_data JSONB NOT NULL,
    submitted_at TIMESTAMPTZ,
    acknowledgment JSONB,
    errors JSONB
);

-- ====================
-- INDEXES FOR PERFORMANCE
-- ====================

-- Trading indexes
CREATE INDEX idx_orders_account_created ON trading.orders(account_id, created_at DESC);
CREATE INDEX idx_orders_symbol_status ON trading.orders(symbol, status) WHERE status IN ('NEW', 'PARTIALLY_FILLED');
CREATE INDEX idx_orders_strategy_time ON trading.orders(strategy_id, created_at DESC) WHERE strategy_id IS NOT NULL;
CREATE INDEX idx_trades_account_time ON trading.trades(account_id, execution_time DESC);
CREATE INDEX idx_trades_symbol_time ON trading.trades(symbol, execution_time DESC);
CREATE INDEX idx_positions_account ON trading.positions(account_id);

-- Market data indexes
CREATE INDEX idx_quotes_symbol_time ON market_data.quotes(symbol, timestamp DESC);
CREATE INDEX idx_bars_symbol_timeframe_time ON market_data.bars(symbol, timeframe, timestamp DESC);
CREATE INDEX idx_options_underlying_expiry ON market_data.options_chain(symbol, expiry_date, timestamp DESC);

-- Risk indexes
CREATE INDEX idx_portfolio_metrics_account_time ON risk.portfolio_metrics(account_id, timestamp DESC);
CREATE INDEX idx_risk_events_account_severity ON risk.risk_events(account_id, severity, timestamp DESC);

-- Audit indexes  
CREATE INDEX idx_audit_events_type_time ON audit.system_events(event_type, timestamp DESC);
CREATE INDEX idx_audit_events_user_time ON audit.system_events(user_id, timestamp DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_events_risk_time ON audit.system_events(risk_level, timestamp DESC) WHERE risk_level IN ('HIGH', 'CRITICAL');

-- ====================
-- TRIGGERS AND FUNCTIONS
-- ====================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to orders table
CREATE TRIGGER update_orders_updated_at 
    BEFORE UPDATE ON trading.orders 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to positions table
CREATE TRIGGER update_positions_updated_at 
    BEFORE UPDATE ON trading.positions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Position update trigger for trade execution
CREATE OR REPLACE FUNCTION update_position_on_trade()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO trading.positions (account_id, symbol, quantity, average_cost, market_value, unrealized_pnl, last_trade_date)
    VALUES (NEW.account_id, NEW.symbol, 
            CASE WHEN NEW.side = 'BUY' THEN NEW.quantity ELSE -NEW.quantity END,
            NEW.price, NEW.quantity * NEW.price, 0, NEW.execution_time)
    ON CONFLICT (account_id, symbol)
    DO UPDATE SET
        quantity = positions.quantity + CASE WHEN NEW.side = 'BUY' THEN NEW.quantity ELSE -NEW.quantity END,
        average_cost = CASE 
            WHEN (positions.quantity + CASE WHEN NEW.side = 'BUY' THEN NEW.quantity ELSE -NEW.quantity END) = 0 THEN 0
            ELSE (positions.average_cost * positions.quantity + NEW.price * NEW.quantity) / 
                 (positions.quantity + CASE WHEN NEW.side = 'BUY' THEN NEW.quantity ELSE -NEW.quantity END)
        END,
        market_value = (positions.quantity + CASE WHEN NEW.side = 'BUY' THEN NEW.quantity ELSE -NEW.quantity END) * NEW.price,
        last_trade_date = NEW.execution_time,
        last_updated = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_position_on_trade
    AFTER INSERT ON trading.trades
    FOR EACH ROW
    EXECUTE FUNCTION update_position_on_trade();

-- ====================
-- DATA RETENTION POLICIES
-- ====================

-- Set retention policies for time-series data
SELECT add_retention_policy('trading.orders', INTERVAL '7 years');
SELECT add_retention_policy('trading.trades', INTERVAL '10 years'); -- Regulatory requirement
SELECT add_retention_policy('market_data.quotes', INTERVAL '2 years');
SELECT add_retention_policy('market_data.bars', INTERVAL '5 years');
SELECT add_retention_policy('market_data.options_chain', INTERVAL '1 year');
SELECT add_retention_policy('risk.portfolio_metrics', INTERVAL '5 years');
SELECT add_retention_policy('risk.risk_events', INTERVAL '7 years');
SELECT add_retention_policy('audit.system_events', INTERVAL '7 years'); -- SOX compliance

-- ====================
-- VIEWS FOR COMMON QUERIES
-- ====================

-- Active orders view
CREATE VIEW trading.active_orders AS
SELECT * FROM trading.orders 
WHERE status IN ('NEW', 'PARTIALLY_FILLED', 'PENDING')
AND (expires_at IS NULL OR expires_at > NOW());

-- Today's trading activity
CREATE VIEW trading.todays_activity AS
SELECT 
    account_id,
    symbol,
    COUNT(*) as trade_count,
    SUM(quantity) as total_quantity,
    SUM(gross_amount) as total_volume,
    AVG(price) as avg_price,
    SUM(commission) as total_commission
FROM trading.trades 
WHERE execution_time >= CURRENT_DATE
GROUP BY account_id, symbol;

-- Current portfolio positions
CREATE VIEW trading.current_positions AS
SELECT * FROM trading.positions 
WHERE quantity != 0;

-- Risk summary view
CREATE VIEW risk.current_risk_summary AS
SELECT DISTINCT ON (account_id)
    account_id,
    total_equity,
    total_pnl,
    day_pnl,
    leverage,
    var_95,
    max_drawdown,
    timestamp
FROM risk.portfolio_metrics 
ORDER BY account_id, timestamp DESC;
