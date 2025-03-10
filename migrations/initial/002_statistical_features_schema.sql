-- Statistical features and analysis results schema

-- Enable required extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Table for storing calculated statistical features
CREATE TABLE IF NOT EXISTS statistical_features (
    -- Time and identifier columns
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    feature_set     TEXT NOT NULL,  -- Identifier for the set of features (e.g., 'volatility', 'trend', etc.)

    -- Feature data stored as JSONB for flexibility
    features        JSONB NOT NULL,
    
    -- Metadata
    calculation_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parameters       JSONB,
    
    -- Constraints
    CONSTRAINT statistical_features_pkey PRIMARY KEY (time, symbol, feature_set)
);

-- Convert to hypertable
SELECT create_hypertable(
    'statistical_features',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_statistical_features_symbol_time
    ON statistical_features (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_statistical_features_set
    ON statistical_features (feature_set, time DESC);

-- Table for pair correlations and cointegration results
CREATE TABLE IF NOT EXISTS pair_relationships (
    -- Pair identifiers
    symbol_a        TEXT NOT NULL,
    symbol_b        TEXT NOT NULL,
    relationship_type TEXT NOT NULL,  -- 'correlation', 'cointegration', etc.
    
    -- Time range for the analysis
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ NOT NULL,
    
    -- Results
    results         JSONB NOT NULL,
    
    -- Metadata
    calculation_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parameters      JSONB,
    
    -- Constraints
    CONSTRAINT pair_relationships_pkey 
        PRIMARY KEY (symbol_a, symbol_b, relationship_type, start_time, end_time)
);

-- Create indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_pair_relationships_symbols
    ON pair_relationships (symbol_a, symbol_b);

CREATE INDEX IF NOT EXISTS idx_pair_relationships_type
    ON pair_relationships (relationship_type);

CREATE INDEX IF NOT EXISTS idx_pair_relationships_time
    ON pair_relationships (start_time, end_time);

-- Table for market regime classification
CREATE TABLE IF NOT EXISTS market_regimes (
    -- Time and identifier columns
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    regime_type     TEXT NOT NULL,  -- Type of regime (volatility, trend, etc.)
    
    -- Regime classification
    regime_value    TEXT NOT NULL,  -- Actual regime value (e.g., 'high_volatility', 'uptrend', etc.)
    confidence      DOUBLE PRECISION,
    
    -- Additional data
    metrics         JSONB,
    
    -- Metadata
    calculation_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parameters      JSONB,
    
    -- Constraints
    CONSTRAINT market_regimes_pkey 
        PRIMARY KEY (time, symbol, regime_type)
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_regimes',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_market_regimes_symbol_time
    ON market_regimes (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_market_regimes_type
    ON market_regimes (regime_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_market_regimes_value
    ON market_regimes (regime_value);

-- Table for trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    -- Time and identifier columns
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    strategy_id     TEXT NOT NULL,
    
    -- Signal data
    signal_type     TEXT NOT NULL,  -- 'entry', 'exit', 'stop_loss', etc.
    direction       TEXT NOT NULL,  -- 'long', 'short', 'neutral'
    strength        DOUBLE PRECISION,  -- Signal strength/confidence (0-1)
    
    -- Price levels
    price           DOUBLE PRECISION,
    target_price    DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    
    -- Additional data
    risk_reward     DOUBLE PRECISION,
    metrics         JSONB,
    
    -- Metadata
    generation_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parameters      JSONB,
    
    -- Constraints
    CONSTRAINT trading_signals_pkey 
        PRIMARY KEY (time, symbol, strategy_id, signal_type)
);

-- Convert to hypertable
SELECT create_hypertable(
    'trading_signals',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_time
    ON trading_signals (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_trading_signals_strategy
    ON trading_signals (strategy_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_trading_signals_type
    ON trading_signals (signal_type, direction);

-- Table for backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    -- Identifiers
    backtest_id     TEXT PRIMARY KEY,
    strategy_id     TEXT NOT NULL,
    
    -- Time range
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ NOT NULL,
    
    -- Basic results
    initial_capital DOUBLE PRECISION NOT NULL,
    final_capital   DOUBLE PRECISION NOT NULL,
    total_return    DOUBLE PRECISION NOT NULL,
    annualized_return DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    
    -- Additional metrics
    win_rate        DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    avg_win         DOUBLE PRECISION,
    avg_loss        DOUBLE PRECISION,
    num_trades      INTEGER,
    
    -- Detailed results
    metrics         JSONB,
    parameters      JSONB,
    
    -- Metadata
    run_time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_duration    INTERVAL,
    notes           TEXT
);

-- Create indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy
    ON backtest_results (strategy_id);

CREATE INDEX IF NOT EXISTS idx_backtest_results_time
    ON backtest_results (start_time, end_time);

-- Table for trades (from backtests or live trading)
CREATE TABLE IF NOT EXISTS trades (
    -- Identifiers
    trade_id        TEXT PRIMARY KEY,
    backtest_id     TEXT,  -- NULL for live trades
    strategy_id     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    
    -- Direction
    direction       TEXT NOT NULL,  -- 'long' or 'short'
    
    -- Entry
    entry_time      TIMESTAMPTZ NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    entry_signal    TEXT,
    position_size   DOUBLE PRECISION NOT NULL,
    
    -- Exit
    exit_time       TIMESTAMPTZ,
    exit_price      DOUBLE PRECISION,
    exit_signal     TEXT,
    
    -- Results
    pnl             DOUBLE PRECISION,
    pnl_percentage  DOUBLE PRECISION,
    duration        INTERVAL,
    
    -- Risk management
    initial_stop    DOUBLE PRECISION,
    initial_target  DOUBLE PRECISION,
    risk_reward     DOUBLE PRECISION,
    
    -- Additional data
    tags            TEXT[],
    metrics         JSONB,
    notes           TEXT,
    
    -- Constraints
    CONSTRAINT fk_backtest
        FOREIGN KEY(backtest_id)
        REFERENCES backtest_results(backtest_id)
        ON DELETE SET NULL
);

-- Create indices for efficient querying
CREATE INDEX IF NOT EXISTS idx_trades_symbol
    ON trades (symbol);

CREATE INDEX IF NOT EXISTS idx_trades_strategy
    ON trades (strategy_id);

CREATE INDEX IF NOT EXISTS idx_trades_backtest
    ON trades (backtest_id);

CREATE INDEX IF NOT EXISTS idx_trades_entry_time
    ON trades (entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_exit_time
    ON trades (exit_time DESC);

-- Function to calculate trade duration on insert/update
CREATE OR REPLACE FUNCTION calculate_trade_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.exit_time IS NOT NULL AND NEW.entry_time IS NOT NULL THEN
        NEW.duration = NEW.exit_time - NEW.entry_time;
    ELSE
        NEW.duration = NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_trade_duration
    BEFORE INSERT OR UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION calculate_trade_duration();