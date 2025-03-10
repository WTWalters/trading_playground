-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Market data hypertable
CREATE TABLE IF NOT EXISTS market_data (
    -- Time and identifier columns
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    provider        TEXT NOT NULL,

    -- Price and volume data
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT NOT NULL,

    -- Quality tracking
    data_quality    SMALLINT NOT NULL DEFAULT 100,
    is_adjusted     BOOLEAN NOT NULL DEFAULT FALSE,
    source_time     TIMESTAMPTZ NOT NULL,  -- Original time from data provider
    update_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata        JSONB,  -- Flexible field for additional provider-specific data

    -- Constraints
    CONSTRAINT market_data_pkey PRIMARY KEY (time, symbol, provider),
    CONSTRAINT market_data_price_check CHECK (
        high >= low AND
        high >= open AND
        high >= close AND
        low <= open AND
        low <= close AND
        volume >= 0
    ),
    CONSTRAINT market_data_quality_check CHECK (
        data_quality >= 0 AND
        data_quality <= 100
    )
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_data',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time
    ON market_data (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_provider_time
    ON market_data (provider, time DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_quality
    ON market_data (data_quality)
    WHERE data_quality < 100;

-- Create continuous aggregates for common timeframes
CREATE MATERIALIZED VIEW market_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    provider,
    first(open, time) as open,
    max(high) as high,
    min(low) as low,
    last(close, time) as close,
    sum(volume) as volume,
    avg(data_quality) as avg_quality,
    count(*) as sample_count
FROM market_data
GROUP BY bucket, symbol, provider
WITH NO DATA;

CREATE MATERIALIZED VIEW market_data_1hour
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    provider,
    first(open, time) as open,
    max(high) as high,
    min(low) as low,
    last(close, time) as close,
    sum(volume) as volume,
    avg(data_quality) as avg_quality,
    count(*) as sample_count
FROM market_data
GROUP BY bucket, symbol, provider
WITH NO DATA;

CREATE MATERIALIZED VIEW market_data_1day
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    symbol,
    provider,
    first(open, time) as open,
    max(high) as high,
    min(low) as low,
    last(close, time) as close,
    sum(volume) as volume,
    avg(data_quality) as avg_quality,
    count(*) as sample_count
FROM market_data
GROUP BY bucket, symbol, provider
WITH NO DATA;

-- Set up refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('market_data_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('market_data_1hour',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('market_data_1day',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Set up compression policy
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time',
    timescaledb.compress_segmentby = 'symbol,provider'
);

-- Add compression policy to compress chunks older than 7 days
SELECT add_compression_policy('market_data',
    compress_after => INTERVAL '7 days');

-- Create data dictionary table for symbols
CREATE TABLE IF NOT EXISTS symbol_reference (
    symbol          TEXT PRIMARY KEY,
    name            TEXT,
    asset_type      TEXT NOT NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create provider configuration table
CREATE TABLE IF NOT EXISTS data_provider_config (
    provider        TEXT PRIMARY KEY,
    config          JSONB NOT NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    last_success    TIMESTAMPTZ,
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create data quality tracking table
CREATE TABLE IF NOT EXISTS data_quality_log (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    provider        TEXT NOT NULL,
    issue_type      TEXT NOT NULL,
    description     TEXT,
    severity        SMALLINT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_symbol
        FOREIGN KEY(symbol)
        REFERENCES symbol_reference(symbol)
);

CREATE INDEX IF NOT EXISTS idx_data_quality_log_time
    ON data_quality_log (time DESC);

-- Create function for updating symbol reference
CREATE OR REPLACE FUNCTION update_symbol_reference_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_symbol_reference_updated_at
    BEFORE UPDATE ON symbol_reference
    FOR EACH ROW
    EXECUTE FUNCTION update_symbol_reference_timestamp();

-- Create function for updating provider config
CREATE OR REPLACE FUNCTION update_provider_config_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_provider_config_updated_at
    BEFORE UPDATE ON data_provider_config
    FOR EACH ROW
    EXECUTE FUNCTION update_provider_config_timestamp();