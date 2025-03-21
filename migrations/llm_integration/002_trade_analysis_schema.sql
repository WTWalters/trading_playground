-- Trade analysis schema for LLM integration

-- Trade context table for storing relevant market conditions during a trade
CREATE TABLE IF NOT EXISTS trade_context (
    trade_id         TEXT NOT NULL,
    entry_context    JSONB NOT NULL,  -- Market conditions at entry
    exit_context     JSONB,           -- Market conditions at exit
    regime_at_entry  TEXT NOT NULL,
    regime_at_exit   TEXT,
    news_ids         BIGINT[],        -- Related news during trade period
    macro_data       JSONB,           -- Economic indicators during trade
    
    CONSTRAINT trade_context_pkey PRIMARY KEY (trade_id),
    CONSTRAINT fk_trade_id
        FOREIGN KEY(trade_id)
        REFERENCES trades(trade_id)
        ON DELETE CASCADE
);

-- Post-mortem analysis table
CREATE TABLE IF NOT EXISTS trade_post_mortem (
    id               BIGSERIAL PRIMARY KEY,
    trade_id         TEXT NOT NULL,
    analysis_date    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_used       TEXT NOT NULL,
    
    -- Analysis components
    performance_analysis  JSONB NOT NULL,  -- Performance metrics assessment
    decision_analysis     JSONB NOT NULL,  -- Entry/exit decision evaluation
    context_analysis      JSONB NOT NULL,  -- Market context analysis
    improvement_suggestions JSONB NOT NULL, -- Suggested improvements
    
    -- Summary fields for quick reference
    primary_factors       TEXT[],  -- Primary factors affecting the trade
    success_score         SMALLINT, -- 0-100 success rating
    error_types           TEXT[],  -- Types of errors made
    
    CONSTRAINT fk_trade_id
        FOREIGN KEY(trade_id)
        REFERENCES trades(trade_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trade_post_mortem_date
    ON trade_post_mortem (analysis_date DESC);

-- Aggregate learning table to track patterns across trades
CREATE TABLE IF NOT EXISTS trade_learning_patterns (
    id               BIGSERIAL PRIMARY KEY,
    pattern_type     TEXT NOT NULL,
    description      TEXT NOT NULL,
    affected_trades  TEXT[] NOT NULL,
    creation_date    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidence       DOUBLE PRECISION NOT NULL,
    recurring_count  INTEGER NOT NULL,
    
    -- Categorization
    category         TEXT NOT NULL,  -- 'psychology', 'technical', 'fundamental'
    subcategory      TEXT,
    
    -- Improvement plan
    action_items     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_learning_patterns_type
    ON trade_learning_patterns (pattern_type);
    
CREATE INDEX IF NOT EXISTS idx_trade_learning_patterns_category
    ON trade_learning_patterns (category);

-- Function to update last_updated timestamp on pattern updates
CREATE OR REPLACE FUNCTION update_pattern_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_pattern_last_updated
    BEFORE UPDATE ON trade_learning_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_pattern_timestamp();

-- Performance metrics tracking table to measure LLM impact
CREATE TABLE IF NOT EXISTS llm_performance_metrics (
    id               BIGSERIAL PRIMARY KEY,
    metric_date      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_type      TEXT NOT NULL,  -- 'regime_accuracy', 'win_rate', etc.
    baseline_value   DOUBLE PRECISION,  -- Performance without LLM
    enhanced_value   DOUBLE PRECISION,  -- Performance with LLM
    improvement_pct  DOUBLE PRECISION,  -- Percentage improvement
    sample_size      INTEGER NOT NULL,  -- Number of trades/detections in sample
    
    -- Additional metadata
    time_period      TEXT NOT NULL,  -- '7d', '30d', '90d', etc.
    notes            TEXT,
    raw_data         JSONB
);

CREATE INDEX IF NOT EXISTS idx_llm_performance_metrics_type
    ON llm_performance_metrics (metric_type, metric_date DESC);

-- Create hypertable for performance metrics to enable time-based queries
SELECT create_hypertable(
    'llm_performance_metrics',
    'metric_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);
