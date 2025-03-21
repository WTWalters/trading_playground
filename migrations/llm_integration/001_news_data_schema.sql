-- News data schema for LLM integration

-- Enable required extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- News data table
CREATE TABLE IF NOT EXISTS news_data (
    -- Identifiers
    id              BIGSERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    
    -- Content and metadata
    content         TEXT NOT NULL,
    summary         TEXT,
    source          TEXT NOT NULL,
    author          TEXT,
    url             TEXT,
    
    -- Time information
    published_time  TIMESTAMPTZ NOT NULL,
    retrieved_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Classification
    categories      TEXT[],
    sentiment       DOUBLE PRECISION,
    relevance_score DOUBLE PRECISION,
    entities        JSONB,
    
    -- Search optimization
    search_vector   tsvector GENERATED ALWAYS AS (
        to_tsvector('english', title || ' ' || content)
    ) STORED
);

-- Create hypertable
SELECT create_hypertable(
    'news_data',
    'published_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Create indices for efficient searching
CREATE INDEX IF NOT EXISTS idx_news_data_time
    ON news_data (published_time DESC);
    
CREATE INDEX IF NOT EXISTS idx_news_data_source
    ON news_data (source);
    
CREATE INDEX IF NOT EXISTS idx_news_search_vector
    ON news_data USING GIN (search_vector);

-- Symbol-specific news mapping
CREATE TABLE IF NOT EXISTS symbol_news_mapping (
    symbol          TEXT NOT NULL,
    news_id         BIGINT NOT NULL,
    relevance_score DOUBLE PRECISION,
    
    CONSTRAINT symbol_news_mapping_pkey
        PRIMARY KEY (symbol, news_id),
    CONSTRAINT fk_news_id
        FOREIGN KEY(news_id)
        REFERENCES news_data(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_symbol_news_mapping
    ON symbol_news_mapping (symbol, relevance_score DESC);

-- LLM analysis of news
CREATE TABLE IF NOT EXISTS news_llm_analysis (
    news_id         BIGINT NOT NULL,
    analysis_type   TEXT NOT NULL,
    analysis_result JSONB NOT NULL,
    model_used      TEXT NOT NULL,
    processing_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT news_llm_analysis_pkey
        PRIMARY KEY (news_id, analysis_type),
    CONSTRAINT fk_news_id
        FOREIGN KEY(news_id)
        REFERENCES news_data(id)
        ON DELETE CASCADE
);

-- Set up compression policy for news data
ALTER TABLE news_data SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'published_time',
    timescaledb.compress_segmentby = 'source'
);

-- Add compression policy to compress chunks older than 30 days
SELECT add_compression_policy('news_data',
    compress_after => INTERVAL '30 days');
