# Database Schema Extensions for LLM Integration

## Overview

This document describes the database schema extensions required to support LLM integration in the TITAN trading platform. All extensions follow the existing TimescaleDB architecture.

## News Data Schema

### `news_data` Table

Stores financial news articles and related information.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| title | TEXT | News article title |
| content | TEXT | Full article content |
| summary | TEXT | Article summary |
| source | TEXT | Source of the news (e.g., Bloomberg, Reuters) |
| author | TEXT | Article author |
| url | TEXT | Original article URL |
| published_time | TIMESTAMPTZ | Article publication time |
| retrieved_time | TIMESTAMPTZ | Time article was retrieved |
| categories | TEXT[] | Array of category tags |
| sentiment | DOUBLE PRECISION | Sentiment score (-1 to 1) |
| relevance_score | DOUBLE PRECISION | Relevance score (0 to 1) |
| entities | JSONB | Extracted entities (people, organizations, etc.) |
| search_vector | tsvector | Vector for full-text search |

### `symbol_news_mapping` Table

Maps news articles to specific symbols.

| Column | Type | Description |
|--------|------|-------------|
| symbol | TEXT | Trading symbol |
| news_id | BIGINT | References news_data(id) |
| relevance_score | DOUBLE PRECISION | Relevance score for this symbol (0-1) |

### `news_llm_analysis` Table

Stores LLM analysis results for news articles.

| Column | Type | Description |
|--------|------|-------------|
| news_id | BIGINT | References news_data(id) |
| analysis_type | TEXT | Type of analysis performed |
| analysis_result | JSONB | JSON result of the analysis |
| model_used | TEXT | LLM model name |
| processing_time | TIMESTAMPTZ | When analysis was performed |

## Trade Analysis Schema

### `trade_context` Table

Stores market context information for trades.

| Column | Type | Description |
|--------|------|-------------|
| trade_id | TEXT | References trades(trade_id) |
| entry_context | JSONB | Market conditions at entry |
| exit_context | JSONB | Market conditions at exit |
| regime_at_entry | TEXT | Market regime at entry |
| regime_at_exit | TEXT | Market regime at exit |
| news_ids | BIGINT[] | Related news during trade period |
| macro_data | JSONB | Economic indicators during trade |

### `trade_post_mortem` Table

Stores LLM-generated post-mortem analysis for trades.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| trade_id | TEXT | References trades(trade_id) |
| analysis_date | TIMESTAMPTZ | When analysis was performed |
| model_used | TEXT | LLM model name |
| performance_analysis | JSONB | Performance metrics assessment |
| decision_analysis | JSONB | Entry/exit decision evaluation |
| context_analysis | JSONB | Market context analysis |
| improvement_suggestions | JSONB | Suggested improvements |
| primary_factors | TEXT[] | Primary factors affecting the trade |
| success_score | SMALLINT | 0-100 success rating |
| error_types | TEXT[] | Types of errors made |

### `trade_learning_patterns` Table

Stores patterns identified across multiple trades.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| pattern_type | TEXT | Type of pattern identified |
| description | TEXT | Description of the pattern |
| affected_trades | TEXT[] | Trade IDs exhibiting this pattern |
| creation_date | TIMESTAMPTZ | When pattern was identified |
| last_updated | TIMESTAMPTZ | When pattern was last updated |
| confidence | DOUBLE PRECISION | Confidence in pattern (0-1) |
| recurring_count | INTEGER | Number of pattern occurrences |
| category | TEXT | Pattern category (psychology, technical, etc.) |
| subcategory | TEXT | More specific categorization |
| action_items | JSONB | Recommended actions |

## Indexing Strategy

The schema includes these indexes for optimal query performance:

- `idx_news_data_time`: On `news_data(published_time DESC)`
- `idx_news_data_source`: On `news_data(source)`
- `idx_news_search_vector`: GIN index on `news_data(search_vector)`
- `idx_symbol_news_mapping`: On `symbol_news_mapping(symbol, relevance_score DESC)`
- `idx_trade_post_mortem_date`: On `trade_post_mortem(analysis_date DESC)`
- `idx_trade_learning_patterns_type`: On `trade_learning_patterns(pattern_type)`
- `idx_trade_learning_patterns_category`: On `trade_learning_patterns(category)`

## TimescaleDB Features

The schema leverages TimescaleDB features:
- Hypertables for time-series data
- Continuous aggregates for efficient queries
- Compression policies for older data
- Retention policies for data lifecycle management