# File: tests/data_ingestion/test_db_manager.py

import pytest
import pandas as pd
from datetime import datetime, timezone
import asyncio
from src.data_ingestion.db_manager import DatabaseManager, MarketDataManager
from src.data_ingestion.config import ConfigurationManager

@pytest.fixture
def config():
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'trading_playground_test',
        'user': 'postgres',
        'password': ''
    }

@pytest.fixture
async def db_manager(config):
    manager = DatabaseManager(config)
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.fixture
def market_data_manager(db_manager):
    return MarketDataManager(db_manager)

@pytest.mark.asyncio
async def test_database_connection(db_manager):
    async with db_manager.connection() as conn:
        result = await conn.fetchval('SELECT 1')
        assert result == 1

@pytest.mark.asyncio
async def test_insert_market_data(market_data_manager):
    # Prepare test data
    data = pd.DataFrame({
        'time': [datetime.now(timezone.utc)],
        'symbol': ['AAPL'],
        'open': [150.0],
        'high': [152.0],
        'low': [149.0],
        'close': [151.0],
        'volume': [1000000]
    })

    # Insert data
    count = await market_data_manager.insert_market_data(data, 'test_provider')
    assert count == 1

    # Verify insertion
    result = await market_data_manager.get_market_data(
        'AAPL',
        data['time'].iloc[0],
        data['time'].iloc[0],
        provider='test_provider'
    )
    assert not result.empty
    assert result.index.get_level_values('symbol')[0] == 'AAPL'

@pytest.mark.asyncio
async def test_log_data_quality_issue(market_data_manager):
    await market_data_manager.log_data_quality_issue(
        symbol='AAPL',
        provider='test_provider',
        issue_type='missing_data',
        description='Missing volume data',
        severity=1
    )

    # Verify logging
    async with market_data_manager.db_manager.connection() as conn:
        record = await conn.fetchrow("""
            SELECT symbol, provider, issue_type
            FROM data_quality_log
            WHERE symbol = 'AAPL'
            ORDER BY created_at DESC
            LIMIT 1
        """)

    assert record['symbol'] == 'AAPL'
    assert record['provider'] == 'test_provider'
    assert record['issue_type'] == 'missing_data'
