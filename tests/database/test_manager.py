"""Tests for database manager implementation."""
import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.config.db_config import DatabaseConfig
from src.database.manager import DatabaseManager

@pytest.fixture
def db_config():
    """Database configuration for testing."""
    return DatabaseConfig(
        host='localhost',
        port=5432,
        database='trading_test',  # Use test database
        user='whitneywalters',    # Your local username
        password='',              # Empty password
        min_connections=1,
        max_connections=5
    )

@pytest_asyncio.fixture
async def db_manager(db_config):
    """Database manager fixture."""
    manager = DatabaseManager(db_config)
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.cleanup_old_data(0)  # Clean up test data
        await manager.close()

@pytest.fixture
def sample_data():
    """Generate sample market data."""
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-10',
        freq='1D'
    )
    
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.uniform(100, 200, len(dates))
    data['high'] = data['open'] + np.random.uniform(0, 5, len(dates))
    data['low'] = data['open'] - np.random.uniform(0, 5, len(dates))
    data['close'] = data['open'] + np.random.uniform(-3, 3, len(dates))
    data['volume'] = np.random.randint(1000, 10000, len(dates))
    
    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1) + 0.1
    data['low'] = data[['open', 'low', 'close']].min(axis=1) - 0.1
    
    return data

@pytest.mark.asyncio
async def test_store_and_retrieve_market_data(db_manager, sample_data):
    """Test storing and retrieving market data."""
    symbol = 'AAPL'
    source = 'yahoo'
    timeframe = '1d'
    
    # Store data
    await db_manager.store_market_data(
        sample_data,
        symbol,
        source,
        timeframe
    )
    
    # Retrieve data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 10)
    
    result = await db_manager.get_market_data(
        symbol,
        start_date,
        end_date,
        timeframe,
        source
    )
    
    # Verify results
    assert not result.empty
    assert len(result) == len(sample_data)
    pd.testing.assert_index_equal(result.index, sample_data.index)
    pd.testing.assert_series_equal(
        result['close'],
        sample_data['close'],
        check_names=False
    )

@pytest.mark.asyncio
async def test_get_latest_dates(db_manager, sample_data):
    """Test retrieving latest dates for symbols."""
    symbols = ['AAPL', 'GOOGL']
    source = 'yahoo'
    timeframe = '1d'
    
    # Store data for multiple symbols
    for symbol in symbols:
        await db_manager.store_market_data(
            sample_data,
            symbol,
            source,
            timeframe
        )
    
    # Get latest dates
    latest_dates = await db_manager.get_latest_dates()
    
    # Verify results
    assert len(latest_dates) == len(symbols)
    for symbol in symbols:
        key = (symbol, '1 day')  # timeframe is stored as interval in DB
        assert key in latest_dates
        assert latest_dates[key].date() == sample_data.index[-1].date()

@pytest.mark.asyncio
async def test_cleanup_old_data(db_manager, sample_data):
    """Test cleaning up old market data."""
    symbol = 'AAPL'
    source = 'yahoo'
    timeframe = '1d'
    
    # Store data
    await db_manager.store_market_data(
        sample_data,
        symbol,
        source,
        timeframe
    )
    
    # Clean up data older than 5 days
    await db_manager.cleanup_old_data(5)
    
    # Verify only recent data remains
    result = await db_manager.get_market_data(
        symbol,
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        timeframe,
        source
    )
    
    assert len(result) <= 5

@pytest.mark.asyncio
async def test_different_timeframes(db_manager):
    """Test handling different timeframes."""
    symbol = 'AAPL'
    source = 'yahoo'
    timeframes = ['1m', '5m', '1h', '1d']
    
    for timeframe in timeframes:
        # Create sample data with appropriate frequency
        dates = pd.date_range(
            start='2024-01-01',
            end='2024-01-02',
            freq=timeframe[:-1] + 'min' if timeframe.endswith('m')
            else timeframe[:-1] + 'H' if timeframe.endswith('h')
            else 'D'
        )
        
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(100, 200, len(dates)),
            'low': np.random.uniform(100, 200, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        data['high'] = data[['open', 'high', 'close']].max(axis=1) + 0.1
        data['low'] = data[['open', 'low', 'close']].min(axis=1) - 0.1
        
        # Store data
        await db_manager.store_market_data(
            data,
            symbol,
            source,
            timeframe
        )
        
        # Retrieve and verify
        result = await db_manager.get_market_data(
            symbol,
            dates[0],
            dates[-1],
            timeframe,
            source
        )
        
        assert not result.empty
        assert len(result) == len(data)

@pytest.mark.asyncio
async def test_handle_empty_data(db_manager):
    """Test handling empty DataFrame."""
    symbol = 'AAPL'
    source = 'yahoo'
    timeframe = '1d'
    empty_data = pd.DataFrame()
    
    # Should not raise an exception
    await db_manager.store_market_data(
        empty_data,
        symbol,
        source,
        timeframe
    )
    
    # Verify no data was stored
    result = await db_manager.get_market_data(
        symbol,
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        timeframe,
        source
    )
    
    assert result.empty

@pytest.mark.asyncio
async def test_data_upsert(db_manager, sample_data):
    """Test updating existing data."""
    symbol = 'AAPL'
    source = 'yahoo'
    timeframe = '1d'
    
    # Store initial data
    await db_manager.store_market_data(
        sample_data,
        symbol,
        source,
        timeframe
    )
    
    # Modify some data
    modified_data = sample_data.copy()
    modified_data['close'] = modified_data['close'] * 1.1
    
    # Store modified data
    await db_manager.store_market_data(
        modified_data,
        symbol,
        source,
        timeframe
    )
    
    # Retrieve and verify
    result = await db_manager.get_market_data(
        symbol,
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        timeframe,
        source
    )
    
    pd.testing.assert_series_equal(
        result['close'],
        modified_data['close'],
        check_names=False
    )
