import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from data_ingestion.orchestrator import DataOrchestrator, DataIngestionConfig
from data_ingestion.validation import DataValidator

@pytest.fixture
def valid_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    base_price = 100
    return pd.DataFrame({
        'open': [base_price + i*0.1 for i in range(len(dates))],
        'high': [base_price + 1 + i*0.1 for i in range(len(dates))],
        'low': [base_price - 1 + i*0.1 for i in range(len(dates))],
        'close': [base_price + 0.5 + i*0.1 for i in range(len(dates))],
        'volume': [1000 + i*100 for i in range(len(dates))]
    }, index=dates)

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    return pd.DataFrame({
        'open': np.random.randn(len(dates)) + 100,
        'high': np.random.randn(len(dates)) + 101,
        'low': np.random.randn(len(dates)) + 99,
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

@pytest.fixture
def validator():
    return DataValidator({
        'null_threshold': 0.01,
        'price_z_threshold': 4.0,
        'volume_z_threshold': 5.0,
        'max_time_gap': pd.Timedelta(days=5)
    })

@pytest.fixture
def mock_db_manager():
    manager = Mock()
    manager.store_market_data = AsyncMock()
    manager.get_latest_dates = AsyncMock(return_value={})
    return manager

@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.fetch_historical_data = AsyncMock()
    return provider

@pytest.fixture
def config():
    return DataIngestionConfig(
        symbols=['AAPL', 'MSFT'],
        timeframes=['1d'],
        providers=['yahoo'],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
        validation_rules={
            'null_threshold': 0.01,
            'price_z_threshold': 4.0,
            'volume_z_threshold': 5.0,
            'max_time_gap': pd.Timedelta(days=5)
        },
        retry_attempts=3,
        retry_delay=1
    )

def test_validate_market_data(validator, valid_data):
    """Test basic market data validation"""
    assert validator.validate_market_data(valid_data) is True

def test_validate_price_consistency(validator, valid_data):
    """Test price relationship validation"""
    assert validator._validate_price_consistency(valid_data) is True

def test_validate_timestamps(validator, valid_data):
    """Test timestamp validation"""
    assert validator._validate_timestamps(valid_data) is True

def test_validate_gaps(validator, valid_data):
    """Test time gap validation"""
    assert validator._validate_gaps(valid_data) is True

def test_validate_outliers(validator, valid_data):
    """Test outlier detection"""
    assert validator._validate_outliers(valid_data) is True

def test_invalid_data_structure(validator):
    """Test validation with missing columns"""
    invalid_data = pd.DataFrame({
        'open': [100, 101],
        'close': [101, 102]  # Missing required columns
    })
    assert validator._validate_basic_structure(invalid_data) is False

def test_invalid_price_consistency(validator, valid_data):
    """Test validation with inconsistent prices"""
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = 98  # High below low
    assert validator._validate_price_consistency(invalid_data) is False

@pytest.mark.asyncio
async def test_fetch_with_retry_success(sample_data, config, mock_db_manager):
    orchestrator = DataOrchestrator(config, mock_db_manager)
    mock_provider = Mock()
    mock_provider.fetch_historical_data = AsyncMock(return_value=sample_data)

    result = await orchestrator._fetch_with_retry(
        mock_provider,
        'AAPL',
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        '1d'
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    assert mock_provider.fetch_historical_data.call_count == 1

@pytest.mark.asyncio
async def test_fetch_with_retry_failure(config, mock_db_manager):
    orchestrator = DataOrchestrator(config, mock_db_manager)
    mock_provider = Mock()
    mock_provider.fetch_historical_data = AsyncMock(side_effect=Exception("API Error"))

    result = await orchestrator._fetch_with_retry(
        mock_provider,
        'AAPL',
        datetime(2024, 1, 1),
        datetime(2024, 1, 10),
        '1d'
    )

    assert result is None
    assert mock_provider.fetch_historical_data.call_count == 3  # retry_attempts

@pytest.mark.asyncio
async def test_validate_and_store_success(sample_data, config, mock_db_manager):
    orchestrator = DataOrchestrator(config, mock_db_manager)

    success = await orchestrator._validate_and_store(
        sample_data,
        'AAPL',
        'yahoo',
        '1d'
    )

    assert success is True
    mock_db_manager.store_market_data.assert_called_once()
