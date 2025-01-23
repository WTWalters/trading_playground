import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from data_ingestion.orchestrator import DataOrchestrator, DataIngestionConfig
from data_ingestion.validation import DataValidator

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

@pytest.mark.asyncio
async def test_ingest_historical_data(sample_data, config, mock_db_manager):
    orchestrator = DataOrchestrator(config, mock_db_manager)
    mock_provider = Mock()
    mock_provider.fetch_historical_data = AsyncMock(return_value=sample_data)
    orchestrator.providers = {'yahoo': mock_provider}

    results = await orchestrator.ingest_historical_data()

    assert len(results['successful']) > 0
    assert len(results['failed']) == 0
    assert results['total_records'] > 0

@pytest.mark.asyncio
async def test_update_market_data(sample_data, config, mock_db_manager):
    orchestrator = DataOrchestrator(config, mock_db_manager)
    mock_provider = Mock()
    mock_provider.fetch_historical_data = AsyncMock(return_value=sample_data)
    orchestrator.providers = {'yahoo': mock_provider}

    results = await orchestrator.update_market_data()

    assert len(results['updated']) > 0
    assert len(results['failed']) == 0
    assert results['total_records'] > 0

@pytest.mark.asyncio
async def test_error_handling(config, mock_db_manager):
    orchestrator = DataOrchestrator(config, mock_db_manager)
    mock_provider = Mock()
    mock_provider.fetch_historical_data = AsyncMock(side_effect=Exception("API Error"))
    orchestrator.providers = {'yahoo': mock_provider}

    results = await orchestrator.ingest_historical_data()

    assert len(results['successful']) == 0
    assert len(results['failed']) > 0
    assert results['total_records'] == 0


def test_validate_market_data(validator, valid_data):
    # Add debug logging
    result = validator.validate_market_data(valid_data)
    print("Validation result:", result)
    print("Price consistency:", validator._validate_price_consistency(valid_data))
    print("Timestamps:", validator._validate_timestamps(valid_data))
    print("Gaps:", validator._validate_gaps(valid_data))
    print("Outliers:", validator._validate_outliers(valid_data))

    assert validator.validate_market_data(valid_data) is True
