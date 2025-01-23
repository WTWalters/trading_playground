import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_ingestion.validation import DataValidator
from trading_playground.data_ingestion.orchestrator import DataOrchestrator, DataIngestionConfig

@pytest.fixture
def validator():
    validation_rules = {
        'null_threshold': 0.01,
        'price_z_threshold': 4.0,
        'volume_z_threshold': 5.0,
        'max_time_gap': pd.Timedelta(days=5)
    }
    return DataValidator(validation_rules)

@pytest.fixture
def valid_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    return pd.DataFrame({
        'open': np.random.randn(len(dates)) + 100,
        'high': np.random.randn(len(dates)) + 101,
        'low': np.random.randn(len(dates)) + 99,
        'close': np.random.randn(len(dates)) + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

def test_validate_basic_structure(validator, valid_data):
    assert validator._validate_basic_structure(valid_data) is True

    # Test missing columns
    invalid_data = valid_data.drop('volume', axis=1)
    assert validator._validate_basic_structure(invalid_data) is False

    # Test null values
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = None
    assert validator._validate_basic_structure(invalid_data) is True  # Single null allowed

    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[:5], 'close'] = None  # 50% nulls
    assert validator._validate_basic_structure(invalid_data) is False

def test_validate_price_consistency(validator, valid_data):
    assert validator._validate_price_consistency(valid_data) is True

    # Test invalid high price
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = 98  # Below low
    assert validator._validate_price_consistency(invalid_data) is False

    # Test invalid low price
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'low'] = 102  # Above high
    assert validator._validate_price_consistency(invalid_data) is False

    # Test negative volume
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'volume'] = -1000
    assert validator._validate_price_consistency(invalid_data) is False

def test_validate_timestamps(validator, valid_data):
    assert validator._validate_timestamps(valid_data) is True

    # Test unordered timestamps
    invalid_data = valid_data.copy()
    invalid_data.index = invalid_data.index[::-1]  # Reverse order
    assert validator._validate_timestamps(invalid_data) is False

    # Test duplicate timestamps
    invalid_data = valid_data.copy()
    invalid_data.index = [valid_data.index[0]] * len(valid_data)
    assert validator._validate_timestamps(invalid_data) is False

def test_validate_gaps(validator, valid_data):
    assert validator._validate_gaps(valid_data) is True

    # Test data with gaps
    dates = list(valid_data.index[:5]) + list(valid_data.index[-3:])
    invalid_data = valid_data.loc[dates]
    assert validator._validate_gaps(invalid_data) is False

def test_validate_outliers(validator, valid_data):
    assert validator._validate_outliers(valid_data) is True

    # Test price outliers
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = 1000  # Extreme value
    assert validator._validate_outliers(invalid_data) is False

    # Test volume outliers
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'volume'] = 1000000  # Extreme value
    assert validator._validate_outliers(invalid_data) is False

def test_validate_market_data_integration(validator, valid_data):
    # Full integration test
    assert validator.validate_market_data(valid_data) is True

    # Test with multiple issues
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = 98  # Price inconsistency
    invalid_data.loc[invalid_data.index[1], 'volume'] = -1000  # Invalid volume
    invalid_data.loc[invalid_data.index[2], 'close'] = None  # Null value

    assert validator.validate_market_data(invalid_data) is False

def test_validate_corporate_actions(validator, valid_data):
    actions = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', end='2024-01-10', freq='2D'),
        'action': ['split', 'dividend'] * 3,
        'value': [2.0, 0.5] * 3
    })

    assert validator.validate_corporate_actions(valid_data, actions) is True
