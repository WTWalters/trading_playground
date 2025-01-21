import pytest
from datetime import datetime
from trading_playground.utils.time_utils import parse_timeframe, convert_timestamp
from trading_playground.utils.validation import validate_ohlcv_data


def test_parse_timeframe():
    value, unit = parse_timeframe('5m')
    assert value == 5
    assert unit == 'minutes'
    
    with pytest.raises(ValueError):
        parse_timeframe('invalid')


def test_convert_timestamp():
    dt = datetime.now()
    assert isinstance(convert_timestamp(dt), datetime)
    assert isinstance(convert_timestamp(dt.timestamp()), datetime)
    assert isinstance(convert_timestamp(dt.isoformat()), datetime)


def test_validate_ohlcv_data(sample_ohlcv_data):
    assert validate_ohlcv_data(sample_ohlcv_data)
    
    # Test with missing columns
    with pytest.raises(ValueError):
        validate_ohlcv_data(sample_ohlcv_data.drop('open', axis=1))
