import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from data_ingestion.data_providers import YahooFinanceProvider, PolygonProvider

@pytest.fixture
def yahoo_provider():
    return YahooFinanceProvider()

@pytest.fixture
def polygon_provider():
    return PolygonProvider("fake_api_key")

def test_yahoo_timeframe_conversion(yahoo_provider):
    """Test timeframe conversion for Yahoo Finance"""
    assert yahoo_provider._convert_timeframe('1d') == '1d'
    assert yahoo_provider._convert_timeframe('1h') == '1h'
    with pytest.raises(ValueError):
        yahoo_provider._convert_timeframe('invalid')

def test_polygon_timeframe_conversion(polygon_provider):
    """Test timeframe conversion for Polygon"""
    multiplier, timespan = polygon_provider._convert_timeframe('1d')
    assert multiplier == '1'
    assert timespan == 'day'
    
    with pytest.raises(ValueError):
        polygon_provider._convert_timeframe('invalid')

@pytest.mark.asyncio
async def test_yahoo_fetch_historical(yahoo_provider):
    """Test Yahoo Finance data fetching"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 10)
    
    mock_data = pd.DataFrame({
        'Open': [100] * 5,
        'High': [101] * 5,
        'Low': [99] * 5,
        'Close': [100.5] * 5,
        'Volume': [1000] * 5
    })
    
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        
        df = await yahoo_provider.fetch_historical_data(
            'AAPL',
            start_date,
            end_date,
            '1d'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

@pytest.mark.asyncio
async def test_polygon_fetch_historical(polygon_provider):
    """Test Polygon.io data fetching"""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 10)
    
    # Mock Polygon Agg object
    class MockAgg:
        def __init__(self):
            self.timestamp = 1704067200000  # 2024-01-01
            self.open = 100
            self.high = 101
            self.low = 99
            self.close = 100.5
            self.volume = 1000
    
    with patch.object(polygon_provider.client, 'list_aggs') as mock_list_aggs:
        mock_list_aggs.return_value = [MockAgg() for _ in range(5)]
        
        df = await polygon_provider.fetch_historical_data(
            'AAPL',
            start_date,
            end_date,
            '1d'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

@pytest.mark.asyncio
async def test_yahoo_error_handling(yahoo_provider):
    """Test Yahoo Finance error handling"""
    with patch('yfinance.Ticker') as mock_ticker:
        mock_ticker.return_value.history.side_effect = Exception("API Error")
        
        df = await yahoo_provider.fetch_historical_data(
            'AAPL',
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
            '1d'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty

@pytest.mark.asyncio
async def test_polygon_error_handling(polygon_provider):
    """Test Polygon.io error handling"""
    with patch.object(polygon_provider.client, 'list_aggs') as mock_list_aggs:
        mock_list_aggs.side_effect = Exception("API Error")
        
        df = await polygon_provider.fetch_historical_data(
            'AAPL',
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
            '1d'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty