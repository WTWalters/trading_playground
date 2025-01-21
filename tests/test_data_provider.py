import pytest
from datetime import datetime, timedelta
from trading_playground.data_ingestion.data_provider import DataProvider


class MockDataProvider(DataProvider):
    """Mock data provider for testing."""
    
    async def fetch_historical_data(self, symbol, start_date, end_date, timeframe):
        return self.sample_ohlcv_data
        
    async def fetch_multiple_symbols(self, symbols, start_date, end_date, timeframe):
        return {symbol: self.sample_ohlcv_data for symbol in symbols}
        
    async def get_latest_price(self, symbol):
        return 100.0


@pytest.mark.asyncio
async def test_validate_timeframe():
    provider = MockDataProvider()
    assert await provider.validate_timeframe('1m')
    assert await provider.validate_timeframe('1h')
    assert not await provider.validate_timeframe('invalid')
