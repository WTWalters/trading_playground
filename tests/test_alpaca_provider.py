import pytest
from datetime import datetime, timedelta
from trading_playground.data_ingestion.alpaca_provider import AlpacaDataProvider

# Note: These tests require valid Alpaca credentials
# Use environment variables or config files in practice

@pytest.fixture
def alpaca_provider():
    return AlpacaDataProvider(
        api_key='YOUR_API_KEY',
        api_secret='YOUR_API_SECRET'
    )

@pytest.mark.asyncio
async def test_fetch_historical_data(alpaca_provider):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    df = await alpaca_provider.fetch_historical_data(
        symbol='AAPL',
        start_date=start_date,
        end_date=end_date,
        timeframe='1d'
    )
    
    assert not df.empty
    assert all(col in df.columns for col in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])

@pytest.mark.asyncio
async def test_fetch_multiple_symbols(alpaca_provider):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    result = await alpaca_provider.fetch_multiple_symbols(
        symbols=['AAPL', 'MSFT'],
        start_date=start_date,
        end_date=end_date,
        timeframe='1d'
    )
    
    assert len(result) == 2
    assert 'AAPL' in result
    assert 'MSFT' in result
    assert not result['AAPL'].empty
    assert not result['MSFT'].empty

@pytest.mark.asyncio
async def test_get_latest_price(alpaca_provider):
    price = await alpaca_provider.get_latest_price('AAPL')
    assert isinstance(price, float)
    assert price > 0
