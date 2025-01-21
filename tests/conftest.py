import pytest
from datetime import datetime, timedelta
import pandas as pd


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=10),
        end=datetime.now(),
        freq='1D'
    )
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * len(dates),
        'high': [105.0] * len(dates),
        'low': [95.0] * len(dates),
        'close': [102.0] * len(dates),
        'volume': [1000.0] * len(dates)
    })
    
    return df
