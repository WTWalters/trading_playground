from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd


class DataProvider(ABC):
    """Base class for all data providers."""
    
    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.
        
        Args:
            symbol: The trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        pass

    @abstractmethod
    async def get_latest_price(
        self,
        symbol: str
    ) -> float:
        """Get the latest price for a symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Latest price as float
        """
        pass

    async def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if the timeframe is supported.
        
        Args:
            timeframe: Data timeframe to validate
            
        Returns:
            True if timeframe is valid, False otherwise
        """
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        return timeframe in valid_timeframes
