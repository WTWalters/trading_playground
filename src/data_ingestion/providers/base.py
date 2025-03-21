"""
Base classes for market data providers.

This module defines the interface that all data providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime


class DataProvider(ABC):
    """
    Abstract base class for market data providers.
    
    All data providers must implement this interface to ensure
    consistent behavior across different data sources.
    """
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """
        Get the unique identifier for this provider.
        
        Returns:
            String identifier for the provider
        """
        pass
        
    @property
    @abstractmethod
    def supports_intraday(self) -> bool:
        """
        Check if this provider supports intraday data.
        
        Returns:
            True if intraday data is supported, False otherwise
        """
        pass
        
    @property
    @abstractmethod
    def max_lookback_days(self) -> Optional[int]:
        """
        Get the maximum lookback period in days.
        
        Returns:
            Maximum number of days or None if unlimited
        """
        pass
        
    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol.
        
        Args:
            symbol: Symbol to fetch data for
            start_date: Start date for the data
            end_date: End date for the data (default: now)
            timeframe: Timeframe as string ('1m', '5m', '15m', '1h', '1d', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
        
    @abstractmethod
    async def is_symbol_valid(self, symbol: str) -> bool:
        """
        Check if a symbol is valid for this provider.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if the symbol is valid, False otherwise
        """
        pass
        
    @abstractmethod
    async def get_supported_timeframes(self) -> List[str]:
        """
        Get a list of timeframes supported by this provider.
        
        Returns:
            List of timeframe strings
        """
        pass
        
    @abstractmethod
    async def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limit information for this provider.
        
        Returns:
            Dictionary with rate limit details
        """
        pass
