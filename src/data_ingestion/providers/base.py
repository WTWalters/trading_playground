# File: src/data_ingestion/providers/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataProvider(ABC):
    """Abstract base class for market data providers."""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.name: str = self.__class__.__name__
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Args:
            symbol: The ticker symbol
            start_date: Start date (UTC)
            end_date: End date (UTC)
            interval: Data interval ('1m', '1h', '1d', etc.)

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        pass

    @abstractmethod
    async def stream_real_time_data(
        self,
        symbols: List[str]
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream real-time market data.

        Args:
            symbols: List of ticker symbols

        Yields:
            Dict containing market data updates
        """
        pass

    @abstractmethod
    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate if symbols are tradable.

        Args:
            symbols: List of symbols to validate

        Returns:
            Dict mapping symbols to their validity
        """
        pass
