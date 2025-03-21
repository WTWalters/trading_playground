"""
Market data service for TITAN Trading System.

Provides access to market data through a bridge to the
existing data manager component.
"""
import logging
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any

from django.conf import settings
from django.db import transaction

from .base_service import BaseService
from ..models.symbols import Symbol
from ..models.prices import Price

from src.database.manager import DatabaseManager
from src.config.db_config import DatabaseConfig


class MarketDataService(BaseService):
    """
    Service for market data operations.
    
    This service provides methods for:
    - Fetching market data from the database
    - Storing market data to the database
    - Converting between Django models and raw data structures
    - Synchronizing data between Django and TimescaleDB
    """
    
    def __init__(self):
        """Initialize the MarketDataService."""
        super().__init__()
        self.db_manager: Optional[DatabaseManager] = None
        
    async def _initialize_resources(self) -> None:
        """
        Initialize the database manager if not already initialized.
        
        This method will be called automatically by methods that require
        the database manager to be initialized.
        """
        if self.db_manager is None:
            self.logger.info("Initializing DatabaseManager")
            config = DatabaseConfig()
            self.db_manager = DatabaseManager(config)
            await self.db_manager.initialize()
    
    async def _cleanup_resources(self) -> None:
        """
        Clean up database resources.
        
        This method will be called by the cleanup method when the service
        is no longer needed.
        """
        if self.db_manager is not None:
            self.logger.info("Closing DatabaseManager")
            await self.db_manager.close()
            self.db_manager = None
    
    async def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch market data from the database.
        
        Args:
            symbol: Symbol identifier (e.g., 'AAPL')
            start_date: Start date for the data range
            end_date: End date for the data range
            timeframe: Data timeframe (e.g., '1d' for daily)
            source: Data source (optional)
            
        Returns:
            DataFrame with market data
        """
        await self._initialize_resources()
        
        self.logger.info(f"Fetching market data for {symbol} from {start_date} to {end_date}")
        return await self.db_manager.get_market_data(
            symbol, start_date, end_date, timeframe, source
        )
    
    @BaseService.sync_wrap
    async def get_market_data_sync(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Synchronous wrapper for get_market_data.
        
        Args:
            symbol: Symbol identifier (e.g., 'AAPL')
            start_date: Start date for the data range
            end_date: End date for the data range
            timeframe: Data timeframe (e.g., '1d' for daily)
            source: Data source (optional)
            
        Returns:
            DataFrame with market data
        """
        return await self.get_market_data(symbol, start_date, end_date, timeframe, source)
    
    async def store_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        source: str,
        timeframe: str
    ) -> None:
        """
        Store market data in the database.
        
        Args:
            data: DataFrame with market data
            symbol: Symbol identifier
            source: Data source
            timeframe: Data timeframe
        """
        await self._initialize_resources()
        
        self.logger.info(f"Storing {len(data)} records for {symbol}")
        await self.db_manager.store_market_data(data, symbol, source, timeframe)
    
    @BaseService.sync_wrap
    async def store_market_data_sync(
        self,
        data: pd.DataFrame,
        symbol: str,
        source: str,
        timeframe: str
    ) -> None:
        """
        Synchronous wrapper for store_market_data.
        
        Args:
            data: DataFrame with market data
            symbol: Symbol identifier
            source: Data source
            timeframe: Data timeframe
        """
        await self.store_market_data(data, symbol, source, timeframe)
    
    def sync_market_data_with_django(
        self,
        data: pd.DataFrame,
        symbol_ticker: str,
        source: str,
        timeframe: str
    ) -> List[Price]:
        """
        Synchronize market data from pandas DataFrame to Django models.
        
        This method creates or updates Price objects in the Django database
        based on the provided DataFrame.
        
        Args:
            data: DataFrame with market data
            symbol_ticker: Symbol ticker string
            source: Data source
            timeframe: Data timeframe
            
        Returns:
            List of created or updated Price objects
        """
        self.logger.info(f"Syncing {len(data)} records for {symbol_ticker} to Django models")
        
        # Get or create Symbol object
        symbol, created = Symbol.objects.get_or_create(
            ticker=symbol_ticker,
            defaults={'name': symbol_ticker}  # Simple default, should be updated later
        )
        
        if created:
            self.logger.info(f"Created new Symbol object for {symbol_ticker}")
        
        # Process the DataFrame in batches to avoid memory issues
        prices = []
        batch_size = 1000
        
        with transaction.atomic():
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                for idx, row in batch.iterrows():
                    # Convert pandas.Timestamp to datetime if needed
                    if hasattr(idx, 'to_pydatetime'):
                        timestamp = idx.to_pydatetime()
                    else:
                        timestamp = idx
                    
                    # Try to get existing Price or create new one
                    try:
                        price = Price.objects.get(
                            symbol=symbol,
                            timestamp=timestamp,
                            source=source,
                            timeframe=timeframe
                        )
                    except Price.DoesNotExist:
                        price = Price(
                            symbol=symbol,
                            timestamp=timestamp,
                            source=source,
                            timeframe=timeframe
                        )
                    
                    # Update fields
                    price.open = row['open']
                    price.high = row['high']
                    price.low = row['low']
                    price.close = row['close']
                    price.volume = row['volume']
                    
                    # Handle adjusted close if present
                    if 'adjusted_close' in row:
                        price.adjusted_close = row['adjusted_close']
                    
                    price.save()
                    prices.append(price)
        
        self.logger.info(f"Synced {len(prices)} Price objects for {symbol_ticker}")
        return prices
    
    def convert_django_to_dataframe(self, prices: List[Price]) -> pd.DataFrame:
        """
        Convert Django Price objects to pandas DataFrame.
        
        Args:
            prices: List of Price objects
            
        Returns:
            DataFrame with market data
        """
        if not prices:
            return pd.DataFrame()
        
        # Extract data from Price objects
        data = {
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'adjusted_close': []
        }
        
        for price in prices:
            data['timestamp'].append(price.timestamp)
            data['open'].append(float(price.open))
            data['high'].append(float(price.high))
            data['low'].append(float(price.low))
            data['close'].append(float(price.close))
            data['volume'].append(int(price.volume))
            data['adjusted_close'].append(
                float(price.adjusted_close) if price.adjusted_close else None
            )
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    @BaseService.sync_wrap
    async def get_latest_dates_sync(self) -> Dict[Tuple[str, str], datetime]:
        """
        Synchronous wrapper for getting the latest dates for each symbol/timeframe.
        
        Returns:
            Dictionary mapping (symbol, timeframe) tuples to latest dates
        """
        await self._initialize_resources()
        return await self.db_manager.get_latest_dates()
