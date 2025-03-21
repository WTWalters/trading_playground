"""
Yahoo Finance data provider implementation.

This module provides a Yahoo Finance data provider implementation
using the yfinance library.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import logging
import time
from .base import DataProvider


class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider implementation.
    
    This provider uses the yfinance library to fetch data
    from Yahoo Finance.
    """
    
    def __init__(self, rate_limit_pause: float = 0.2):
        """
        Initialize the Yahoo Finance provider.
        
        Args:
            rate_limit_pause: Pause between requests to avoid rate limiting
        """
        self.logger = logging.getLogger(__name__)
        self.rate_limit_pause = rate_limit_pause
        self._last_request_time = 0
        
    @property
    def provider_id(self) -> str:
        return "yahoo"
        
    @property
    def supports_intraday(self) -> bool:
        return True
        
    @property
    def max_lookback_days(self) -> Optional[int]:
        # Yahoo Finance typically has decades of daily data
        # and about 60 days of 1-minute data
        return None  # Unlimited for daily data
        
    def _convert_timeframe(self, timeframe: str) -> str:
        """
        Convert standard timeframe to Yahoo Finance interval.
        
        Args:
            timeframe: Standard timeframe string
            
        Returns:
            Yahoo Finance interval string
            
        Raises:
            ValueError: If the timeframe is not supported
        """
        conversions = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            '1h': '60m',
            '90m': '90m',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo',
            '3mo': '3mo'
        }
        
        if timeframe not in conversions:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported timeframes: {list(conversions.keys())}"
            )
            
        return conversions[timeframe]
        
    async def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits by adding delays between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self.rate_limit_pause:
            await asyncio.sleep(self.rate_limit_pause - elapsed)
            
        self._last_request_time = time.time()
        
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date (default: now)
            timeframe: Time interval ('1m', '5m', '15m', '1h', '1d', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            await self._respect_rate_limit()
            
            interval = self._convert_timeframe(timeframe)
            ticker = yf.Ticker(symbol)
            
            # Use current time if end_date not provided
            if end_date is None:
                end_date = datetime.now()
                
            # For intraday data, Yahoo has limitations
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                # Yahoo only provides ~60 days of 1-minute data
                # and limits other intraday ranges too
                max_days = {
                    '1m': 7,   # Actually 7 days
                    '2m': 60,  # Actually 60 days
                    '5m': 60,  # Actually 60 days
                    '15m': 60, # Actually 60 days
                    '30m': 60, # Actually 60 days
                    '60m': 730, # Actually 730 days (2 years)
                    '90m': 60   # Actually 60 days
                }
                
                max_lookback = max_days.get(interval, 7)
                
                # Limit start_date to prevent errors
                earliest_allowed = end_date - timedelta(days=max_lookback)
                if start_date < earliest_allowed:
                    self.logger.warning(
                        f"Yahoo Finance limits {interval} data to {max_lookback} days. "
                        f"Adjusting start date from {start_date} to {earliest_allowed}"
                    )
                    start_date = earliest_allowed
            
            # Execute the request in a separate thread to prevent blocking
            self.logger.info(f"Fetching {symbol} from {start_date} to {end_date} ({interval})")
            df = await asyncio.to_thread(
                ticker.history,
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Ensure the DataFrame has the correct structure
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                
            # Yahoo returns columns with capitalized names - standardize them
            df.columns = [col.lower() for col in df.columns]
            
            # Extract the required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                self.logger.error(f"Missing required columns for {symbol}: {missing}")
                return pd.DataFrame()
                
            # Create a new DataFrame with just the required columns
            result_df = df[required_cols].copy()
            
            # Ensure the index is a DatetimeIndex
            if not isinstance(result_df.index, pd.DatetimeIndex):
                self.logger.warning(f"Converting index to DatetimeIndex for {symbol}")
                result_df.index = pd.to_datetime(result_df.index)
                
            # Check for invalid data
            invalid_data = (
                (result_df['high'] < result_df['low']) | 
                (result_df['high'] < result_df['open']) | 
                (result_df['high'] < result_df['close']) |
                (result_df['low'] > result_df['open']) | 
                (result_df['low'] > result_df['close']) |
                (result_df['volume'] < 0)
            )
            
            invalid_count = invalid_data.sum()
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} invalid data points for {symbol}")
                
                # Drop invalid rows
                result_df = result_df[~invalid_data]
                
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def is_symbol_valid(self, symbol: str) -> bool:
        """
        Check if a symbol is valid on Yahoo Finance.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if the symbol is valid
        """
        try:
            await self._respect_rate_limit()
            
            # Use a minimal history request to check validity
            ticker = yf.Ticker(symbol)
            
            # Execute in a thread to prevent blocking
            info = await asyncio.to_thread(
                lambda: ticker.info
            )
            
            # If certain essential keys are missing, the symbol is invalid
            if 'symbol' not in info or info.get('regularMarketPrice') is None:
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"Symbol validation error for {symbol}: {str(e)}")
            return False
    
    async def get_supported_timeframes(self) -> List[str]:
        """
        Get timeframes supported by Yahoo Finance.
        
        Returns:
            List of supported timeframe strings
        """
        return [
            '1m', '2m', '5m', '15m', '30m', 
            '60m', '1h', '90m', 
            '1d', '5d', '1wk', '1mo', '3mo'
        ]
    
    async def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limit information for Yahoo Finance.
        
        Returns:
            Dictionary with rate limit details
        """
        # Yahoo Finance doesn't publish official rate limits
        # These are conservative estimates
        return {
            "max_requests_per_second": 2,
            "max_requests_per_hour": 1000,
            "recommended_pause": self.rate_limit_pause,
            "notes": "Yahoo Finance does not publish official rate limits. " +
                     "These are conservative estimates to avoid IP blocks."
        }
    
    async def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Get additional metadata for a symbol.
        
        Args:
            symbol: Symbol to get metadata for
            
        Returns:
            Dictionary with symbol metadata
        """
        try:
            await self._respect_rate_limit()
            
            ticker = yf.Ticker(symbol)
            
            # Execute in a thread to prevent blocking
            info = await asyncio.to_thread(
                lambda: ticker.info
            )
            
            # Extract useful metadata
            metadata = {
                "symbol": info.get("symbol"),
                "name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency"),
                "country": info.get("country"),
                "asset_type": self._determine_asset_type(info)
            }
            
            return {k: v for k, v in metadata.items() if v is not None}
            
        except Exception as e:
            self.logger.error(f"Error fetching metadata for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}
    
    def _determine_asset_type(self, info: Dict[str, Any]) -> str:
        """
        Determine the asset type from Yahoo Finance info.
        
        Args:
            info: Yahoo Finance info dictionary
            
        Returns:
            Asset type string
        """
        # Try to determine asset type
        if info.get("quoteType") == "EQUITY":
            return "stock"
        elif info.get("quoteType") == "ETF":
            return "etf"
        elif info.get("quoteType") == "MUTUALFUND":
            return "mutual_fund"
        elif info.get("quoteType") == "CRYPTOCURRENCY":
            return "crypto"
        elif info.get("quoteType") == "CURRENCY":
            return "forex"
        elif info.get("quoteType") == "FUTURE":
            return "future"
        elif info.get("quoteType") == "INDEX":
            return "index"
        else:
            return "unknown"
