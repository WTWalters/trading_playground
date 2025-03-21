"""
Polygon.io data provider implementation.

This module provides a Polygon.io data provider implementation
using the polygon-api-client library.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import asyncio
import logging
import time
from polygon import RESTClient
from .base import DataProvider


class PolygonProvider(DataProvider):
    """
    Polygon.io data provider implementation.
    
    This provider uses the polygon-api-client library to fetch data
    from Polygon.io.
    """
    
    def __init__(
        self, 
        api_key: str,
        rate_limit_pause: float = 0.12,  # Polygon allows up to 5 req/sec on free plan
        premium: bool = False
    ):
        """
        Initialize the Polygon provider.
        
        Args:
            api_key: Polygon API key
            rate_limit_pause: Pause between requests to avoid rate limiting
            premium: Whether this is a premium/paid account
        """
        self.api_key = api_key
        self.premium = premium
        self.client = RESTClient(api_key)
        self.rate_limit_pause = rate_limit_pause
        self._last_request_time = 0
        self.logger = logging.getLogger(__name__)
        
    @property
    def provider_id(self) -> str:
        return "polygon"
        
    @property
    def supports_intraday(self) -> bool:
        return True
        
    @property
    def max_lookback_days(self) -> Optional[int]:
        # Polygon.io has different lookback limitations based on plan
        # Free tier has 2 years of data
        return 365 * 2 if not self.premium else None
        
    def _convert_timeframe(self, timeframe: str) -> Tuple[str, str]:
        """
        Convert standard timeframe to Polygon multiplier/timespan.
        
        Args:
            timeframe: Standard timeframe string
            
        Returns:
            Tuple of (multiplier, timespan)
            
        Raises:
            ValueError: If the timeframe is not supported
        """
        conversions = {
            '1m': ('1', 'minute'),
            '5m': ('5', 'minute'),
            '15m': ('15', 'minute'),
            '30m': ('30', 'minute'),
            '1h': ('1', 'hour'),
            '4h': ('4', 'hour'),
            '1d': ('1', 'day'),
            '1wk': ('1', 'week'),
            '1mo': ('1', 'month')
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
        Fetch historical data from Polygon.io.
        
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
            
            # Convert timeframe
            multiplier, timespan = self._convert_timeframe(timeframe)
            
            # Use current time if end_date not provided
            if end_date is None:
                end_date = datetime.now()
                
            # Polygon API parameters accept date objects, not datetimes
            # And it's exclusive of the end date, so add one day
            start = start_date.date()
            end = end_date.date() + timedelta(days=1)
            
            # Check if we're exceeding max lookback
            if not self.premium and self.max_lookback_days is not None:
                earliest_allowed = datetime.now().date() - timedelta(days=self.max_lookback_days)
                if start < earliest_allowed:
                    self.logger.warning(
                        f"Polygon free tier limits historical data to {self.max_lookback_days} days. "
                        f"Adjusting start date from {start} to {earliest_allowed}"
                    )
                    start = earliest_allowed
            
            self.logger.info(
                f"Fetching {symbol} from {start} to {end} "
                f"({multiplier} {timespan})"
            )
            
            # Fetch data from Polygon
            aggs = []
            
            # Execute in a thread pool to avoid blocking the event loop
            def fetch_aggs():
                return list(self.client.list_aggs(
                    ticker=symbol,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start,
                    to=end,
                    limit=50000  # Maximum allowed by Polygon
                ))
            
            aggs_list = await asyncio.to_thread(fetch_aggs)
            
            # Convert to list of dictionaries
            for agg in aggs_list:
                aggs.append({
                    'timestamp': agg.timestamp,
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            # Create DataFrame from aggregates
            if not aggs:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(aggs)
            
            # Convert timestamp from milliseconds to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure all required columns are present
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                self.logger.error(f"Missing required columns for {symbol}: {missing}")
                return pd.DataFrame()
            
            # Keep only required columns in a standard order
            df = df[required_cols]
            
            # Ensure numeric data types
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for and handle NaN values
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} NaN values in {symbol} data")
                # Fill missing values where possible
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def is_symbol_valid(self, symbol: str) -> bool:
        """
        Check if a symbol is valid on Polygon.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if the symbol is valid, False otherwise
        """
        try:
            await self._respect_rate_limit()
            
            # Try to get ticker details to check validity
            def check_ticker():
                try:
                    ticker_details = self.client.get_ticker_details(symbol)
                    return ticker_details is not None
                except Exception:
                    return False
            
            is_valid = await asyncio.to_thread(check_ticker)
            return is_valid
            
        except Exception as e:
            self.logger.debug(f"Symbol validation error for {symbol}: {str(e)}")
            return False
    
    async def get_supported_timeframes(self) -> List[str]:
        """
        Get timeframes supported by Polygon.
        
        Returns:
            List of supported timeframe strings
        """
        return [
            '1m', '5m', '15m', '30m', 
            '1h', '4h', 
            '1d', '1wk', '1mo'
        ]
    
    async def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limit information for Polygon.
        
        Returns:
            Dictionary with rate limit details
        """
        if self.premium:
            # Premium tier has higher limits
            return {
                "max_requests_per_second": 100,
                "max_requests_per_minute": 6000,
                "recommended_pause": 0.01,
                "notes": "Premium tier with higher rate limits"
            }
        else:
            # Free tier limits
            return {
                "max_requests_per_second": 5,
                "max_requests_per_minute": 200,
                "recommended_pause": self.rate_limit_pause,
                "notes": "Free tier with limited rate and historical data"
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
            
            # Fetch ticker details in a thread
            def get_details():
                return self.client.get_ticker_details(symbol)
            
            details = await asyncio.to_thread(get_details)
            
            if not details:
                return {"symbol": symbol, "error": "No details found"}
            
            # Extract useful metadata
            metadata = {
                "symbol": details.ticker,
                "name": details.name,
                "market": details.market,
                "locale": details.locale,
                "currency": details.currency_name,
                "asset_type": self._determine_asset_type(details.market, details.type),
                "active": details.active,
                "primary_exchange": details.primary_exchange,
                "cik": details.cik,
                "composite_figi": details.composite_figi,
            }
            
            return {k: v for k, v in metadata.items() if v is not None}
            
        except Exception as e:
            self.logger.error(f"Error fetching metadata for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}
    
    def _determine_asset_type(self, market: str, type_: Optional[str] = None) -> str:
        """
        Determine asset type from Polygon's market and type fields.
        
        Args:
            market: Market identifier from Polygon
            type_: Type identifier from Polygon
            
        Returns:
            Asset type string
        """
        # Map Polygon markets to asset types
        market_map = {
            "stocks": "stock",
            "fx": "forex",
            "crypto": "crypto",
            "otc": "otc",
            "indices": "index"
        }
        
        # If we have a market match, use it
        if market and market.lower() in market_map:
            return market_map[market.lower()]
        
        # Try to use the type field
        if type_:
            type_map = {
                "CS": "stock",           # Common Stock
                "ETF": "etf",            # Exchange Traded Fund
                "FUND": "mutual_fund",   # Mutual Fund
                "ADR": "adr",            # American Depositary Receipt
                "INDEX": "index",        # Index
                "FOREX": "forex",        # Forex
                "CRYPTO": "crypto"       # Cryptocurrency
            }
            
            if type_.upper() in type_map:
                return type_map[type_.upper()]
        
        # Default when we can't determine
        return "unknown"
