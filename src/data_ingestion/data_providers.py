from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from datetime import datetime
import aiohttp
import asyncio
import logging
from polygon import RESTClient

class DataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol
        """
        pass

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Yahoo Finance interval"""
        conversions = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d'
        }
        if timeframe not in conversions:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return conversions[timeframe]
        
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        try:
            interval = self._convert_timeframe(timeframe)
            ticker = yf.Ticker(symbol)
            
            df = await asyncio.to_thread(
                ticker.history,
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            df.columns = [col.lower() for col in df.columns]
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns for {symbol}")
                return pd.DataFrame()
                
            return df[required_cols]
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

class PolygonProvider(DataProvider):
    """Polygon.io data provider implementation"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.logger = logging.getLogger(__name__)
        
    def _convert_timeframe(self, timeframe: str) -> tuple[str, str]:
        """Convert standard timeframe to Polygon multiplier/timespan"""
        conversions = {
            '1m': ('1', 'minute'),
            '5m': ('5', 'minute'),
            '15m': ('15', 'minute'),
            '30m': ('30', 'minute'),
            '1h': ('1', 'hour'),
            '1d': ('1', 'day')
        }
        if timeframe not in conversions:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return conversions[timeframe]
        
    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical data from Polygon.io"""
        try:
            multiplier, timespan = self._convert_timeframe(timeframe)
            
            aggs = []
            for agg in self.client.list_aggs(
                symbol,
                multiplier,
                timespan,
                start_date,
                end_date or datetime.now(),
                limit=50000
            ):
                aggs.append({
                    'timestamp': agg.timestamp,
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
                
            if not aggs:
                return pd.DataFrame()
                
            df = pd.DataFrame(aggs)
            df.set_index('timestamp', inplace=True)
            df.index = pd.to_datetime(df.index, unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()