from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .data_provider import DataProvider
from ..utils.validation import validate_ohlcv_data


class AlpacaDataProvider(DataProvider):
    """Alpaca Markets data provider implementation."""
    
    def __init__(self, api_key: str, api_secret: str):
        """Initialize Alpaca client.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
        """
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self._timeframe_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame.Minute * 5,
            '15m': TimeFrame.Minute * 15,
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch historical data from Alpaca."""
        if not await self.validate_timeframe(timeframe):
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe_map[timeframe],
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df.reset_index()
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'trade_count': 'trades',
            'vwap': 'vwap'
        })
        
        validate_ohlcv_data(df)
        return df

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols from Alpaca."""
        if not await self.validate_timeframe(timeframe):
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self._timeframe_map[timeframe],
            start=start_date,
            end=end_date
        )

        bars = self.client.get_stock_bars(request)
        
        # Split multi-symbol response into individual dataframes
        result = {}
        for symbol in symbols:
            symbol_data = bars.df.xs(symbol, level=0).reset_index()
            symbol_data = symbol_data.rename(columns={
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'trade_count': 'trades',
                'vwap': 'vwap'
            })
            validate_ohlcv_data(symbol_data)
            result[symbol] = symbol_data
            
        return result

    async def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol from Alpaca."""
        end = datetime.now()
        start = end - timedelta(minutes=1)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end
        )

        bars = self.client.get_stock_bars(request)
        if bars.df.empty:
            raise ValueError(f"No recent data available for {symbol}")
            
        return bars.df.iloc[-1].close

    async def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if the timeframe is supported by Alpaca."""
        return timeframe in self._timeframe_map
