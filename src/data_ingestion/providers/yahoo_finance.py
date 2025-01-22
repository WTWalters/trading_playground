# File: src/data_ingestion/providers/yahoo_finance.py

import yfinance as yf
from typing import Dict, List, Optional, AsyncGenerator
import pandas as pd
from datetime import datetime, timezone
import asyncio
import logging
from .base import DataProvider

logger = logging.getLogger(__name__)

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider implementation."""

    INTERVAL_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo"
    }

    def initialize(self) -> None:
        """Initialize the Yahoo Finance provider."""
        self.session = yf.Ticker("")  # Empty ticker to initialize session
        logger.info("Yahoo Finance provider initialized")

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        try:
            # Validate interval
            if interval not in self.INTERVAL_MAP:
                raise ValueError(f"Invalid interval: {interval}")

            # Create ticker object
            ticker = yf.Ticker(symbol)

            # Fetch data (runs in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=self.INTERVAL_MAP[interval]
                )
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Rename columns to match our schema
            df.index.name = 'time'
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            # Add symbol column
            df['symbol'] = symbol

            # Ensure UTC timezone
            df.index = df.index.tz_localize('UTC')

            # Select and reorder columns
            columns = ['symbol', 'open', 'high', 'low', 'close', 'volume']
            df = df[columns]

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    async def stream_real_time_data(
        self,
        symbols: List[str]
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream real-time data from Yahoo Finance.
        Note: Yahoo Finance doesn't provide true streaming - this is a simulation
        using repeated API calls.
        """
        while True:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(
                        None,
                        lambda: ticker.fast_info
                    )

                    if data:
                        yield {
                            'symbol': symbol,
                            'time': datetime.now(timezone.utc),
                            'last_price': data.get('lastPrice', None),
                            'last_volume': data.get('lastVolume', None),
                            'bid': data.get('bid', None),
                            'ask': data.get('ask', None)
                        }

                except Exception as e:
                    logger.error(f"Error streaming data for {symbol}: {str(e)}")
                    continue

            await asyncio.sleep(1)  # Rate limiting

    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate symbols against Yahoo Finance."""
        results = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(None, lambda: ticker.info)
                results[symbol] = 'regularMarketPrice' in info
            except Exception as e:
                logger.error(f"Error validating symbol {symbol}: {str(e)}")
                results[symbol] = False

        return results
