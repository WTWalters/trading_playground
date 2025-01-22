# File: src/data_ingestion/providers/polygon.py

from typing import Dict, List, Optional, AsyncGenerator
import pandas as pd
from datetime import datetime, timezone
import aiohttp
import asyncio
import logging
from .base import DataProvider

logger = logging.getLogger(__name__)

class PolygonProvider(DataProvider):
    """Polygon.io data provider implementation."""

    BASE_URL = "https://api.polygon.io"

    INTERVAL_MAP = {
        "1m": "minute",
        "1h": "hour",
        "1d": "day",
        "1wk": "week",
        "1mo": "month"
    }

    def initialize(self) -> None:
        """Initialize the Polygon provider."""
        self.api_key = self.config.get('api_key')
        if not self.api_key:
            raise ValueError("Polygon API key not provided")

        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        logger.info("Polygon provider initialized")

    async def close(self) -> None:
        """Close the session."""
        if hasattr(self, 'session'):
            await self.session.close()

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Polygon."""
        try:
            if interval not in self.INTERVAL_MAP:
                raise ValueError(f"Invalid interval: {interval}")

            # Convert dates to timestamps
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            # Build URL
            url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/{self.INTERVAL_MAP[interval]}/{start_ts}/{end_ts}"

            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"API returned status {response.status}")

                data = await response.json()

                if data['status'] != 'OK':
                    raise Exception(f"API returned status {data['status']}")

                if not data.get('results'):
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(data['results'])

                # Rename columns to match our schema
                df.rename(columns={
                    't': 'time',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                }, inplace=True)

                # Convert timestamp to datetime
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)

                # Add symbol column
                df['symbol'] = symbol

                # Select and reorder columns
                columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                df = df[columns]

                return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    async def stream_real_time_data(
        self,
        symbols: List[str]
    ) -> AsyncGenerator[Dict, None]:
        """Stream real-time data from Polygon websocket."""
        ws_url = f"wss://socket.polygon.io/stocks"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Authenticate
                    await ws.send_json({
                        "action": "auth",
                        "params": self.api_key
                    })

                    # Subscribe to symbols
                    await ws.send_json({
                        "action": "subscribe",
                        "params": [f"T.{symbol}" for symbol in symbols]
                    })

                    while True:
                        msg = await ws.receive_json()

                        if msg[0]['ev'] == 'T':  # Trade event
                            trade = msg[0]
                            yield {
                                'symbol': trade['sym'],
                                'time': pd.Timestamp(trade['t'], unit='ms', tz='UTC'),
                                'price': trade['p'],
                                'volume': trade['s'],
                                'exchange': trade['x']
                            }

        except Exception as e:
            logger.error(f"Websocket error: {str(e)}")
            raise

    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate symbols against Polygon."""
        results = {}

        for symbol in symbols:
            try:
                url = f"{self.BASE_URL}/v3/reference/tickers/{symbol}"

                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[symbol] = data['status'] == 'OK'
                    else:
                        results[symbol] = False

            except Exception as e:
                logger.error(f"Error validating symbol {symbol}: {str(e)}")
                results[symbol] = False

        return results

    def __del__(self):
        """Ensure session is closed on deletion."""
        if hasattr(self, 'session') and not self.session.closed:
            asyncio.run(self.session.close())
