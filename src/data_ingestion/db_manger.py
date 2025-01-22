# File: src/data_ingestion/db_manager.py

from typing import Dict, List, Optional, Union
import asyncio
import asyncpg
import pandas as pd
from datetime import datetime, timezone
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations for market data."""

    def __init__(self, config: Dict[str, str]):
        """
        Initialize database manager.

        Args:
            config: Database configuration dictionary containing:
                - host: database host
                - port: database port
                - database: database name
                - user: database user
                - password: database password
        """
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                min_size=5,
                max_size=20
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def connection(self):
        """Context manager for database connections."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise

class MarketDataManager:
    """Manages market data operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def insert_market_data(
        self,
        data: pd.DataFrame,
        provider: str,
        is_adjusted: bool = False
    ) -> int:
        """
        Insert market data into the database.

        Args:
            data: DataFrame with columns: time, symbol, open, high, low, close, volume
            provider: Data provider name
            is_adjusted: Whether the data is adjusted for corporate actions

        Returns:
            Number of rows inserted
        """
        if data.empty:
            return 0

        required_columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Prepare data for insertion
        records = []
        for _, row in data.iterrows():
            record = {
                'time': pd.Timestamp(row['time']).tz_localize('UTC') if row['time'].tzinfo is None else row['time'],
                'symbol': row['symbol'],
                'provider': provider,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume']),
                'is_adjusted': is_adjusted,
                'source_time': pd.Timestamp(row.get('source_time', row['time'])).tz_localize('UTC'),
                'data_quality': row.get('data_quality', 100),
                'metadata': row.get('metadata', {})
            }
            records.append(record)

        # Insert data
        async with self.db_manager.connection() as conn:
            count = await conn.executemany("""
                INSERT INTO market_data (
                    time, symbol, provider, open, high, low, close, volume,
                    is_adjusted, source_time, data_quality, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                ) ON CONFLICT (time, symbol, provider) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    is_adjusted = EXCLUDED.is_adjusted,
                    source_time = EXCLUDED.source_time,
                    data_quality = EXCLUDED.data_quality,
                    metadata = EXCLUDED.metadata,
                    update_time = NOW()
                """,
                [(r['time'], r['symbol'], r['provider'], r['open'], r['high'],
                  r['low'], r['close'], r['volume'], r['is_adjusted'],
                  r['source_time'], r['data_quality'], r['metadata'])
                 for r in records]
            )

        return len(records)

    async def get_market_data(
        self,
        symbols: Union[str, List[str]],
        start_time: datetime,
        end_time: datetime,
        provider: Optional[str] = None,
        timeframe: str = '1min'
    ) -> pd.DataFrame:
        """
        Retrieve market data from the database.

        Args:
            symbols: Single symbol or list of symbols
            start_time: Start time (UTC)
            end_time: End time (UTC)
            provider: Optional provider filter
            timeframe: Timeframe for aggregation ('1min', '1hour', '1day')

        Returns:
            DataFrame with market data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        timeframe_views = {
            '1min': 'market_data_1min',
            '1hour': 'market_data_1hour',
            '1day': 'market_data_1day'
        }

        view_name = timeframe_views.get(timeframe, 'market_data')
        provider_filter = "AND provider = $4" if provider else ""

        async with self.db_manager.connection() as conn:
            records = await conn.fetch(f"""
                SELECT bucket as time, symbol, provider,
                       open, high, low, close, volume,
                       avg_quality as data_quality
                FROM {view_name}
                WHERE symbol = ANY($1)
                AND bucket >= $2
                AND bucket <= $3
                {provider_filter}
                ORDER BY bucket ASC
            """, symbols, start_time, end_time, *([provider] if provider else []))

        if not records:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(records)
        df.set_index(['time', 'symbol'], inplace=True)

        return df

    async def log_data_quality_issue(
        self,
        symbol: str,
        provider: str,
        issue_type: str,
        description: str,
        severity: int,
        time: Optional[datetime] = None
    ) -> None:
        """Log a data quality issue."""
        if time is None:
            time = datetime.now(timezone.utc)

        async with self.db_manager.connection() as conn:
            await conn.execute("""
                INSERT INTO data_quality_log (
                    time, symbol, provider, issue_type,
                    description, severity
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, time, symbol, provider, issue_type, description, severity)

    async def update_provider_status(
        self,
        provider: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Update provider status."""
        async with self.db_manager.connection() as conn:
            if success:
                await conn.execute("""
                    UPDATE data_provider_config
                    SET last_success = NOW(),
                        last_error = NULL
                    WHERE provider = $1
                """, provider)
            else:
                await conn.execute("""
                    UPDATE data_provider_config
                    SET last_error = $2
                    WHERE provider = $1
                """, provider, error_message)
