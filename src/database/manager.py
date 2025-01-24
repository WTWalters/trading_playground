"""Database manager module for market data storage and retrieval."""
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
import asyncpg
from asyncpg import Pool
from ..config.db_config import DatabaseConfig

class DatabaseManager:
    """Manages database operations for market data."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database manager with configuration."""
        self.config = config
        self.pool: Optional[Pool] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize database connection pool and create schema."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections
            )
            await self._create_schema()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
            
    async def _create_schema(self) -> None:
        """Create necessary database schema if not exists."""
        async with self.pool.acquire() as conn:
            # Create the TimescaleDB extension
            await conn.execute(
                "CREATE EXTENSION IF NOT EXISTS timescaledb;"
            )
            
            # Create market data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time        TIMESTAMPTZ NOT NULL,
                    symbol      TEXT NOT NULL,
                    open        DOUBLE PRECISION,
                    high        DOUBLE PRECISION,
                    low         DOUBLE PRECISION,
                    close       DOUBLE PRECISION,
                    volume      BIGINT,
                    source      TEXT NOT NULL,
                    timeframe   INTERVAL NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create hypertable
            await conn.execute("""
                SELECT create_hypertable(
                    'market_data', 'time',
                    if_not_exists => TRUE
                );
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
                ON market_data (symbol, time DESC);
                
                CREATE INDEX IF NOT EXISTS idx_market_data_source 
                ON market_data (source);
            """)
            
    async def store_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        source: str,
        timeframe: str
    ) -> None:
        """Store market data in the database."""
        if data.empty:
            return
            
        # Convert timeframe string to interval
        interval_map = {
            '1m': '1 minute',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day'
        }
        interval = interval_map.get(timeframe, '1 day')
        
        try:
            # Prepare data for insertion
            records = [
                (
                    index,          # time
                    symbol,         # symbol
                    row['open'],    # open
                    row['high'],    # high
                    row['low'],     # low
                    row['close'],   # close
                    int(row['volume']), # volume
                    source,         # source
                    interval        # timeframe
                )
                for index, row in data.iterrows()
            ]
            
            async with self.pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO market_data (
                        time, symbol, open, high, low, close, 
                        volume, source, timeframe
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (time, symbol) DO UPDATE
                    SET 
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source,
                        timeframe = EXCLUDED.timeframe,
                        updated_at = CURRENT_TIMESTAMP
                """, records)
                
            self.logger.info(
                f"Stored {len(records)} records for {symbol} ({timeframe})"
            )
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")
            raise
            
    async def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve market data from the database."""
        try:
            query = """
                SELECT time, open, high, low, close, volume
                FROM market_data
                WHERE symbol = $1
                AND time BETWEEN $2 AND $3
            """
            params = [symbol, start_date, end_date]
            
            if source:
                query += " AND source = $4"
                params.append(source)
                
            query += " ORDER BY time ASC"
            
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, *params)
                
            if not records:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(
                records,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )
            df.set_index('time', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            raise
            
    async def get_latest_dates(self) -> Dict[Tuple[str, str], datetime]:
        """Get the latest date for each symbol/timeframe combination."""
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch("""
                    SELECT symbol, timeframe::text, MAX(time) as latest_time
                    FROM market_data
                    GROUP BY symbol, timeframe
                """)
                
            return {
                (record['symbol'], record['timeframe']): record['latest_time']
                for record in records
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving latest dates: {e}")
            raise
            
    async def cleanup_old_data(self, days_to_keep: int) -> None:
        """Remove market data older than specified days."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    DELETE FROM market_data
                    WHERE time < NOW() - INTERVAL '1 day' * $1
                """, days_to_keep)
                
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise
            
    async def close(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connections closed")
