"""
Database Manager for TimescaleDB operations.

This module handles all database operations for the trading platform, including:
- Market data storage and retrieval
- Symbol reference management
- Data provider configuration
- Data quality tracking
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import asyncpg
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from ..config.db_config import DatabaseConfig

class DatabaseManager:
    """
    Manages database operations for TimescaleDB.
    
    This class handles all database interactions, providing a clean
    interface for storing and retrieving time-series market data
    and related information.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database manager with configuration.
        
        Args:
            config: Database configuration settings
        """
        self.config = config
        self.pool = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """
        Initialize the database connection pool.
        
        Creates a pool of connections to the TimescaleDB database
        based on the provided configuration.
        """
        try:
            self.logger.info(f"Initializing database connection to {self.config.host}:{self.config.port}/{self.config.database}")
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections
            )
            self.logger.info("Database connection pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            raise
            
    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection pool closed")
            
    async def _ensure_connection(self) -> None:
        """Ensure the connection pool is initialized."""
        if self.pool is None:
            await self.initialize()
    
    async def store_market_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        provider: str, 
        data_quality: int = 100,
        is_adjusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store market data in the database.
        
        Args:
            data: DataFrame with OHLCV data (time index, open, high, low, close, volume columns)
            symbol: The ticker symbol
            provider: Data provider identifier
            data_quality: Quality score for the data (0-100)
            is_adjusted: Whether the data is adjusted for splits/dividends
            metadata: Optional metadata to store with the data
            
        Returns:
            Number of records inserted
        """
        if data.empty:
            self.logger.warning(f"Empty data provided for {symbol}, skipping database insertion")
            return 0
            
        await self._ensure_connection()
            
        # Validate DataFrame structure
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns in data: {missing}")
            
        # Convert metadata to JSON if provided
        meta_json = json.dumps(metadata) if metadata else None
        
        # Prepare records for insertion
        records = []
        for idx, row in data.iterrows():
            # Handle different index types
            if isinstance(idx, (pd.Timestamp, datetime)):
                timestamp = idx
            else:
                self.logger.warning(f"Non-datetime index in data for {symbol}, using current time")
                timestamp = datetime.now()
                
            record = (
                timestamp,                   # time
                symbol,                      # symbol
                provider,                    # provider
                float(row['open']),          # open
                float(row['high']),          # high
                float(row['low']),           # low
                float(row['close']),         # close
                int(row['volume']),          # volume
                data_quality,                # data_quality
                is_adjusted,                 # is_adjusted
                timestamp,                   # source_time (using index time as source time)
                meta_json                    # metadata
            )
            records.append(record)
            
        # Insert data in chunks to avoid memory issues with large datasets
        chunk_size = 1000
        total_inserted = 0
        
        try:
            async with self.pool.acquire() as conn:
                self.logger.info(f"Inserting {len(records)} records for {symbol} from {provider}")
                
                for i in range(0, len(records), chunk_size):
                    chunk = records[i:i + chunk_size]
                    async with conn.transaction():
                        result = await conn.executemany('''
                            INSERT INTO market_data 
                            (time, symbol, provider, open, high, low, close, volume, 
                             data_quality, is_adjusted, source_time, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            ON CONFLICT (time, symbol, provider) 
                            DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume,
                                data_quality = EXCLUDED.data_quality,
                                is_adjusted = EXCLUDED.is_adjusted,
                                update_time = NOW(),
                                metadata = EXCLUDED.metadata
                        ''', chunk)
                        
                        total_inserted += len(chunk)
                        
                self.logger.info(f"Successfully inserted {total_inserted} records for {symbol}")
                return total_inserted
                
        except Exception as e:
            self.logger.error(f"Failed to insert market data for {symbol}: {str(e)}")
            raise
            
    async def get_market_data(
        self, 
        symbol: str, 
        start_time: datetime,
        end_time: Optional[datetime] = None,
        provider: Optional[str] = None,
        timeframe: str = 'raw',
        include_metadata: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve market data from the database.
        
        Args:
            symbol: The ticker symbol
            start_time: Start time for the data
            end_time: End time for the data (defaults to current time)
            provider: Specific provider to query (optional)
            timeframe: Timeframe to query ('raw', '1m', '1h', '1d')
            include_metadata: Whether to include metadata in the result
            
        Returns:
            DataFrame with OHLCV data
        """
        await self._ensure_connection()
        
        # Set end_time to now if not provided
        if end_time is None:
            end_time = datetime.now()
            
        # Determine which table to query based on timeframe
        table_map = {
            'raw': 'market_data',
            '1m': 'market_data_1min',
            '1h': 'market_data_1hour',
            '1d': 'market_data_1day'
        }
        
        table = table_map.get(timeframe, 'market_data')
        
        # Handle bucketed time column name
        time_col = 'time' if table == 'market_data' else 'bucket'
        
        # Build the query
        columns = [
            f"{time_col} as time", 
            "open", 
            "high", 
            "low", 
            "close", 
            "volume"
        ]
        
        if include_metadata and table == 'market_data':
            columns.append("metadata")
            
        # Add quality columns for aggregates
        if table != 'market_data':
            columns.extend(["avg_quality", "sample_count"])
            
        query = f"""
            SELECT {', '.join(columns)}
            FROM {table}
            WHERE symbol = $1
              AND {time_col} >= $2
              AND {time_col} <= $3
        """
        
        params = [symbol, start_time, end_time]
        
        if provider and table == 'market_data':
            query += " AND provider = $4"
            params.append(provider)
            
        query += f" ORDER BY {time_col} ASC"
        
        try:
            async with self.pool.acquire() as conn:
                self.logger.info(f"Fetching {timeframe} data for {symbol} from {start_time} to {end_time}")
                rows = await conn.fetch(query, *params)
                
                if not rows:
                    self.logger.warning(f"No data found for {symbol} in the specified time range")
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                df_data = {}
                
                # Extract all column names from the first row
                columns = rows[0].keys()
                
                for col in columns:
                    df_data[col] = [row[col] for row in rows]
                    
                df = pd.DataFrame(df_data)
                
                # Set time as index
                if 'time' in df.columns:
                    df.set_index('time', inplace=True)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to fetch market data for {symbol}: {str(e)}")
            raise
            
    async def get_latest_market_data(
        self,
        symbol: str,
        lookback: timedelta = timedelta(days=1),
        provider: Optional[str] = None,
        timeframe: str = 'raw'
    ) -> pd.DataFrame:
        """
        Get the most recent market data for a symbol.
        
        Args:
            symbol: The ticker symbol
            lookback: How far back to look for data
            provider: Specific provider to query (optional)
            timeframe: Timeframe to query ('raw', '1m', '1h', '1d')
            
        Returns:
            DataFrame with the latest OHLCV data
        """
        end_time = datetime.now()
        start_time = end_time - lookback
        
        return await self.get_market_data(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            provider=provider,
            timeframe=timeframe
        )
        
    async def get_latest_dates(
        self, 
        symbols: Optional[List[str]] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the latest available dates for symbols in the database.
        
        Args:
            symbols: List of symbols to check (optional, if None checks all)
            provider: Specific provider to check (optional)
            
        Returns:
            Dictionary mapping symbols to their latest available dates
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                if symbols is None:
                    # Get all symbols if not specified
                    query = """
                        SELECT DISTINCT symbol
                        FROM market_data
                    """
                    if provider:
                        query += " WHERE provider = $1"
                        rows = await conn.fetch(query, provider)
                    else:
                        rows = await conn.fetch(query)
                        
                    symbols = [row['symbol'] for row in rows]
                
                result = {}
                
                # Fetch latest date for each symbol
                for symbol in symbols:
                    query = """
                        SELECT MAX(time) as latest_time
                        FROM market_data
                        WHERE symbol = $1
                    """
                    
                    params = [symbol]
                    
                    if provider:
                        query += " AND provider = $2"
                        params.append(provider)
                        
                    row = await conn.fetchrow(query, *params)
                    
                    if row and row['latest_time']:
                        result[symbol] = row['latest_time']
                    else:
                        result[symbol] = None
                        
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get latest dates: {str(e)}")
            raise
            
    async def log_data_quality_issue(
        self,
        symbol: str,
        provider: str,
        time: datetime,
        issue_type: str,
        description: str,
        severity: int
    ) -> None:
        """
        Log a data quality issue.
        
        Args:
            symbol: The ticker symbol
            provider: Data provider identifier
            time: Time of the data point with issues
            issue_type: Type of issue (e.g., 'missing_data', 'outlier')
            description: Detailed description of the issue
            severity: Severity level (1-10, with 10 being most severe)
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Check if symbol exists in reference table and insert if not
                symbol_exists = await conn.fetchval(
                    "SELECT COUNT(*) FROM symbol_reference WHERE symbol = $1",
                    symbol
                )
                
                if not symbol_exists:
                    await conn.execute(
                        """
                        INSERT INTO symbol_reference (symbol, asset_type)
                        VALUES ($1, 'unknown')
                        """,
                        symbol
                    )
                
                # Log the quality issue
                await conn.execute(
                    """
                    INSERT INTO data_quality_log
                    (time, symbol, provider, issue_type, description, severity)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    time, symbol, provider, issue_type, description, severity
                )
                
                self.logger.info(f"Logged data quality issue for {symbol}: {issue_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to log data quality issue: {str(e)}")
            raise
            
    async def update_symbol_reference(
        self,
        symbol: str,
        name: Optional[str] = None,
        asset_type: Optional[str] = None,
        active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update or create a symbol reference entry.
        
        Args:
            symbol: The ticker symbol
            name: Full name of the security
            asset_type: Type of asset (e.g., 'stock', 'etf', 'future')
            active: Whether the symbol is currently active
            metadata: Additional metadata for the symbol
        """
        await self._ensure_connection()
        
        # Convert metadata to JSON if provided
        meta_json = json.dumps(metadata) if metadata else None
        
        try:
            async with self.pool.acquire() as conn:
                # Check if symbol exists
                exists = await conn.fetchval(
                    "SELECT COUNT(*) FROM symbol_reference WHERE symbol = $1",
                    symbol
                )
                
                if exists:
                    # Build dynamic update query based on provided fields
                    updates = []
                    params = [symbol]  # First parameter is always the symbol
                    
                    if name is not None:
                        updates.append(f"name = ${len(params) + 1}")
                        params.append(name)
                        
                    if asset_type is not None:
                        updates.append(f"asset_type = ${len(params) + 1}")
                        params.append(asset_type)
                        
                    if active is not None:
                        updates.append(f"active = ${len(params) + 1}")
                        params.append(active)
                        
                    if meta_json is not None:
                        updates.append(f"metadata = ${len(params) + 1}")
                        params.append(meta_json)
                    
                    if updates:
                        query = f"""
                            UPDATE symbol_reference
                            SET {', '.join(updates)}
                            WHERE symbol = $1
                        """
                        
                        await conn.execute(query, *params)
                        self.logger.info(f"Updated symbol reference for {symbol}")
                else:
                    # Insert new symbol reference
                    if asset_type is None:
                        asset_type = 'unknown'  # Default asset type
                        
                    await conn.execute(
                        """
                        INSERT INTO symbol_reference
                        (symbol, name, asset_type, active, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        symbol, name, asset_type, 
                        True if active is None else active,
                        meta_json
                    )
                    
                    self.logger.info(f"Created symbol reference for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update symbol reference: {str(e)}")
            raise
            
    async def get_symbols(
        self,
        asset_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get list of symbols from the reference table.
        
        Args:
            asset_type: Filter by asset type (optional)
            active_only: Only return active symbols
            
        Returns:
            List of symbol dictionaries with metadata
        """
        await self._ensure_connection()
        
        query = """
            SELECT symbol, name, asset_type, active, metadata, created_at, updated_at
            FROM symbol_reference
            WHERE 1=1
        """
        
        params = []
        
        if active_only:
            query += " AND active = TRUE"
            
        if asset_type:
            query += f" AND asset_type = ${len(params) + 1}"
            params.append(asset_type)
            
        query += " ORDER BY symbol"
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                result = []
                for row in rows:
                    symbol_data = {
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'asset_type': row['asset_type'],
                        'active': row['active'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    
                    # Parse metadata JSON if present
                    if row['metadata']:
                        try:
                            symbol_data['metadata'] = json.loads(row['metadata'])
                        except:
                            symbol_data['metadata'] = {}
                    else:
                        symbol_data['metadata'] = {}
                        
                    result.append(symbol_data)
                    
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {str(e)}")
            raise
            
    async def update_provider_config(
        self,
        provider: str,
        config: Dict[str, Any],
        active: bool = True
    ) -> None:
        """
        Update or create a data provider configuration.
        
        Args:
            provider: Provider identifier
            config: Configuration dictionary
            active: Whether the provider is active
        """
        await self._ensure_connection()
        
        config_json = json.dumps(config)
        
        try:
            async with self.pool.acquire() as conn:
                # Check if provider exists
                exists = await conn.fetchval(
                    "SELECT COUNT(*) FROM data_provider_config WHERE provider = $1",
                    provider
                )
                
                if exists:
                    await conn.execute(
                        """
                        UPDATE data_provider_config
                        SET config = $2, active = $3
                        WHERE provider = $1
                        """,
                        provider, config_json, active
                    )
                    
                    self.logger.info(f"Updated configuration for provider {provider}")
                else:
                    await conn.execute(
                        """
                        INSERT INTO data_provider_config
                        (provider, config, active)
                        VALUES ($1, $2, $3)
                        """,
                        provider, config_json, active
                    )
                    
                    self.logger.info(f"Created configuration for provider {provider}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update provider config: {str(e)}")
            raise
            
    async def get_provider_config(
        self,
        provider: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a data provider.
        
        Args:
            provider: Provider identifier
            
        Returns:
            Provider configuration dictionary or None if not found
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT config, active, last_success, last_error, created_at, updated_at
                    FROM data_provider_config
                    WHERE provider = $1
                    """,
                    provider
                )
                
                if not row:
                    return None
                    
                result = {
                    'active': row['active'],
                    'last_success': row['last_success'],
                    'last_error': row['last_error'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
                
                # Parse config JSON
                if row['config']:
                    try:
                        result['config'] = json.loads(row['config'])
                    except:
                        result['config'] = {}
                else:
                    result['config'] = {}
                    
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get provider config: {str(e)}")
            raise
            
    async def update_provider_status(
        self,
        provider: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update the status of a data provider.
        
        Args:
            provider: Provider identifier
            success: Whether the last operation was successful
            error_message: Error message if operation failed
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                if success:
                    await conn.execute(
                        """
                        UPDATE data_provider_config
                        SET last_success = NOW(), last_error = NULL
                        WHERE provider = $1
                        """,
                        provider
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE data_provider_config
                        SET last_error = $2
                        WHERE provider = $1
                        """,
                        provider, error_message
                    )
                    
                self.logger.info(f"Updated status for provider {provider}: success={success}")
                
        except Exception as e:
            self.logger.error(f"Failed to update provider status: {str(e)}")
            raise
            
    async def get_data_quality_issues(
        self,
        symbol: Optional[str] = None,
        provider: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        issue_type: Optional[str] = None,
        min_severity: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get data quality issues matching the filters.
        
        Args:
            symbol: Filter by symbol (optional)
            provider: Filter by provider (optional)
            start_time: Filter by minimum time (optional)
            end_time: Filter by maximum time (optional)
            issue_type: Filter by issue type (optional)
            min_severity: Filter by minimum severity (optional)
            limit: Maximum number of issues to return
            
        Returns:
            List of data quality issue dictionaries
        """
        await self._ensure_connection()
        
        query = """
            SELECT id, time, symbol, provider, issue_type, description, severity, created_at
            FROM data_quality_log
            WHERE 1=1
        """
        
        params = []
        
        if symbol:
            query += f" AND symbol = ${len(params) + 1}"
            params.append(symbol)
            
        if provider:
            query += f" AND provider = ${len(params) + 1}"
            params.append(provider)
            
        if start_time:
            query += f" AND time >= ${len(params) + 1}"
            params.append(start_time)
            
        if end_time:
            query += f" AND time <= ${len(params) + 1}"
            params.append(end_time)
            
        if issue_type:
            query += f" AND issue_type = ${len(params) + 1}"
            params.append(issue_type)
            
        if min_severity is not None:
            query += f" AND severity >= ${len(params) + 1}"
            params.append(min_severity)
            
        query += " ORDER BY time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
            
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                result = []
                for row in rows:
                    issue = {
                        'id': row['id'],
                        'time': row['time'],
                        'symbol': row['symbol'],
                        'provider': row['provider'],
                        'issue_type': row['issue_type'],
                        'description': row['description'],
                        'severity': row['severity'],
                        'created_at': row['created_at']
                    }
                    
                    result.append(issue)
                    
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get data quality issues: {str(e)}")
            raise
            
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Get total number of records
                total_records = await conn.fetchval(
                    "SELECT COUNT(*) FROM market_data"
                )
                
                # Get record count by symbol
                symbol_counts = await conn.fetch(
                    """
                    SELECT symbol, COUNT(*) as count
                    FROM market_data
                    GROUP BY symbol
                    ORDER BY count DESC
                    LIMIT 10
                    """
                )
                
                # Get total symbols
                total_symbols = await conn.fetchval(
                    "SELECT COUNT(DISTINCT symbol) FROM market_data"
                )
                
                # Get data date range
                date_range = await conn.fetchrow(
                    """
                    SELECT 
                        MIN(time) as oldest,
                        MAX(time) as newest
                    FROM market_data
                    """
                )
                
                # Get chunk information from TimescaleDB
                chunks = await conn.fetch(
                    """
                    SELECT 
                        chunk_schema, 
                        chunk_name, 
                        range_start, 
                        range_end, 
                        is_compressed
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = 'market_data'
                    ORDER BY range_start DESC
                    LIMIT 10
                    """
                )
                
                # Assemble result
                result = {
                    'total_records': total_records,
                    'total_symbols': total_symbols,
                    'date_range': {
                        'oldest': date_range['oldest'] if date_range else None,
                        'newest': date_range['newest'] if date_range else None
                    },
                    'top_symbols': [
                        {'symbol': row['symbol'], 'count': row['count']}
                        for row in symbol_counts
                    ],
                    'recent_chunks': [
                        {
                            'schema': row['chunk_schema'],
                            'name': row['chunk_name'],
                            'range_start': row['range_start'],
                            'range_end': row['range_end'],
                            'is_compressed': row['is_compressed']
                        }
                        for row in chunks
                    ]
                }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            raise
            
    async def create_continuous_aggregate(
        self,
        view_name: str,
        query: str,
        refresh_policy: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a continuous aggregate for efficient time-series queries.
        
        Args:
            view_name: Name of the continuous aggregate view
            query: SQL query defining the view
            refresh_policy: Optional refresh policy configuration
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Check if view already exists
                view_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM timescaledb_information.continuous_aggregates 
                        WHERE view_name = $1
                    )
                    """, 
                    view_name
                )
                
                if view_exists:
                    self.logger.warning(f"Continuous aggregate {view_name} already exists, skipping creation")
                    return
                
                # Create the continuous aggregate view
                await conn.execute(f"CREATE MATERIALIZED VIEW {view_name} WITH (timescaledb.continuous) AS {query}")
                
                # Set refresh policy if provided
                if refresh_policy:
                    interval = refresh_policy.get('interval', '1h')
                    start_offset = refresh_policy.get('start_offset', '1d')
                    end_offset = refresh_policy.get('end_offset', '1h')
                    
                    await conn.execute(f"""
                        SELECT add_continuous_aggregate_policy('{view_name}',
                            start_offset => INTERVAL '{start_offset}',
                            end_offset => INTERVAL '{end_offset}',
                            schedule_interval => INTERVAL '{interval}')
                    """)
                    
                self.logger.info(f"Created continuous aggregate view: {view_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to create continuous aggregate: {str(e)}")
            raise

    async def drop_continuous_aggregate(self, view_name: str) -> None:
        """
        Drop a continuous aggregate view.
        
        Args:
            view_name: Name of the continuous aggregate view to drop
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name} CASCADE")
                self.logger.info(f"Dropped continuous aggregate view: {view_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to drop continuous aggregate: {str(e)}")
            raise

    async def setup_compression_policy(
        self,
        table_name: str = 'market_data',
        compress_after: str = '7 days',
        segment_by: Optional[List[str]] = None
    ) -> None:
        """
        Setup compression policy for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            compress_after: Time interval after which to compress chunks
            segment_by: Columns to use for segmenting data
        """
        await self._ensure_connection()
        
        if segment_by is None:
            segment_by = ['symbol', 'provider']
        
        try:
            async with self.pool.acquire() as conn:
                # Check TimescaleDB version to determine approach
                version_query = "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                version_str = await conn.fetchval(version_query)
                major_version = int(version_str.split('.')[0]) if version_str else 1
                
                self.logger.info(f"Detected TimescaleDB version: {version_str}")
                
                # Enable compression on the table
                compression_enabled = False
                try:
                    # First try checking with the newer API (v2.x)
                    compression_enabled = await conn.fetchval(
                        """
                        SELECT compression_enabled 
                        FROM timescaledb_information.hypertables
                        WHERE hypertable_name = $1
                        """,
                        table_name
                    )
                except Exception as e:
                    # Fall back to checking with a direct query for older versions
                    self.logger.info(f"Using alternative method to check compression status: {str(e)}")
                    # For older versions, we can just proceed and let the ALTER TABLE command handle it
                    pass
                
                if not compression_enabled:
                    segment_by_clause = f"timescaledb.compress_segmentby = '{','.join(segment_by)}'" if segment_by else ""
                    
                    # Choose appropriate orderby columns for optimal compression
                    orderby_clause = "timescaledb.compress_orderby = 'time'"
                    
                    if segment_by_clause:
                        await conn.execute(f"""
                            ALTER TABLE {table_name} SET ({segment_by_clause}, {orderby_clause})
                        """)
                    else:
                        await conn.execute(f"""
                            ALTER TABLE {table_name} SET ({orderby_clause})
                        """)
                    
                    await conn.execute(f"""
                        ALTER TABLE {table_name} SET (timescaledb.compress = true)
                    """)
                    
                    self.logger.info(f"Enabled compression for {table_name}")
                
                # Create compression policy based on version
                if major_version >= 2:
                    # For TimescaleDB 2.x, try to remove existing policy first
                    try:
                        await conn.execute(f"SELECT remove_compression_policy('{table_name}')")
                        self.logger.info(f"Removed existing compression policy for {table_name}")
                    except Exception as e:
                        self.logger.info(f"No existing compression policy to remove: {str(e)}")
                        
                    # Add new policy
                    await conn.execute(f"""
                        SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after}')
                    """)
                else:
                    # For TimescaleDB 1.x
                    # The older versions used a different function
                    try:
                        await conn.execute(f"""
                            SELECT add_compress_chunks_policy('{table_name}', INTERVAL '{compress_after}')
                        """)
                    except Exception as e:
                        self.logger.warning(f"Could not add compression policy using v1.x method: {str(e)}")
                        self.logger.info("Trying alternate compression approach for older TimescaleDB")
                        # Some very old versions might not have the policy API at all
                        # In this case, mention manual compression is needed
                        self.logger.info("For this version, you may need to manually compress chunks")
                
                self.logger.info(f"Compression policy set for {table_name}: compress after {compress_after}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup compression policy: {str(e)}")
            raise

    async def setup_retention_policy(
        self,
        table_name: str = 'market_data',
        retention_period: str = '1 year'
    ) -> None:
        """
        Setup data retention policy for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            retention_period: Time interval to keep data
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Check for existing retention policy
                policy_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM timescaledb_information.policies
                        WHERE hypertable_name = $1 AND policy_type = 'retention'
                    )
                    """,
                    table_name
                )
                
                if policy_exists:
                    # Remove existing policy before creating a new one
                    await conn.execute(f"""
                        SELECT remove_retention_policy('{table_name}')
                    """)
                
                # Create retention policy
                await conn.execute(f"""
                    SELECT add_retention_policy('{table_name}', INTERVAL '{retention_period}')
                """)
                
                self.logger.info(f"Retention policy set for {table_name}: retain for {retention_period}")
                
        except Exception as e:
            self.logger.error(f"Failed to setup retention policy: {str(e)}")
            raise

    async def initialize_database_schema(self, recreate: bool = False) -> None:
        """
        Initialize the database schema with all required tables and indices.
        
        Args:
            recreate: Whether to drop and recreate existing objects
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Check TimescaleDB extension
                is_timescaledb = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
                )
                
                if not is_timescaledb:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                    self.logger.info("TimescaleDB extension created")
                
                # Create tables if they don't exist or if recreate is True
                if recreate:
                    await conn.execute("DROP TABLE IF EXISTS market_data CASCADE")
                    await conn.execute("DROP TABLE IF EXISTS symbol_reference CASCADE")
                    await conn.execute("DROP TABLE IF EXISTS data_provider_config CASCADE")
                    await conn.execute("DROP TABLE IF EXISTS data_quality_log CASCADE")
                
                # Create tables
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS symbol_reference (
                        symbol          TEXT PRIMARY KEY,
                        name            TEXT,
                        asset_type      TEXT NOT NULL,
                        active          BOOLEAN NOT NULL DEFAULT TRUE,
                        metadata        JSONB,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_provider_config (
                        provider        TEXT PRIMARY KEY,
                        config          JSONB NOT NULL,
                        active          BOOLEAN NOT NULL DEFAULT TRUE,
                        last_success    TIMESTAMPTZ,
                        last_error      TEXT,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        time            TIMESTAMPTZ NOT NULL,
                        symbol          TEXT NOT NULL,
                        provider        TEXT NOT NULL,
                        open            DOUBLE PRECISION NOT NULL,
                        high            DOUBLE PRECISION NOT NULL,
                        low             DOUBLE PRECISION NOT NULL,
                        close           DOUBLE PRECISION NOT NULL,
                        volume          BIGINT NOT NULL,
                        data_quality    SMALLINT NOT NULL DEFAULT 100,
                        is_adjusted     BOOLEAN NOT NULL DEFAULT FALSE,
                        source_time     TIMESTAMPTZ NOT NULL,
                        update_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        metadata        JSONB,
                        CONSTRAINT market_data_pkey PRIMARY KEY (time, symbol, provider)
                    )
                """)
                
                # Create hypertable for market data if it doesn't exist
                is_hypertable = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = 'market_data'
                    )
                    """
                )
                
                if not is_hypertable:
                    await conn.execute(
                        "SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE)"
                    )
                    self.logger.info("Created market_data hypertable")
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_log (
                        id              SERIAL PRIMARY KEY,
                        time            TIMESTAMPTZ NOT NULL,
                        symbol          TEXT NOT NULL,
                        provider        TEXT NOT NULL,
                        issue_type      TEXT NOT NULL,
                        description     TEXT NOT NULL,
                        severity        SMALLINT NOT NULL,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                
                # Create indices for performance
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol)"
                )
                
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_provider ON market_data (provider)"
                )
                
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_market_data_time_symbol ON market_data (time, symbol)"
                )
                
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_data_quality_log_symbol ON data_quality_log (symbol)"
                )
                
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_data_quality_log_time ON data_quality_log (time)"
                )
                
                self.logger.info("Database schema initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {str(e)}")
            raise

    async def create_standard_continuous_aggregates(self) -> None:
        """
        Create standard continuous aggregates for common time intervals.
        """
        await self._ensure_connection()
        
        try:
            # Create 1-minute aggregate
            await self.create_continuous_aggregate(
                "market_data_1min",
                """
                SELECT 
                    time_bucket('1 minute', time) AS bucket,
                    symbol,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume,
                    avg(data_quality) AS avg_quality,
                    count(*) AS sample_count
                FROM market_data
                GROUP BY bucket, symbol
                """,
                {
                    'interval': '15 minutes',
                    'start_offset': '1 day',
                    'end_offset': '5 minutes'
                }
            )
            
            # Create 1-hour aggregate
            await self.create_continuous_aggregate(
                "market_data_1hour",
                """
                SELECT 
                    time_bucket('1 hour', time) AS bucket,
                    symbol,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume,
                    avg(data_quality) AS avg_quality,
                    count(*) AS sample_count
                FROM market_data
                GROUP BY bucket, symbol
                """,
                {
                    'interval': '1 hour',
                    'start_offset': '7 days',
                    'end_offset': '1 hour'
                }
            )
            
            # Create 1-day aggregate
            await self.create_continuous_aggregate(
                "market_data_1day",
                """
                SELECT 
                    time_bucket('1 day', time) AS bucket,
                    symbol,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume,
                    avg(data_quality) AS avg_quality,
                    count(*) AS sample_count
                FROM market_data
                GROUP BY bucket, symbol
                """,
                {
                    'interval': '1 day',
                    'start_offset': '30 days',
                    'end_offset': '1 day'
                }
            )
            
            self.logger.info("Standard continuous aggregates created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create standard continuous aggregates: {str(e)}")
            raise
            
    async def refresh_continuous_aggregate(self, view_name: str) -> None:
        """
        Manually refresh a continuous aggregate view.
        
        Args:
            view_name: Name of the continuous aggregate view to refresh
        """
        await self._ensure_connection()
        
        try:
            async with self.pool.acquire() as conn:
                # Check if view exists
                view_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM timescaledb_information.continuous_aggregates 
                        WHERE view_name = $1
                    )
                    """, 
                    view_name
                )
                
                if not view_exists:
                    self.logger.warning(f"Continuous aggregate {view_name} does not exist, cannot refresh")
                    return
                
                # Refresh the continuous aggregate
                await conn.execute(f"CALL refresh_continuous_aggregate('{view_name}', NULL, NULL)")
                self.logger.info(f"Refreshed continuous aggregate view: {view_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to refresh continuous aggregate: {str(e)}")
            raise
