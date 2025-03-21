"""
Database Manager compatibility helpers for different TimescaleDB versions.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

async def setup_compression_policy_compat(
    self,
    table_name: str = 'market_data',
    compress_after: str = '7 days',
    segment_by: Optional[List[str]] = None
) -> None:
    """
    Setup compression policy with version compatibility.
    
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
            # Check TimescaleDB version
            version_info = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            self.logger.info(f"TimescaleDB version: {version_info}")
            
            # Try to enable compression directly without checking policies
            segment_by_clause = f"timescaledb.compress_segmentby = '{','.join(segment_by)}'" if segment_by else ""
            orderby_clause = "timescaledb.compress_orderby = 'time'"
            
            # Enable compression
            try:
                # First set the compression parameters
                if segment_by_clause:
                    await conn.execute(f"""
                        ALTER TABLE {table_name} SET ({segment_by_clause}, {orderby_clause})
                    """)
                else:
                    await conn.execute(f"""
                        ALTER TABLE {table_name} SET ({orderby_clause})
                    """)
                
                # Then enable compression
                await conn.execute(f"""
                    ALTER TABLE {table_name} SET (timescaledb.compress = true)
                """)
                
                self.logger.info(f"Enabled compression for {table_name}")
            except Exception as e:
                self.logger.warning(f"Error enabling compression: {str(e)}")
                raise
            
            # Try to add compression policy - multiple approaches for different versions
            try:
                # Try newer version function
                await conn.execute(f"""
                    SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after}')
                """)
                self.logger.info(f"Compression policy set for {table_name} using add_compression_policy")
            except Exception as e1:
                self.logger.warning(f"Could not use add_compression_policy: {str(e1)}")
                
                try:
                    # Try older version function
                    await conn.execute(f"""
                        SELECT add_job('compress_chunks', INTERVAL '{compress_after}', 
                                      '{{ "hypertable_name": "{table_name}" }}')
                    """)
                    self.logger.info(f"Compression policy set for {table_name} using add_job")
                except Exception as e2:
                    self.logger.warning(f"Could not use add_job: {str(e2)}")
                    self.logger.info("Compression is enabled but automatic policy could not be set. Manual compression will be needed.")
                
    except Exception as e:
        self.logger.error(f"Failed to setup compression policy: {str(e)}")
        raise

async def setup_retention_policy_compat(
    self,
    table_name: str = 'market_data',
    retention_period: str = '1 year'
) -> None:
    """
    Setup data retention policy with version compatibility.
    
    Args:
        table_name: Name of the hypertable
        retention_period: Time interval to keep data
    """
    await self._ensure_connection()
    
    try:
        async with self.pool.acquire() as conn:
            # Check TimescaleDB version
            version_info = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            self.logger.info(f"TimescaleDB version: {version_info}")
            
            # Try different approaches for retention policy
            try:
                # Try newer version function
                await conn.execute(f"""
                    SELECT add_retention_policy('{table_name}', INTERVAL '{retention_period}')
                """)
                self.logger.info(f"Retention policy set for {table_name} using add_retention_policy")
            except Exception as e1:
                self.logger.warning(f"Could not use add_retention_policy: {str(e1)}")
                
                try:
                    # Try older version function with drop_chunks
                    await conn.execute(f"""
                        SELECT add_job('drop_chunks', INTERVAL '1d', 
                                      '{{ "hypertable_name": "{table_name}", "older_than": "interval {retention_period}" }}')
                    """)
                    self.logger.info(f"Retention policy set for {table_name} using add_job with drop_chunks")
                except Exception as e2:
                    self.logger.warning(f"Could not use add_job for retention: {str(e2)}")
                    self.logger.info("Retention policy could not be set automatically. Manual chunk management will be needed.")
    
    except Exception as e:
        self.logger.error(f"Failed to setup retention policy: {str(e)}")
        raise
