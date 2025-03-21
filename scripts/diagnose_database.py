#!/usr/bin/env python
"""
Diagnose database issues by checking what data is available and how to access it.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.db_config import DatabaseConfig
from src.database.manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def diagnose_database():
    """Run diagnostics on the database to identify issues with data access."""
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # 1. Check what's in the database
        async with db_manager.pool.acquire() as conn:
            # Check row count
            count = await conn.fetchval("SELECT COUNT(*) FROM market_data")
            logger.info(f"Total rows in market_data: {count}")
            
            # Check what symbols exist
            symbols = await conn.fetch("SELECT DISTINCT symbol FROM market_data")
            logger.info(f"Symbols in database: {[s['symbol'] for s in symbols]}")
            
            # Check what sources exist
            sources = await conn.fetch("SELECT DISTINCT source FROM market_data")
            logger.info(f"Sources in database: {[s['source'] for s in sources]}")
            
            # Check what timeframes exist
            timeframes = await conn.fetch("SELECT DISTINCT timeframe FROM market_data")
            logger.info(f"Timeframes in database: {[str(t['timeframe']) for t in timeframes]}")
            
            # Check date ranges for some symbols
            if symbols:
                for symbol in [s['symbol'] for s in symbols][:5]:  # First 5 symbols
                    date_range = await conn.fetchrow(
                        "SELECT MIN(time), MAX(time) FROM market_data WHERE symbol = $1",
                        symbol
                    )
                    logger.info(f"Date range for {symbol}: {date_range['min']} to {date_range['max']}")
        
        # 2. Try to retrieve data for each symbol using different source parameters
        for symbol in [s['symbol'] for s in symbols][:5]:  # First 5 symbols
            for source in [None] + [s['source'] for s in sources]:
                # Try with original date range
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()
                
                df = await db_manager.get_market_data(
                    symbol, start_date, end_date, '1d', source
                )
                
                source_str = source if source else "None"
                logger.info(f"Retrieved {len(df)} rows for {symbol} with source={source_str}")
                
                if not df.empty:
                    logger.info(f"  First row: {df.index[0]}")
                    logger.info(f"  Last row: {df.index[-1]}")
        
        # 3. Check the get_latest_dates function
        latest_dates = await db_manager.get_latest_dates()
        logger.info(f"Latest dates: {latest_dates}")
        
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(diagnose_database())
