#!/usr/bin/env python
"""
Debug script to help diagnose backtest data issues.
This script focuses on checking the data availability for synthetic sources.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

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

async def debug_data_availability():
    """Check data availability in the database."""
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # Define the symbols to check
        symbols = ['SPY', 'IVV', 'QQQ', 'XLK', 'GLD', 'SLV']
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)
        
        logger.info("Checking data availability for synthetic source...")
        
        # Check each symbol
        for symbol in symbols:
            logger.info(f"Checking {symbol}...")
            
            # Check data without source filter
            data_no_source = await db_manager.get_market_data(
                symbol, start_date, end_date, '1d'
            )
            logger.info(f"{symbol} - Without source filter: {len(data_no_source)} records")
            
            # Check synthetic data
            data_synthetic = await db_manager.get_market_data(
                symbol, start_date, end_date, '1d', 'synthetic'
            )
            logger.info(f"{symbol} - With 'synthetic' source: {len(data_synthetic)} records")
            
            # If we found synthetic data, print the date range
            if not data_synthetic.empty:
                logger.info(f"{symbol} - Date range: {data_synthetic.index.min()} to {data_synthetic.index.max()}")
        
        # Run a direct database query to check all available sources
        async with db_manager.pool.acquire() as conn:
            sources = await conn.fetch(
                "SELECT DISTINCT source FROM market_data"
            )
            logger.info(f"Available sources in database: {[r['source'] for r in sources]}")
            
            # Count records by source
            for source in [r['source'] for r in sources]:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM market_data WHERE source = $1", source
                )
                logger.info(f"Records with source '{source}': {count}")
            
            # Check specific symbols and sources
            for symbol in symbols:
                for source in [r['source'] for r in sources]:
                    count = await conn.fetchval(
                        "SELECT COUNT(*) FROM market_data WHERE symbol = $1 AND source = $2", 
                        symbol, source
                    )
                    logger.info(f"{symbol} with source '{source}': {count} records")
                    
                    if count > 0:
                        # Get date range
                        date_range = await conn.fetch(
                            "SELECT MIN(time) as min_time, MAX(time) as max_time FROM market_data WHERE symbol = $1 AND source = $2",
                            symbol, source
                        )
                        min_time = date_range[0]['min_time']
                        max_time = date_range[0]['max_time']
                        logger.info(f"{symbol} with source '{source}': {min_time} to {max_time}")
        
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(debug_data_availability())
