#!/usr/bin/env python
"""
Verify that the test data was correctly loaded and can be retrieved.
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

async def verify_data():
    """
    Verify that the test data was correctly loaded and can be retrieved.
    """
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Check each symbol
        symbols = ['SPY', 'IVV', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLU', 'GLD', 'SLV']
        
        for symbol in symbols:
            logger.info(f"Checking data for {symbol}...")
            
            # Get data
            df = await db_manager.get_market_data(
                symbol, start_date, end_date, '1d', 'synthetic'
            )
            
            if df.empty:
                logger.error(f"No data found for {symbol}")
            else:
                logger.info(f"Found {len(df)} records for {symbol}")
                logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                logger.info(f"Sample data: {df.iloc[0].to_dict()}")
            
            logger.info("-" * 40)
        
        # Check if we can get data without specifying the source
        logger.info("Checking data retrieval without specifying source...")
        df = await db_manager.get_market_data(
            'SPY', start_date, end_date, '1d'
        )
        
        if df.empty:
            logger.error("No data found without source specification")
        else:
            logger.info(f"Found {len(df)} records without source specification")
            
        # Get the latest dates from database
        latest_dates = await db_manager.get_latest_dates()
        logger.info(f"Latest dates in database: {latest_dates}")
        
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(verify_data())
