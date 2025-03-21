#!/usr/bin/env python
"""
Load a simple test dataset with explicit timestamps.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz

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

async def load_simple_test():
    """
    Load a simple test dataset with explicit timestamps.
    """
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # Set fixed seed for reproducibility
        np.random.seed(42)
        
        # Generate dates for a month of daily data with explicit timezone
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Make sure dates are timezone-aware
        dates = [d.replace(tzinfo=pytz.UTC) for d in dates]
        
        # Create a simple price series
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        # Create dataframe with explicit column names matching the database schema
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 100000, len(dates))
        }, index=pd.DatetimeIndex(dates))
        
        # Ensure index has timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        logger.info(f"Created test dataframe with {len(df)} rows")
        logger.info(f"Index: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Timezone: {df.index.tz}")
        
        # Store just one symbol for testing
        symbol = 'TEST'
        source = 'test_source'
        timeframe = '1d'
        
        logger.info(f"Storing data for {symbol}...")
        await db_manager.store_market_data(df, symbol, source, timeframe)
        logger.info(f"Stored {len(df)} records")
        
        # Now try to retrieve it
        logger.info(f"Retrieving data for {symbol}...")
        retrieved_df = await db_manager.get_market_data(
            symbol, start_date, end_date, timeframe, source
        )
        
        if retrieved_df.empty:
            logger.error(f"No data retrieved for {symbol}")
        else:
            logger.info(f"Retrieved {len(retrieved_df)} records")
            logger.info(f"Retrieved index: {retrieved_df.index[0]} to {retrieved_df.index[-1]}")
            logger.info(f"Retrieved timezone: {retrieved_df.index.tz}")
            
            # Check if data matches
            logger.info(f"Original first row: {df.iloc[0].to_dict()}")
            logger.info(f"Retrieved first row: {retrieved_df.iloc[0].to_dict()}")
        
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(load_simple_test())
