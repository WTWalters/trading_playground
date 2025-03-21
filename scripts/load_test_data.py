#!/usr/bin/env python
"""
Load test data for the cointegration framework.
This script generates synthetic ETF data that includes known cointegrated pairs.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

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

async def load_test_data():
    """
    Generate synthetic ETF data with known cointegrated pairs and load into the database.
    """
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # Set fixed seed for reproducibility
        np.random.seed(42)
        
        # Generate dates for a year of daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Define ETF symbols
        symbols = ['SPY', 'IVV', 'QQQ', 'XLF', 'XLE', 'XLV', 'XLK', 'XLU', 'GLD', 'SLV']
        
        # Generate base price series with random walk
        base_series1 = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates)))
        base_series2 = 200 * np.cumprod(1 + np.random.normal(0.0006, 0.012, len(dates)))
        base_series3 = 50 * np.cumprod(1 + np.random.normal(0.0004, 0.009, len(dates)))
        
        # Create price series for each symbol
        price_data = {}
        
        # SPY and IVV - Cointegrated pair (S&P 500 ETFs)
        price_data['SPY'] = base_series1
        price_data['IVV'] = 0.25 * base_series1 + np.random.normal(0, 0.5, len(dates))
        
        # QQQ and XLK - Cointegrated pair (Technology ETFs)
        price_data['QQQ'] = base_series2
        price_data['XLK'] = 0.5 * base_series2 + np.random.normal(0, 1.0, len(dates))
        
        # XLF - Financial ETF (independent)
        price_data['XLF'] = 40 * np.cumprod(1 + np.random.normal(0.0003, 0.011, len(dates)))
        
        # XLE - Energy ETF (independent)
        price_data['XLE'] = 60 * np.cumprod(1 + np.random.normal(0.0002, 0.015, len(dates)))
        
        # XLV - Healthcare ETF (independent)
        price_data['XLV'] = 70 * np.cumprod(1 + np.random.normal(0.0004, 0.008, len(dates)))
        
        # XLU - Utilities ETF (weakly correlated with XLV)
        price_data['XLU'] = 0.3 * price_data['XLV'] + 50 * np.cumprod(1 + np.random.normal(0.0001, 0.007, len(dates)))
        
        # GLD and SLV - Cointegrated pair (Precious metals ETFs)
        price_data['GLD'] = base_series3
        price_data['SLV'] = 0.1 * base_series3 + np.random.normal(0, 0.3, len(dates))
        
        # Store each series in the database
        for symbol in symbols:
            # Create dataframe
            df = pd.DataFrame({
                'open': price_data[symbol] * 0.998,
                'high': price_data[symbol] * 1.005,
                'low': price_data[symbol] * 0.995,
                'close': price_data[symbol],
                'volume': np.random.randint(100000, 10000000, len(dates))
            }, index=dates)
            
            # Store in database
            await db_manager.store_market_data(
                df, symbol, 'synthetic', '1d'
            )
            
            logger.info(f"Stored {len(df)} records for {symbol}")
            
        logger.info("Test data loading complete!")
        
        # Log the known cointegrated pairs
        logger.info("Known cointegrated pairs:")
        logger.info("1. SPY/IVV - S&P 500 ETFs")
        logger.info("2. QQQ/XLK - Technology ETFs")
        logger.info("3. GLD/SLV - Precious metals ETFs")
        
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(load_test_data())
