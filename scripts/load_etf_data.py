#!/usr/bin/env python
"""
Load ETF data directly from Yahoo Finance into the database.
This script bypasses the provider factory system and uses yfinance directly.
"""

import asyncio
import argparse
import yfinance as yf
import pandas as pd
import sys
import os
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

async def load_etf_data(symbols, start_date, end_date, timeframe='1d'):
    """
    Load ETF data from Yahoo Finance and store it in the database.
    
    Args:
        symbols: List of ETF symbols to load
        start_date: Start date for data
        end_date: End date for data
        timeframe: Data timeframe (1d for daily)
    """
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=timeframe)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue
                
            # Rename columns to match database schema
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Store in database
            await db_manager.store_market_data(
                df, symbol, 'yahoo', timeframe
            )
            
            logger.info(f"Stored {len(df)} records for {symbol}")
            
        logger.info("Data loading complete!")
            
    finally:
        # Close database connection
        await db_manager.close()

async def main():
    parser = argparse.ArgumentParser(description="Load ETF data into the database")
    parser.add_argument("--symbols", type=str, required=True,
                        help="Comma-separated list of symbols")
    parser.add_argument("--days", type=int, default=365,
                        help="Number of days of data to fetch")
    parser.add_argument("--timeframe", type=str, default="1d",
                        help="Data timeframe (1d, 1h, etc.)")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    await load_etf_data(symbols, start_date, end_date, args.timeframe)

if __name__ == "__main__":
    asyncio.run(main())
