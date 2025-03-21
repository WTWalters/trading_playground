#!/usr/bin/env python
"""
Check the database connection and schema.
"""

import asyncio
import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.db_config import DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_database():
    """
    Check the database connection and schema.
    """
    import asyncpg
    
    # Get database configuration
    db_config = DatabaseConfig()
    
    logger.info(f"Database configuration:")
    logger.info(f"Host: {db_config.host}")
    logger.info(f"Port: {db_config.port}")
    logger.info(f"Database: {db_config.database}")
    logger.info(f"User: {db_config.user}")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            user=db_config.user,
            password=db_config.password
        )
        
        logger.info("Successfully connected to database")
        
        # Check if TimescaleDB extension exists
        extension_check = await conn.fetchrow(
            "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
        )
        
        if extension_check:
            logger.info("TimescaleDB extension is installed")
        else:
            logger.warning("TimescaleDB extension is NOT installed")
        
        # Check if market_data table exists
        table_check = await conn.fetchrow(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'market_data'"
        )
        
        if table_check:
            logger.info("market_data table exists")
            
            # Check if it's a hypertable
            hypertable_check = await conn.fetchrow(
                "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data'"
            )
            
            if hypertable_check:
                logger.info("market_data is a hypertable")
            else:
                logger.warning("market_data is NOT a hypertable")
                
            # Check table structure
            columns = await conn.fetch(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'market_data'"
            )
            
            logger.info("market_data columns:")
            for col in columns:
                logger.info(f"  {col['column_name']} ({col['data_type']})")
                
            # Check row count
            count = await conn.fetchval("SELECT COUNT(*) FROM market_data")
            logger.info(f"market_data row count: {count}")
            
            # Check symbols
            symbols = await conn.fetch("SELECT DISTINCT symbol FROM market_data")
            logger.info(f"Symbols in market_data: {[s['symbol'] for s in symbols]}")
            
            # Check sources
            sources = await conn.fetch("SELECT DISTINCT source FROM market_data")
            logger.info(f"Sources in market_data: {[s['source'] for s in sources]}")
            
            # Check latest data
            latest = await conn.fetch("SELECT symbol, MAX(time) FROM market_data GROUP BY symbol LIMIT 5")
            logger.info("Latest data for some symbols:")
            for row in latest:
                logger.info(f"  {row['symbol']}: {row['max']}")
                
            # Check sample rows
            sample = await conn.fetch("SELECT * FROM market_data LIMIT 5")
            logger.info("Sample rows from market_data:")
            for i, row in enumerate(sample):
                logger.info(f"Row {i+1}: {dict(row)}")
        else:
            logger.error("market_data table does NOT exist")
        
        # Close connection
        await conn.close()
        
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")

if __name__ == "__main__":
    asyncio.run(check_database())
