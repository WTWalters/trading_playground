"""
Test script for DatabaseManager functionality.

This script tests the core functionality of the DatabaseManager class
with a focus on connection handling and basic operations.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.db_config import DatabaseConfig
from src.data_ingestion.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("DB_Manager_Test")

async def test_db_connection():
    """Test database connection and basic operations"""
    logger.info("Testing database connection")
    
    # Initialize database configuration
    db_config = DatabaseConfig(
        host="localhost",  # Default for local PostgreSQL
        port=5432,         # Default PostgreSQL port
        user="whitneywalters",   # Your system username
        password="",        # Leave empty if using peer authentication
        database="trading", # The database we just created
        min_connections=1,
        max_connections=5
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    try:
        # Initialize connection
        await db_manager.initialize()
        logger.info("Database connection initialized successfully")
        
        # Test closing the connection
        await db_manager.close()
        logger.info("Database connection closed successfully")
        
        # Test re-opening the connection
        await db_manager.initialize()
        logger.info("Database connection re-opened successfully")
        
        # Close again
        await db_manager.close()
        
        logger.info("Connection tests passed")
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

async def test_schema_initialization():
    """Test database schema initialization"""
    logger.info("Testing schema initialization")
    
    # Initialize database configuration
    db_config = DatabaseConfig(
        host="localhost",  # Default for local PostgreSQL
        port=5432,         # Default PostgreSQL port
        user="whitneywalters",   # Your system username
        password="",        # Leave empty if using peer authentication
        database="trading", # The database we just created
        min_connections=1,
        max_connections=5
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    try:
        # Initialize schema
        await db_manager.initialize_database_schema()
        logger.info("Schema initialization successful")
        
        # Close connection
        await db_manager.close()
        
        logger.info("Schema initialization test passed")
        return True
    except Exception as e:
        logger.error(f"Schema initialization test failed: {e}")
        return False

async def test_basic_operations():
    """Test basic database operations"""
    logger.info("Testing basic database operations")
    
    # Initialize database configuration
    db_config = DatabaseConfig(
        host="localhost",  # Default for local PostgreSQL
        port=5432,         # Default PostgreSQL port
        user="whitneywalters",   # Your system username
        password="",        # Leave empty if using peer authentication
        database="trading", # The database we just created
        min_connections=1,
        max_connections=5
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    try:
        # Create sample data
        now = datetime.now()
        dates = pd.date_range(start=now - timedelta(days=10), periods=10, freq='D')
        
        data = pd.DataFrame({
            'open': [100 + i for i in range(10)],
            'high': [110 + i for i in range(10)],
            'low': [90 + i for i in range(10)],
            'close': [105 + i for i in range(10)],
            'volume': [1000000 + i * 10000 for i in range(10)]
        }, index=dates)
        
        # Test symbol reference creation
        await db_manager.update_symbol_reference(
            symbol="TEST",
            name="Test Symbol",
            asset_type="test",
            active=True
        )
        logger.info("Symbol reference created successfully")
        
        # Test data storage
        count = await db_manager.store_market_data(
            data=data,
            symbol="TEST",
            provider="test_provider"
        )
        logger.info(f"Stored {count} records successfully")
        
        # Test data retrieval
        retrieved_data = await db_manager.get_market_data(
            symbol="TEST",
            start_time=now - timedelta(days=10),
            end_time=now,
            provider="test_provider"
        )
        
        if not retrieved_data.empty and len(retrieved_data) == 10:
            logger.info("Data retrieval successful")
        else:
            logger.error(f"Data retrieval returned incorrect number of records: {len(retrieved_data)}")
            return False
        
        # Test latest dates
        latest_dates = await db_manager.get_latest_dates(
            symbols=["TEST"],
            provider="test_provider"
        )
        
        if "TEST" in latest_dates and latest_dates["TEST"] is not None:
            logger.info("Latest dates retrieval successful")
        else:
            logger.error("Latest dates retrieval failed")
            return False
        
        # Close connection
        await db_manager.close()
        
        logger.info("Basic operations test passed")
        return True
    except Exception as e:
        logger.error(f"Basic operations test failed: {e}")
        return False

async def test_timescale_features():
    """Test TimescaleDB-specific features"""
    logger.info("Testing TimescaleDB-specific features")
    
    # Initialize database configuration
    db_config = DatabaseConfig(
        host="localhost",  # Default for local PostgreSQL
        port=5432,         # Default PostgreSQL port
        user="whitneywalters",   # Your system username
        password="",        # Leave empty if using peer authentication
        database="trading", # The database we just created
        min_connections=1,
        max_connections=5
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    try:
        # Make sure schema is initialized
        await db_manager.initialize_database_schema()
        
        # Create sample data if not already existing
        symbol = "TIMESCALE_TEST"
        provider = "test_provider"
        
        # Add symbol to reference table
        await db_manager.update_symbol_reference(
            symbol=symbol,
            name="TimescaleDB Test Symbol",
            asset_type="test",
            active=True
        )
        
        # Create test data
        now = datetime.now()
        # Create 100 points with 1-hour interval
        dates = pd.date_range(start=now - timedelta(days=5), periods=100, freq='H')
        
        data = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [110 + i * 0.1 for i in range(100)],
            'low': [90 + i * 0.1 for i in range(100)],
            'close': [105 + i * 0.1 for i in range(100)],
            'volume': [1000000 + i * 1000 for i in range(100)]
        }, index=dates)
        
        # Store data
        count = await db_manager.store_market_data(
            data=data,
            symbol=symbol,
            provider=provider
        )
        logger.info(f"Stored {count} records for TimescaleDB tests")
        
        # Test continuous aggregate creation
        try:
            await db_manager.create_continuous_aggregate(
                "test_hourly_agg",
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
                WHERE symbol = 'TIMESCALE_TEST'
                GROUP BY bucket, symbol
                """
            )
            logger.info("Successfully created continuous aggregate")
        except Exception as e:
            logger.error(f"Failed to create continuous aggregate: {e}")
            return False
        
        # Test setup compression policy
        try:
            await db_manager.setup_compression_policy(
                compress_after='1 day'
            )
            logger.info("Successfully setup compression policy")
        except Exception as e:
            logger.error(f"Failed to setup compression policy: {e}")
            return False
        
        # Test retention policy
        try:
            await db_manager.setup_retention_policy(
                retention_period='30 days'
            )
            logger.info("Successfully setup retention policy")
        except Exception as e:
            logger.error(f"Failed to setup retention policy: {e}")
            return False
        
        # Test database stats
        try:
            stats = await db_manager.get_database_stats()
            if 'total_records' in stats:
                logger.info(f"Database stats retrieved successfully: {stats['total_records']} records")
            else:
                logger.error("Failed to retrieve database stats")
                return False
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return False
        
        # Clean up
        await db_manager.close()
        
        logger.info("TimescaleDB features test passed")
        return True
    except Exception as e:
        logger.error(f"TimescaleDB features test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting DatabaseManager tests")
    
    # Run tests
    connection_test = await test_db_connection()
    schema_test = await test_schema_initialization()
    operations_test = await test_basic_operations()
    timescale_test = await test_timescale_features()
    
    # Summary
    logger.info("Test Summary:")
    logger.info(f"- Connection Test: {'PASSED' if connection_test else 'FAILED'}")
    logger.info(f"- Schema Test: {'PASSED' if schema_test else 'FAILED'}")
    logger.info(f"- Operations Test: {'PASSED' if operations_test else 'FAILED'}")
    logger.info(f"- TimescaleDB Features Test: {'PASSED' if timescale_test else 'FAILED'}")
    
    # Overall result
    all_passed = connection_test and schema_test and operations_test and timescale_test
    logger.info(f"Overall Test Result: {'PASSED' if all_passed else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())
