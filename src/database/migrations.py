"""Database migration utilities."""
import asyncio
import argparse
import asyncpg
import logging
from src.config.db_config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_database(config: DatabaseConfig) -> None:
    """Create database if it doesn't exist."""
    # Connect to default postgres database
    conn = await asyncpg.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database='postgres'
    )
    
    try:
        # Check if database exists
        result = await conn.fetch(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            config.database
        )
        
        if not result:
            await conn.execute(f'CREATE DATABASE {config.database}')
            logger.info(f"Created database: {config.database}")
        else:
            logger.info(f"Database already exists: {config.database}")
            
    finally:
        await conn.close()

async def setup_timescaledb(config: DatabaseConfig) -> None:
    """Set up TimescaleDB extension and initial schema."""
    conn = await asyncpg.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database
    )
    
    try:
        # Create TimescaleDB extension
        await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb;')
        logger.info("TimescaleDB extension enabled")
        
        # Create market data table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                time        TIMESTAMPTZ NOT NULL,
                symbol      TEXT NOT NULL,
                open        DOUBLE PRECISION,
                high        DOUBLE PRECISION,
                low         DOUBLE PRECISION,
                close       DOUBLE PRECISION,
                volume      BIGINT,
                source      TEXT NOT NULL,
                timeframe   INTERVAL NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Created market_data table")
        
        # Create hypertable
        await conn.execute("""
            SELECT create_hypertable(
                'market_data', 'time',
                if_not_exists => TRUE
            );
        """)
        logger.info("Created hypertable")
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
            ON market_data (symbol, time DESC);
            
            CREATE INDEX IF NOT EXISTS idx_market_data_source 
            ON market_data (source);
        """)
        logger.info("Created indexes")
        
    finally:
        await conn.close()

async def run_migrations(config: DatabaseConfig) -> None:
    """Run all database migrations."""
    try:
        await create_database(config)
        await setup_timescaledb(config)
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Error during migrations: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--database', default='trading', help='Database name')
    parser.add_argument('--user', default='postgres', help='Database user')
    parser.add_argument('--password', default='postgres', help='Database password')
    
    args = parser.parse_args()
    
    config = DatabaseConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )
    
    asyncio.run(run_migrations(config))
