"""
Proof of Concept: TITAN Trading Platform Data Pipeline

This script demonstrates a complete data pipeline using the DatabaseManager:
- Database schema initialization
- Data ingestion from Yahoo Finance
- Data validation and storage
- TimescaleDB continuous aggregates setup
- Basic data analysis
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.db_config import DatabaseConfig
from src.data_ingestion.db_manager import DatabaseManager
# Import compatibility helpers
from src.data_ingestion.db_manager_compat import setup_compression_policy_compat, setup_retention_policy_compat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("POC_Pipeline")

# Target symbols
SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()
PROVIDER = "yahoo_finance"

async def fetch_market_data():
    """Fetch market data using yfinance"""
    logger.info(f"Fetching market data for {SYMBOLS} from {START_DATE.date()} to {END_DATE.date()}")
    
    data_frames = {}
    
    for symbol in SYMBOLS:
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=START_DATE, end=END_DATE)
            
            # Ensure expected columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in data for {symbol}")
                continue
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only the columns we need
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Basic validation
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                continue
                
            # Check for NaN values
            if df.isna().any().any():
                logger.warning(f"NaN values found in data for {symbol}. Filling with appropriate values.")
                # Fill NaN values - forward fill for prices, zeros for volume
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='ffill')
                df['volume'] = df['volume'].fillna(0)
            
            # Store in our dictionary
            data_frames[symbol] = df
            logger.info(f"Successfully fetched data for {symbol}: {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return data_frames

def validate_data(data):
    """
    Validate market data and fix common issues
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Tuple of (validated_data, quality_score, issues)
    """
    issues = []
    quality_score = 100
    
    if data.empty:
        return data, 0, ["Empty dataset"]
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            issues.append(f"Missing column: {col}")
            return data, 0, issues
    
    # Create a copy to avoid modifying the original
    validated = data.copy()
    
    # Check for NaN values
    nan_count = validated.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values: {nan_count}")
        quality_score -= min(40, (nan_count / len(validated)) * 100)
        
        # Fill NaN values
        validated[['open', 'high', 'low', 'close']] = validated[['open', 'high', 'low', 'close']].fillna(method='ffill').fillna(method='bfill')
        validated['volume'] = validated['volume'].fillna(0)
    
    # Check for negative values
    neg_prices = (validated[['open', 'high', 'low', 'close']] <= 0).any().any()
    neg_volumes = (validated['volume'] < 0).any()
    
    if neg_prices:
        issues.append("Negative prices detected")
        quality_score -= 20
        
        # Fix negative prices
        for col in ['open', 'high', 'low', 'close']:
            validated[col] = validated[col].abs()
    
    if neg_volumes:
        issues.append("Negative volumes detected")
        quality_score -= 10
        
        # Fix negative volumes
        validated['volume'] = validated['volume'].clip(lower=0)
    
    # Check price relationships (high >= open, high >= close, low <= open, low <= close)
    relationship_errors = ((validated['high'] < validated['open']) | 
                         (validated['high'] < validated['close']) | 
                         (validated['low'] > validated['open']) | 
                         (validated['low'] > validated['close'])).sum()
    
    if relationship_errors > 0:
        issues.append(f"Price relationship errors: {relationship_errors}")
        quality_score -= min(30, (relationship_errors / len(validated)) * 100)
        
        # Fix relationship errors
        validated['high'] = validated[['high', 'open', 'close']].max(axis=1)
        validated['low'] = validated[['low', 'open', 'close']].min(axis=1)
    
    # Check for outliers using z-score
    z_scores = (validated[['open', 'high', 'low', 'close']] - 
               validated[['open', 'high', 'low', 'close']].mean()) / validated[['open', 'high', 'low', 'close']].std()
    
    outliers = (abs(z_scores) > 3).any(axis=1)
    outlier_count = outliers.sum()
    
    if outlier_count > 0:
        issues.append(f"Outliers detected: {outlier_count}")
        quality_score -= min(10, (outlier_count / len(validated)) * 50)
    
    # Ensure quality score is between 0 and 100
    quality_score = max(0, min(100, quality_score))
    
    return validated, quality_score, issues

async def main():
    """Main POC pipeline function"""
    logger.info("Starting TITAN Trading Platform Data Pipeline POC")
    
    # Initialize database configuration
    db_config = DatabaseConfig(
        host="localhost",  # Default for local PostgreSQL
        port=5432,         # Default PostgreSQL port
        user="whitneywalters",   # Your system username
        password="",        # Leave empty if using peer authentication
        database="trading", # The database we just created
        min_connections=1,
        max_connections=10
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    try:
        # Initialize database schema
        logger.info("Initializing database schema")
        await db_manager.initialize_database_schema()
        
        # Fetch market data
        market_data = await fetch_market_data()
        
        # Store data in database
        for symbol, data in market_data.items():
            logger.info(f"Processing data for {symbol}")
            
            # Validate data
            validated_data, quality_score, issues = validate_data(data)
            
            if issues:
                logger.warning(f"Data quality issues for {symbol}: {', '.join(issues)}")
                
                # Log issues in the database
                for issue in issues:
                    await db_manager.log_data_quality_issue(
                        symbol=symbol,
                        provider=PROVIDER,
                        time=datetime.now(),
                        issue_type="validation",
                        description=issue,
                        severity=8 if quality_score < 50 else 5
                    )
            
            # Add symbol to reference table
            await db_manager.update_symbol_reference(
                symbol=symbol,
                name=symbol,  # Using symbol as name for simplicity
                asset_type="stock",
                active=True,
                metadata={"sector": "Technology", "provider": PROVIDER}
            )
            
            # Store market data
            count = await db_manager.store_market_data(
                data=validated_data,
                symbol=symbol,
                provider=PROVIDER,
                data_quality=quality_score,
                is_adjusted=True
            )
            
            logger.info(f"Stored {count} records for {symbol} with quality score {quality_score}")
        
        # Set up compression policy using compatibility function
        logger.info("Setting up compression policy")
        # Add compatibility function to DatabaseManager instance
        setattr(db_manager, "setup_compression_policy_compat", setup_compression_policy_compat.__get__(db_manager))
        await db_manager.setup_compression_policy_compat(
            compress_after='30 days'
        )
        
        # Set up retention policy using compatibility function
        logger.info("Setting up retention policy")
        setattr(db_manager, "setup_retention_policy_compat", setup_retention_policy_compat.__get__(db_manager))
        await db_manager.setup_retention_policy_compat(
            retention_period='1 year'
        )
        
        # Create continuous aggregates
        logger.info("Creating standard continuous aggregates")
        await db_manager.create_standard_continuous_aggregates()
        
        # Get database statistics
        logger.info("Getting database statistics")
        stats = await db_manager.get_database_stats()
        
        logger.info(f"Database Statistics:")
        logger.info(f"- Total Records: {stats['total_records']}")
        logger.info(f"- Total Symbols: {stats['total_symbols']}")
        logger.info(f"- Date Range: {stats['date_range']['oldest']} to {stats['date_range']['newest']}")
        logger.info(f"- Top Symbols: {', '.join([s['symbol'] for s in stats['top_symbols']])}")
        
        # Perform basic analysis
        logger.info("Performing basic analysis on the data")
        
        # Get daily data for all symbols for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        symbol_data = {}
        returns_data = {}
        
        for symbol in SYMBOLS:
            # Get daily data
            daily_data = await db_manager.get_market_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
                timeframe='1d'
            )
            
            if daily_data.empty:
                logger.warning(f"No daily data found for {symbol}")
                continue
                
            symbol_data[symbol] = daily_data
            
            # Calculate daily returns
            daily_returns = daily_data['close'].pct_change().dropna()
            returns_data[symbol] = daily_returns
            
            # Calculate statistics
            annualized_vol = daily_returns.std() * np.sqrt(252)
            avg_daily_return = daily_returns.mean()
            sharpe_ratio = (avg_daily_return * 252) / annualized_vol if annualized_vol != 0 else 0
            
            logger.info(f"Statistics for {symbol}:")
            logger.info(f"- Annualized Volatility: {annualized_vol:.2%}")
            logger.info(f"- Average Daily Return: {avg_daily_return:.4%}")
            logger.info(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Calculate correlation matrix if we have data for multiple symbols
        if len(returns_data) > 1:
            # Create DataFrame with all returns
            all_returns = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            corr_matrix = all_returns.corr()
            
            logger.info("Correlation Matrix:")
            for symbol1 in corr_matrix.index:
                correlations = []
                for symbol2 in corr_matrix.columns:
                    if symbol1 != symbol2:
                        correlations.append(f"{symbol2}: {corr_matrix.loc[symbol1, symbol2]:.2f}")
                
                logger.info(f"{symbol1} correlations: {', '.join(correlations)}")
        
        logger.info("POC pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in POC pipeline: {e}")
        raise
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
