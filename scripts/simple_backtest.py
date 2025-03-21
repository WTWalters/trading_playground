#!/usr/bin/env python
"""
Simple backtest script that focuses on backtesting a single predefined pair.
This avoids complex filtering and selection to demonstrate core functionality.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.db_config import DatabaseConfig
from src.database.manager import DatabaseManager
from src.market_analysis.backtest import MeanReversionBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_simple_backtest(output_dir='results/simple_backtest'):
    """Run a backtest on hard-coded pairs to demonstrate core functionality."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define fixed test pairs with known properties
    # These are the pairs we know should work from the synthetic test data
    # Hedge ratios from the cointegrated_pairs.csv file
    test_pairs = [
        # symbol1, symbol2, hedge_ratio
        ('SPY', 'IVV', 0.252),  # Updated to match exact value from analysis
        ('QQQ', 'XLK', 0.494),  # Updated to match exact value from analysis
        ('GLD', 'SLV', 0.100)   # Updated to match exact value from analysis
    ]
    
    # Try to load hedge ratios from cointegration results if available
    try:
        import pandas as pd
        import glob
        
        # Look for cointegration results in various places
        possible_files = glob.glob('results/*/cointegrated_pairs.csv') + \
                        glob.glob('results/*/*/cointegrated_pairs.csv')
        
        if possible_files:
            latest_file = max(possible_files, key=os.path.getmtime)
            logger.info(f"Found cointegration results: {latest_file}")
            
            pairs_df = pd.read_csv(latest_file)
            updated_pairs = []
            
            for symbol1, symbol2, default_ratio in test_pairs:
                pair_name = f"{symbol1}/{symbol2}"
                # Look for the pair in the results
                pair_row = pairs_df[pairs_df['pair'] == pair_name]
                
                if not pair_row.empty:
                    hedge_ratio = pair_row.iloc[0]['hedge_ratio']
                    logger.info(f"Using hedge ratio {hedge_ratio} for {pair_name} from analysis")
                    updated_pairs.append((symbol1, symbol2, hedge_ratio))
                else:
                    # Use default if not found
                    logger.info(f"Using default hedge ratio {default_ratio} for {pair_name}")
                    updated_pairs.append((symbol1, symbol2, default_ratio))
            
            # Replace test_pairs with updated values
            test_pairs = updated_pairs
    except Exception as e:
        logger.warning(f"Could not load hedge ratios from cointegration results: {e}")
        logger.info("Using default hedge ratios")

    
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # Initialize backtester
        backtester = MeanReversionBacktester(db_manager)
        
        # Run backtest for each pair
        for symbol1, symbol2, hedge_ratio in test_pairs:
            # Set dates - make sure they match our synthetic data generation
            # For synthetic data we need to use the exact range when the data was created
            # Try fetching a small amount of data to check availability
            test_data = await db_manager.get_market_data(
                symbol1, 
                datetime.now() - timedelta(days=400), 
                datetime.now(), 
                '1d', 
                'synthetic'
            )
            
            if len(test_data) > 0:
                # Use the actual data range from the synthetic dataset
                start_date = test_data.index.min()
                end_date = test_data.index.max()
            else:
                # Fallback to a reasonable range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
            
            logger.info(f"Backtesting {symbol1}/{symbol2} with date range: {start_date} to {end_date}")
            logger.info(f"Using hedge ratio: {hedge_ratio}")
            
            # Run backtest with fixed parameters
            result = await backtester.backtest_pair(
                symbol1=symbol1,
                symbol2=symbol2,
                hedge_ratio=hedge_ratio,
                start_date=start_date,
                end_date=end_date,
                entry_threshold=2.0,
                exit_threshold=0.0,
                risk_per_trade=2.0,
                initial_capital=100000.0,
                source='synthetic'  # Make sure we're using the synthetic data source
            )
            
            if result is not None:
                # Save results
                pair_output_dir = f"{output_dir}/{symbol1}_{symbol2}"
                result.save_report(pair_output_dir)
                
                logger.info(f"Results saved to {pair_output_dir}")
                logger.info(f"  Return: {result.metrics['total_return']:.2f}%")
                logger.info(f"  Sharpe: {result.metrics['sharpe_ratio']:.2f}")
                logger.info(f"  Trades: {result.metrics['num_trades']}")
                logger.info(f"  Win Rate: {result.metrics['win_rate']:.2f}%")
            else:
                logger.warning(f"No results for {symbol1}/{symbol2}")
    
    finally:
        # Close database connection
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(run_simple_backtest())
