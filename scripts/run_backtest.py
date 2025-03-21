#!/usr/bin/env python
"""
Run a backtest on the top cointegrated pairs found in analysis.
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
from src.market_analysis.backtest import MeanReversionBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_backtest(input_dir=None, output_dir='results/backtest'):
    """
    Run a backtest on the top pairs from the input directory.
    
    Args:
        input_dir: Directory with cointegration analysis results
        output_dir: Directory to save backtest results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for input directory
    if input_dir is None:
        # Check both possible locations
        if os.path.exists('results/analysis'):
            input_dir = 'results/analysis'
        elif os.path.exists('results/complete_test'):
            input_dir = 'results/complete_test'
        else:
            # Try to find the latest pipeline directory
            pipeline_dirs = [d for d in os.listdir('results') if d.startswith('pipeline_')]
            if pipeline_dirs:
                latest_dir = sorted(pipeline_dirs)[-1]
                input_dir = f'results/{latest_dir}/01_cointegration'
                if os.path.exists(input_dir):
                    logger.info(f"Found pipeline directory: {input_dir}")
                else:
                    logger.error(f"Pipeline directory structure not as expected in {latest_dir}")
                    return
            else:
                logger.error("No input directory specified and couldn't find default locations")
                return
    
    logger.info(f"Using input directory: {input_dir}")
    
    # Load the pairs data
    try:
        pairs_df = pd.read_csv(f"{input_dir}/cointegrated_pairs.csv")
        half_life_df = pd.read_csv(f"{input_dir}/half_life_results.csv")
        
        if pairs_df.empty:
            logger.error("No pairs found in input directory")
            return
        
        logger.info(f"Loaded {len(pairs_df)} pairs from {input_dir}")
        
        # Merge with half-life information
        pairs_df = pd.merge(pairs_df, half_life_df, on='pair', how='left')
        
        # Sort by statistical quality (using p-value and stability)
        # Debug column names
        logger.info(f"Available columns in pairs_df: {pairs_df.columns.tolist()}")
        logger.info(f"Available columns in half_life_df: {half_life_df.columns.tolist()}")
        
        # Safety conversion in case column types don't match
        # Handle cases where correlation might be renamed during the merge
        correlation_col = 'correlation'
        if 'correlation' not in pairs_df.columns:
            if 'correlation_x' in pairs_df.columns:
                correlation_col = 'correlation_x'
            elif 'correlation_y' in pairs_df.columns:
                correlation_col = 'correlation_y'
                
        logger.info(f"Using correlation column: {correlation_col}")
        
        pairs_df[correlation_col] = pd.to_numeric(pairs_df[correlation_col], errors='coerce')
        pairs_df['eg_pvalue'] = pd.to_numeric(pairs_df['eg_pvalue'], errors='coerce')
        pairs_df['johansen_pvalue'] = pd.to_numeric(pairs_df['johansen_pvalue'], errors='coerce')
        
        pairs_df['quality_score'] = (
            pairs_df[correlation_col].abs() * 0.4 + 
            (1 / (pairs_df['eg_pvalue'] + 0.00001)) * 0.3 +
            (1 / (pairs_df['johansen_pvalue'] + 0.00001)) * 0.3
        )
        
        # Filter for mean-reverting pairs with reasonable half-life
        # Use safer filtering to handle potential missing values or type mismatches
        quality_pairs = pairs_df.copy()
        
        # Convert boolean column safely
        quality_pairs['is_mean_reverting'] = quality_pairs['is_mean_reverting'].astype(str).str.lower() == 'true'
        
        # Apply filters one by one with error handling
        try:
            # Filter mean-reverting pairs
            if 'is_mean_reverting' in quality_pairs.columns:
                quality_pairs = quality_pairs[quality_pairs['is_mean_reverting'] == True]
            
            # Filter by half-life upper bound
            if 'half_life' in quality_pairs.columns:
                quality_pairs = quality_pairs[quality_pairs['half_life'] < 15]
            
            # Filter by half-life lower bound
            if 'half_life' in quality_pairs.columns:
                quality_pairs = quality_pairs[quality_pairs['half_life'] > 0.1]
                
            # Sort by quality score
            quality_pairs = quality_pairs.sort_values('quality_score', ascending=False)
        except Exception as e:
            logger.warning(f"Error in filtering pairs: {e}")
            # Fallback: just sort by correlation if filtering fails
            quality_pairs = pairs_df.sort_values(correlation_col, ascending=False)
        
        # Take the top 3 pairs for backtesting
        top_pairs = quality_pairs.head(3)
        
        logger.info(f"Selected top {len(top_pairs)} pairs for backtesting:")
        for _, pair in top_pairs.iterrows():
            correlation_value = pair[correlation_col] if correlation_col in pair else 'N/A'
            logger.info(f"  {pair['pair']} - Half-life: {pair['half_life']:.2f} days, Correlation: {correlation_value}")
        
        # Initialize database
        db_config = DatabaseConfig()
        db_manager = DatabaseManager(db_config)
        await db_manager.initialize()
        
        try:
            # Initialize backtester
            backtester = MeanReversionBacktester(db_manager)
            
            # Run backtest for each pair
            backtest_results = []
            
            for _, pair in top_pairs.iterrows():
                # Parse pair symbols
                symbol1, symbol2 = pair['pair'].split('/')
                
                # Determine the date range for which we have synthetic data
                # We need to query the exact range where our synthetic data exists
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
                    logger.info(f"Using actual data range from synthetic dataset: {start_date} to {end_date}")
                else:
                    # Fallback to default range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)  # One year backtest
                    logger.warning(f"No synthetic data found for {symbol1}, using default date range")
                
                logger.info(f"Backtesting {symbol1}/{symbol2} from {start_date.date()} to {end_date.date()}")
                
                # Run backtest with different parameter variations
                for entry_z in [2.0, 2.5, 3.0]:
                    for exit_z in [0.0, 0.5, 1.0]:
                        for risk_pct in [1.0, 2.0]:
                            # Skip some combinations to reduce total number
                            if entry_z == 2.5 and exit_z == 0.5:
                                continue
                                
                            variation_name = f"{symbol1}_{symbol2}_entry{entry_z}_exit{exit_z}_risk{risk_pct}"
                            
                            # Run backtest
                            result = await backtester.backtest_pair(
                                symbol1=symbol1,
                                symbol2=symbol2,
                                hedge_ratio=pair['hedge_ratio'],
                                start_date=start_date,
                                end_date=end_date,
                                entry_threshold=entry_z,
                                exit_threshold=exit_z,
                                risk_per_trade=risk_pct,
                                initial_capital=100000.0,
                                source='synthetic'  # Explicitly use synthetic data source
                            )
                            
                            if result is not None:
                                # Save results
                                pair_output_dir = f"{output_dir}/{variation_name}"
                                result.save_report(pair_output_dir)
                                
                                # Add to summary
                                backtest_results.append({
                                    'pair': pair['pair'],
                                    'entry_threshold': entry_z,
                                    'exit_threshold': exit_z,
                                    'risk_per_trade': risk_pct,
                                    'total_return': result.metrics['total_return'],
                                    'sharpe_ratio': result.metrics['sharpe_ratio'],
                                    'max_drawdown': result.metrics['max_drawdown'],
                                    'num_trades': result.metrics['num_trades'],
                                    'win_rate': result.metrics['win_rate'],
                                    'profit_factor': result.metrics['profit_factor'],
                                    'half_life': pair['half_life']
                                })
                                
                                logger.info(f"  {variation_name}: Return: {result.metrics['total_return']:.2f}%, Sharpe: {result.metrics['sharpe_ratio']:.2f}")
            
            # Create summary report
            if backtest_results:
                summary_df = pd.DataFrame(backtest_results)
                summary_df.to_csv(f"{output_dir}/backtest_summary.csv", index=False)
                
                # Create markdown summary report
                with open(f"{output_dir}/backtest_summary.md", "w") as f:
                    f.write("# Mean Reversion Backtest Results\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("## Top Performing Configurations by Return\n\n")
                    top_return = summary_df.sort_values('total_return', ascending=False).head(5)
                    f.write("| Pair | Entry Z | Exit Z | Risk % | Return (%) | Sharpe | Max DD (%) | # Trades | Win Rate (%) |\n")
                    f.write("|------|---------|--------|--------|------------|--------|------------|----------|-------------|\n")
                    for _, row in top_return.iterrows():
                        f.write(f"| {row['pair']} | {row['entry_threshold']:.1f} | {row['exit_threshold']:.1f} | {row['risk_per_trade']:.1f} | {row['total_return']:.2f} | {row['sharpe_ratio']:.2f} | {row['max_drawdown']:.2f} | {row['num_trades']} | {row['win_rate']:.1f} |\n")
                    
                    f.write("\n## Top Performing Configurations by Sharpe Ratio\n\n")
                    top_sharpe = summary_df.sort_values('sharpe_ratio', ascending=False).head(5)
                    f.write("| Pair | Entry Z | Exit Z | Risk % | Return (%) | Sharpe | Max DD (%) | # Trades | Win Rate (%) |\n")
                    f.write("|------|---------|--------|--------|------------|--------|------------|----------|-------------|\n")
                    for _, row in top_sharpe.iterrows():
                        f.write(f"| {row['pair']} | {row['entry_threshold']:.1f} | {row['exit_threshold']:.1f} | {row['risk_per_trade']:.1f} | {row['total_return']:.2f} | {row['sharpe_ratio']:.2f} | {row['max_drawdown']:.2f} | {row['num_trades']} | {row['win_rate']:.1f} |\n")
                    
                    f.write("\n## Performance by Pair\n\n")
                    for pair_name in summary_df['pair'].unique():
                        f.write(f"### {pair_name}\n\n")
                        pair_data = summary_df[summary_df['pair'] == pair_name]
                        pair_best = pair_data.sort_values('sharpe_ratio', ascending=False).iloc[0]
                        
                        f.write(f"- Best configuration: Entry Z = {pair_best['entry_threshold']:.1f}, Exit Z = {pair_best['exit_threshold']:.1f}, Risk = {pair_best['risk_per_trade']:.1f}%\n")
                        f.write(f"- Return: {pair_best['total_return']:.2f}%\n")
                        f.write(f"- Sharpe Ratio: {pair_best['sharpe_ratio']:.2f}\n")
                        f.write(f"- Max Drawdown: {pair_best['max_drawdown']:.2f}%\n")
                        f.write(f"- Win Rate: {pair_best['win_rate']:.1f}%\n")
                        f.write(f"- Number of Trades: {pair_best['num_trades']}\n")
                        f.write(f"- Half-life: {pair_best['half_life']:.2f} days\n\n")
                    
                    # Overall conclusion
                    f.write("\n## Conclusion\n\n")
                    
                    # Best overall strategy
                    best_overall = summary_df.sort_values('sharpe_ratio', ascending=False).iloc[0]
                    f.write(f"The best overall strategy is trading the **{best_overall['pair']}** pair with:\n\n")
                    f.write(f"- Entry Z-score: {best_overall['entry_threshold']:.1f}\n")
                    f.write(f"- Exit Z-score: {best_overall['exit_threshold']:.1f}\n")
                    f.write(f"- Risk per trade: {best_overall['risk_per_trade']:.1f}%\n\n")
                    f.write(f"This strategy produced:\n\n")
                    f.write(f"- Total Return: {best_overall['total_return']:.2f}%\n")
                    f.write(f"- Sharpe Ratio: {best_overall['sharpe_ratio']:.2f}\n")
                    f.write(f"- Maximum Drawdown: {best_overall['max_drawdown']:.2f}%\n")
                    f.write(f"- Win Rate: {best_overall['win_rate']:.1f}%\n\n")
                    
                    # Parameter sensitivity analysis
                    f.write("### Parameter Sensitivity Analysis\n\n")
                    f.write("**Entry Z-score**: ")
                    entry_analysis = summary_df.groupby('entry_threshold')[['total_return', 'sharpe_ratio', 'win_rate']].mean()
                    best_entry = entry_analysis['sharpe_ratio'].idxmax()
                    f.write(f"The optimal entry threshold appears to be around {best_entry:.1f}. ")
                    f.write(f"Higher thresholds generally result in fewer trades but higher win rates.\n\n")
                    
                    f.write("**Exit Z-score**: ")
                    exit_analysis = summary_df.groupby('exit_threshold')[['total_return', 'sharpe_ratio', 'win_rate']].mean()
                    best_exit = exit_analysis['sharpe_ratio'].idxmax()
                    f.write(f"The optimal exit threshold appears to be around {best_exit:.1f}. ")
                    f.write(f"Lower exit thresholds capture more mean reversion but may exit too early.\n\n")
                    
                    f.write("**Risk per Trade**: ")
                    risk_analysis = summary_df.groupby('risk_per_trade')[['total_return', 'sharpe_ratio', 'max_drawdown']].mean()
                    best_risk = risk_analysis['sharpe_ratio'].idxmax()
                    f.write(f"The optimal risk percentage appears to be {best_risk:.1f}%. ")
                    f.write(f"Higher risk leads to higher returns but also increases maximum drawdown.\n\n")
                    
                    # Next steps
                    f.write("### Next Steps\n\n")
                    f.write("1. Implement the top strategy in live trading\n")
                    f.write("2. Monitor pair stability and recalibrate if necessary\n")
                    f.write("3. Consider expanding to additional pairs with similar characteristics\n")
                    f.write("4. Add stress testing with more extreme market conditions\n")
                    f.write("5. Implement additional risk management features (trailing stops, position sizing)\n")
                
                logger.info(f"Backtest summary saved to {output_dir}/backtest_summary.md")
                
        finally:
            # Close database connection
            await db_manager.close()
            
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_backtest())
