#!/usr/bin/env python
"""
Robust pipeline script that runs the complete analysis with error handling and fallbacks.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_robust_pipeline():
    """Run the complete analysis pipeline with robust error handling."""
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_dir = f"results/pipeline_{timestamp}"
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # Configure log file
    log_file = f"{pipeline_dir}/pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Starting robust pipeline run in {pipeline_dir}")
    
    # Step 1: Run cointegration analysis
    cointegration_dir = f"{pipeline_dir}/01_cointegration"
    os.makedirs(cointegration_dir, exist_ok=True)
    
    try:
        logger.info("Running cointegration analysis...")
        from scripts.run_cointegration_analysis import run_cointegration_analysis
        await run_cointegration_analysis(output_dir=cointegration_dir)
        logger.info("Cointegration analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error in cointegration analysis: {e}")
        logger.warning("Will attempt to continue with other steps if possible.")
    
    # Step 2: Performance testing (optional)
    performance_dir = f"{pipeline_dir}/02_performance"
    os.makedirs(performance_dir, exist_ok=True)
    
    try:
        if os.path.exists(f"{cointegration_dir}/cointegrated_pairs.csv"):
            logger.info("Running parallel performance testing...")
            from scripts.run_performance_test import run_performance_test
            await run_performance_test(
                input_dir=cointegration_dir,
                output_dir=performance_dir
            )
            logger.info("Performance testing completed successfully.")
        else:
            logger.warning("Skipping performance testing due to missing input files.")
    except Exception as e:
        logger.error(f"Error in performance testing: {e}")
        logger.warning("Continuing with next steps.")
    
    # Step 3: Backtesting
    backtest_dir = f"{pipeline_dir}/03_backtest"
    os.makedirs(backtest_dir, exist_ok=True)
    
    try:
        logger.info("Running backtest analysis...")
        from scripts.run_backtest import run_backtest
        await run_backtest(
            input_dir=cointegration_dir,
            output_dir=backtest_dir
        )
        logger.info("Backtesting completed successfully.")
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        logger.info("Attempting to run simple backtest as fallback...")
        
        # Fallback to simple backtest if the main one fails
        try:
            from scripts.simple_backtest import run_simple_backtest
            await run_simple_backtest(output_dir=f"{backtest_dir}/simple")
            logger.info("Simple backtest completed successfully.")
        except Exception as e2:
            logger.error(f"Error in simple backtest fallback: {e2}")
    
    # Create pipeline summary
    try:
        with open(f"{pipeline_dir}/pipeline_summary.md", "w") as f:
            f.write(f"# TITAN Trading Pipeline Summary\n\n")
            f.write(f"**Run Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Cointegration results
            f.write("## Cointegration Analysis\n\n")
            if os.path.exists(f"{cointegration_dir}/cointegrated_pairs.csv"):
                import pandas as pd
                pairs_df = pd.read_csv(f"{cointegration_dir}/cointegrated_pairs.csv")
                
                f.write(f"Found {len(pairs_df)} cointegrated pairs.\n\n")
                f.write("### Top Pairs by Correlation\n\n")
                f.write("| Pair | Correlation | Hedge Ratio | EG p-value | Johansen p-value |\n")
                f.write("|------|------------|-------------|------------|------------------|\n")
                
                # Handle the case where correlation might have different column names
                correlation_col = 'correlation'
                if correlation_col not in pairs_df.columns:
                    if 'correlation_x' in pairs_df.columns:
                        correlation_col = 'correlation_x'
                    elif 'correlation_y' in pairs_df.columns:
                        correlation_col = 'correlation_y'
                
                top_pairs = pairs_df.sort_values(correlation_col, ascending=False).head(5)
                for _, row in top_pairs.iterrows():
                    corr_value = row[correlation_col] if correlation_col in row else 'N/A'
                    f.write(f"| {row['pair']} | {corr_value:.4f} | {row['hedge_ratio']:.4f} | {row['eg_pvalue']:.6f} | {row['johansen_pvalue']:.6f} |\n")
            else:
                f.write("No cointegration results available.\n\n")
            
            # Backtest results
            f.write("\n## Backtest Results\n\n")
            if os.path.exists(f"{backtest_dir}/backtest_summary.csv"):
                backtest_df = pd.read_csv(f"{backtest_dir}/backtest_summary.csv")
                
                f.write(f"Tested {len(backtest_df)} strategy variations.\n\n")
                
                # Best strategy by Sharpe ratio
                best_sharpe = backtest_df.sort_values('sharpe_ratio', ascending=False).iloc[0]
                f.write("### Best Strategy (by Sharpe Ratio)\n\n")
                f.write(f"- Pair: **{best_sharpe['pair']}**\n")
                f.write(f"- Entry Z-score: {best_sharpe['entry_threshold']:.1f}\n")
                f.write(f"- Exit Z-score: {best_sharpe['exit_threshold']:.1f}\n")
                f.write(f"- Risk per trade: {best_sharpe['risk_per_trade']:.1f}%\n")
                f.write(f"- Total Return: {best_sharpe['total_return']:.2f}%\n")
                f.write(f"- Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}\n")
                f.write(f"- Max Drawdown: {best_sharpe['max_drawdown']:.2f}%\n")
                f.write(f"- Win Rate: {best_sharpe['win_rate']:.1f}%\n")
                
            elif os.path.exists(f"{backtest_dir}/simple"):
                f.write("Used simple backtest as fallback. See detailed results in the simple backtest directory.\n\n")
            else:
                f.write("No backtest results available.\n\n")
            
            # Next steps
            f.write("\n## Next Steps\n\n")
            f.write("1. Review the cointegration results to validate pair selection\n")
            f.write("2. Examine backtest performance for the top strategies\n")
            f.write("3. Compare current results with previous runs\n")
            f.write("4. Set up monitoring for the selected pairs\n")
            f.write("5. Consider implementing adaptive parameter adjustment\n")
        
        logger.info(f"Pipeline summary written to {pipeline_dir}/pipeline_summary.md")
    except Exception as e:
        logger.error(f"Error creating pipeline summary: {e}")
    
    logger.info(f"Pipeline run completed. Results in {pipeline_dir}")
    logger.info(f"Log file: {log_file}")
    
    return pipeline_dir

if __name__ == "__main__":
    asyncio.run(run_robust_pipeline())
