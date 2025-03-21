#!/usr/bin/env python
"""
A robust version of the complete statistical arbitrage pipeline with error handling:
1. Load test data and analyze cointegration
2. Benchmark performance of standard vs parallel implementation
3. Run backtests on the most promising pairs
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
    """
    Run the complete statistical arbitrage pipeline with robust error handling.
    """
    # Create pipeline outputs directory
    pipeline_dir = f"results/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(pipeline_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure log file
    log_file = f"{pipeline_dir}/pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Starting robust pipeline run in {pipeline_dir}")
    
    # Track completion status for each step
    step_status = {
        "cointegration": False,
        "benchmark": False,
        "backtest": False
    }
    
    # Step 1: Run cointegration analysis
    logger.info("STEP 1: RUNNING COINTEGRATION ANALYSIS")
    analysis_dir = f"{pipeline_dir}/01_cointegration"
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        from scripts.run_complete_test import run_complete_test
        await run_complete_test(output_dir=analysis_dir)
        step_status["cointegration"] = True
        logger.info("Cointegration analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error in cointegration analysis: {e}")
        logger.warning("Will attempt to continue with other steps if possible.")
    
    # Step 2: Benchmark performance
    logger.info("STEP 2: BENCHMARKING PERFORMANCE")
    benchmark_dir = f"{pipeline_dir}/02_benchmark"
    Path(benchmark_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        from scripts.benchmark_performance import run_benchmark
        await run_benchmark(num_symbols=15, output_dir=benchmark_dir)
        step_status["benchmark"] = True
        logger.info("Performance benchmarking completed successfully.")
    except Exception as e:
        logger.error(f"Error in performance benchmarking: {e}")
        logger.warning("Continuing with next steps.")
    
    # Step 3: Run backtests (only if cointegration analysis was successful)
    logger.info("STEP 3: RUNNING BACKTESTS")
    backtest_dir = f"{pipeline_dir}/03_backtest"
    Path(backtest_dir).mkdir(parents=True, exist_ok=True)
    
    if step_status["cointegration"]:
        try:
            from scripts.run_backtest import run_backtest
            await run_backtest(input_dir=analysis_dir, output_dir=backtest_dir)
            step_status["backtest"] = True
            logger.info("Backtesting completed successfully.")
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            logger.info("Attempting to run simple backtest as fallback...")
            
            # Fallback to simple backtest if the main one fails
            try:
                from scripts.simple_backtest import run_simple_backtest
                await run_simple_backtest(output_dir=f"{backtest_dir}/simple")
                logger.info("Simple backtest completed as fallback.")
            except Exception as e2:
                logger.error(f"Error in simple backtest fallback: {e2}")
    else:
        logger.warning("Skipping backtesting due to failed cointegration analysis.")
        # Try simple backtest as a fallback
        try:
            logger.info("Attempting simple backtest as fallback...")
            from scripts.simple_backtest import run_simple_backtest
            await run_simple_backtest(output_dir=f"{backtest_dir}/simple")
            logger.info("Simple backtest completed as fallback.")
        except Exception as e:
            logger.error(f"Error in simple backtest fallback: {e}")
    
    # Generate summary report regardless of step completion status
    logger.info("GENERATING PIPELINE SUMMARY")
    try:
        with open(f"{pipeline_dir}/pipeline_summary.md", "w") as f:
            f.write("# Statistical Arbitrage Pipeline Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall status
            f.write("## Pipeline Status\n\n")
            f.write("| Step | Status | Location |\n")
            f.write("|------|--------|----------|\n")
            f.write(f"| Cointegration Analysis | {'✅ Completed' if step_status['cointegration'] else '❌ Failed'} | [{analysis_dir}]({analysis_dir}) |\n")
            f.write(f"| Performance Benchmarking | {'✅ Completed' if step_status['benchmark'] else '❌ Failed'} | [{benchmark_dir}]({benchmark_dir}) |\n")
            f.write(f"| Backtesting | {'✅ Completed' if step_status['backtest'] else '❌ Failed'} | [{backtest_dir}]({backtest_dir}) |\n\n")
            
            # Cointegration Results (if available)
            f.write("## Cointegration Results\n\n")
            if step_status["cointegration"] and os.path.exists(f"{analysis_dir}/cointegrated_pairs.csv"):
                import pandas as pd
                pairs_df = pd.read_csv(f"{analysis_dir}/cointegrated_pairs.csv")
                
                # Find the correlation column
                correlation_col = 'correlation'
                if correlation_col not in pairs_df.columns:
                    if 'correlation_x' in pairs_df.columns:
                        correlation_col = 'correlation_x'
                    elif 'correlation_y' in pairs_df.columns:
                        correlation_col = 'correlation_y'
                
                f.write(f"Found {len(pairs_df)} cointegrated pairs.\n\n")
                f.write("### Top Pairs by Correlation\n\n")
                f.write("| Pair | Correlation | Hedge Ratio | EG p-value | Johansen p-value |\n")
                f.write("|------|------------|-------------|------------|------------------|\n")
                
                # Sort by correlation and take top 5
                if len(pairs_df) > 0:
                    top_pairs = pairs_df.sort_values(correlation_col, ascending=False).head(5)
                    for _, row in top_pairs.iterrows():
                        corr_value = row[correlation_col] if correlation_col in row else 'N/A'
                        f.write(f"| {row['pair']} | {corr_value:.4f} | {row['hedge_ratio']:.4f} | {row['eg_pvalue']:.6f} | {row['johansen_pvalue']:.6f} |\n")
                else:
                    f.write("No cointegrated pairs found.\n\n")
            else:
                f.write("No cointegration results available.\n\n")
            
            # Backtest Results (if available)
            f.write("\n## Backtest Results\n\n")
            backtest_summary_file = f"{backtest_dir}/backtest_summary.csv"
            simple_backtest_dir = f"{backtest_dir}/simple"
            
            if step_status["backtest"] and os.path.exists(backtest_summary_file):
                import pandas as pd
                backtest_df = pd.read_csv(backtest_summary_file)
                
                f.write(f"Tested {len(backtest_df)} strategy variations.\n\n")
                
                # Best strategy by Sharpe ratio
                if len(backtest_df) > 0:
                    best_sharpe = backtest_df.sort_values('sharpe_ratio', ascending=False).iloc[0]
                    f.write("### Best Strategy (by Sharpe Ratio)\n\n")
                    f.write(f"- Pair: **{best_sharpe['pair']}**\n")
                    f.write(f"- Entry Z-score: {best_sharpe['entry_threshold']:.1f}\n")
                    f.write(f"- Exit Z-score: {best_sharpe['exit_threshold']:.1f}\n")
                    f.write(f"- Risk per trade: {best_sharpe['risk_per_trade']:.1f}%\n")
                    f.write(f"- Total Return: {best_sharpe['total_return']:.2f}%\n")
                    f.write(f"- Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}\n")
                    f.write(f"- Max Drawdown: {best_sharpe['max_drawdown']:.2f}%\n")
                    f.write(f"- Win Rate: {best_sharpe['win_rate']:.1f}%\n\n")
                    
                    # Best strategy by Return
                    best_return = backtest_df.sort_values('total_return', ascending=False).iloc[0]
                    f.write("### Best Strategy (by Total Return)\n\n")
                    f.write(f"- Pair: **{best_return['pair']}**\n")
                    f.write(f"- Entry Z-score: {best_return['entry_threshold']:.1f}\n")
                    f.write(f"- Exit Z-score: {best_return['exit_threshold']:.1f}\n")
                    f.write(f"- Risk per trade: {best_return['risk_per_trade']:.1f}%\n")
                    f.write(f"- Total Return: {best_return['total_return']:.2f}%\n")
                    f.write(f"- Sharpe Ratio: {best_return['sharpe_ratio']:.2f}\n")
                    f.write(f"- Max Drawdown: {best_return['max_drawdown']:.2f}%\n")
                    f.write(f"- Win Rate: {best_return['win_rate']:.1f}%\n\n")
                else:
                    f.write("No backtest strategies were successful.\n\n")
                    
            elif os.path.exists(simple_backtest_dir):
                f.write("Used simple backtest as fallback. See detailed results in the simple backtest directory.\n\n")
            else:
                f.write("No backtest results available.\n\n")
            
            # Performance Results (if available)
            f.write("\n## Performance Benchmarking Results\n\n")
            if step_status["benchmark"] and os.path.exists(f"{benchmark_dir}/benchmark_results.csv"):
                import pandas as pd
                benchmark_df = pd.read_csv(f"{benchmark_dir}/benchmark_results.csv")
                
                if len(benchmark_df) > 0:
                    f.write("### Performance Comparison\n\n")
                    f.write("| Implementation | Time (s) | Pairs Tested | Speedup Factor |\n")
                    f.write("|----------------|----------|-------------|----------------|\n")
                    
                    for _, row in benchmark_df.iterrows():
                        f.write(f"| {row['implementation']} | {row['time_seconds']:.2f} | {row['pairs_tested']} | {row['speedup_factor']:.2f} |\n")
                        
                    # Add projection if available
                    if 'projected_time_100_symbols' in benchmark_df.columns:
                        f.write("\n### Scaling Projection\n\n")
                        f.write("| Implementation | Projected Time for 100 Symbols (min) |\n")
                        f.write("|----------------|--------------------------------------|\n")
                        for _, row in benchmark_df.iterrows():
                            proj_time = row['projected_time_100_symbols'] / 60 if 'projected_time_100_symbols' in row else 'N/A'
                            if isinstance(proj_time, (int, float)):
                                f.write(f"| {row['implementation']} | {proj_time:.2f} |\n")
                            else:
                                f.write(f"| {row['implementation']} | {proj_time} |\n")
                else:
                    f.write("No benchmark results available.\n\n")
            else:
                f.write("No performance benchmark results available.\n\n")
            
            # Next Steps
            f.write("\n## Next Steps\n\n")
            f.write("1. Review the backtest results to identify the optimal trading parameters\n")
            f.write("2. Implement the adaptive parameter system for different market regimes\n")
            f.write("3. Add walk-forward testing for more realistic performance evaluation\n")
            f.write("4. Set up monitoring for pair stability in production\n")
            f.write("5. Implement real-time signal generation with the optimal parameters\n")
            
            # Technical Recommendations
            f.write("\n## Technical Recommendations\n\n")
            f.write("Based on the pipeline results, here are recommendations for next development steps:\n\n")
            
            # If we had errors, add recommendations to fix them
            missing_steps = [step for step, status in step_status.items() if not status]
            if missing_steps:
                f.write("### Error Recovery\n\n")
                for step in missing_steps:
                    f.write(f"- Fix the {step} step that failed in this pipeline run\n")
                f.write("\n")
                
            # Add general recommendations
            f.write("### System Improvements\n\n")
            f.write("- Implement the MarketRegimeDetector class to adapt parameters to changing market conditions\n")
            f.write("- Add a robust data validation layer between pipeline components\n")
            f.write("- Create a unified monitoring dashboard for the statistical arbitrage system\n")
            f.write("- Develop a component for detecting pair breakdown in real-time\n")
            
        logger.info(f"Pipeline summary saved to {pipeline_dir}/pipeline_summary.md")
    except Exception as e:
        logger.error(f"Error creating pipeline summary: {e}")
    
    # Final status message
    completed_steps = sum(1 for status in step_status.values() if status)
    total_steps = len(step_status)
    
    if completed_steps == total_steps:
        logger.info(f"Pipeline completed successfully! All {total_steps} steps completed.")
    else:
        logger.warning(f"Pipeline completed with issues. {completed_steps}/{total_steps} steps succeeded.")
        
    logger.info(f"Results directory: {pipeline_dir}")
    
    return pipeline_dir

if __name__ == "__main__":
    asyncio.run(run_robust_pipeline())
