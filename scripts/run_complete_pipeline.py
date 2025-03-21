#!/usr/bin/env python
"""
Run the complete statistical arbitrage pipeline:
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

async def run_pipeline():
    """
    Run the complete statistical arbitrage pipeline.
    """
    # Create pipeline outputs directory
    pipeline_dir = f"results/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(pipeline_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Run cointegration analysis
        logger.info("STEP 1: RUNNING COINTEGRATION ANALYSIS")
        analysis_dir = f"{pipeline_dir}/01_cointegration"
        
        from scripts.run_complete_test import run_complete_test
        await run_complete_test(output_dir=analysis_dir)
        
        # Step 2: Benchmark performance
        logger.info("STEP 2: BENCHMARKING PERFORMANCE")
        benchmark_dir = f"{pipeline_dir}/02_benchmark"
        
        from scripts.benchmark_performance import run_benchmark
        await run_benchmark(num_symbols=15, output_dir=benchmark_dir)
        
        # Step 3: Run backtests
        logger.info("STEP 3: RUNNING BACKTESTS")
        backtest_dir = f"{pipeline_dir}/03_backtest"
        
        from scripts.run_backtest import run_backtest
        await run_backtest(input_dir=analysis_dir, output_dir=backtest_dir)
        
        # Generate summary report
        logger.info("GENERATING PIPELINE SUMMARY")
        with open(f"{pipeline_dir}/pipeline_summary.md", "w") as f:
            f.write("# Statistical Arbitrage Pipeline Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Pipeline Steps\n\n")
            f.write("1. **Cointegration Analysis**: Identified cointegrated pairs with mean-reverting properties\n")
            f.write("2. **Performance Benchmarking**: Compared standard and parallel implementations\n")
            f.write("3. **Backtesting**: Validated trading strategy with historical data\n\n")
            
            f.write("## Results Location\n\n")
            f.write(f"- Cointegration Analysis: [{analysis_dir}]({analysis_dir})\n")
            f.write(f"- Performance Benchmark: [{benchmark_dir}]({benchmark_dir})\n")
            f.write(f"- Backtest Results: [{backtest_dir}]({backtest_dir})\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review the backtest results to identify the optimal trading parameters\n")
            f.write("2. Implement the strategy with real-time data\n")
            f.write("3. Set up monitoring for pair stability\n")
            f.write("4. Deploy with appropriate risk management controls\n")
        
        logger.info(f"Pipeline completed successfully! Summary report saved to {pipeline_dir}/pipeline_summary.md")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_pipeline())
