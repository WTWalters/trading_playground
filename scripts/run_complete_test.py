#!/usr/bin/env python
"""
Run a complete test of the cointegration framework.
This script loads synthetic test data and then analyzes it in a single session.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.db_config import DatabaseConfig
from src.database.manager import DatabaseManager
from src.market_analysis.cointegration import CointegrationTester
from src.market_analysis.mean_reversion import MeanReversionAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_complete_test(output_dir='results/complete_test'):
    """
    Run a complete test of the cointegration framework.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        ###########################################
        # PART 1: LOAD SYNTHETIC TEST DATA
        ###########################################
        
        logger.info("LOADING SYNTHETIC TEST DATA")
        
        # Set fixed seed for reproducibility
        np.random.seed(42)
        
        # Generate dates for a year of daily data
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Define ETF symbols
        symbols = ['SPY', 'IVV', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLU', 'GLD', 'SLV']
        
        # Generate base price series with random walk
        base_series1 = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates)))
        base_series2 = 200 * np.cumprod(1 + np.random.normal(0.0006, 0.012, len(dates)))
        base_series3 = 50 * np.cumprod(1 + np.random.normal(0.0004, 0.009, len(dates)))
        
        # Create price series for each symbol
        price_data = {}
        
        # SPY and IVV - Cointegrated pair (S&P 500 ETFs)
        price_data['SPY'] = base_series1
        price_data['IVV'] = 0.25 * base_series1 + np.random.normal(0, 0.5, len(dates))
        
        # QQQ and XLK - Cointegrated pair (Technology ETFs)
        price_data['QQQ'] = base_series2
        price_data['XLK'] = 0.5 * base_series2 + np.random.normal(0, 1.0, len(dates))
        
        # XLF - Financial ETF (independent)
        price_data['XLF'] = 40 * np.cumprod(1 + np.random.normal(0.0003, 0.011, len(dates)))
        
        # XLE - Energy ETF (independent)
        price_data['XLE'] = 60 * np.cumprod(1 + np.random.normal(0.0002, 0.015, len(dates)))
        
        # XLV - Healthcare ETF (independent)
        price_data['XLV'] = 70 * np.cumprod(1 + np.random.normal(0.0004, 0.008, len(dates)))
        
        # XLU - Utilities ETF (weakly correlated with XLV)
        price_data['XLU'] = 0.3 * price_data['XLV'] + 50 * np.cumprod(1 + np.random.normal(0.0001, 0.007, len(dates)))
        
        # GLD and SLV - Cointegrated pair (Precious metals ETFs)
        price_data['GLD'] = base_series3
        price_data['SLV'] = 0.1 * base_series3 + np.random.normal(0, 0.3, len(dates))
        
        # Store each series in the database
        for symbol in symbols:
            # Create dataframe
            df = pd.DataFrame({
                'open': price_data[symbol] * 0.998,
                'high': price_data[symbol] * 1.005,
                'low': price_data[symbol] * 0.995,
                'close': price_data[symbol],
                'volume': np.random.randint(100000, 10000000, len(dates))
            }, index=dates)
            
            # Store in database
            await db_manager.store_market_data(
                df, symbol, 'synthetic', '1d'
            )
            
            logger.info(f"Stored {len(df)} records for {symbol}")
            
        logger.info("Test data loading complete!")
        
        # Log the known cointegrated pairs
        logger.info("Known cointegrated pairs:")
        logger.info("1. SPY/IVV - S&P 500 ETFs")
        logger.info("2. QQQ/XLK - Technology ETFs")
        logger.info("3. GLD/SLV - Precious metals ETFs")
        
        ###########################################
        # PART 2: ANALYZE COINTEGRATION
        ###########################################
        
        logger.info("\nANALYZING COINTEGRATION")
        
        # Initialize our testing components
        cointegration_tester = CointegrationTester(db_manager)
        mean_reversion_analyzer = MeanReversionAnalyzer(db_manager)
        
        # 1. Find cointegrated pairs
        logger.info(f"Searching for cointegrated pairs among {len(symbols)} symbols...")
        pairs = await cointegration_tester.select_pairs(
            symbols, start_date, end_date, '1d', min_correlation=0.6, source='synthetic'
        )
        
        logger.info(f"Found {len(pairs)} cointegrated pairs")
        
        # Save pair results
        pairs_df = pd.DataFrame([
            {
                'pair': f"{p['symbol1']}/{p['symbol2']}",
                'correlation': p['correlation'],
                'hedge_ratio': p['hedge_ratio'],
                'eg_pvalue': p['engle_granger_result']['adf_results']['p_value'],
                'johansen_pvalue': p['johansen_result']['p_value'],
            }
            for p in pairs
        ])
        
        pairs_df.to_csv(f"{output_dir}/cointegrated_pairs.csv", index=False)
        
        # 2. Analyze stability for the top 5 pairs
        top_pairs = sorted(pairs, key=lambda x: x['correlation'], reverse=True)[:5]
        
        stability_results = []
        for pair in top_pairs:
            logger.info(f"Analyzing stability for {pair['symbol1']}/{pair['symbol2']}...")
            stability = await cointegration_tester.calculate_cointegration_stability(
                pair['symbol1'], pair['symbol2'], start_date, end_date, source='synthetic'
            )
            
            stability_results.append({
                'pair': f"{pair['symbol1']}/{pair['symbol2']}",
                'stability_ratio': stability['stability_ratio'],
                'hedge_ratio_mean': stability['hedge_ratio']['mean'],
                'hedge_ratio_std': stability['hedge_ratio']['std'],
                'is_stable': stability['is_stable']
            })
        
        # Save stability results
        stability_df = pd.DataFrame(stability_results)
        stability_df.to_csv(f"{output_dir}/stability_results.csv", index=False)
        
        # 3. Calculate half-life and analyze mean-reversion for each pair
        half_life_results = []
        for pair in pairs:
            # Get the signals
            signals = await mean_reversion_analyzer.generate_mean_reversion_signals(
                pair['symbol1'], pair['symbol2'], pair['hedge_ratio'],
                start_date, end_date, entry_threshold=2.0, source='synthetic'
            )
            
            # Get the pair statistics
            stats = await mean_reversion_analyzer.analyze_pair_statistics(
                pair['symbol1'], pair['symbol2'], pair['hedge_ratio'],
                start_date, end_date, source='synthetic'
            )
            
            half_life_results.append({
                'pair': f"{pair['symbol1']}/{pair['symbol2']}",
                'half_life': stats['half_life'],
                'is_mean_reverting': stats['is_mean_reverting'],
                'correlation': stats['correlation'],
                'spread_volatility': stats['volatility']['spread'],
                'zscore_threshold_90': stats['zscore_thresholds']['90th_percentile']
            })
            
            # For the best pair, save the signals and create visualization
            if pair == top_pairs[0]:
                # Save signals
                signals_df = pd.DataFrame(signals['signals'])
                signals_df.to_csv(f"{output_dir}/best_pair_signals.csv", index=False)
                
                # Create visualization
                create_pair_visualization(
                    signals['signals'], 
                    pair['symbol1'], 
                    pair['symbol2'],
                    f"{output_dir}/best_pair_visualization.png"
                )
        
        # Save half-life results
        half_life_df = pd.DataFrame(half_life_results)
        half_life_df.to_csv(f"{output_dir}/half_life_results.csv", index=False)
        
        # 4. Generate the summary report
        generate_summary_report(pairs, stability_results, half_life_results, output_dir)
        
        logger.info(f"All results saved to {output_dir}")
        
    finally:
        # Close database connection
        await db_manager.close()

def create_pair_visualization(signals, symbol1, symbol2, output_path):
    """Create visualization for pair trading signals."""
    try:
        import matplotlib.pyplot as plt
        
        signals_df = pd.DataFrame(signals)
        signals_df['time'] = pd.to_datetime(signals_df['time'])
        signals_df.set_index('time', inplace=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot normalized prices
        normalized1 = signals_df[symbol1] / signals_df[symbol1].iloc[0]
        normalized2 = signals_df[symbol2] / signals_df[symbol2].iloc[0]
        
        ax1.plot(normalized1, label=symbol1)
        ax1.plot(normalized2, label=symbol2)
        ax1.set_title(f"Normalized Prices: {symbol1} vs {symbol2}")
        ax1.legend()
        ax1.grid(True)
        
        # Plot spread
        ax2.plot(signals_df['spread'], color='green')
        ax2.set_title("Spread")
        ax2.grid(True)
        
        # Plot Z-score and signals
        ax3.plot(signals_df['zscore'], color='blue')
        ax3.axhline(y=2.0, color='r', linestyle='--', alpha=0.3)
        ax3.axhline(y=-2.0, color='r', linestyle='--', alpha=0.3)
        ax3.axhline(y=0.0, color='black', linestyle='-', alpha=0.2)
        
        # Color the background based on signals
        for i in range(len(signals_df)):
            if signals_df['signal'].iloc[i] == 1:  # Long spread
                ax3.axvspan(signals_df.index[i], signals_df.index[min(i+1, len(signals_df)-1)], 
                            alpha=0.2, color='green')
            elif signals_df['signal'].iloc[i] == -1:  # Short spread
                ax3.axvspan(signals_df.index[i], signals_df.index[min(i+1, len(signals_df)-1)], 
                            alpha=0.2, color='red')
        
        ax3.set_title("Z-Score and Signals")
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

def generate_summary_report(pairs, stability_results, half_life_results, output_dir):
    """Generate summary report for the cointegration analysis."""
    # Create a summary report with statistics
    half_life_df = pd.DataFrame(half_life_results)
    stability_df = pd.DataFrame(stability_results) if stability_results else pd.DataFrame()
    
    with open(f"{output_dir}/summary_report.md", "w") as f:
        f.write("# Cointegration Testing Framework Summary Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total pairs analyzed: {len(pairs)}\n")
        
        # Calculate total possible pairs
        n = len(set([p['symbol1'] for p in pairs] + [p['symbol2'] for p in pairs]))
        total_possible_pairs = n * (n - 1) // 2
        
        f.write(f"- Cointegrated pairs found: {len(pairs)}\n")
        if total_possible_pairs > 0:
            f.write(f"- Percentage of cointegrated pairs: {len(pairs) / total_possible_pairs * 100:.2f}%\n\n")
        
        if not half_life_df.empty:
            f.write("## Half-Life Distribution\n\n")
            f.write(f"- Mean half-life: {half_life_df['half_life'].mean():.2f} days\n")
            f.write(f"- Median half-life: {half_life_df['half_life'].median():.2f} days\n")
            f.write(f"- Min half-life: {half_life_df['half_life'].min():.2f} days\n")
            f.write(f"- Max half-life: {half_life_df['half_life'].max():.2f} days\n\n")
        
        if not stability_df.empty:
            f.write("## Cointegration Stability\n\n")
            f.write(f"- Stable pairs: {stability_df['is_stable'].sum()} out of {len(stability_df)}\n")
            f.write(f"- Average stability ratio: {stability_df['stability_ratio'].mean():.2f}\n")
            f.write(f"- Average hedge ratio standard deviation: {stability_df['hedge_ratio_std'].mean():.4f}\n\n")
        
        f.write("## Top 5 Pairs by Correlation\n\n")
        if len(pairs) > 0:
            f.write("| Pair | Correlation | Hedge Ratio | Half-Life (days) | Mean-Reverting |\n")
            f.write("|------|-------------|-------------|------------------|----------------|\n")
            
            top_pairs = sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)[:5]
            for pair in top_pairs:
                pair_name = f"{pair['symbol1']}/{pair['symbol2']}"
                half_life_row = half_life_df[half_life_df['pair'] == pair_name]
                
                if not half_life_row.empty:
                    half_life = half_life_row['half_life'].values[0]
                    mean_reverting = "Yes" if half_life_row['is_mean_reverting'].values[0] else "No"
                else:
                    half_life = float('nan')
                    mean_reverting = "Unknown"
                
                f.write(f"| {pair_name} | {pair['correlation']:.4f} | {pair['hedge_ratio']:.4f} | {half_life:.2f} | {mean_reverting} |\n")
        
        f.write("\n## Trading Implications\n\n")
        f.write("Based on the cointegration analysis, we recommend focusing on pairs with:\n\n")
        f.write("1. Strong cointegration (low p-value in Engle-Granger and Johansen tests)\n")
        f.write("2. Stable hedge ratios (low standard deviation over time)\n")
        f.write("3. Reasonable half-life (between 5-30 days for daily trading)\n")
        f.write("4. High spread volatility (for better profit potential)\n\n")
        
        f.write("These pairs provide the best opportunity for statistical arbitrage strategies.\n")

if __name__ == "__main__":
    asyncio.run(run_complete_test())
