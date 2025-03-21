#!/usr/bin/env python
"""
Test script for the cointegration testing framework.

This script demonstrates how to use the CointegrationTester and MeanReversionAnalyzer
classes to identify and analyze pairs for statistical arbitrage trading.
"""

import asyncio
import argparse
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import numpy as np
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

async def test_cointegration(symbols=None, start_date=None, end_date=None, output_dir='results'):
    """Test the cointegration framework on a list of ETF pairs.
    
    Args:
        symbols: List of symbols to test (default: popular ETFs)
        start_date: Start date for analysis (default: 1 year ago)
        end_date: End date for analysis (default: today)
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set default symbols if none provided (popular ETFs)
    if symbols is None:
        symbols = [
            # Equity ETFs
            'SPY', 'IVV', 'VOO',  # S&P 500
            'QQQ', 'XLK',         # Tech
            'XLF', 'VFH',         # Financial
            'XLE', 'VDE',         # Energy
            'XLV', 'VHT',         # Health
            'XLU', 'VPU',         # Utilities
            'XLY', 'VCR',         # Consumer Discretionary
            'XLP', 'VDC',         # Consumer Staples
            'XLI', 'VIS',         # Industrials
            'XLB', 'VAW',         # Materials
            'XLRE', 'VNQ',        # Real Estate
            'IWM', 'IWB',         # Russell indexes
            # International ETFs
            'EFA', 'VEA',         # Developed Markets
            'EEM', 'VWO',         # Emerging Markets
            # Fixed Income ETFs
            'AGG', 'BND',         # Aggregate Bond
            'LQD', 'VCIT',        # Corporate Bond
        ]
    
    # Set default dates if none provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # Initialize database manager
    db_config = DatabaseConfig()
    db_manager = DatabaseManager(db_config)
    await db_manager.initialize()
    
    try:
        # Initialize our testing components
        cointegration_tester = CointegrationTester(db_manager)
        mean_reversion_analyzer = MeanReversionAnalyzer(db_manager)
        
        # 1. Find cointegrated pairs
        logger.info(f"Searching for cointegrated pairs among {len(symbols)} symbols...")
        # Use 'synthetic' as the source for our test data
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
                pair['symbol1'], pair['symbol2'], start_date, end_date
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
                start_date, end_date, entry_threshold=2.0
            )
            
            # Get the pair statistics
            stats = await mean_reversion_analyzer.analyze_pair_statistics(
                pair['symbol1'], pair['symbol2'], pair['hedge_ratio'],
                start_date, end_date
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
    """Create visualization for pair trading signals.
    
    Args:
        signals: Signal data
        symbol1: First symbol
        symbol2: Second symbol
        output_path: Output path for visualization
    """
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

def generate_summary_report(pairs, stability_results, half_life_results, output_dir):
    """Generate summary report for the cointegration analysis.
    
    Args:
        pairs: Cointegrated pairs
        stability_results: Stability analysis results
        half_life_results: Half-life analysis results
        output_dir: Output directory for report
    """
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
                
                # Fixed the nested conditional
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

async def main():
    parser = argparse.ArgumentParser(description="Test the cointegration framework")
    parser.add_argument("--symbols", nargs="+", help="List of symbols to test")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse symbols as a list if provided
    symbols = args.symbols
    if symbols and isinstance(symbols, list) and len(symbols) == 1 and ',' in symbols[0]:
        symbols = symbols[0].split(',')
        
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    await test_cointegration(symbols, start_date, end_date, args.output)

if __name__ == "__main__":
    asyncio.run(main())
