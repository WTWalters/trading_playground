#!/usr/bin/env python3
"""
Script to bulk import market data for multiple symbols.

Usage:
  python3 bulk_import.py --symbols AAPL,MSFT,GOOGL --provider yahoo --days 30 --timeframe 1d
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
import os
import dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.db_config import DatabaseConfig
from src.data_ingestion.providers.factory import DataProviderFactory
from src.data_ingestion.orchestrator import DataOrchestrator


async def main():
    """Main function to bulk import market data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Bulk import market data")
    parser.add_argument(
        "--symbols", 
        required=True, 
        help="Comma-separated list of symbols to fetch"
    )
    parser.add_argument(
        "--provider", 
        default="yahoo", 
        help="Provider (yahoo or polygon)"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Number of days to fetch"
    )
    parser.add_argument(
        "--timeframe", 
        default="1d", 
        help="Timeframe (1m, 5m, 1h, 1d, etc.)"
    )
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=5, 
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get API keys from environment
    polygon_api_key = os.getenv("POLYGON_API_KEY", "")
    
    # Create database config
    db_config = DatabaseConfig()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    
    if not symbols:
        print("Error: No valid symbols provided")
        return
    
    # Create provider instances
    factory = DataProviderFactory()
    providers = {
        "yahoo": factory.get_provider("yahoo"),
        "polygon": factory.get_provider("polygon", {"api_key": polygon_api_key}) if polygon_api_key else None
    }
    
    # Remove None providers
    providers = {k: v for k, v in providers.items() if v is not None}
    
    if args.provider not in providers:
        print(f"Provider '{args.provider}' not available. Check API keys or use 'yahoo'.")
        return
    
    # Create orchestrator
    orchestrator = DataOrchestrator(
        db_config=db_config,
        providers=providers,
        auto_correct=True,
        min_quality_score=50.0
    )
    
    # Initialize orchestrator
    print(f"Initializing data orchestrator...")
    await orchestrator.initialize()
    
    try:
        print(f"Importing {len(symbols)} symbols from {args.provider}...")
        print(f"Symbols: {', '.join(symbols)}")
        
        # Update multiple symbols
        result = await orchestrator.update_multiple_symbols(
            symbols=symbols,
            provider_id=args.provider,
            lookback_days=args.days,
            timeframe=args.timeframe,
            concurrency_limit=args.concurrency
        )
        
        # Print summary
        print("\nImport Summary:")
        print(f"Total symbols: {result['total_symbols']}")
        print(f"Successful: {result['successful_symbols']}")
        print(f"Failed: {len(result['failed_symbols'])}")
        print(f"Total records added: {result['total_records_added']}")
        
        if result['failed_symbols']:
            print("\nFailed symbols:")
            for symbol in result['failed_symbols']:
                print(f"  - {symbol}")
        
        # Print individual results if in debug mode
        if args.debug:
            print("\nDetailed Results:")
            for detail in result['details']:
                status = "✅" if detail['success'] else "❌"
                print(f"{status} {detail['symbol']}: {detail.get('records_added', 0)} records")
                if not detail['success']:
                    print(f"    Error: {detail['message']}")
        
    finally:
        # Cleanup
        await orchestrator.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())