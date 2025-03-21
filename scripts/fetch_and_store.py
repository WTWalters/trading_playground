#!/usr/bin/env python3
"""
Script to fetch and store market data.

Usage:
  python3 fetch_and_store.py --symbol AAPL --provider yahoo --days 30 --timeframe 1d
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
from src.data_ingestion.db_manager import DatabaseManager
from src.data_ingestion.validation import DataValidator
from src.data_ingestion.providers.factory import DataProviderFactory
from src.data_ingestion.orchestrator import DataOrchestrator


async def main():
    """Main function to fetch and store market data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch and store market data")
    parser.add_argument("--symbol", required=True, help="Symbol to fetch")
    parser.add_argument("--provider", default="yahoo", help="Provider (yahoo or polygon)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch")
    parser.add_argument("--timeframe", default="1d", help="Timeframe (1m, 5m, 1h, 1d, etc.)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
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
    await orchestrator.initialize()
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        print(f"Fetching {args.symbol} from {args.provider} for the last {args.days} days...")
        
        # Fetch and store data
        result = await orchestrator.fetch_and_store(
            symbol=args.symbol,
            provider_id=args.provider,
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe
        )
        
        # Print result
        if result["success"]:
            print(f"Success! Added {result['records_added']} records for {args.symbol}")
            if result.get("issues_corrected"):
                print(f"Corrected issues: {result['issues_corrected']}")
            print(f"Data quality score: {result.get('quality_score', 'N/A')}")
        else:
            print(f"Error: {result['message']}")
            if "issues" in result:
                print(f"Issues: {list(result['issues'].keys())}")
        
    finally:
        # Cleanup
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
