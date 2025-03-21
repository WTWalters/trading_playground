"""
Data orchestration module for managing the full data ingestion pipeline.

This module provides the central orchestration for:
1. Fetching data from providers
2. Validating and correcting data
3. Storing data in the database
4. Logging issues and quality metrics
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta

from ..config.db_config import DatabaseConfig
from .db_manager import DatabaseManager
from .validation import DataValidator
from .data_providers import DataProvider


class DataOrchestrator:
    """
    Orchestrates the data ingestion process.
    
    This class coordinates:
    1. Data providers
    2. Data validation
    3. Database operations
    
    It manages the entire data ingestion pipeline, ensuring data quality
    and providing proper error handling and retry logic.
    """
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        providers: Dict[str, DataProvider],
        auto_correct: bool = True,
        min_quality_score: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the data orchestrator.
        
        Args:
            db_config: Database configuration
            providers: Dictionary mapping provider IDs to provider instances
            auto_correct: Whether to auto-correct data issues
            min_quality_score: Minimum quality score for data to be accepted
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Delay between retries in seconds
        """
        self.db_manager = DatabaseManager(db_config)
        self.validator = DataValidator(auto_correct=auto_correct)
        self.providers = providers
        self.min_quality_score = min_quality_score
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize the orchestrator and its components."""
        try:
            await self.db_manager.initialize()
            self.logger.info("Data orchestrator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize data orchestrator: {str(e)}")
            raise
            
    async def close(self) -> None:
        """Close all resources."""
        await self.db_manager.close()
            
    async def fetch_and_store(
        self,
        symbol: str,
        provider_id: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d'
    ) -> Dict[str, Any]:
        """
        Fetch data from a provider, validate, and store in the database.
        
        Args:
            symbol: Symbol to fetch
            provider_id: ID of the provider to use
            start_date: Start date for data
            end_date: End date for data (default: now)
            timeframe: Timeframe to fetch ('1m', '5m', '1h', '1d', etc.)
            
        Returns:
            Dictionary with operation results
        """
        if provider_id not in self.providers:
            error_msg = f"Unknown provider: {provider_id}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "symbol": symbol,
                "provider": provider_id,
                "records_added": 0
            }
            
        provider = self.providers[provider_id]
        
        # Set end_date to now if not provided
        if end_date is None:
            end_date = datetime.now()
            
        self.logger.info(
            f"Fetching {symbol} from {provider_id} "
            f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
        )
        
        # Fetch data with retries
        data = None
        fetch_success = False
        error_msg = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                data = await provider.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
                
                if data is not None and not data.empty:
                    fetch_success = True
                    break
                else:
                    error_msg = "Provider returned empty data"
                    self.logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: {error_msg} "
                        f"for {symbol} from {provider_id}"
                    )
            except Exception as e:
                error_msg = f"Provider error: {str(e)}"
                self.logger.warning(
                    f"Attempt {attempt}/{self.max_retries}: {error_msg} "
                    f"for {symbol} from {provider_id}"
                )
                
            # Wait before retry
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)  # Exponential backoff
        
        # Check if fetch was successful
        if not fetch_success:
            self.logger.error(
                f"Failed to fetch {symbol} from {provider_id} after {self.max_retries} attempts: {error_msg}"
            )
            
            # Update provider status
            await self.db_manager.update_provider_status(
                provider=provider_id,
                success=False,
                error_message=error_msg
            )
            
            return {
                "success": False,
                "message": error_msg,
                "symbol": symbol,
                "provider": provider_id,
                "records_added": 0
            }
        
        # Validate data
        self.logger.info(f"Validating {len(data)} records for {symbol}")
        validation_result = self.validator.validate_ohlcv(
            data=data,
            symbol=symbol,
            min_quality_score=self.min_quality_score
        )
        
        # Process validation result
        if not validation_result.valid:
            error_msg = f"Data failed validation with score {validation_result.quality_score:.1f}"
            self.logger.error(
                f"{error_msg} for {symbol} from {provider_id}: {list(validation_result.issues.keys())}"
            )
            
            # Log data quality issues
            for issue_type, details in validation_result.issues.items():
                if isinstance(details, dict) and 'count' in details:
                    count = details['count']
                    description = f"{issue_type}: {count} instances"
                else:
                    description = str(details)
                    
                await self.db_manager.log_data_quality_issue(
                    symbol=symbol,
                    provider=provider_id,
                    time=datetime.now(),
                    issue_type=issue_type,
                    description=description,
                    severity=8  # High severity for validation failures
                )
            
            return {
                "success": False,
                "message": error_msg,
                "symbol": symbol,
                "provider": provider_id,
                "quality_score": validation_result.quality_score,
                "issues": validation_result.issues,
                "records_added": 0
            }
        
        # Use validated and potentially corrected data
        validated_data = validation_result.df
        
        # Store in database
        self.logger.info(f"Storing {len(validated_data)} records for {symbol} in database")
        
        try:
            data_quality = int(validation_result.quality_score)
            records_added = await self.db_manager.store_market_data(
                data=validated_data,
                symbol=symbol,
                provider=provider_id,
                data_quality=data_quality,
                is_adjusted=False,  # Assume unadjusted data for now
                metadata={
                    "timeframe": timeframe,
                    "quality_score": validation_result.quality_score,
                    "has_corrections": len(validation_result.issues) > 0
                }
            )
            
            # Update provider status
            await self.db_manager.update_provider_status(
                provider=provider_id,
                success=True
            )
            
            # Update symbol reference if needed
            await self.db_manager.update_symbol_reference(
                symbol=symbol,
                asset_type="unknown",  # Default to unknown, can be updated later
                active=True
            )
            
            # Log any corrected issues as low severity
            if validation_result.has_issues:
                for issue_type, details in validation_result.issues.items():
                    if "_corrected" in issue_type:
                        # This was an auto-corrected issue
                        await self.db_manager.log_data_quality_issue(
                            symbol=symbol,
                            provider=provider_id,
                            time=datetime.now(),
                            issue_type=issue_type,
                            description=str(details),
                            severity=3  # Low severity for corrected issues
                        )
            
            return {
                "success": True,
                "message": f"Successfully stored {records_added} records",
                "symbol": symbol,
                "provider": provider_id,
                "quality_score": validation_result.quality_score,
                "records_added": records_added,
                "issues_corrected": [k for k in validation_result.issues if "_corrected" in k]
            }
            
        except Exception as e:
            error_msg = f"Database error: {str(e)}"
            self.logger.error(f"Failed to store data for {symbol}: {error_msg}")
            
            return {
                "success": False,
                "message": error_msg,
                "symbol": symbol,
                "provider": provider_id,
                "records_added": 0
            }
            
    async def update_symbol_data(
        self,
        symbol: str,
        provider_id: str,
        lookback_days: int = 7,
        timeframe: str = '1d'
    ) -> Dict[str, Any]:
        """
        Update a symbol's data by fetching recent data.
        
        Args:
            symbol: Symbol to update
            provider_id: ID of the provider to use
            lookback_days: Number of days to look back
            timeframe: Timeframe to fetch
            
        Returns:
            Dictionary with operation results
        """
        # Get the latest date for this symbol
        latest_dates = await self.db_manager.get_latest_dates([symbol], provider_id)
        latest_date = latest_dates.get(symbol)
        
        # Calculate start and end dates
        end_date = datetime.now()
        
        if latest_date:
            # Use latest date minus 1 day to ensure overlap
            start_date = latest_date - timedelta(days=1)
        else:
            # No existing data, use lookback from current date
            start_date = end_date - timedelta(days=lookback_days)
            
        # Fetch and store the data
        return await self.fetch_and_store(
            symbol=symbol,
            provider_id=provider_id,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
    async def backfill_symbol_data(
        self,
        symbol: str,
        provider_id: str,
        days: int,
        timeframe: str = '1d'
    ) -> Dict[str, Any]:
        """
        Backfill historical data for a symbol.
        
        Args:
            symbol: Symbol to update
            provider_id: ID of the provider to use
            days: Number of days to backfill
            timeframe: Timeframe to fetch
            
        Returns:
            Dictionary with operation results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return await self.fetch_and_store(
            symbol=symbol,
            provider_id=provider_id,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
    async def update_multiple_symbols(
        self,
        symbols: List[str],
        provider_id: str,
        lookback_days: int = 7,
        timeframe: str = '1d',
        concurrency_limit: int = 5
    ) -> Dict[str, Any]:
        """
        Update multiple symbols concurrently.
        
        Args:
            symbols: List of symbols to update
            provider_id: ID of the provider to use
            lookback_days: Number of days to look back
            timeframe: Timeframe to fetch
            concurrency_limit: Maximum number of concurrent requests
            
        Returns:
            Dictionary with operation results
        """
        self.logger.info(f"Updating {len(symbols)} symbols from {provider_id}")
        
        # Function to process a single symbol
        async def process_symbol(symbol: str) -> Dict[str, Any]:
            try:
                return await self.update_symbol_data(
                    symbol=symbol,
                    provider_id=provider_id,
                    lookback_days=lookback_days,
                    timeframe=timeframe
                )
            except Exception as e:
                self.logger.error(f"Error updating {symbol}: {str(e)}")
                return {
                    "success": False,
                    "message": f"Exception: {str(e)}",
                    "symbol": symbol,
                    "provider": provider_id,
                    "records_added": 0
                }
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def bounded_process(symbol: str) -> Dict[str, Any]:
            async with semaphore:
                return await process_symbol(symbol)
        
        # Process all symbols concurrently with bounded concurrency
        tasks = [bounded_process(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # Summarize results
        success_count = sum(1 for r in results if r["success"])
        records_added = sum(r.get("records_added", 0) for r in results)
        failed_symbols = [r["symbol"] for r in results if not r["success"]]
        
        summary = {
            "success": success_count == len(symbols),
            "total_symbols": len(symbols),
            "successful_symbols": success_count,
            "failed_symbols": failed_symbols,
            "total_records_added": records_added,
            "details": results
        }
        
        self.logger.info(
            f"Updated {success_count}/{len(symbols)} symbols, "
            f"added {records_added} records"
        )
        
        if failed_symbols:
            self.logger.warning(f"Failed to update symbols: {failed_symbols}")
            
        return summary
    
    async def get_data_for_analysis(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d',
        provider: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols for analysis.
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date for data
            end_date: End date for data (default: now)
            timeframe: Timeframe to fetch
            provider: Specific provider to use (optional)
            
        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        result = {}
        
        for symbol in symbols:
            try:
                df = await self.db_manager.get_market_data(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date,
                    provider=provider,
                    timeframe=timeframe
                )
                
                if not df.empty:
                    result[symbol] = df
                else:
                    self.logger.warning(f"No data found for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return result
    
    async def get_data_quality_summary(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get a summary of data quality issues.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with data quality summary
        """
        start_time = datetime.now() - timedelta(days=days)
        
        issues = await self.db_manager.get_data_quality_issues(
            start_time=start_time,
            limit=1000  # Get a large sample
        )
        
        if not issues:
            return {
                "total_issues": 0,
                "by_type": {},
                "by_symbol": {},
                "by_provider": {}
            }
            
        # Analyze issues
        by_type = {}
        by_symbol = {}
        by_provider = {}
        
        for issue in issues:
            # Count by issue type
            issue_type = issue["issue_type"]
            by_type[issue_type] = by_type.get(issue_type, 0) + 1
            
            # Count by symbol
            symbol = issue["symbol"]
            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
            
            # Count by provider
            provider = issue["provider"]
            by_provider[provider] = by_provider.get(provider, 0) + 1
            
        # Sort dictionaries by count in descending order
        by_type = dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True))
        by_symbol = dict(sorted(by_symbol.items(), key=lambda x: x[1], reverse=True))
        by_provider = dict(sorted(by_provider.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "total_issues": len(issues),
            "period_days": days,
            "by_type": by_type,
            "by_symbol": by_symbol,
            "by_provider": by_provider
        }
