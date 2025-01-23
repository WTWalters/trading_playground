from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel, field_validator
import asyncpg

from .data_providers import DataProvider, YahooFinanceProvider, PolygonProvider
from .db_manager import DatabaseManager
from .validation import DataValidator

class DataIngestionConfig(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    symbols: List[str]
    timeframes: List[str]
    providers: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    validation_rules: Dict[str, Any]  # Changed from any to Any
    retry_attempts: int = 3
    retry_delay: int = 5

    @field_validator('timeframes')
    def validate_timeframes(cls, v):
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        for tf in v:
            if tf not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {tf}")
        return v

class DataOrchestrator:
    """Orchestrates data ingestion from multiple sources"""

    def __init__(self, config: DataIngestionConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.providers: Dict[str, DataProvider] = self._initialize_providers()
        self.validator = DataValidator(config.validation_rules)

    def _initialize_providers(self) -> Dict[str, DataProvider]:
        providers = {}
        for provider_name in self.config.providers:
            if provider_name == 'yahoo':
                providers[provider_name] = YahooFinanceProvider()
            elif provider_name == 'polygon':
                providers[provider_name] = PolygonProvider()
        return providers

    async def _fetch_with_retry(
        self,
        provider: DataProvider,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                data = await provider.fetch_historical_data(
                    symbol, start_date, end_date, timeframe
                )
                return data
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {symbol}: {str(e)}"
                )
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self.logger.error(
                        f"All retry attempts failed for {symbol}"
                    )
                    return None

    async def _validate_and_store(
        self,
        data: pd.DataFrame,
        symbol: str,
        provider: str,
        timeframe: str
    ) -> bool:
        """Validate and store market data"""
        try:
            # Validate data
            if not self.validator.validate_market_data(data):
                self.logger.error(f"Validation failed for {symbol}")
                return False

            # Store data
            await self.db_manager.store_market_data(
                data, symbol, provider, timeframe
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error in validation/storage for {symbol}: {str(e)}"
            )
            return False

    async def ingest_historical_data(self) -> Dict[str, any]:
        """Ingest historical data for all configured symbols"""
        results = {
            'successful': [],
            'failed': [],
            'total_records': 0
        }

        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                for provider_name, provider in self.providers.items():
                    data = await self._fetch_with_retry(
                        provider,
                        symbol,
                        self.config.start_date,
                        self.config.end_date,
                        timeframe
                    )

                    if data is not None and not data.empty:
                        success = await self._validate_and_store(
                            data, symbol, provider_name, timeframe
                        )
                        if success:
                            results['successful'].append(
                                f"{symbol}-{timeframe}-{provider_name}"
                            )
                            results['total_records'] += len(data)
                        else:
                            results['failed'].append(
                                f"{symbol}-{timeframe}-{provider_name}"
                            )
                    else:
                        results['failed'].append(
                            f"{symbol}-{timeframe}-{provider_name}"
                        )

        return results

    async def update_market_data(self) -> Dict[str, any]:
        """Update market data for all symbols to latest available"""
        latest_dates = await self.db_manager.get_latest_dates()
        results = {
            'updated': [],
            'failed': [],
            'total_records': 0
        }

        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                start_date = latest_dates.get(
                    (symbol, timeframe),
                    self.config.start_date
                )

                if start_date >= datetime.now():
                    continue

                for provider_name, provider in self.providers.items():
                    data = await self._fetch_with_retry(
                        provider,
                        symbol,
                        start_date,
                        None,  # Fetch to latest
                        timeframe
                    )

                    if data is not None and not data.empty:
                        success = await self._validate_and_store(
                            data, symbol, provider_name, timeframe
                        )
                        if success:
                            results['updated'].append(
                                f"{symbol}-{timeframe}-{provider_name}"
                            )
                            results['total_records'] += len(data)
                        else:
                            results['failed'].append(
                                f"{symbol}-{timeframe}-{provider_name}"
                            )
                    else:
                        results['failed'].append(
                            f"{symbol}-{timeframe}-{provider_name}"
                        )

        return results
