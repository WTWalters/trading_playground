# File: src/data_ingestion/providers/factory.py

from typing import Dict, Type
from .base import DataProvider
from .yahoo_finance import YahooFinanceProvider
from .polygon import PolygonProvider

class DataProviderFactory:
    """Factory for creating data provider instances."""

    _providers: Dict[str, Type[DataProvider]] = {
        'yahoo': YahooFinanceProvider,
        'polygon': PolygonProvider
    }

    @classmethod
    def create(cls, provider_name: str, config: Dict[str, any]) -> DataProvider:
        """
        Create a data provider instance.

        Args:
            provider_name: Name of the provider ('yahoo' or 'polygon')
            config: Provider configuration

        Returns:
            DataProvider instance
        """
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        return provider_class(config)
