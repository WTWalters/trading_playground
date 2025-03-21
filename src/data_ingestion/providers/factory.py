"""
Factory module for creating data provider instances.

This module handles the creation of appropriate data provider instances
based on provider ID and configuration.
"""

from typing import Dict, Any, Optional, Type
import logging

from .base import DataProvider
from .yahoo_finance import YahooFinanceProvider
from .polygon import PolygonProvider


class DataProviderFactory:
    """
    Factory for creating data provider instances.
    
    This class maintains a registry of available provider classes
    and creates instances based on provider ID and configuration.
    """
    
    def __init__(self):
        """Initialize the factory with default providers."""
        self.providers: Dict[str, Type[DataProvider]] = {
            'yahoo': YahooFinanceProvider,
            'polygon': PolygonProvider
        }
        self.logger = logging.getLogger(__name__)
        
    def register_provider(self, provider_id: str, provider_class: Type[DataProvider]) -> None:
        """
        Register a new provider class.
        
        Args:
            provider_id: ID for the provider
            provider_class: Provider class to register
        """
        self.providers[provider_id] = provider_class
        self.logger.info(f"Registered provider: {provider_id}")
        
    def get_provider(
        self, 
        provider_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> DataProvider:
        """
        Get a provider instance.
        
        Args:
            provider_id: ID of the provider to create
            config: Configuration for the provider (optional)
            
        Returns:
            DataProvider instance
            
        Raises:
            ValueError: If the provider ID is unknown
        """
        if provider_id not in self.providers:
            raise ValueError(f"Unknown provider: {provider_id}")
            
        provider_class = self.providers[provider_id]
        
        if provider_id == 'yahoo':
            # Yahoo Finance provider doesn't require config
            return provider_class()
            
        elif provider_id == 'polygon':
            # Polygon provider requires API key
            if not config or 'api_key' not in config:
                raise ValueError(f"Polygon provider requires 'api_key' in config")
                
            return provider_class(api_key=config['api_key'])
            
        else:
            # Generic initialization for other providers
            if config:
                return provider_class(**config)
            else:
                return provider_class()
                
    def create_all_providers(
        self, 
        configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, DataProvider]:
        """
        Create instances of all configured providers.
        
        Args:
            configs: Dictionary mapping provider IDs to their configs
            
        Returns:
            Dictionary mapping provider IDs to provider instances
        """
        providers = {}
        
        for provider_id, config in configs.items():
            try:
                providers[provider_id] = self.get_provider(provider_id, config)
                self.logger.info(f"Created provider: {provider_id}")
            except Exception as e:
                self.logger.error(f"Failed to create provider {provider_id}: {str(e)}")
                
        return providers
