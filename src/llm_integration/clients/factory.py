"""
Factory for creating LLM client instances.

This module provides a factory class for creating instances of different LLM clients.
"""

import logging
from typing import Dict, Optional, Type

from .base import LLMClient
from .claude import ClaudeClient
from ...llm_integration.config import LLMConfig

class LLMClientFactory:
    """
    Factory for creating LLM client instances.
    
    This class handles the creation of appropriate LLM client instances
    based on provider ID and configuration.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the factory.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Register available providers
        self.providers = {
            'claude': ClaudeClient,
            # Add other providers as they're implemented
            # 'deepseek': DeepseekClient,
            # 'gemini': GeminiClient,
            # 'ollama': OllamaClient,
        }
    
    def register_provider(self, provider_id: str, provider_class: Type[LLMClient]) -> None:
        """
        Register a new provider class.
        
        Args:
            provider_id: ID for the provider
            provider_class: Provider class to register
        """
        self.providers[provider_id] = provider_class
        self.logger.info(f"Registered LLM provider: {provider_id}")
    
    def get_client(self, provider_id: Optional[str] = None) -> LLMClient:
        """
        Get an LLM client instance.
        
        Args:
            provider_id: ID of the provider to create (default from config if None)
            
        Returns:
            LLMClient instance
            
        Raises:
            ValueError: If the provider ID is unknown or configuration is invalid
        """
        if provider_id is None:
            provider_id = self.config.default_provider
        
        if provider_id not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_id}")
        
        provider_class = self.providers[provider_id]
        
        # Check for API key availability
        if provider_id == 'claude' and not self.config.claude_api_key:
            raise ValueError(f"Claude API key not configured")
        elif provider_id == 'deepseek' and not self.config.deepseek_api_key:
            raise ValueError(f"DeepSeek API key not configured")
        elif provider_id == 'gemini' and not self.config.gemini_api_key:
            raise ValueError(f"Gemini API key not configured")
        
        # Create and return client instance
        return provider_class(self.config)
    
    def available_providers(self) -> Dict[str, bool]:
        """
        Get a dictionary of available providers and their availability status.
        
        Returns:
            Dictionary mapping provider IDs to availability (True/False)
        """
        availability = {}
        
        for provider_id in self.providers.keys():
            try:
                # Check if we can create this provider
                if provider_id == 'claude':
                    availability[provider_id] = bool(self.config.claude_api_key)
                elif provider_id == 'deepseek':
                    availability[provider_id] = bool(self.config.deepseek_api_key)
                elif provider_id == 'gemini':
                    availability[provider_id] = bool(self.config.gemini_api_key)
                elif provider_id == 'ollama':
                    availability[provider_id] = True  # Ollama is always available if implemented
                else:
                    availability[provider_id] = False
            except Exception:
                availability[provider_id] = False
        
        return availability
