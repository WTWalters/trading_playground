"""
Configuration module for LLM integration.

This module provides configuration classes and loading functions for
LLM provider settings and API keys.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)

class LLMConfig(BaseSettings):
    """LLM configuration settings loaded from environment variables."""
    
    # Anthropic Claude settings
    claude_api_key: Optional[str] = None
    claude_default_model: str = "claude-3-opus-20240229"
    claude_timeout: int = 120
    
    # DeepSeek settings
    deepseek_api_key: Optional[str] = None
    deepseek_default_model: str = "deepseek-chat"
    deepseek_timeout: int = 60
    
    # Google Gemini settings
    gemini_api_key: Optional[str] = None
    gemini_default_model: str = "gemini-pro"
    gemini_timeout: int = 60
    
    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_default_model: str = "llama2"
    ollama_timeout: int = 60
    
    # General settings
    default_provider: str = "claude"  # Options: claude, deepseek, gemini, ollama
    max_retries: int = 3
    retry_delay: int = 1
    cache_results: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    # Location of prompts directory
    prompts_dir: str = "prompts"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def load_llm_config(env_path: Optional[Path] = None) -> LLMConfig:
    """
    Load LLM configuration from environment variables and .env file.
    
    Args:
        env_path: Optional path to .env file. Defaults to project root .env
        
    Returns:
        Loaded LLMConfig instance
    """
    if env_path is None:
        # Try to find .env in common locations
        potential_paths = [
            Path(".env"),  # Current directory
            Path("../.env"),  # Parent directory
            Path.home() / ".env",  # Home directory
        ]
        
        for path in potential_paths:
            if path.exists():
                env_path = path
                logger.info(f"Found .env file at {env_path}")
                break
    
    if env_path and env_path.exists():
        config = LLMConfig(_env_file=env_path)
        logger.info(f"Loaded LLM configuration from {env_path}")
    else:
        config = LLMConfig()
        logger.info("Loaded LLM configuration from environment variables")
    
    # Validate configuration
    validate_config(config)
    
    return config

def validate_config(config: LLMConfig) -> None:
    """
    Validate the LLM configuration.
    
    Args:
        config: LLMConfig instance to validate
        
    Raises:
        ValueError: If validation fails
    """
    # Check if selected provider is configured
    provider = config.default_provider
    
    if provider == "claude" and not config.claude_api_key:
        logger.warning("Claude selected as default provider, but API key not set")
    elif provider == "deepseek" and not config.deepseek_api_key:
        logger.warning("DeepSeek selected as default provider, but API key not set")
    elif provider == "gemini" and not config.gemini_api_key:
        logger.warning("Gemini selected as default provider, but API key not set")
    
    # Validate prompts directory
    prompts_dir = Path(config.prompts_dir)
    if not prompts_dir.exists():
        logger.warning(f"Prompts directory not found: {prompts_dir}")
