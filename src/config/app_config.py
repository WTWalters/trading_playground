"""
Application configuration module.

This module provides centralized configuration settings for the entire application.
"""

from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    log_to_console: bool = True
    log_to_file: bool = False
    
    class Config:
        env_prefix = "LOG_"


class DataProviderConfig(BaseSettings):
    """Data provider configuration settings."""
    default_provider: str = "yahoo"
    polygon_api_key: str = ""
    yahoo_rate_limit_pause: float = 0.2
    max_retries: int = 3
    retry_delay: float = 1.0
    concurrency_limit: int = 5
    
    class Config:
        env_prefix = "DATA_"


class ValidationConfig(BaseSettings):
    """Data validation configuration settings."""
    auto_correct: bool = True
    min_quality_score: float = 60.0
    log_quality_issues: bool = True
    
    class Config:
        env_prefix = "VALIDATION_"


class BacktestConfig(BaseSettings):
    """Backtesting configuration settings."""
    default_initial_capital: float = 100000.0
    default_risk_per_trade: float = 0.02
    default_commission: float = 0.0
    log_trades: bool = True
    
    class Config:
        env_prefix = "BACKTEST_"


class ApplicationConfig(BaseSettings):
    """Main application configuration."""
    app_name: str = "TITAN Trading Platform"
    app_version: str = "0.1.0"
    environment: str = "development"
    data_dir: str = Field(default_factory=lambda: str(Path.home() / "titan_data"))
    temp_dir: str = Field(default_factory=lambda: str(Path.home() / "titan_data" / "temp"))
    
    # Sub-configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    providers: DataProviderConfig = Field(default_factory=DataProviderConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Default symbols
    default_symbols: List[str] = ["SPY", "QQQ", "IWM", "GLD", "TLT", "VIX"]
    
    class Config:
        env_prefix = "APP_"


def load_config() -> ApplicationConfig:
    """
    Load application configuration from environment and files.
    
    Returns:
        ApplicationConfig: Application configuration instance
    """
    # Load from environment variables
    config = ApplicationConfig()
    
    # Create data directories if they don't exist
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.temp_dir, exist_ok=True)
    
    return config


# Global application config instance
app_config = load_config()