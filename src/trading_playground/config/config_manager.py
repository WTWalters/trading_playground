# src/trading_playground/config/config_manager.py
from pathlib import Path
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field, SecretStr, ConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    model_config = ConfigDict(frozen=True)
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="trading_system", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: Optional[SecretStr] = Field(default=None, description="Database password")

    class Config:
        frozen = True


class AlpacaConfig(BaseModel):
    """Alpaca API configuration settings."""
    model_config = ConfigDict(frozen=True)
    api_key: SecretStr = Field(..., description="Alpaca API key")
    api_secret: SecretStr = Field(..., description="Alpaca API secret")
    paper_trading: bool = Field(True, description="Use paper trading")
    base_url: str = Field(
        "https://paper-api.alpaca.markets",
        description="API base URL"
    )

    class Config:
        frozen = True


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    model_config = ConfigDict(frozen=True)
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    handlers: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "console": {"type": "console"},
            "file": {
                "type": "file",
                "filename": "logs/trading_system.log"
            }
        }
    )

    class Config:
        frozen = True


class TradingConfig(BaseModel):
    """Main configuration container."""
    model_config = ConfigDict(frozen=True)
    database: DatabaseConfig
    alpaca: AlpacaConfig
    logging: LoggingConfig

    class Config:
        frozen = True


class ConfigurationError(Exception):
    """Raised when there is an error in configuration loading or validation."""
    pass


class ConfigManager:
    """Manages application configuration with environment variable support."""

    _instance = None
    ENV_PREFIX = "TRADING_"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_config'):
            self._config = None
            self._load_config()

    def _get_config_path(self) -> Path:
        """Get configuration file path with environment override support."""
        env_path = os.getenv(f"{self.ENV_PREFIX}CONFIG")
        if env_path:
            return Path(env_path)

        # Default paths in order of precedence
        config_paths = [
            Path.cwd() / "config.local.yml",
            Path.cwd() / "config.yml",
            Path.home() / ".trading_playground" / "config.yml",
            Path(__file__).parent / "config.yml"
        ]

        for path in config_paths:
            if path.exists():
                return path

        raise ConfigurationError("No configuration file found")

    def _load_env_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        # Load .env file if it exists
        dotenv_path = Path.cwd() / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # Remove prefix and convert to lowercase
                clean_key = key[len(self.ENV_PREFIX):].lower()
                # Split by double underscore for nested configs
                parts = clean_key.split('__')

                # Build nested dictionary
                current = env_config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value

        return env_config

    def _load_config(self):
        """Load configuration from files and environment variables."""
        try:
            config_path = self._get_config_path()

            # Load base config
            with open(config_path) as f:
                file_config = yaml.safe_load(f)

            # Load environment variables
            env_config = self._load_env_variables()

            # Merge configurations (environment takes precedence)
            merged_config = {**file_config, **env_config}

            # Create pydantic model
            self._config = TradingConfig(**merged_config)

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    @property
    def config(self) -> TradingConfig:
        """Get the configuration."""
        return self._config

    def reload(self):
        """Reload configuration."""
        self._load_config()


# Create a singleton instance
config_manager = ConfigManager()

def get_config() -> TradingConfig:
    """Get the application configuration."""
    return config_manager.config
