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


class TradingConfig(BaseModel):
    """Main configuration container."""
    model_config = ConfigDict(frozen=True)
    database: DatabaseConfig
    alpaca: AlpacaConfig
    logging: LoggingConfig


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
            cls._instance._config = None
            cls._instance._last_env_vars = None
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_config'):
            self._config = None
            self._last_env_vars = None
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

    def _get_env_snapshot(self) -> Dict[str, str]:
        """Get a snapshot of relevant environment variables."""
        return {
            k: v for k, v in os.environ.items() 
            if k.startswith(self.ENV_PREFIX)
        }

    def _load_env_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        # Load .env file if it exists
        dotenv_path = Path.cwd() / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

        env_config = {}
        # Update last environment snapshot
        self._last_env_vars = self._get_env_snapshot()

        for key, value in self._last_env_vars.items():
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

    def _load_config(self) -> None:
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

    def _should_reload(self) -> bool:
        """Check if configuration should be reloaded due to environment changes."""
        if self._last_env_vars is None:
            return True
        current_env = self._get_env_snapshot()
        return current_env != self._last_env_vars

    @property
    def config(self) -> TradingConfig:
        """Get the configuration."""
        if self._config is None or self._should_reload():
            self._load_config()
        return self._config

    def reload(self):
        """Reload configuration."""
        self._load_config()


# Create a singleton instance
config_manager = ConfigManager()

def get_config() -> TradingConfig:
    """Get the application configuration."""
    return config_manager.config