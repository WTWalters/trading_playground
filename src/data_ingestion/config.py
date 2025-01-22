# File: src/data_ingestion/config.py

from typing import Dict, Optional
import json
import os
from pathlib import Path

class ConfigurationManager:
    """Manages configuration settings for the data ingestion system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses environment variables.
        """
        self.config_path = config_path
        self._config: Dict = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from file or environment."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
        else:
            self._config = {
                'database': {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': int(os.getenv('DB_PORT', '5432')),
                    'database': os.getenv('DB_NAME', 'trading_playground'),
                    'user': os.getenv('DB_USER', 'postgres'),
                    'password': os.getenv('DB_PASSWORD', ''),
                }
            }

    @property
    def database_config(self) -> Dict[str, str]:
        """Get database configuration."""
        return self._config['database']

    def save_config(self) -> None:
        """Save configuration to file."""
        if not self.config_path:
            raise ValueError("No config path specified")

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=4)

    def update_database_config(self, **kwargs) -> None:
        """
        Update database configuration.

        Args:
            **kwargs: Database configuration parameters to update
        """
        self._config['database'].update(kwargs)
        if self.config_path:
            self.save_config()
