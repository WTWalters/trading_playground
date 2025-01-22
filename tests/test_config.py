import pytest
import os
from pathlib import Path
from trading_playground.config.config_manager import (
    ConfigManager,
    get_config,
    ConfigurationError,
    config_manager
)


def reset_config():
    """Reset the configuration singleton."""
    ConfigManager._instance = None
    if hasattr(config_manager, '_config'):
        config_manager._config = None


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables before and after each test."""
    # Store original environment
    original_env = {k: v for k, v in os.environ.items()}
    
    # Reset configuration
    reset_config()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    
    # Reset configuration again
    reset_config()


@pytest.fixture
def sample_config_file(tmp_path):
    config_content = """
database:
    host: localhost
    port: 5432
    database: trading_test
alpaca:
    api_key: test_key
    api_secret: test_secret
    paper_trading: true
logging:
    level: INFO
    """
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def env_vars():
    """Set up test environment variables."""
    # Reset configuration first
    reset_config()
    
    # Set test environment variables
    os.environ["TRADING_DATABASE__HOST"] = "testhost"
    os.environ["TRADING_ALPACA__API_KEY"] = "env_test_key"
    os.environ["TRADING_ALPACA__API_SECRET"] = "env_test_secret"  # Added missing required field
    
    yield
    
    # Clean up
    del os.environ["TRADING_DATABASE__HOST"]
    del os.environ["TRADING_ALPACA__API_KEY"]
    del os.environ["TRADING_ALPACA__API_SECRET"]


def test_config_loading(sample_config_file):
    config = get_config()
    assert config is not None
    assert config.database.host == "localhost"
    assert config.database.port == 5432


def test_environment_override(env_vars):
    config = get_config()
    assert config.database.host == "testhost"
    assert config.alpaca.api_key.get_secret_value() == "env_test_key"
    assert config.alpaca.api_secret.get_secret_value() == "env_test_secret"


def test_config_singleton():
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_secure_secrets():
    config = get_config()
    # Ensure secrets are not exposed in string representation
    assert str(config.alpaca.api_secret) != config.alpaca.api_secret.get_secret_value()


def test_missing_config():
    with pytest.raises(ConfigurationError):
        ConfigManager()._get_config_path = lambda: Path("nonexistent.yml")
        ConfigManager()._load_config()