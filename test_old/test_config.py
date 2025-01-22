# tests/test_config.py
import pytest
import os
from pathlib import Path
from trading_playground.config.config_manager import (
    ConfigManager,
    get_config,
    ConfigurationError,
    _config_instance
)


def reset_config():
    """Reset the configuration singleton."""
    global _config_instance
    ConfigManager._instance = None
    ConfigManager._config = None
    _config_instance = None


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
def env_vars():
    """Set up test environment variables."""
    # Store original values
    original_env = {}
    for key in ['TRADING_DATABASE__HOST', 'TRADING_ALPACA__API_KEY']:
        original_env[key] = os.environ.get(key)

    # Reset configuration first
    reset_config()

    # Set test environment variables
    os.environ["TRADING_DATABASE__HOST"] = "testhost"
    os.environ["TRADING_ALPACA__API_KEY"] = "env_test_key"

    yield

    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value

    # Reset configuration after cleanup
    reset_config()


def test_config_loading():
    config = get_config()
    assert config is not None
    assert config.database.port == 5432


def test_environment_override(env_vars):
    print("\nCurrent environment variables:",
          {k:v for k,v in os.environ.items() if k.startswith("TRADING_")})

    config = get_config()
    print("Config database host:", config.database.host)
    print("Full config:", config.model_dump())

    assert config.database.host == "testhost"
    assert config.alpaca.api_key.get_secret_value() == "env_test_key"


def test_config_singleton():
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_secure_secrets():
    # First reset the configuration
    reset_config()

    # Set the test secret
    test_secret = "test_secret_value"
    os.environ["TRADING_ALPACA__API_SECRET"] = test_secret

    config = get_config()

    # The string representation should hide the value
    assert str(config.alpaca.api_secret) == "**********"
    # But the actual value should be retrievable
    assert config.alpaca.api_secret.get_secret_value() == test_secret

    # Clean up
    del os.environ["TRADING_ALPACA__API_SECRET"]


def test_missing_config():
    with pytest.raises(ConfigurationError):
        ConfigManager()._get_config_path = lambda: Path("nonexistent.yml")
        ConfigManager()._load_config()
