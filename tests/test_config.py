import pytest
from pathlib import Path
import os
from src.config import load_config, APIConfig

@pytest.fixture
def mock_env_file(tmp_path):
    env_content = """
POLYGON_API_KEY=test_polygon_key
ALPHA_VANTAGE_KEY=test_alpha_key
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content.strip())
    return env_file

def test_load_config_with_env_file(mock_env_file):
    config = load_config(mock_env_file)
    assert config.polygon_api_key == "test_polygon_key"
    assert config.alpha_vantage_key == "test_alpha_key"

def test_load_config_with_env_vars(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "env_polygon_key")
    monkeypatch.setenv("ALPHA_VANTAGE_KEY", "env_alpha_key")
    config = load_config(env_path=Path("nonexistent.env"))
    assert config.polygon_api_key == "env_polygon_key"
    assert config.alpha_vantage_key == "env_alpha_key"

def test_empty_config():
    config = APIConfig()
    assert config.polygon_api_key is None
    assert config.alpha_vantage_key is None