from pathlib import Path
from typing import Optional
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class APIConfig(BaseSettings):
    """API Configuration loaded from environment variables."""
    polygon_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None

def load_config(env_path: Optional[Path] = None) -> APIConfig:
    """Load configuration from environment variables and .env file."""
    if env_path is None:
        env_path = Path('.env')
        
    if env_path.exists():
        return APIConfig(_env_file=env_path)
    return APIConfig()