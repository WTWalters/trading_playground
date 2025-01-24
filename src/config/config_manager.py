from pathlib import Path
from typing import Optional
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class APIConfig(BaseSettings):
    polygon_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=None)  # Default to no env file

def load_config(env_path: Optional[Path] = None) -> APIConfig:
    """Load configuration from environment variables and .env file"""
    if env_path is None:
        env_path = Path('.env')
        
    if env_path.exists():
        return APIConfig(model_config=SettingsConfigDict(env_file=str(env_path)))
    return APIConfig()