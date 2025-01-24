from pathlib import Path
from typing import Optional
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

class APIConfig(BaseSettings):
    polygon_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

def load_config() -> APIConfig:
    """Load configuration from environment variables and .env file"""
    env_path = Path('.env')
    load_dotenv(env_path)
    return APIConfig()