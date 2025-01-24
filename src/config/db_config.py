"""Database configuration module."""
from typing import Optional
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'trading'
    user: str = 'postgres'
    password: str = 'postgres'
    min_connections: int = 1
    max_connections: int = 10
    
    class Config:
        env_prefix = 'DB_'  # Will look for DB_HOST, DB_PORT, etc.
