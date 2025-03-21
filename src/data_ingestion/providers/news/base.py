"""
Base class for news data providers.

This module defines the interface that all news data providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import logging

class NewsDataProvider(ABC):
    """
    Abstract base class for news data providers.
    
    All news data providers must implement this interface to ensure
    consistent behavior across different data sources.
    """
    
    def __init__(self):
        """Initialize the news data provider."""
        self.logger = logging.getLogger(__name__)
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """
        Get the unique identifier for this provider.
        
        Returns:
            String identifier for the provider
        """
        pass
    
    @property
    @abstractmethod
    def max_lookback_days(self) -> Optional[int]:
        """
        Get the maximum lookback period in days.
        
        Returns:
            Maximum number of days or None if unlimited
        """
        pass
    
    @abstractmethod
    async def fetch_recent_news(
        self,
        topics: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent financial news for specified topics.
        
        Args:
            topics: List of topics/keywords to search for
            start_date: Start date for the news
            end_date: End date for the news (default: now)
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with news data (date, title, content, source, url, etc.)
        """
        pass
    
    @abstractmethod
    async def fetch_specific_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_results: int = 50
    ) -> pd.DataFrame:
        """
        Fetch news specifically about a symbol.
        
        Args:
            symbol: Symbol to fetch news for
            start_date: Start date for the news
            end_date: End date for the news (default: now)
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with news data
        """
        pass
    
    @abstractmethod
    async def fetch_market_events(
        self,
        event_types: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch specific market events like Fed announcements, earnings, etc.
        
        Args:
            event_types: List of event types to fetch
            start_date: Start date for the events
            end_date: End date for the events (default: now)
            
        Returns:
            DataFrame with event data
        """
        pass
