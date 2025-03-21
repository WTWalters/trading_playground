"""
Alpha Vantage news API provider implementation.

This module provides a news data provider that uses the Alpha Vantage News API.
"""

import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .base import NewsDataProvider
from src.config.config_manager import load_config

class AlphaVantageNewsProvider(NewsDataProvider):
    """
    Provider for Alpha Vantage News API.
    
    Retrieves financial news and market events from Alpha Vantage.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Alpha Vantage news provider.
        
        Args:
            api_key: Alpha Vantage API key (optional, loads from config if None)
        """
        super().__init__()
        
        if api_key is None:
            # Load from config
            config = load_config()
            api_key = config.alpha_vantage_key
            
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
            
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    @property
    def provider_id(self) -> str:
        """Get the provider ID."""
        return "alpha_vantage_news"
    
    @property
    def max_lookback_days(self) -> Optional[int]:
        """Get maximum lookback period in days."""
        return 30  # Alpha Vantage typically provides about a month of news
    
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
            DataFrame with news data
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Alpha Vantage API has a single endpoint for news, we'll filter by keywords
        all_news = []
        
        # Process each topic
        for topic in topics:
            try:
                # Construct API parameters
                params = {
                    "function": "NEWS_SENTIMENT",
                    "apikey": self.api_key,
                    "topics": topic,
                    "sort": "RELEVANCE",
                    "limit": min(max_results, 200)  # API limit
                }
                
                # Make the API request
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(self.base_url, params=params) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            self.logger.error(f"Alpha Vantage API error: {response.status} - {error_text}")
                            continue
                            
                        response_data = await response.json()
                        
                        if "feed" not in response_data:
                            self.logger.warning(f"No news feed in Alpha Vantage response for topic: {topic}")
                            continue
                        
                        # Extract articles
                        articles = response_data["feed"]
                        
                        # Filter by date
                        filtered_articles = []
                        for article in articles:
                            try:
                                # Parse article time
                                time_published = datetime.fromisoformat(article.get("time_published", "").replace("Z", "+00:00"))
                                
                                # Check if within date range
                                if start_date <= time_published <= end_date:
                                    filtered_articles.append(article)
                            except (ValueError, KeyError) as e:
                                self.logger.warning(f"Error processing article time: {e}")
                                continue
                        
                        all_news.extend(filtered_articles)
                        
            except Exception as e:
                self.logger.error(f"Error fetching news for topic {topic}: {e}")
                continue
        
        # Convert to DataFrame
        if not all_news:
            return pd.DataFrame()
            
        # Process the news feed
        processed_news = []
        for article in all_news:
            try:
                # Extract base fields
                processed_article = {
                    "title": article.get("title", ""),
                    "content": article.get("summary", ""),
                    "source": article.get("source", ""),
                    "author": article.get("authors", []),
                    "url": article.get("url", ""),
                    "published_time": datetime.fromisoformat(article.get("time_published", "").replace("Z", "+00:00")),
                    "retrieved_time": datetime.now(),
                    "categories": [],
                    "sentiment": article.get("overall_sentiment_score", 0),
                    "relevance_score": article.get("relevance_score", 0) if "relevance_score" in article else 1.0,
                }
                
                # Extract topics/categories
                if "topics" in article:
                    processed_article["categories"] = [topic.get("topic") for topic in article["topics"]]
                
                # Extract entities
                entities = {}
                if "ticker_sentiment" in article:
                    ticker_sentiments = article["ticker_sentiment"]
                    for ticker_data in ticker_sentiments:
                        ticker = ticker_data.get("ticker")
                        sentiment = ticker_data.get("ticker_sentiment_score")
                        if ticker and sentiment:
                            entities[ticker] = {
                                "type": "TICKER",
                                "name": ticker,
                                "sentiment": sentiment
                            }
                
                processed_article["entities"] = entities
                processed_news.append(processed_article)
                
            except Exception as e:
                self.logger.warning(f"Error processing article: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(processed_news)
        
        # Limit to max_results
        if len(df) > max_results:
            df = df.sort_values("published_time", ascending=False).head(max_results)
            
        return df
    
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
        if end_date is None:
            end_date = datetime.now()
            
        try:
            # Construct API parameters
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "tickers": symbol,
                "sort": "RELEVANCE",
                "limit": min(max_results, 200)  # API limit
            }
            
            # Make the API request
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Alpha Vantage API error: {response.status} - {error_text}")
                        return pd.DataFrame()
                        
                    response_data = await response.json()
                    
                    if "feed" not in response_data:
                        self.logger.warning(f"No news feed in Alpha Vantage response for symbol: {symbol}")
                        return pd.DataFrame()
                    
                    # Extract articles
                    articles = response_data["feed"]
                    
                    # Filter by date
                    filtered_articles = []
                    for article in articles:
                        try:
                            # Parse article time
                            time_published = datetime.fromisoformat(article.get("time_published", "").replace("Z", "+00:00"))
                            
                            # Check if within date range
                            if start_date <= time_published <= end_date:
                                filtered_articles.append(article)
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"Error processing article time: {e}")
                            continue
            
            # Process the articles
            processed_news = []
            for article in filtered_articles:
                try:
                    # Extract base fields
                    processed_article = {
                        "title": article.get("title", ""),
                        "content": article.get("summary", ""),
                        "source": article.get("source", ""),
                        "author": article.get("authors", []),
                        "url": article.get("url", ""),
                        "published_time": datetime.fromisoformat(article.get("time_published", "").replace("Z", "+00:00")),
                        "retrieved_time": datetime.now(),
                        "categories": [],
                        "sentiment": article.get("overall_sentiment_score", 0),
                        "relevance_score": 0.0,
                    }
                    
                    # Extract topics/categories
                    if "topics" in article:
                        processed_article["categories"] = [topic.get("topic") for topic in article["topics"]]
                    
                    # Extract ticker-specific sentiment and relevance
                    if "ticker_sentiment" in article:
                        ticker_sentiments = article["ticker_sentiment"]
                        for ticker_data in ticker_sentiments:
                            if ticker_data.get("ticker") == symbol:
                                processed_article["sentiment"] = ticker_data.get("ticker_sentiment_score", 0)
                                processed_article["relevance_score"] = ticker_data.get("relevance_score", 0)
                                break
                    
                    processed_news.append(processed_article)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article: {e}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(processed_news)
            
            # Sort by relevance and date
            if not df.empty:
                df = df.sort_values(["relevance_score", "published_time"], ascending=[False, False])
                
                # Limit to max_results
                if len(df) > max_results:
                    df = df.head(max_results)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching news for symbol {symbol}: {e}")
            return pd.DataFrame()
    
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
        # Alpha Vantage doesn't have a dedicated events API
        # We'll search for event-related keywords instead
        
        event_keywords = {
            "fed_announcement": ["Federal Reserve", "Fed announcement", "FOMC", "Powell", "interest rate decision"],
            "earnings_release": ["earnings report", "earnings release", "quarterly results", "financial results"],
            "economic_data": ["economic data", "GDP", "unemployment", "inflation", "jobs report", "CPI", "PPI"],
            "ipo": ["IPO", "initial public offering", "goes public", "market debut"]
        }
        
        # Map event types to keywords
        search_topics = []
        for event_type in event_types:
            if event_type in event_keywords:
                search_topics.extend(event_keywords[event_type])
            else:
                search_topics.append(event_type)  # Use directly if not in mapping
        
        # Fetch news with these keywords
        news_df = await self.fetch_recent_news(
            topics=search_topics,
            start_date=start_date,
            end_date=end_date,
            max_results=200  # Higher limit for events
        )
        
        if news_df.empty:
            return pd.DataFrame()
        
        # Add event_type column based on content matching
        def determine_event_type(row):
            content = (row['title'] + ' ' + row['content']).lower()
            for event_type, keywords in event_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in content:
                        return event_type
            return "other"
        
        news_df['event_type'] = news_df.apply(determine_event_type, axis=1)
        
        # Filter to requested event types
        result_df = news_df[news_df['event_type'].isin(event_types)]
        
        return result_df
