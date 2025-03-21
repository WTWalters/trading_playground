"""
Trade context collection module.

This module provides functionality for collecting and storing relevant
market context information for trades to support post-mortem analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import json
import asyncio

from ...database.manager import DatabaseManager
from ...market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector
from ...market_analysis.trade import Trade

class TradeContextCollector:
    """
    Collects and stores relevant market context for trades.
    
    This class gathers market conditions, news, and regime information
    at trade entry and exit points for later analysis.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        regime_detector: Optional[EnhancedRegimeDetector] = None
    ):
        """
        Initialize the trade context collector.
        
        Args:
            db_manager: Database manager for data access
            regime_detector: Optional regime detector (creates default if None)
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Create default regime detector if not provided
        if regime_detector is None:
            self.regime_detector = EnhancedRegimeDetector()
        else:
            self.regime_detector = regime_detector
    
    async def collect_trade_context(self, trade: Trade) -> Dict[str, Any]:
        """
        Collect and store context information for a trade.
        
        Args:
            trade: Trade object to collect context for
            
        Returns:
            Dictionary with the collected context information
        """
        try:
            # Validate trade object
            if not trade.entry_time:
                raise ValueError("Trade must have entry time")
            
            # Define time windows for context collection
            entry_start = trade.entry_time - timedelta(days=5)
            entry_end = trade.entry_time + timedelta(hours=6)
            
            exit_start = None
            exit_end = None
            if trade.exit_time:
                exit_start = trade.exit_time - timedelta(days=1)
                exit_end = trade.exit_time + timedelta(hours=6)
            
            # Collect market data around entry
            entry_market_data = await self._fetch_market_data(
                trade.symbol, entry_start, entry_end
            )
            
            # Detect regime at entry
            entry_regime = self._detect_regime(entry_market_data)
            
            # Collect relevant news around entry
            entry_news = await self._fetch_relevant_news(
                trade.symbol, entry_start, entry_end
            )
            
            # Prepare entry context
            entry_context = {
                "market_data": self._summarize_market_data(entry_market_data),
                "regime": entry_regime,
                "news_sentiment": self._summarize_news_sentiment(entry_news),
                "macro_indicators": await self._fetch_macro_indicators(trade.entry_time)
            }
            
            # Handle exit context if trade is closed
            exit_context = None
            exit_regime = None
            exit_news_ids = []
            
            if trade.exit_time and exit_start and exit_end:
                # Collect market data around exit
                exit_market_data = await self._fetch_market_data(
                    trade.symbol, exit_start, exit_end
                )
                
                # Detect regime at exit
                exit_regime = self._detect_regime(exit_market_data)
                
                # Collect relevant news around exit
                exit_news = await self._fetch_relevant_news(
                    trade.symbol, exit_start, exit_end
                )
                
                # Prepare exit context
                exit_context = {
                    "market_data": self._summarize_market_data(exit_market_data),
                    "regime": exit_regime,
                    "news_sentiment": self._summarize_news_sentiment(exit_news),
                    "macro_indicators": await self._fetch_macro_indicators(trade.exit_time)
                }
                
                # Get news IDs for exit
                exit_news_ids = [n.get('id') for n in exit_news if 'id' in n]
            
            # Get news IDs for entry
            entry_news_ids = [n.get('id') for n in entry_news if 'id' in n]
            
            # Collect intermediate news (between entry and exit)
            intermediate_news_ids = []
            if trade.exit_time and trade.entry_time != trade.exit_time:
                intermediate_news = await self._fetch_relevant_news(
                    trade.symbol, 
                    trade.entry_time + timedelta(hours=6),
                    trade.exit_time - timedelta(hours=6)
                )
                intermediate_news_ids = [n.get('id') for n in intermediate_news if 'id' in n]
            
            # All news IDs related to this trade
            all_news_ids = entry_news_ids + intermediate_news_ids + exit_news_ids
            
            # Create trade context record
            trade_context = {
                "trade_id": trade.id,
                "entry_context": entry_context,
                "exit_context": exit_context,
                "regime_at_entry": entry_regime.get("primary_regime") if entry_regime else None,
                "regime_at_exit": exit_regime.get("primary_regime") if exit_regime else None,
                "news_ids": all_news_ids,
                "macro_data": {
                    "entry": entry_context.get("macro_indicators", {}),
                    "exit": exit_context.get("macro_indicators", {}) if exit_context else {}
                }
            }
            
            # Store in database
            await self._store_trade_context(trade_context)
            
            return trade_context
            
        except Exception as e:
            self.logger.error(f"Error collecting trade context: {str(e)}")
            return {"error": str(e), "trade_id": trade.id}
    
    async def _fetch_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch market data for a symbol in the given time range"""
        return await self.db_manager.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1h"  # Using hourly data for context
        )
    
    def _detect_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime from market data"""
        if market_data.empty:
            return None
            
        result = self.regime_detector.detect_regime(market_data)
        
        # Convert regime detection result to dictionary
        return {
            "primary_regime": result.primary_regime.value,
            "secondary_regime": result.secondary_regime.value if result.secondary_regime else None,
            "confidence": result.confidence,
            "volatility_regime": result.volatility_regime.value if result.volatility_regime else None,
            "correlation_regime": result.correlation_regime.value if result.correlation_regime else None,
            "liquidity_regime": result.liquidity_regime.value if result.liquidity_regime else None,
            "transition_probability": result.transition_probability
        }
    
    async def _fetch_relevant_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch relevant news for a symbol in the given time range"""
        try:
            # Query the database for relevant news
            query = """
            SELECT n.*
            FROM news_data n
            JOIN symbol_news_mapping m ON n.id = m.news_id
            WHERE m.symbol = $1
            AND n.published_time BETWEEN $2 AND $3
            ORDER BY m.relevance_score DESC, n.published_time DESC
            LIMIT 20
            """
            
            # Execute the query
            async with self.db_manager.pool.acquire() as conn:
                records = await conn.fetch(query, symbol, start_date, end_date)
            
            # Process records into list of dictionaries
            news_list = []
            for record in records:
                news_list.append({
                    "id": record["id"],
                    "title": record["title"],
                    "content": record["content"],
                    "source": record["source"],
                    "published_time": record["published_time"],
                    "sentiment": record["sentiment"],
                    "categories": record["categories"] if "categories" in record else []
                })
                
            return news_list
            
        except Exception as e:
            self.logger.error(f"Error fetching relevant news: {e}")
            return []
    
    def _summarize_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create a summary of market data for context"""
        if market_data.empty:
            return {}
            
        return {
            "open": float(market_data.iloc[0]['open']),
            "high": float(market_data['high'].max()),
            "low": float(market_data['low'].min()),
            "close": float(market_data.iloc[-1]['close']),
            "volume": float(market_data['volume'].sum()),
            "volatility": float(market_data['close'].pct_change().std() * 100),
            "price_change_pct": float((market_data.iloc[-1]['close'] / market_data.iloc[0]['open'] - 1) * 100),
            "periods": len(market_data)
        }
    
    def _summarize_news_sentiment(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize news sentiment for context"""
        if not news:
            return {"count": 0, "sentiment": "neutral"}
            
        # Calculate average sentiment and extract key topics
        sentiments = [n.get('sentiment', 0) for n in news if 'sentiment' in n]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Map average sentiment to category
        sentiment_category = "neutral"
        if avg_sentiment > 0.2:
            sentiment_category = "positive"
        elif avg_sentiment < -0.2:
            sentiment_category = "negative"
            
        return {
            "count": len(news),
            "sentiment": sentiment_category,
            "sentiment_score": avg_sentiment,
            "key_topics": self._extract_key_topics(news)
        }
    
    def _extract_key_topics(self, news: List[Dict[str, Any]]) -> List[str]:
        """Extract key topics from news articles"""
        # Simple implementation - extract categories
        topics = set()
        
        for article in news:
            if 'categories' in article and article['categories']:
                topics.update(article['categories'])
                        
        return list(topics)[:5]  # Return top 5 topics
    
    async def _fetch_macro_indicators(self, date: datetime) -> Dict[str, Any]:
        """Fetch macro economic indicators for a given date"""
        try:
            # Query for macro indicators (using VIX as a proxy for now)
            query = """
            SELECT 
                close as vix,
                date
            FROM market_data
            WHERE symbol = 'VIX'
            AND date <= $1
            ORDER BY date DESC
            LIMIT 1
            """
            
            # Execute the query
            async with self.db_manager.pool.acquire() as conn:
                record = await conn.fetchrow(query, date)
            
            if record:
                return {
                    "vix": float(record["vix"]),
                    "date": record["date"].isoformat()
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching macro indicators: {e}")
            return {}
    
    async def _store_trade_context(self, trade_context: Dict[str, Any]) -> None:
        """Store trade context in the database"""
        try:
            # Check if record already exists
            check_query = """
            SELECT trade_id FROM trade_context
            WHERE trade_id = $1
            """
            
            async with self.db_manager.pool.acquire() as conn:
                existing = await conn.fetchrow(check_query, trade_context["trade_id"])
                
                if existing:
                    # Update existing record
                    update_query = """
                    UPDATE trade_context
                    SET 
                        entry_context = $2,
                        exit_context = $3,
                        regime_at_entry = $4,
                        regime_at_exit = $5,
                        news_ids = $6,
                        macro_data = $7
                    WHERE trade_id = $1
                    """
                    
                    await conn.execute(
                        update_query,
                        trade_context["trade_id"],
                        json.dumps(trade_context["entry_context"]),
                        json.dumps(trade_context["exit_context"]) if trade_context["exit_context"] else None,
                        trade_context["regime_at_entry"],
                        trade_context["regime_at_exit"],
                        trade_context["news_ids"],
                        json.dumps(trade_context["macro_data"])
                    )
                else:
                    # Insert new record
                    insert_query = """
                    INSERT INTO trade_context
                    (trade_id, entry_context, exit_context, regime_at_entry, 
                    regime_at_exit, news_ids, macro_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    
                    await conn.execute(
                        insert_query,
                        trade_context["trade_id"],
                        json.dumps(trade_context["entry_context"]),
                        json.dumps(trade_context["exit_context"]) if trade_context["exit_context"] else None,
                        trade_context["regime_at_entry"],
                        trade_context["regime_at_exit"],
                        trade_context["news_ids"],
                        json.dumps(trade_context["macro_data"])
                    )
                    
        except Exception as e:
            self.logger.error(f"Error storing trade context: {e}")
