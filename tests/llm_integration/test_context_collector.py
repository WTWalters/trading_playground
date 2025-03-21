"""
Tests for trade context collection.

This module tests the TradeContextCollector class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import json

from src.market_analysis.trade import Trade, TradeDirection, TradeStatus
from src.llm_integration.trade_analysis.context_collector import TradeContextCollector
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector, RegimeType

class TestContextCollector:
    """Test trade context collection."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db_manager = MagicMock()
        
        # Mock connection
        conn = MagicMock()
        db_manager.pool = MagicMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = conn
        
        # Mock market data retrieval
        db_manager.get_market_data = AsyncMock(side_effect=self._mock_get_market_data)
        
        return db_manager
    
    @pytest.fixture
    def regime_detector(self):
        """Create a mock regime detector."""
        detector = MagicMock()
        result = MagicMock()
        result.primary_regime = RegimeType.TRENDING
        result.secondary_regime = RegimeType.HIGH_VOLATILITY
        result.confidence = 0.8
        result.volatility_regime = RegimeType.HIGH_VOLATILITY
        result.correlation_regime = RegimeType.LOW_CORRELATION
        result.transition_probability = 0.2
        detector.detect_regime.return_value = result
        
        return detector
    
    @pytest.fixture
    def context_collector(self, db_manager, regime_detector):
        """Create a context collector instance for testing."""
        collector = TradeContextCollector(
            db_manager=db_manager,
            regime_detector=regime_detector
        )
        
        # Mock helper methods
        collector._fetch_relevant_news = AsyncMock(return_value=self._create_mock_news())
        collector._fetch_macro_indicators = AsyncMock(return_value={"vix": 18.5})
        collector._store_trade_context = AsyncMock()
        
        return collector
    
    @pytest.fixture
    def sample_trade(self):
        """Create a sample trade for testing."""
        now = datetime.now()
        
        return Trade(
            entry_price=100.0,
            position_size=10,
            direction=TradeDirection.LONG,
            entry_time=now - timedelta(days=5),
            exit_price=110.0,
            exit_time=now - timedelta(days=2),
            stop_loss=95.0,
            take_profit=115.0,
            status=TradeStatus.CLOSED,
            profit=100.0,  # (110 - 100) * 10
            risk_amount=50.0,  # (100 - 95) * 10
            id="test_trade_001",
            tags={"swing", "tech"},
            notes="Test trade"
        )
    
    async def _mock_get_market_data(self, symbol, start_date, end_date, timeframe):
        """Mock implementation of get_market_data."""
        # Create sample market data
        periods = int((end_date - start_date).total_seconds() / 3600) + 1
        dates = pd.date_range(start=start_date, end=end_date, periods=periods)
        n = len(dates)
        
        # Generate sample price data
        base_price = 100.0
        prices = [base_price]
        for i in range(1, n):
            # Add trend and noise
            trend = 0.001
            noise = np.random.normal(0, 0.005)
            prices.append(prices[-1] * (1 + trend + noise))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': [p * (1 - 0.002) for p in prices],
            'high': [p * (1 + 0.005) for p in prices],
            'low': [p * (1 - 0.005) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, size=n)
        }, index=dates)
        
        return df
    
    def _create_mock_news(self):
        """Create mock news data."""
        return [
            {
                "id": 1,
                "title": "Market Update: Tech Stocks Rally",
                "content": "Technology stocks rallied today on positive earnings.",
                "source": "Financial News",
                "published_time": datetime.now() - timedelta(days=5),
                "sentiment": 0.3,
                "categories": ["technology", "stocks"]
            },
            {
                "id": 2,
                "title": "Fed Signals Policy Shift",
                "content": "Federal Reserve signals potential policy shift.",
                "source": "Market Watch",
                "published_time": datetime.now() - timedelta(days=4),
                "sentiment": -0.1,
                "categories": ["economy", "federal_reserve"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_collect_trade_context(self, context_collector, sample_trade):
        """Test collecting context for a trade."""
        # Call the method
        result = await context_collector.collect_trade_context(sample_trade)
        
        # Verify structure
        assert "trade_id" in result
        assert result["trade_id"] == sample_trade.id
        assert "entry_context" in result
        assert "exit_context" in result
        assert "regime_at_entry" in result
        assert "regime_at_exit" in result
        assert "news_ids" in result
        assert "macro_data" in result
        
        # Verify entry context structure
        entry_context = result["entry_context"]
        assert "market_data" in entry_context
        assert "regime" in entry_context
        assert "news_sentiment" in entry_context
        assert "macro_indicators" in entry_context
        
        # Check market data was fetched correctly
        context_collector.db_manager.get_market_data.assert_called()
        
        # Check regime detection was called
        context_collector.regime_detector.detect_regime.assert_called()
        
        # Check news was fetched
        context_collector._fetch_relevant_news.assert_called()
        
        # Check macro indicators were fetched
        context_collector._fetch_macro_indicators.assert_called()
        
        # Check context was stored
        context_collector._store_trade_context.assert_called_once_with(result)
    
    @pytest.mark.asyncio
    async def test_collect_trade_context_entry_only(self, context_collector):
        """Test collecting context for a trade with only entry (open trade)."""
        # Create a trade with only entry information
        now = datetime.now()
        open_trade = Trade(
            entry_price=100.0,
            position_size=10,
            direction=TradeDirection.LONG,
            entry_time=now - timedelta(days=1),
            status=TradeStatus.OPEN,
            id="test_open_trade",
            stop_loss=95.0,
            take_profit=110.0
        )
        
        # Call the method
        result = await context_collector.collect_trade_context(open_trade)
        
        # Verify structure
        assert "trade_id" in result
        assert result["trade_id"] == open_trade.id
        assert "entry_context" in result
        assert "exit_context" in result  # Should be None for open trade
        assert result["exit_context"] is None
        
        # Check that only entry-related methods were called
        assert context_collector.db_manager.get_market_data.call_count == 1
        assert context_collector.regime_detector.detect_regime.call_count == 1
    
    def test_summarize_market_data(self, context_collector):
        """Test market data summarization."""
        # Create sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=24, freq='1H')
        data = pd.DataFrame({
            'open': [100.0] * len(dates),
            'high': [110.0] * len(dates),
            'low': [90.0] * len(dates),
            'close': [105.0] * len(dates),
            'volume': [1000] * len(dates)
        }, index=dates)
        
        # Call the method
        result = context_collector._summarize_market_data(data)
        
        # Verify result structure
        assert 'open' in result
        assert 'high' in result
        assert 'low' in result
        assert 'close' in result
        assert 'volume' in result
        assert 'volatility' in result
        assert 'price_change_pct' in result
        assert 'periods' in result
        
        # Check specific values
        assert result['open'] == 100.0
        assert result['high'] == 110.0
        assert result['low'] == 90.0
        assert result['close'] == 105.0
        assert result['periods'] == len(dates)
    
    def test_summarize_news_sentiment_positive(self, context_collector):
        """Test news sentiment summarization with positive sentiment."""
        news = [
            {"sentiment": 0.3},
            {"sentiment": 0.4},
            {"sentiment": 0.5}
        ]
        
        result = context_collector._summarize_news_sentiment(news)
        
        assert result["sentiment"] == "positive"
        assert result["count"] == 3
        assert result["sentiment_score"] > 0.2
    
    def test_summarize_news_sentiment_negative(self, context_collector):
        """Test news sentiment summarization with negative sentiment."""
        news = [
            {"sentiment": -0.3},
            {"sentiment": -0.4},
            {"sentiment": -0.5}
        ]
        
        result = context_collector._summarize_news_sentiment(news)
        
        assert result["sentiment"] == "negative"
        assert result["count"] == 3
        assert result["sentiment_score"] < -0.2
    
    def test_extract_key_topics(self, context_collector):
        """Test key topic extraction from news."""
        news = [
            {"categories": ["finance", "stocks"]},
            {"categories": ["technology", "earnings"]},
            {"categories": ["stocks", "markets"]},
            {"categories": ["economy", "federal_reserve"]},
            {"categories": ["finance", "banking"]}
        ]
        
        result = context_collector._extract_key_topics(news)
        
        assert isinstance(result, list)
        assert len(result) <= 5  # Should not exceed 5 topics
        assert "finance" in result  # Should include common topics
        assert "stocks" in result
