"""
Tests for LLM-enhanced regime detection.

This module tests the LLMRegimeAnalyzer class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import json

from src.llm_integration.config import LLMConfig
from src.llm_integration.market_analysis.regime_analyzer import LLMRegimeAnalyzer
from src.market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector, RegimeType

class TestRegimeAnalyzer:
    """Test LLM-enhanced regime analyzer."""
    
    @pytest.fixture
    def db_manager(self):
        """Create a mock database manager."""
        db_manager = MagicMock()
        pool = MagicMock()
        conn = MagicMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = conn
        db_manager.get_market_data = self._mock_get_market_data
        return db_manager
    
    @pytest.fixture
    def llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.provider_id = "test_provider"
        client.analyze_text.return_value = {
            "json_response": {
                "primary_regime": "TRENDING",
                "confidence": 85,
                "secondary_regime": "RISK_ON",
                "secondary_confidence": 70,
                "key_indicators": ["strong momentum", "increasing volume"],
                "market_implications": "Likely continued upward trend",
                "risk_assessment": "medium"
            },
            "raw_response": json.dumps({
                "primary_regime": "TRENDING",
                "confidence": 85,
                "secondary_regime": "RISK_ON",
                "secondary_confidence": 70,
                "key_indicators": ["strong momentum", "increasing volume"],
                "market_implications": "Likely continued upward trend",
                "risk_assessment": "medium"
            }),
            "model": "test-model",
            "success": True
        }
        return client
    
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
        
        # Mock turning point detection
        detector._detect_turning_point.return_value = (False, 0.2)
        
        return detector
    
    @pytest.fixture
    def analyzer(self, db_manager, llm_client, regime_detector):
        """Create an analyzer instance for testing."""
        config = LLMConfig()
        analyzer = LLMRegimeAnalyzer(
            db_manager=db_manager,
            config=config,
            llm_client=llm_client,
            regime_detector=regime_detector
        )
        
        # Mock prompts
        analyzer.prompts = {
            "regime_classification": "Analyze regime",
            "market_turning_point": "Detect turning points",
            "fed_statement": "Analyze Fed statement",
            "earnings_call": "Analyze earnings call",
            "economic_report": "Analyze economic report"
        }
        
        return analyzer
    
    async def _mock_get_market_data(self, symbol, start_date, end_date, timeframe):
        """Mock implementation of get_market_data."""
        # Create sample market data
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
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
    
    def _mock_retrieve_relevant_news(self):
        """Create mock news data."""
        return pd.DataFrame({
            'id': range(5),
            'title': [f"News {i}" for i in range(5)],
            'content': [f"Content {i}" for i in range(5)],
            'source': ["Source A", "Source B"] * 2 + ["Source C"],
            'published_time': [datetime.now() - timedelta(hours=i) for i in range(5)],
            'sentiment': [0.2, -0.3, 0.5, 0.1, -0.1],
            'categories': [["finance"], ["markets"], ["stocks"], ["finance", "markets"], ["economy"]]
        })
    
    @pytest.mark.asyncio
    async def test_analyze_recent_news(self, analyzer):
        """Test analyzing recent news."""
        # Mock retrieving news
        analyzer._retrieve_relevant_news = MagicMock(
            return_value=self._mock_retrieve_relevant_news()
        )
        analyzer._store_analysis_result = MagicMock()
        
        # Call the method
        result = await analyzer.analyze_recent_news(timeframe="7d")
        
        # Verify result
        assert result["primary_regime"] == "TRENDING"
        assert result["confidence"] == 85
        assert "key_indicators" in result
        assert len(result["key_indicators"]) > 0
        
        # Verify LLM client was called correctly
        analyzer.llm_client.analyze_text.assert_called_once()
        
        # Verify analysis was stored
        analyzer._store_analysis_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_specific_document(self, analyzer):
        """Test analyzing a specific document."""
        # Mock store function
        analyzer._store_document_analysis = MagicMock()
        
        # Document content and metadata
        document = "Federal Reserve maintains interest rates at current levels..."
        metadata = {"date": datetime.now(), "source": "Federal Reserve"}
        
        # Call the method
        result = await analyzer.analyze_specific_document(
            document_type="fed_statement",
            document_content=document,
            document_metadata=metadata
        )
        
        # Verify result
        assert result["primary_regime"] == "TRENDING"
        assert "document_type" in result
        assert result["document_type"] == "fed_statement"
        assert "document_metadata" in result
        
        # Verify LLM client was called with correct prompt
        analyzer.llm_client.analyze_text.assert_called_with(
            document,
            analyzer.prompts["fed_statement"]
        )
        
        # Verify analysis was stored
        analyzer._store_document_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enhance_regime_detection(self, analyzer):
        """Test enhanced regime detection."""
        # Create market data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        market_data = await self._mock_get_market_data(
            "SPY", start_date, end_date, "1d"
        )
        
        # Mock retrieving analysis
        analyzer._retrieve_recent_analysis = MagicMock(return_value={
            "primary_regime": "TRENDING",
            "confidence": 85,
            "key_indicators": ["strong momentum"]
        })
        
        # Call the method
        result = await analyzer.enhance_regime_detection(market_data)
        
        # Verify result
        assert "primary_regime" in result
        assert "statistical_regime" in result
        assert "llm_regime" in result
        assert "regimes_agree" in result
        
        # Check combination logic
        assert result["weightings"] in ["agreement", "statistical", "llm"]
    
    @pytest.mark.asyncio
    async def test_detect_market_turning_points(self, analyzer):
        """Test market turning point detection."""
        # Create market data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        market_data = await self._mock_get_market_data(
            "SPY", start_date, end_date, "1d"
        )
        
        # Mock retrieving news
        analyzer._retrieve_relevant_news = MagicMock(
            return_value=self._mock_retrieve_relevant_news()
        )
        
        # Set up turning point response
        analyzer.llm_client.analyze_text.return_value = {
            "json_response": {
                "is_turning_point": True,
                "confidence": 75,
                "direction": "bearish",
                "key_indicators": ["weakening breadth", "defensive rotation"],
                "narrative": "The market appears to be transitioning from a bullish trend to a more cautious stance."
            },
            "model": "test-model",
            "success": True
        }
        
        # Call the method
        result = await analyzer.detect_market_turning_points(market_data)
        
        # Verify result
        assert "is_turning_point" in result
        assert result["is_turning_point"] is True
        assert "direction" in result
        assert result["direction"] == "bearish"
        assert "statistical_confidence" in result
        assert "llm_confidence" in result
        assert "key_indicators" in result
        
        # Verify turning point detection was called in the regime detector
        analyzer.regime_detector._detect_turning_point.assert_called_once()
    
    def test_combine_regime_analyses_agreement(self, analyzer):
        """Test combining analyses when regimes agree."""
        # Both analyses agree on TRENDING
        stat_analysis = {
            "primary_regime": "TRENDING",
            "confidence": 0.8,
            "secondary_regime": "HIGH_VOLATILITY"
        }
        
        llm_analysis = {
            "primary_regime": "TRENDING",
            "confidence": 75,
            "key_indicators": ["strong momentum"]
        }
        
        # Call the method
        result = analyzer._combine_regime_analyses(stat_analysis, llm_analysis)
        
        # Verify result shows agreement
        assert result["regimes_agree"] is True
        assert result["weightings"] == "agreement"
        assert result["primary_regime"] == "TRENDING"
        assert result["confidence"] == 0.8  # Higher of the two
    
    def test_combine_regime_analyses_disagreement(self, analyzer):
        """Test combining analyses when regimes disagree."""
        # Analyses disagree
        stat_analysis = {
            "primary_regime": "MEAN_REVERTING",
            "confidence": 0.6,
            "secondary_regime": "LOW_VOLATILITY"
        }
        
        llm_analysis = {
            "primary_regime": "TRENDING",
            "confidence": 90,
            "key_indicators": ["strong momentum"]
        }
        
        # Call the method
        result = analyzer._combine_regime_analyses(stat_analysis, llm_analysis)
        
        # Verify result shows disagreement
        assert result["regimes_agree"] is False
        assert result["primary_regime"] == "TRENDING"  # LLM wins due to higher confidence
        assert result["statistical_regime"] == "MEAN_REVERTING"
        assert result["llm_regime"] == "TRENDING"
