"""
Tests for LLM client implementations.

This module tests the functionality of LLM clients and the factory.
"""

import pytest
import os
import json
from unittest.mock import MagicMock, patch
import asyncio

from src.llm_integration.config import LLMConfig
from src.llm_integration.clients import LLMClient, ClaudeClient, LLMClientFactory

class TestLLMConfig:
    """Test LLM configuration loading."""
    
    def test_default_config(self):
        """Test that default configuration can be loaded."""
        config = LLMConfig()
        
        # Check default values
        assert config.default_provider == "claude"
        assert config.claude_default_model == "claude-3-opus-20240229"
        assert config.max_retries == 3
    
    def test_env_loading(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("CLAUDE_API_KEY", "test_api_key")
        monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        
        # Load config
        config = LLMConfig()
        
        # Check values
        assert config.claude_api_key == "test_api_key"
        assert config.default_provider == "gemini"
        assert config.max_retries == 5

class TestLLMClientFactory:
    """Test LLM client factory."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        config = LLMConfig()
        factory = LLMClientFactory(config)
        
        # Should have at least Claude registered
        assert "claude" in factory.providers
    
    def test_get_default_client(self, monkeypatch):
        """Test getting default client."""
        # Set API key for testing
        monkeypatch.setenv("CLAUDE_API_KEY", "test_api_key")
        
        config = LLMConfig()
        factory = LLMClientFactory(config)
        
        # Get default client (should be Claude based on default config)
        client = factory.get_client()
        
        assert isinstance(client, ClaudeClient)
        assert client.provider_id == "claude"
    
    def test_client_selection(self, monkeypatch):
        """Test selecting specific client."""
        # Set API keys for testing
        monkeypatch.setenv("CLAUDE_API_KEY", "test_claude_key")
        
        config = LLMConfig()
        factory = LLMClientFactory(config)
        
        # Get Claude client
        client = factory.get_client("claude")
        
        assert isinstance(client, ClaudeClient)
        assert client.provider_id == "claude"
    
    def test_missing_api_key(self):
        """Test handling of missing API key."""
        # Create config without API keys
        config = LLMConfig(claude_api_key=None)
        factory = LLMClientFactory(config)
        
        # Trying to get Claude client should raise ValueError
        with pytest.raises(ValueError):
            factory.get_client("claude")
    
    def test_available_providers(self, monkeypatch):
        """Test available_providers method."""
        # Set API key for testing
        monkeypatch.setenv("CLAUDE_API_KEY", "test_api_key")
        
        config = LLMConfig()
        factory = LLMClientFactory(config)
        
        # Get available providers
        available = factory.available_providers()
        
        # Claude should be available
        assert "claude" in available
        assert available["claude"] is True

class TestClaudeClient:
    """Test Claude client implementation."""
    
    @pytest.fixture
    def client(self):
        """Create a Claude client for testing."""
        config = LLMConfig(claude_api_key="test_api_key")
        return ClaudeClient(config)
    
    def test_properties(self, client):
        """Test basic properties."""
        assert client.provider_id == "claude"
        assert client.default_model == "claude-3-opus-20240229"
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_analyze_text(self, mock_post, client):
        """Test analyze_text method with mocked API response."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"result": "test"})
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Call method
        result = await client.analyze_text(
            text="test text",
            prompt="Analyze this: ",
            model="claude-3-opus-20240229"
        )
        
        # Verify result
        assert "success" in result
        assert result["success"] is True
        assert "json_response" in result
        assert result["json_response"]["result"] == "test"
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_extract_json(self, mock_post, client):
        """Test extract_json method with mocked API response."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"category": "test", "confidence": 0.9})
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Call method
        result = await client.extract_json(
            text="test text",
            prompt="Classify this: ",
            schema={"type": "object", "properties": {"category": {"type": "string"}}}
        )
        
        # Verify result
        assert "category" in result
        assert result["category"] == "test"
        assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_classify(self, mock_post, client):
        """Test classify method with mocked API response."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"category": "category2", "confidence": 0.8})
                }
            ]
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Call method
        result = await client.classify(
            text="test text",
            categories=["category1", "category2", "category3"],
            prompt="Classify this text: "
        )
        
        # Verify result
        assert "category2" in result
        assert result["category2"] == 0.8
        assert "category1" in result
        assert result["category1"] == 0.0
