"""
Anthropic Claude API client implementation.

This module provides a client for interacting with Anthropic's Claude API.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import asyncio

from .base import LLMClient
from ...llm_integration.config import LLMConfig

class ClaudeClient(LLMClient):
    """
    Client for Anthropic's Claude API.
    
    This client implements the LLMClient interface for Claude.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the Claude client.
        
        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.api_key = config.claude_api_key
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.timeout = aiohttp.ClientTimeout(total=config.claude_timeout)
    
    @property
    def provider_id(self) -> str:
        """Get the provider ID."""
        return "claude"
    
    @property
    def default_model(self) -> str:
        """Get the default model."""
        return self.config.claude_default_model
    
    async def analyze_text(
        self, 
        text: str, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze text using Claude.
        
        Args:
            text: The text to analyze
            prompt: The prompt template to guide the analysis
            model: Optional model override (default from config if None)
            temperature: Sampling temperature (0.0 for deterministic output)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Claude-specific parameters
            
        Returns:
            Dictionary with analysis results
            
        Raises:
            Exception: If analysis fails
        """
        if not self.api_key:
            raise ValueError("Claude API key not set")
        
        if model is None:
            model = self.default_model
        
        # Combine prompt and text
        if text:
            content = f"{prompt}\n\n{text}"
        else:
            content = prompt
            
        if max_tokens is None:
            max_tokens = 4096  # Default for Claude
        
        # Prepare request
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any additional parameters
        if "system" in kwargs:
            data["system"] = kwargs["system"]
            
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Use retry mechanism
        async def make_request():
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.api_url, json=data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Claude API error: {response.status} - {error_text}")
                    
                    response_data = await response.json()
                    
                    if "error" in response_data:
                        raise Exception(f"Claude API error: {response_data['error']}")
                    
                    # Extract the assistant's message
                    content = response_data.get("content", [])
                    
                    # The content is a list of blocks, extract text
                    text_blocks = [block.get("text", "") for block in content 
                                  if block.get("type") == "text"]
                    
                    result_text = "\n".join(text_blocks)
                    
                    # Try to parse JSON if the response looks like JSON
                    try:
                        result_json = self._parse_json_response(result_text)
                        return {
                            "raw_response": result_text,
                            "json_response": result_json,
                            "model": model,
                            "success": True
                        }
                    except (ValueError, json.JSONDecodeError):
                        # If not JSON, return as text
                        return {
                            "raw_response": result_text,
                            "model": model,
                            "success": True
                        }
        
        return await self._handle_retry(make_request)
    
    async def extract_json(
        self,
        text: str,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured JSON data from text using Claude.
        
        Args:
            text: The text to analyze
            prompt: The prompt template to guide the extraction
            schema: JSON schema describing expected output structure
            model: Optional model override
            **kwargs: Additional Claude-specific parameters
            
        Returns:
            Extracted JSON data as a dictionary
            
        Raises:
            Exception: If extraction fails
        """
        # Add schema to the prompt
        schema_json = json.dumps(schema, indent=2)
        enhanced_prompt = f"{prompt}\n\nYour response MUST follow this JSON schema:\n```json\n{schema_json}\n```\n\nPlease provide a valid JSON object that follows this schema exactly."
        
        # Use analyze_text for the API call
        result = await self.analyze_text(
            text=text,
            prompt=enhanced_prompt,
            model=model,
            temperature=0.0,  # Use 0 for deterministic output
            **kwargs
        )
        
        # Return the parsed JSON
        if "json_response" in result:
            return result["json_response"]
        else:
            # Try to parse the raw response
            try:
                return self._parse_json_response(result["raw_response"])
            except ValueError as e:
                raise ValueError(f"Failed to extract JSON: {e}")
    
    async def classify(
        self,
        text: str,
        categories: List[str],
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Classify text into provided categories using Claude.
        
        Args:
            text: The text to classify
            categories: List of possible categories
            prompt: The prompt template to guide the classification
            model: Optional model override
            **kwargs: Additional Claude-specific parameters
            
        Returns:
            Dictionary mapping categories to confidence scores (0-1)
            
        Raises:
            Exception: If classification fails
        """
        # Create a schema for the expected response format
        schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": categories
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "reasoning": {
                    "type": "string"
                }
            },
            "required": ["category", "confidence"]
        }
        
        # Enhance the prompt with categories and instructions
        categories_str = ", ".join(categories)
        enhanced_prompt = f"{prompt}\n\nCategories: {categories_str}\n\nAnalyze the text and determine which category it belongs to. Provide a confidence score between 0 and 1, where 1 indicates complete certainty."
        
        # Extract JSON with the schema
        result = await self.extract_json(
            text=text,
            prompt=enhanced_prompt,
            schema=schema,
            model=model,
            **kwargs
        )
        
        # Convert to expected output format
        category = result.get("category")
        confidence = result.get("confidence", 0.0)
        
        # Create confidence scores for all categories
        scores = {cat: 0.0 for cat in categories}
        scores[category] = confidence
        
        return scores
