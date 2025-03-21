"""
Base client for LLM API integration.

This module defines the interface that all LLM clients must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json
import asyncio
import time

from ...llm_integration.config import LLMConfig

class LLMClient(ABC):
    """
    Abstract base class for LLM API clients.
    
    All LLM clients must implement this interface to ensure
    consistent behavior across different LLM providers.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
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
    def default_model(self) -> str:
        """
        Get the default model for this provider.
        
        Returns:
            String identifier for the default model
        """
        pass
    
    @abstractmethod
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
        Analyze text using the LLM.
        
        Args:
            text: The text to analyze
            prompt: The prompt template to guide the analysis
            model: Optional model override (default from config if None)
            temperature: Sampling temperature (0.0 for deterministic output)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary with analysis results
            
        Raises:
            Exception: If analysis fails
        """
        pass
    
    @abstractmethod
    async def extract_json(
        self,
        text: str,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured JSON data from text using the LLM.
        
        Args:
            text: The text to analyze
            prompt: The prompt template to guide the extraction
            schema: JSON schema describing expected output structure
            model: Optional model override
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Extracted JSON data as a dictionary
            
        Raises:
            Exception: If extraction fails
        """
        pass
    
    @abstractmethod
    async def classify(
        self,
        text: str,
        categories: List[str],
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Classify text into provided categories using the LLM.
        
        Args:
            text: The text to classify
            categories: List of possible categories
            prompt: The prompt template to guide the classification
            model: Optional model override
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary mapping categories to confidence scores (0-1)
            
        Raises:
            Exception: If classification fails
        """
        pass
    
    async def _handle_retry(
        self, 
        func, 
        *args, 
        retries: Optional[int] = None, 
        **kwargs
    ) -> Any:
        """
        Handle retries for API calls.
        
        Args:
            func: Async function to call
            *args: Arguments to pass to func
            retries: Number of retries (default from config if None)
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result from func
            
        Raises:
            Exception: If all retries fail
        """
        if retries is None:
            retries = self.config.max_retries
            
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < retries:
                    # Calculate exponential backoff delay
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{retries + 1} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_error
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON data from LLM response text.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Extracted JSON as dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted
        """
        # Find JSON blocks in the response (handle markdown code blocks)
        json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        import re
        
        # Try to extract JSON from markdown code blocks
        json_matches = re.findall(json_pattern, response_text)
        
        if json_matches:
            # Use the first JSON block found
            try:
                return json.loads(json_matches[0])
            except json.JSONDecodeError:
                # If the first one fails, try others
                for match in json_matches[1:]:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
        
        # If no JSON blocks or all failed, try treating the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Last resort: try to find anything that looks like JSON
            try:
                # Find text that looks like JSON object (between curly braces)
                curly_pattern = r"\{[\s\S]*\}"
                curly_matches = re.findall(curly_pattern, response_text)
                
                if curly_matches:
                    return json.loads(curly_matches[0])
            except Exception:
                pass
            
            # If all attempts fail
            raise ValueError("Could not extract valid JSON from response")
