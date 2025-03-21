"""
LLM-based regime analysis enhancement module.

This module provides the LLMRegimeAnalyzer class which enhances the existing
regime detection system with LLM-based analysis of news and market context.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
import json
import asyncio
import os
from pathlib import Path

from ...config.config_manager import load_config
from ...llm_integration.config import LLMConfig, load_llm_config
from ...market_analysis.regime_detection.enhanced_detector import EnhancedRegimeDetector, MacroRegimeType, RegimeType
from ...database.manager import DatabaseManager
from ...llm_integration.clients import LLMClient, LLMClientFactory

class LLMRegimeAnalyzer:
    """
    LLM-based enhancement for market regime detection.
    
    This class adds contextual awareness to the statistical regime detection
    by analyzing financial news, economic reports, and other text sources.
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None,
        regime_detector: Optional[EnhancedRegimeDetector] = None
    ):
        """
        Initialize the LLM regime analyzer.
        
        Args:
            db_manager: Database manager for storing and retrieving data
            config: LLM configuration (loads from default if None)
            llm_client: Optional LLM client (creates default if None)
            regime_detector: Optional regime detector (creates default if None)
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Load config if not provided
        if config is None:
            self.config = load_llm_config()
        else:
            self.config = config
            
        # Create default LLM client if not provided
        if llm_client is None:
            client_factory = LLMClientFactory(self.config)
            self.llm_client = client_factory.get_client()
        else:
            self.llm_client = llm_client
            
        # Create default regime detector if not provided
        if regime_detector is None:
            self.regime_detector = EnhancedRegimeDetector()
        else:
            self.regime_detector = regime_detector
            
        # Load prompts
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        prompts = {}
        prompts_dir = Path(self.config.prompts_dir) / "regime_analysis"
        
        # List of prompts to load
        prompt_files = [
            "regime_classification.txt",
            "market_turning_point.txt",
            "fed_statement.txt",
            "earnings_call.txt",
            "economic_report.txt"
        ]
        
        for filename in prompt_files:
            prompt_path = prompts_dir / filename
            prompt_name = filename.replace(".txt", "")
            
            try:
                if prompt_path.exists():
                    with open(prompt_path, "r") as f:
                        prompts[prompt_name] = f.read()
                else:
                    # Use default prompt if file not found
                    self.logger.warning(f"Prompt file not found: {prompt_path}")
                    prompts[prompt_name] = self._get_default_prompt(prompt_name)
            except Exception as e:
                self.logger.error(f"Error loading prompt {prompt_name}: {e}")
                prompts[prompt_name] = self._get_default_prompt(prompt_name)
        
        return prompts
    
    def _get_default_prompt(self, prompt_name: str) -> str:
        """Get default prompt template for a given name."""
        defaults = {
            "regime_classification": """
            Analyze the following financial news and determine the most likely market regime.
            Classify the regime as one of: TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY, 
            RISK_ON, RISK_OFF, EXPANSION, CONTRACTION, or TRANSITION.
            
            For each classification, provide:
            1. Confidence level (0-100%)
            2. Key indicators or signals from the text
            3. Potential market implications
            
            Format your response as a JSON object with the following structure:
            {
              "primary_regime": "REGIME_TYPE",
              "confidence": 85,
              "secondary_regime": "REGIME_TYPE",
              "secondary_confidence": 60,
              "key_indicators": ["indicator1", "indicator2"...],
              "market_implications": "description of implications",
              "risk_assessment": "high/medium/low"
            }
            """,
            "market_turning_point": """
            Analyze the following financial news and determine if there are signals of a potential market turning point.
            A turning point is a significant change in market direction or regime that may not be captured by technical indicators alone.
            
            Consider:
            1. Shifts in central bank policy or rhetoric
            2. Changes in market sentiment or psychology
            3. Structural economic changes
            4. Geopolitical developments
            5. Sector rotation patterns
            
            Provide your analysis as a JSON object with the following structure:
            {
              "is_turning_point": true/false,
              "confidence": 75,
              "direction": "bullish/bearish/undefined",
              "key_indicators": ["indicator1", "indicator2"...],
              "expected_timeframe": "short-term/medium-term/long-term",
              "narrative": "brief explanation of the turning point thesis"
            }
            """,
            "fed_statement": """
            Analyze the following Federal Reserve statement or minutes and extract key insights
            about monetary policy, economic outlook, and implications for financial markets.
            
            Focus on:
            1. Changes in policy stance or forward guidance
            2. Inflation and employment outlook
            3. Balance sheet policies
            4. Risk assessments
            5. Dissenting views among committee members
            
            Format your response as a JSON object with the following structure:
            {
              "policy_stance": "hawkish/dovish/neutral/mixed",
              "confidence": 80,
              "key_changes": ["change1", "change2"...],
              "inflation_outlook": "summary of inflation perspective",
              "growth_outlook": "summary of growth perspective",
              "market_implications": {
                "equities": "positive/negative/neutral",
                "bonds": "positive/negative/neutral",
                "currencies": "dollar strengthening/weakening",
                "explanation": "brief explanation"
              }
            }
            """
        }
        # Return default prompt or empty string if not found
        return defaults.get(prompt_name, "")
    
    async def analyze_recent_news(
        self, 
        timeframe: str = "7d", 
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze recent financial news to enhance regime detection.
        
        Args:
            timeframe: Time period to analyze (e.g. "7d", "1m")
            symbols: Optional list of symbols to focus on
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Convert timeframe to datetime
            end_date = datetime.now()
            if timeframe.endswith("d"):
                days = int(timeframe[:-1])
                start_date = end_date - timedelta(days=days)
            elif timeframe.endswith("m"):
                months = int(timeframe[:-1])
                start_date = end_date - timedelta(days=months*30)
            else:
                start_date = end_date - timedelta(days=7)
            
            # Retrieve news from database
            news_data = await self._retrieve_relevant_news(start_date, end_date, symbols)
            
            if news_data.empty:
                self.logger.warning("No news data available for analysis")
                return {"status": "no_data", "message": "No news data available for analysis"}
            
            # Combine news into a single text for analysis
            combined_text = self._prepare_news_for_analysis(news_data)
            
            # Use LLM to analyze the news
            analysis_result = await self.llm_client.analyze_text(
                combined_text, 
                self.prompts["regime_classification"]
            )
            
            # Parse and validate the analysis
            parsed_result = self._parse_llm_result(analysis_result)
            
            # Store the analysis in the database
            await self._store_analysis_result(parsed_result, start_date, end_date, symbols)
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error in news analysis: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_specific_document(
        self,
        document_type: str,
        document_content: str,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a specific document like Fed statement or earnings call.
        
        Args:
            document_type: Type of document (fed_statement, earnings_call, etc.)
            document_content: Text content of the document
            document_metadata: Metadata about the document (date, source, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Select appropriate prompt based on document type
            prompt = self.prompts.get(document_type, self.prompts["regime_classification"])
            
            # Use LLM to analyze the document
            analysis_result = await self.llm_client.analyze_text(
                document_content, 
                prompt
            )
            
            # Parse and validate the analysis
            parsed_result = self._parse_llm_result(analysis_result)
            
            # Add metadata to the result
            parsed_result["document_type"] = document_type
            parsed_result["document_metadata"] = document_metadata
            
            # Store the analysis in the database
            if document_metadata.get("date"):
                doc_date = document_metadata["date"]
                await self._store_document_analysis(parsed_result, document_type, doc_date)
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Error in document analysis: {str(e)}")
            return {"error": str(e)}
    
    async def enhance_regime_detection(
        self,
        market_data: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Combine statistical regime detection with LLM analysis.
        
        Args:
            market_data: Market price/volume data
            macro_data: Optional macro economic indicators
            
        Returns:
            Enhanced regime detection result
        """
        try:
            # Get statistical regime detection result
            stat_regime_result = self.regime_detector.detect_regime(market_data, macro_data)
            
            # Get the start and end dates from the market data
            start_date = market_data.index.min()
            end_date = market_data.index.max()
            
            # Retrieve recent news analysis for this time period
            llm_analysis = await self._retrieve_recent_analysis(start_date, end_date)
            
            # Generate text summary of the statistical regime detection
            stat_regime_summary = {
                "primary_regime": stat_regime_result.primary_regime.value,
                "secondary_regime": stat_regime_result.secondary_regime.value if stat_regime_result.secondary_regime else None,
                "confidence": stat_regime_result.confidence,
                "volatility_regime": stat_regime_result.volatility_regime.value if stat_regime_result.volatility_regime else None,
                "correlation_regime": stat_regime_result.correlation_regime.value if stat_regime_result.correlation_regime else None,
                "transition_probability": stat_regime_result.transition_probability
            }
            
            # Combine statistical and LLM-based regime analysis
            combined_result = self._combine_regime_analyses(stat_regime_summary, llm_analysis)
            
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced regime detection: {str(e)}")
            return {"error": str(e)}
    
    async def detect_market_turning_points(
        self,
        market_data: pd.DataFrame,
        news_lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Use LLM analysis to detect potential market turning points.
        
        Args:
            market_data: Market price/volume data
            news_lookback_days: Days of news to analyze
            
        Returns:
            Dictionary with turning point analysis
        """
        try:
            # Get end date from market data
            end_date = market_data.index.max()
            start_date = end_date - timedelta(days=news_lookback_days)
            
            # Retrieve news for this period
            news_data = await self._retrieve_relevant_news(start_date, end_date)
            
            # Combine news into a single text for analysis
            combined_text = self._prepare_news_for_analysis(news_data)
            
            # Statistical turning point detection from regime detector
            stat_result = self.regime_detector._detect_turning_point(market_data)
            is_turning_point, confidence = stat_result
            
            # Use LLM to analyze for turning points
            analysis_result = await self.llm_client.analyze_text(
                combined_text, 
                self.prompts["market_turning_point"]
            )
            
            # Parse and validate the analysis
            llm_result = self._parse_llm_result(analysis_result)
            
            # Combine statistical and LLM results
            combined_result = {
                "is_turning_point": is_turning_point or llm_result.get("is_turning_point", False),
                "statistical_confidence": confidence,
                "llm_confidence": llm_result.get("confidence", 0),
                "direction": llm_result.get("direction", "unknown"),
                "key_indicators": llm_result.get("key_indicators", []),
                "narrative": llm_result.get("narrative", "")
            }
            
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Error in turning point detection: {str(e)}")
            return {"error": str(e)}
    
    # Helper methods
    async def _retrieve_relevant_news(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve relevant news from the database.
        
        Args:
            start_date: Start date for news
            end_date: End date for news
            symbols: Optional symbols to filter by
            
        Returns:
            DataFrame with news articles
        """
        try:
            # Define the SQL query based on whether symbols are provided
            if symbols and len(symbols) > 0:
                # Query with symbol filter
                query = """
                SELECT n.*
                FROM news_data n
                JOIN symbol_news_mapping m ON n.id = m.news_id
                WHERE n.published_time BETWEEN $1 AND $2
                AND m.symbol = ANY($3)
                ORDER BY n.published_time DESC
                LIMIT 200
                """
                params = [start_date, end_date, symbols]
            else:
                # Query without symbol filter
                query = """
                SELECT *
                FROM news_data
                WHERE published_time BETWEEN $1 AND $2
                ORDER BY published_time DESC
                LIMIT 200
                """
                params = [start_date, end_date]
            
            # Execute the query
            async with self.db_manager.pool.acquire() as conn:
                records = await conn.fetch(query, *params)
            
            # Convert to DataFrame
            if not records:
                return pd.DataFrame()
                
            # Process records into DataFrame
            news_data = []
            for record in records:
                news_data.append({
                    "id": record["id"],
                    "title": record["title"],
                    "content": record["content"],
                    "source": record["source"],
                    "published_time": record["published_time"],
                    "sentiment": record["sentiment"],
                    "categories": record["categories"],
                    "entities": record["entities"] if "entities" in record else {}
                })
                
            return pd.DataFrame(news_data)
            
        except Exception as e:
            self.logger.error(f"Error retrieving news: {e}")
            return pd.DataFrame()
    
    def _prepare_news_for_analysis(self, news_data: pd.DataFrame) -> str:
        """
        Prepare news data for LLM analysis.
        
        Args:
            news_data: DataFrame with news articles
            
        Returns:
            Formatted text for analysis
        """
        if news_data.empty:
            return "No news data available for analysis."
            
        # Sort by date
        news_data = news_data.sort_values("published_time", ascending=False)
        
        # Build text
        text_parts = ["RECENT FINANCIAL NEWS:"]
        
        # Add each article with formatting
        for _, article in news_data.iterrows():
            date_str = article["published_time"].strftime("%Y-%m-%d %H:%M")
            source = article["source"]
            title = article["title"]
            content = article["content"]
            
            # Add article header
            text_parts.append(f"\n--- {date_str} | {source} ---")
            text_parts.append(f"TITLE: {title}")
            
            # Add content (truncate if too long)
            max_content_length = 500  # Characters
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                
            text_parts.append(f"CONTENT: {content}")
        
        # Limit total text length
        combined_text = "\n".join(text_parts)
        max_length = 10000  # Characters
        
        if len(combined_text) > max_length:
            # Truncate while preserving whole articles
            truncated_parts = []
            total_length = 0
            
            for part in text_parts:
                if total_length + len(part) > max_length:
                    break
                    
                truncated_parts.append(part)
                total_length += len(part)
                
            combined_text = "\n".join(truncated_parts)
            combined_text += "\n\n[Additional articles omitted due to length constraints]"
        
        return combined_text
    
    def _parse_llm_result(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the LLM analysis result.
        
        Args:
            llm_result: Raw result from LLM
            
        Returns:
            Parsed and validated result
        """
        try:
            # Check if the result contains a JSON response
            if "json_response" in llm_result:
                result = llm_result["json_response"]
            else:
                # Try to parse JSON from raw response
                raw_response = llm_result.get("raw_response", "")
                
                # If empty or error, return empty result
                if not raw_response or "error" in llm_result:
                    return {"error": "Invalid LLM response"}
                
                # Try to extract JSON
                try:
                    # Extract content within curly braces
                    import re
                    json_pattern = r"\{.*\}"
                    match = re.search(json_pattern, raw_response, re.DOTALL)
                    
                    if match:
                        json_str = match.group(0)
                        result = json.loads(json_str)
                    else:
                        return {"error": "No JSON found in response"}
                        
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON in response"}
            
            # Add LLM model information
            result["model_used"] = llm_result.get("model", "unknown")
            result["analysis_time"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM result: {e}")
            return {"error": str(e)}
    
    async def _store_analysis_result(
        self,
        analysis: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> None:
        """
        Store the analysis result in the database.
        
        Args:
            analysis: Analysis result to store
            start_date: Start date of analyzed period
            end_date: End date of analyzed period
            symbols: Optional symbols involved
        """
        try:
            # Store in news_llm_analysis table
            query = """
            INSERT INTO llm_performance_metrics
            (metric_date, metric_type, baseline_value, enhanced_value, 
            improvement_pct, sample_size, time_period, notes, raw_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            # Prepare parameters
            params = [
                datetime.now(),  # metric_date
                "regime_analysis",  # metric_type
                None,  # baseline_value (no baseline for this)
                analysis.get("confidence", 0),  # enhanced_value
                None,  # improvement_pct (not applicable)
                1,  # sample_size
                f"{(end_date - start_date).days}d",  # time_period
                f"Symbols: {', '.join(symbols) if symbols else 'All'}",  # notes
                json.dumps(analysis)  # raw_data
            ]
            
            # Execute the query
            async with self.db_manager.pool.acquire() as conn:
                await conn.execute(query, *params)
                
        except Exception as e:
            self.logger.error(f"Error storing analysis result: {e}")
    
    async def _store_document_analysis(
        self,
        analysis: Dict[str, Any],
        document_type: str,
        document_date: datetime
    ) -> None:
        """
        Store document analysis in the database.
        
        Args:
            analysis: Analysis result to store
            document_type: Type of document
            document_date: Date of the document
        """
        try:
            # Store in llm_performance_metrics table
            query = """
            INSERT INTO llm_performance_metrics
            (metric_date, metric_type, baseline_value, enhanced_value, 
            improvement_pct, sample_size, time_period, notes, raw_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            # Prepare parameters
            params = [
                document_date,  # metric_date
                f"document_analysis_{document_type}",  # metric_type
                None,  # baseline_value (no baseline for documents)
                analysis.get("confidence", 0),  # enhanced_value
                None,  # improvement_pct (not applicable)
                1,  # sample_size
                "1d",  # time_period
                f"Document type: {document_type}",  # notes
                json.dumps(analysis)  # raw_data
            ]
            
            # Execute the query
            async with self.db_manager.pool.acquire() as conn:
                await conn.execute(query, *params)
                
        except Exception as e:
            self.logger.error(f"Error storing document analysis: {e}")
    
    async def _retrieve_recent_analysis(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Retrieve recent LLM analysis results from the database.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with analysis data
        """
        try:
            # Query the database for recent analysis
            query = """
            SELECT raw_data
            FROM llm_performance_metrics
            WHERE metric_type = 'regime_analysis'
            AND metric_date BETWEEN $1 AND $2
            ORDER BY metric_date DESC
            LIMIT 1
            """
            
            # Execute the query
            async with self.db_manager.pool.acquire() as conn:
                record = await conn.fetchrow(query, start_date, end_date)
            
            if record and "raw_data" in record:
                # Parse the raw_data JSON
                try:
                    return json.loads(record["raw_data"])
                except json.JSONDecodeError:
                    return {}
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent analysis: {e}")
            return {}
    
    def _combine_regime_analyses(
        self,
        statistical_analysis: Dict[str, Any],
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine statistical and LLM-based regime analyses.
        
        Args:
            statistical_analysis: Results from statistical detection
            llm_analysis: Results from LLM analysis
            
        Returns:
            Combined analysis
        """
        # If either analysis is empty, return the other
        if not statistical_analysis:
            return llm_analysis
        if not llm_analysis:
            return statistical_analysis
            
        # Get primary regimes
        stat_regime = statistical_analysis.get("primary_regime")
        llm_regime = llm_analysis.get("primary_regime")
        
        # Get confidence scores
        stat_confidence = statistical_analysis.get("confidence", 0)
        llm_confidence = llm_analysis.get("confidence", 0)
        
        # Determine if regimes agree
        regimes_agree = stat_regime == llm_regime
        
        # Calculate weighted confidence
        # Give more weight to statistical if no agreement
        if regimes_agree:
            # Both agree, use higher confidence
            combined_confidence = max(stat_confidence, llm_confidence)
            primary_regime = stat_regime
            combined_weight = "agreement"
        else:
            # Disagreement - weight based on relative confidence
            stat_weight = 0.7  # Statistical gets higher baseline weight
            llm_weight = 0.3
            
            # Adjust weights based on confidence
            total_confidence = stat_confidence + llm_confidence
            if total_confidence > 0:
                stat_adjusted = (stat_confidence / total_confidence) * stat_weight
                llm_adjusted = (llm_confidence / total_confidence) * llm_weight
            else:
                stat_adjusted = stat_weight
                llm_adjusted = llm_weight
            
            # Select regime based on weighted confidence
            if stat_adjusted >= llm_adjusted:
                primary_regime = stat_regime
                combined_confidence = stat_confidence
                combined_weight = "statistical"
            else:
                primary_regime = llm_regime
                combined_confidence = llm_confidence
                combined_weight = "llm"
        
        # Create combined result
        combined_result = {
            "primary_regime": primary_regime,
            "confidence": combined_confidence,
            "regimes_agree": regimes_agree,
            "weightings": combined_weight,
            "statistical_regime": stat_regime,
            "statistical_confidence": stat_confidence,
            "llm_regime": llm_regime,
            "llm_confidence": llm_confidence,
            "key_indicators": llm_analysis.get("key_indicators", []),
            "market_implications": llm_analysis.get("market_implications", ""),
            "risk_assessment": llm_analysis.get("risk_assessment", "medium"),
            "analysis_time": datetime.now().isoformat()
        }
        
        # Add secondary regime if available
        if "secondary_regime" in statistical_analysis or "secondary_regime" in llm_analysis:
            # Prefer LLM's secondary regime if available
            if "secondary_regime" in llm_analysis:
                combined_result["secondary_regime"] = llm_analysis["secondary_regime"]
                combined_result["secondary_confidence"] = llm_analysis.get("secondary_confidence", 0)
            else:
                combined_result["secondary_regime"] = statistical_analysis.get("secondary_regime")
                combined_result["secondary_confidence"] = statistical_analysis.get("secondary_confidence", 0)
        
        return combined_result
