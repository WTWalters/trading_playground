"""
Pair analysis service for TITAN Trading System.

Provides functionality for analyzing cointegrated pairs and
managing pair relationships.
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

from django.db import transaction

from .base_service import BaseService
from .market_data_service import MarketDataService
from ..models.symbols import Symbol
from ..models.pairs import TradingPair, PairSpread

from src.market_analysis.cointegration import CointegrationTester
from src.config.db_config import DatabaseConfig


class PairAnalysisService(BaseService):
    """
    Service for pair analysis operations.
    
    This service provides methods for:
    - Finding cointegrated pairs
    - Testing cointegration between specific symbols
    - Calculating pair stability
    - Managing trading pair relationships
    """
    
    def __init__(self):
        """Initialize the PairAnalysisService."""
        super().__init__()
        self.market_data_service = MarketDataService()
        self.cointegration_tester = None
        
    async def _initialize_resources(self) -> None:
        """
        Initialize required resources.
        
        This initializes the market data service and cointegration tester.
        """
        await self.market_data_service._initialize_resources()
        
        if self.cointegration_tester is None:
            self.logger.info("Initializing CointegrationTester")
            self.cointegration_tester = CointegrationTester(
                self.market_data_service.db_manager
            )
    
    async def _cleanup_resources(self) -> None:
        """
        Clean up resources.
        
        This cleans up the market data service and other resources.
        """
        await self.market_data_service._cleanup_resources()
    
    async def select_pairs(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        min_correlation: float = 0.6,
        significance_level: float = 0.05,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Select cointegrated pairs from a list of symbols.
        
        Args:
            symbols: List of symbol identifiers
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            min_correlation: Minimum correlation threshold
            significance_level: Statistical significance level
            source: Data source (optional)
            
        Returns:
            List of dictionaries with pair information
        """
        await self._initialize_resources()
        
        self.logger.info(f"Selecting pairs from {len(symbols)} symbols")
        return await self.cointegration_tester.select_pairs(
            symbols,
            start_date,
            end_date,
            timeframe=timeframe,
            min_correlation=min_correlation,
            significance_level=significance_level,
            source=source
        )
    
    @BaseService.sync_wrap
    async def select_pairs_sync(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        min_correlation: float = 0.6,
        significance_level: float = 0.05,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for select_pairs.
        
        Args:
            symbols: List of symbol identifiers
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            min_correlation: Minimum correlation threshold
            significance_level: Statistical significance level
            source: Data source (optional)
            
        Returns:
            List of dictionaries with pair information
        """
        return await self.select_pairs(
            symbols, start_date, end_date, timeframe,
            min_correlation, significance_level, source
        )
    
    async def test_pair(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test cointegration between two specific symbols.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with cointegration test results
        """
        await self._initialize_resources()
        
        self.logger.info(f"Testing cointegration between {symbol1} and {symbol2}")
        
        # Fetch data for both symbols
        df1 = await self.market_data_service.get_market_data(
            symbol1, start_date, end_date, timeframe, source
        )
        df2 = await self.market_data_service.get_market_data(
            symbol2, start_date, end_date, timeframe, source
        )
        
        # Run Engle-Granger test
        eg_result = await self.cointegration_tester.engle_granger_test(
            df1['close'].values, df2['close'].values
        )
        
        # Run Johansen test
        johansen_result = await self.cointegration_tester.johansen_test(
            df1['close'].values, df2['close'].values
        )
        
        # Calculate correlation
        correlation = float(df1['close'].corr(df2['close']))
        
        # Return combined results
        return {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'correlation': correlation,
            'hedge_ratio': eg_result['hedge_ratio'],
            'engle_granger_result': eg_result,
            'johansen_result': johansen_result,
            'is_cointegrated': eg_result['is_cointegrated'] or johansen_result['is_cointegrated'],
            'timeframe': timeframe,
            'period': {
                'start': start_date,
                'end': end_date
            }
        }
    
    @BaseService.sync_wrap
    async def test_pair_sync(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for test_pair.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with cointegration test results
        """
        return await self.test_pair(
            symbol1, symbol2, start_date, end_date, timeframe, source
        )
    
    async def calculate_stability(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        window_size: int = 60,
        step_size: int = 20,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate stability of cointegration relationship over time.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            window_size: Size of rolling window in days
            step_size: Step size for rolling window
            source: Data source (optional)
            
        Returns:
            Dictionary with stability metrics
        """
        await self._initialize_resources()
        
        self.logger.info(f"Calculating stability for {symbol1} and {symbol2}")
        return await self.cointegration_tester.calculate_cointegration_stability(
            symbol1, symbol2, start_date, end_date, window_size, step_size, timeframe, source
        )
    
    @BaseService.sync_wrap
    async def calculate_stability_sync(
        self,
        symbol1: str,
        symbol2: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        window_size: int = 60,
        step_size: int = 20,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for calculate_stability.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            window_size: Size of rolling window in days
            step_size: Step size for rolling window
            source: Data source (optional)
            
        Returns:
            Dictionary with stability metrics
        """
        return await self.calculate_stability(
            symbol1, symbol2, start_date, end_date, timeframe,
            window_size, step_size, source
        )
    
    def create_trading_pair(self, pair_result: Dict[str, Any], user=None) -> TradingPair:
        """
        Create a TradingPair object from cointegration results.
        
        Args:
            pair_result: Cointegration test results dictionary
            user: User who created the pair (optional)
            
        Returns:
            Created TradingPair object
        """
        self.logger.info(f"Creating trading pair for {pair_result['symbol1']} and {pair_result['symbol2']}")
        
        # Get or create Symbol objects
        symbol1, _ = Symbol.objects.get_or_create(
            ticker=pair_result['symbol1'],
            defaults={'name': pair_result['symbol1']}
        )
        symbol2, _ = Symbol.objects.get_or_create(
            ticker=pair_result['symbol2'],
            defaults={'name': pair_result['symbol2']}
        )
        
        # Extract p-value from results
        if 'engle_granger_result' in pair_result and 'adf_results' in pair_result['engle_granger_result']:
            p_value = pair_result['engle_granger_result']['adf_results']['p_value']
        else:
            p_value = pair_result.get('p_value', 0.5)
        
        # Calculate half-life if available, otherwise use default
        half_life = self._calculate_half_life(pair_result)
        
        # Create and save TradingPair
        pair = TradingPair(
            symbol_1=symbol1,
            symbol_2=symbol2,
            cointegration_pvalue=p_value,
            half_life=half_life,
            correlation=pair_result['correlation'],
            hedge_ratio=pair_result['hedge_ratio'],
            stability_score=pair_result.get('stability_score', 0.5),
            is_active=True,
            lookback_days=252  # Default value
        )
        pair.save()
        
        return pair
    
    def create_trading_pairs(self, pair_results: List[Dict[str, Any]], user=None) -> List[TradingPair]:
        """
        Create multiple TradingPair objects from cointegration results.
        
        Args:
            pair_results: List of cointegration test results dictionaries
            user: User who created the pairs (optional)
            
        Returns:
            List of created TradingPair objects
        """
        pairs = []
        
        with transaction.atomic():
            for result in pair_results:
                pair = self.create_trading_pair(result, user)
                pairs.append(pair)
                
        return pairs
    
    def _calculate_half_life(self, pair_result: Dict[str, Any]) -> float:
        """
        Calculate half-life from cointegration results.
        
        Args:
            pair_result: Cointegration test results
            
        Returns:
            Half-life in days
        """
        # Extract beta coefficient from regression results
        if ('engle_granger_result' in pair_result and 
            'regression_results' in pair_result['engle_granger_result'] and
            'params' in pair_result['engle_granger_result']['regression_results']):
            params = pair_result['engle_granger_result']['regression_results']['params']
            if len(params) > 0:
                # Half-life = -log(2) / log(1 + beta)
                beta = params[0]  # Assuming the coefficient is at index 0
                import math
                half_life = -math.log(2) / math.log(1 + beta)
                return max(1.0, abs(half_life))  # Ensure positive and at least 1.0
        
        # Default value if calculation isn't possible
        return 10.0
