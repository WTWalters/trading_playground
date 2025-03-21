"""
Regime detection service for TITAN Trading System.

Provides functionality for detecting and analyzing market regimes.
"""
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple

from django.db import transaction

from .base_service import BaseService
from .market_data_service import MarketDataService
from ..models.symbols import Symbol
from ..models.regimes import MarketRegime, RegimeTransition

from src.market_analysis.regime_detection.detector import RegimeDetector, RegimeType


class RegimeDetectionService(BaseService):
    """
    Service for market regime detection operations.
    
    This service provides methods for:
    - Detecting market regimes
    - Analyzing regime transitions
    - Managing regime data
    """
    
    def __init__(self):
        """Initialize the RegimeDetectionService."""
        super().__init__()
        self.market_data_service = MarketDataService()
        self.regime_detector = RegimeDetector()
        
    async def _initialize_resources(self) -> None:
        """Initialize required resources."""
        await self.market_data_service._initialize_resources()
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources."""
        await self.market_data_service._cleanup_resources()
    
    async def detect_regime(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect market regime for a symbol.
        
        Args:
            symbol: Symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with regime detection results
        """
        await self._initialize_resources()
        
        self.logger.info(f"Detecting regime for {symbol} from {start_date} to {end_date}")
        
        # Fetch market data
        market_data = await self.market_data_service.get_market_data(
            symbol, start_date, end_date, timeframe, source
        )
        
        # Detect regime
        result = self.regime_detector.detect_regime(market_data)
        
        # Convert to dictionary
        return {
            'symbol': symbol,
            'primary_regime': result.primary_regime.value,
            'secondary_regime': result.secondary_regime.value if result.secondary_regime else None,
            'confidence': result.confidence,
            'volatility_regime': result.volatility_regime.value,
            'correlation_regime': result.correlation_regime.value,
            'liquidity_regime': result.liquidity_regime.value,
            'trend_regime': result.trend_regime.value,
            'regime_start_date': result.regime_start_date,
            'stability_score': result.stability_score,
            'transition_probability': {
                regime.value: prob for regime, prob in result.transition_probability.items()
            },
            'features_contribution': result.features_contribution,
            'analysis_date': end_date
        }
    
    @BaseService.sync_wrap
    async def detect_regime_sync(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for detect_regime.
        
        Args:
            symbol: Symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with regime detection results
        """
        return await self.detect_regime(symbol, start_date, end_date, timeframe, source)
    
    async def detect_regime_shifts(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect regime shifts over a period.
        
        Args:
            symbol: Symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            List of regime shift dictionaries
        """
        await self._initialize_resources()
        
        self.logger.info(f"Detecting regime shifts for {symbol} from {start_date} to {end_date}")
        
        # Fetch market data
        market_data = await self.market_data_service.get_market_data(
            symbol, start_date, end_date, timeframe, source
        )
        
        # Detect regime shifts
        shifts = self.regime_detector.detect_regime_shifts(market_data)
        
        # Convert to list of dictionaries
        result = []
        for timestamp, regime in shifts:
            result.append({
                'timestamp': timestamp,
                'regime': regime.value
            })
        
        return result
    
    @BaseService.sync_wrap
    async def detect_regime_shifts_sync(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for detect_regime_shifts.
        
        Args:
            symbol: Symbol identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            List of regime shift dictionaries
        """
        return await self.detect_regime_shifts(symbol, start_date, end_date, timeframe, source)
    
    def create_regime_record(self, regime_result: Dict[str, Any]) -> MarketRegime:
        """
        Create a MarketRegime record from detection results.
        
        Args:
            regime_result: Regime detection results dictionary
            
        Returns:
            Created MarketRegime object
        """
        self.logger.info(f"Creating market regime record for {regime_result['symbol']}")
        
        # Get or create Symbol
        symbol_ticker = regime_result['symbol']
        symbol, _ = Symbol.objects.get_or_create(
            ticker=symbol_ticker,
            defaults={'name': symbol_ticker}
        )
        
        # Create and save MarketRegime
        regime = MarketRegime(
            symbol=symbol,
            detection_date=regime_result['analysis_date'],
            start_date=regime_result['regime_start_date'],
            primary_regime=regime_result['primary_regime'],
            secondary_regime=regime_result['secondary_regime'],
            confidence=regime_result['confidence'],
            volatility_regime=regime_result['volatility_regime'],
            trend_regime=regime_result['trend_regime'],
            liquidity_regime=regime_result['liquidity_regime'],
            correlation_regime=regime_result['correlation_regime'],
            stability_score=regime_result['stability_score'],
            features=regime_result.get('features_contribution', {})
        )
        regime.save()
        
        return regime
    
    def create_regime_transition(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        transition_date: datetime
    ) -> RegimeTransition:
        """
        Create a RegimeTransition record.
        
        Args:
            from_regime: Source regime
            to_regime: Destination regime
            transition_date: Date of transition
            
        Returns:
            Created RegimeTransition object
        """
        self.logger.info(f"Creating regime transition record: {from_regime.primary_regime} -> {to_regime.primary_regime}")
        
        # Create and save RegimeTransition
        transition = RegimeTransition(
            from_regime=from_regime,
            to_regime=to_regime,
            transition_date=transition_date,
            symbol=from_regime.symbol  # Both regimes should have the same symbol
        )
        transition.save()
        
        return transition
    
    def get_current_regime(self, symbol_ticker: str) -> Optional[MarketRegime]:
        """
        Get the current market regime for a symbol.
        
        Args:
            symbol_ticker: Symbol ticker string
            
        Returns:
            Most recent MarketRegime object or None if not found
        """
        try:
            # Get Symbol
            symbol = Symbol.objects.get(ticker=symbol_ticker)
            
            # Get most recent regime
            return MarketRegime.objects.filter(symbol=symbol).order_by('-detection_date').first()
            
        except Symbol.DoesNotExist:
            self.logger.warning(f"Symbol {symbol_ticker} not found")
            return None
        except MarketRegime.DoesNotExist:
            self.logger.warning(f"No regime found for {symbol_ticker}")
            return None
    
    def get_regime_history(
        self,
        symbol_ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[MarketRegime]:
        """
        Get regime history for a symbol.
        
        Args:
            symbol_ticker: Symbol ticker string
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            List of MarketRegime objects
        """
        try:
            # Get Symbol
            symbol = Symbol.objects.get(ticker=symbol_ticker)
            
            # Create query
            query = MarketRegime.objects.filter(symbol=symbol)
            
            # Apply date filters if provided
            if start_date:
                query = query.filter(detection_date__gte=start_date)
            if end_date:
                query = query.filter(detection_date__lte=end_date)
                
            # Return ordered list
            return list(query.order_by('detection_date'))
            
        except Symbol.DoesNotExist:
            self.logger.warning(f"Symbol {symbol_ticker} not found")
            return []
