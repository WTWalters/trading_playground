"""
Parameter management service for TITAN Trading System.

Provides functionality for adaptive parameter optimization
and management across different market regimes.
"""
import logging
from datetime import datetime
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any, Tuple

from django.db import transaction

from .base_service import BaseService
from .market_data_service import MarketDataService
from .regime_detection_service import RegimeDetectionService
from ..models.symbols import Symbol
from ..models.pairs import TradingPair
from ..models.regimes import MarketRegime
from ..models.backtesting import BacktestRun

from src.market_analysis.parameter_management.walk_forward import WalkForwardOptimizer
from src.market_analysis.parameter_management.parameter_manager import ParameterManager


class ParameterService(BaseService):
    """
    Service for parameter management operations.
    
    This service provides methods for:
    - Optimizing strategy parameters
    - Managing regime-specific parameters
    - Performing walk-forward optimization
    """
    
    def __init__(self):
        """Initialize the ParameterService."""
        super().__init__()
        self.market_data_service = MarketDataService()
        self.regime_service = RegimeDetectionService()
        self.parameter_manager = ParameterManager()
        
    async def _initialize_resources(self) -> None:
        """Initialize required resources."""
        await self.market_data_service._initialize_resources()
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources."""
        await self.market_data_service._cleanup_resources()
    
    async def optimize_parameters(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        parameter_ranges: Dict[str, List[float]],
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters for a pair.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for optimization
            end_date: End date for optimization
            parameter_ranges: Dictionary of parameter names to lists of values to test
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with optimization results
        """
        await self._initialize_resources()
        
        self.logger.info(f"Optimizing parameters for {symbol1}/{symbol2} from {start_date} to {end_date}")
        
        # Create optimizer
        optimizer = WalkForwardOptimizer(self.market_data_service.db_manager)
        
        # Run optimization
        results = await optimizer.optimize(
            symbol1=symbol1,
            symbol2=symbol2,
            hedge_ratio=hedge_ratio,
            start_date=start_date,
            end_date=end_date,
            parameter_ranges=parameter_ranges,
            timeframe=timeframe,
            source=source
        )
        
        return results
    
    @BaseService.sync_wrap
    async def optimize_parameters_sync(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        parameter_ranges: Dict[str, List[float]],
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for optimize_parameters.
        
        Args:
            (Same as optimize_parameters)
            
        Returns:
            Dictionary with optimization results
        """
        return await self.optimize_parameters(
            symbol1, symbol2, hedge_ratio, start_date, end_date,
            parameter_ranges, timeframe, source
        )
    
    async def perform_walk_forward_test(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        train_size: int = 252,
        test_size: int = 63,
        parameter_ranges: Optional[Dict[str, List[float]]] = None,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization test.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for walk-forward test
            end_date: End date for walk-forward test
            train_size: Number of periods for training window
            test_size: Number of periods for test window
            parameter_ranges: Dictionary of parameter names to lists of values to test
            timeframe: Data timeframe
            source: Data source (optional)
            
        Returns:
            Dictionary with walk-forward test results
        """
        await self._initialize_resources()
        
        self.logger.info(f"Performing walk-forward test for {symbol1}/{symbol2}")
        
        # Use default parameter ranges if not provided
        if parameter_ranges is None:
            parameter_ranges = {
                'entry_threshold': [1.5, 2.0, 2.5, 3.0],
                'exit_threshold': [0.0, 0.5, 1.0],
                'stop_loss_pct': [None, 5.0, 10.0],
                'risk_per_trade': [1.0, 2.0, 3.0]
            }
        
        # Create optimizer
        optimizer = WalkForwardOptimizer(self.market_data_service.db_manager)
        
        # Run walk-forward test
        results = await optimizer.walk_forward_test(
            symbol1=symbol1,
            symbol2=symbol2,
            hedge_ratio=hedge_ratio,
            start_date=start_date,
            end_date=end_date,
            train_size=train_size,
            test_size=test_size,
            parameter_ranges=parameter_ranges,
            timeframe=timeframe,
            source=source
        )
        
        return results
    
    @BaseService.sync_wrap
    async def perform_walk_forward_test_sync(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        train_size: int = 252,
        test_size: int = 63,
        parameter_ranges: Optional[Dict[str, List[float]]] = None,
        timeframe: str = '1d',
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for perform_walk_forward_test.
        
        Args:
            (Same as perform_walk_forward_test)
            
        Returns:
            Dictionary with walk-forward test results
        """
        return await self.perform_walk_forward_test(
            symbol1, symbol2, hedge_ratio, start_date, end_date,
            train_size, test_size, parameter_ranges, timeframe, source
        )
    
    def get_optimal_parameters(
        self,
        trading_pair: TradingPair,
        current_regime: Optional[MarketRegime] = None
    ) -> Dict[str, Any]:
        """
        Get optimal parameters for a trading pair based on current regime.
        
        Args:
            trading_pair: TradingPair object
            current_regime: Current market regime (optional)
            
        Returns:
            Dictionary with optimal parameters
        """
        self.logger.info(f"Getting optimal parameters for pair {trading_pair.id}")
        
        # Get current regime if not provided
        if current_regime is None:
            current_regime = self.regime_service.get_current_regime(
                trading_pair.symbol_1.ticker
            )
        
        # Get recent backtest runs for this pair
        recent_backtests = BacktestRun.objects.filter(
            trading_pair=trading_pair
        ).order_by('-end_date')[:5]
        
        # If no backtests available, return default parameters
        if not recent_backtests:
            self.logger.warning(f"No backtest data available for pair {trading_pair.id}")
            return {
                'entry_threshold': 2.0,
                'exit_threshold': 0.0,
                'stop_loss_pct': None,
                'risk_per_trade': 2.0
            }
        
        # Parse parameters from backtest runs
        backtest_params = []
        backtest_performances = []
        
        for backtest in recent_backtests:
            # Get parameters
            params = json.loads(backtest.parameters)
            
            # Get performance
            try:
                result = backtest.result
                performance = result.sharpe_ratio
            except:
                performance = 0.0
            
            backtest_params.append(params)
            backtest_performances.append(performance)
        
        # Find best parameters
        best_idx = backtest_performances.index(max(backtest_performances))
        best_params = backtest_params[best_idx]
        
        # Adjust parameters based on regime if available
        if current_regime:
            # Adjust entry threshold based on volatility regime
            if current_regime.volatility_regime == 'high_volatility':
                # Increase threshold in high volatility
                best_params['entry_threshold'] = min(
                    best_params.get('entry_threshold', 2.0) * 1.2, 3.0
                )
            elif current_regime.volatility_regime == 'low_volatility':
                # Decrease threshold in low volatility
                best_params['entry_threshold'] = max(
                    best_params.get('entry_threshold', 2.0) * 0.8, 1.5
                )
            
            # Adjust risk based on trend regime
            if current_regime.trend_regime == 'trending':
                # Decrease risk in trending markets (less mean reversion)
                best_params['risk_per_trade'] = max(
                    best_params.get('risk_per_trade', 2.0) * 0.8, 1.0
                )
            elif current_regime.trend_regime == 'mean_reverting':
                # Increase risk in mean-reverting markets
                best_params['risk_per_trade'] = min(
                    best_params.get('risk_per_trade', 2.0) * 1.2, 3.0
                )
        
        return best_params
    
    def store_regime_parameters(
        self,
        trading_pair: TradingPair,
        regime: MarketRegime,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Store optimal parameters for a specific regime.
        
        Args:
            trading_pair: TradingPair object
            regime: MarketRegime object
            parameters: Dictionary of parameter values
        """
        self.logger.info(f"Storing regime parameters for pair {trading_pair.id}, regime {regime.primary_regime}")
        
        # Extract key parameters
        entry_threshold = parameters.get('entry_threshold', 2.0)
        exit_threshold = parameters.get('exit_threshold', 0.0)
        risk_per_trade = parameters.get('risk_per_trade', 2.0)
        
        # Store in parameter manager
        self.parameter_manager.set_regime_parameters(
            pair_id=str(trading_pair.id),
            regime=regime.primary_regime,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            risk_per_trade=risk_per_trade
        )
