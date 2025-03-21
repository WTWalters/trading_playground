# Service Layer Architecture

This document outlines the service layer architecture that will bridge the Django integration with the existing TITAN Trading System components.

## Overview

The service layer is a crucial component that provides a clean interface between:
- Django's synchronous web framework
- The existing asynchronous trading system

This layer handles the conversion between synchronous and asynchronous code, data translation, and business logic.

## Service Layer Structure

```
trading/
├── services/
│   ├── __init__.py
│   ├── base_service.py             # Base service with sync/async bridging
│   ├── pair_analysis_service.py    # Cointegration and pair selection
│   ├── regime_detection_service.py # Market regime detection
│   ├── market_data_service.py      # Data access and manipulation
│   ├── backtesting_service.py      # Backtesting operations
│   ├── signal_generation_service.py # Trading signal generation
│   └── parameter_service.py        # Adaptive parameter management
```

## BaseService Class

All services inherit from a common `BaseService` class that provides:
- Asynchronous function execution in a synchronous context
- Logging and error handling
- Resource management

```python
# trading/services/base_service.py
import asyncio
import logging
from functools import wraps

class BaseService:
    """Base class for service layer components."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def run_async(self, async_func, *args, **kwargs):
        """Run an asynchronous function in a synchronous context."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
            
    @staticmethod
    def sync_wrap(async_func):
        """Decorator to create a synchronous wrapper for an async function."""
        @wraps(async_func)
        def wrapper(self, *args, **kwargs):
            return self.run_async(async_func, self, *args, **kwargs)
        return wrapper
```

## Example Service Implementation

Here's an example of how a service would be implemented:

```python
# trading/services/pair_analysis_service.py
from .base_service import BaseService
from src.database.manager import DatabaseManager
from src.market_analysis.cointegration import CointegrationTester
from src.config.db_config import DatabaseConfig
from ..models import Symbol, TradingPair

class PairAnalysisService(BaseService):
    """Service for cointegration analysis operations."""
    
    def __init__(self):
        super().__init__()
        self.db_manager = None
        self.cointegration_tester = None
        
    async def _initialize(self):
        """Initialize required components."""
        if self.db_manager is None:
            config = DatabaseConfig()
            self.db_manager = DatabaseManager(config)
            await self.db_manager.initialize()
            
        if self.cointegration_tester is None:
            self.cointegration_tester = CointegrationTester(self.db_manager)
            
    async def select_pairs(self, symbols, start_date, end_date, **kwargs):
        """Select cointegrated pairs from a list of symbols."""
        await self._initialize()
        return await self.cointegration_tester.select_pairs(
            symbols, start_date, end_date, **kwargs
        )
        
    @BaseService.sync_wrap
    async def select_pairs_sync(self, symbols, start_date, end_date, **kwargs):
        """Synchronous wrapper for select_pairs."""
        return await self.select_pairs(symbols, start_date, end_date, **kwargs)
    
    def create_trading_pairs_from_results(self, results, user=None):
        """Create TradingPair objects from cointegration results."""
        pairs = []
        for result in results:
            # Get or create Symbol objects
            symbol1, _ = Symbol.objects.get_or_create(
                ticker=result['symbol1'],
                defaults={'name': result['symbol1']}
            )
            symbol2, _ = Symbol.objects.get_or_create(
                ticker=result['symbol2'],
                defaults={'name': result['symbol2']}
            )
            
            # Create TradingPair
            pair = TradingPair(
                symbol_1=symbol1,
                symbol_2=symbol2,
                cointegration_pvalue=result['engle_granger_result']['adf_results']['p_value'],
                half_life=self._calculate_half_life(result),
                correlation=result['correlation'],
                hedge_ratio=result['hedge_ratio'],
                stability_score=0.5  # Default value, would be calculated later
            )
            pair.save()
            pairs.append(pair)
        
        return pairs
        
    def _calculate_half_life(self, result):
        """Calculate half-life from cointegration results."""
        # Implementation would go here
        return 10.0  # Default value for example
```

## Using Services in Views

Services would be used in views to handle business logic:

```python
# api/views/pairs.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from trading.models import TradingPair, Symbol
from trading.services.pair_analysis_service import PairAnalysisService
from ..serializers import TradingPairSerializer, PairAnalysisRequestSerializer

class TradingPairViewSet(viewsets.ModelViewSet):
    """API endpoints for trading pairs."""
    
    queryset = TradingPair.objects.filter(is_active=True)
    serializer_class = TradingPairSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['post'])
    def analyze(self, request):
        """Analyze a potential trading pair."""
        serializer = PairAnalysisRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Use the service layer
        service = PairAnalysisService()
        try:
            # Call the synchronous wrapper
            result = service.select_pairs_sync(
                [serializer.validated_data['symbol_1'], 
                 serializer.validated_data['symbol_2']],
                serializer.validated_data['start_date'],
                serializer.validated_data['end_date'],
                min_correlation=serializer.validated_data.get('min_correlation', 0.6),
                significance_level=serializer.validated_data.get('significance_level', 0.05)
            )
            
            # Create TradingPair objects if desired
            if serializer.validated_data.get('save_results', False):
                pairs = service.create_trading_pairs_from_results(result, request.user)
                return Response({
                    'pairs': TradingPairSerializer(pairs, many=True).data,
                    'analysis': result
                })
            
            return Response(result)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
```

## Benefits of the Service Layer

1. **Separation of Concerns**: Keeps business logic separate from views
2. **Synchronous/Asynchronous Bridging**: Handles conversion between sync and async code
3. **Reusability**: Services can be used by multiple views and APIs
4. **Testability**: Services can be easily mocked for testing
5. **Abstraction**: Hides the complexity of the underlying systems
6. **Migration Path**: Allows gradual migration from old to new system

## Service Layer in Celery Tasks

For long-running operations, services would be used in Celery tasks:

```python
# trading/tasks.py
from celery import shared_task
from .services.backtesting_service import BacktestingService

@shared_task
def run_backtest(backtest_id):
    """Run a backtest in the background."""
    service = BacktestingService()
    service.run_backtest_by_id(backtest_id)
```

## Next Steps

The service layer will be implemented in Phase 2 of the project, following these steps:

1. Create the `BaseService` class
2. Implement core services (MarketData, PairAnalysis, etc.)
3. Add data translation utilities
4. Create synchronous wrappers for all async functions
5. Implement Celery tasks for long-running operations
6. Add comprehensive error handling and logging
7. Create unit tests for all services
