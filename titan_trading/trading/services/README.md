# TITAN Trading System - Service Layer

This directory contains service classes that bridge the Django integration with the existing TITAN Trading System components. These services handle the conversion between Django's synchronous web framework and the existing asynchronous trading components.

## Service Layer Architecture

The service layer follows these design principles:

1. **Separation of Concerns**: Each service focuses on a specific aspect of the trading system (market data, pair analysis, backtesting, etc.)
2. **Async/Sync Bridging**: Services provide both asynchronous methods and synchronous wrappers
3. **Consistent Error Handling**: All services handle errors consistently and provide informative logging
4. **Data Translation**: Services convert between Django models and the data structures used by existing components
5. **Resource Management**: Services properly initialize and clean up resources

## Available Services

### BaseService

The foundation for all services, providing:
- Async/sync bridging through the `run_async` method and `sync_wrap` decorator
- Resource initialization and cleanup
- Consistent error handling and logging

```python
from trading.services.base_service import BaseService

class MyService(BaseService):
    async def async_method(self, param):
        # Async implementation
        pass
        
    @BaseService.sync_wrap
    async def sync_method(self, param):
        # This creates a synchronous wrapper
        return await self.async_method(param)
```

### MarketDataService

Provides access to market data:
- Fetching price data from TimescaleDB
- Storing new market data
- Converting between Django models and pandas DataFrames

```python
from trading.services.market_data_service import MarketDataService

# For synchronous code (e.g., Django views)
service = MarketDataService()
df = service.get_market_data_sync('AAPL', start_date, end_date)

# For asynchronous code
async def example():
    service = MarketDataService()
    df = await service.get_market_data('AAPL', start_date, end_date)
```

### PairAnalysisService

Handles cointegration testing and pair management:
- Finding cointegrated pairs
- Testing specific symbol pairs
- Calculating pair stability
- Creating TradingPair records

```python
from trading.services.pair_analysis_service import PairAnalysisService

service = PairAnalysisService()
pairs = service.select_pairs_sync(['AAPL', 'MSFT', 'GOOG'], start_date, end_date)
```

### BacktestService

Manages backtesting operations:
- Running backtests on pairs
- Storing backtest results
- Analyzing backtest performance

```python
from trading.services.backtest_service import BacktestService

service = BacktestService()
result = service.backtest_pair_sync(
    'AAPL', 'MSFT', hedge_ratio, start_date, end_date,
    entry_threshold=2.0, exit_threshold=0.0
)
```

### RegimeDetectionService

Handles market regime analysis:
- Detecting market regimes
- Identifying regime shifts
- Managing regime records

```python
from trading.services.regime_detection_service import RegimeDetectionService

service = RegimeDetectionService()
regime = service.detect_regime_sync('SPY', start_date, end_date)
```

### SignalGenerationService

Manages trading signal generation:
- Generating trading signals for pairs
- Managing signal records
- Retrieving active signals

```python
from trading.services.signal_generation_service import SignalGenerationService

service = SignalGenerationService()
signals = service.generate_pair_signals_sync(
    'AAPL', 'MSFT', hedge_ratio, start_date, end_date
)
```

### ParameterService

Handles adaptive parameter management:
- Optimizing strategy parameters
- Performing walk-forward testing
- Managing regime-specific parameters

```python
from trading.services.parameter_service import ParameterService

service = ParameterService()
optimal_params = service.optimize_parameters_sync(
    'AAPL', 'MSFT', hedge_ratio, start_date, end_date,
    parameter_ranges={'entry_threshold': [1.5, 2.0, 2.5, 3.0]}
)
```

## Using Services in Views

Services should be used in views to handle business logic:

```python
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response

from trading.services.pair_analysis_service import PairAnalysisService
from trading.services.market_data_service import MarketDataService

class PairAnalysisView(APIView):
    def post(self, request):
        # Extract parameters from request
        symbol1 = request.data.get('symbol1')
        symbol2 = request.data.get('symbol2')
        start_date = request.data.get('start_date')
        end_date = request.data.get('end_date')
        
        # Use the service
        service = PairAnalysisService()
        try:
            result = service.test_pair_sync(
                symbol1, symbol2, start_date, end_date
            )
            return Response(result)
        except Exception as e:
            return Response({'error': str(e)}, status=500)
```

## Using Services in Celery Tasks

For long-running operations, services can be used in Celery tasks:

```python
from celery import shared_task
from trading.services.backtest_service import BacktestService

@shared_task
def run_backtest(symbol1, symbol2, hedge_ratio, start_date, end_date):
    service = BacktestService()
    result = service.backtest_pair_sync(
        symbol1, symbol2, hedge_ratio, start_date, end_date
    )
    return result
```

## Error Handling

All services include comprehensive error handling:

```python
from trading.services.market_data_service import MarketDataService

service = MarketDataService()
try:
    df = service.get_market_data_sync('INVALID', start_date, end_date)
except Exception as e:
    # Handle the error
    print(f"Error: {e}")
```

## Resource Cleanup

For services that manage resources (like database connections), remember to clean up:

```python
service = MarketDataService()
try:
    # Use the service
    df = service.get_market_data_sync('AAPL', start_date, end_date)
finally:
    # Clean up resources
    service.cleanup()
```

## Best Practices

1. **Use Synchronous Methods in Views**: Always use the synchronous wrapper methods in Django views.
2. **Error Handling**: Always wrap service calls in try-except blocks to handle errors gracefully.
3. **Resource Cleanup**: Call the `cleanup` method when finished with services that manage resources.
4. **Transaction Management**: For operations that modify multiple database records, use Django's transaction management.
5. **Service Composition**: Services can use other services, but be careful about circular dependencies.
6. **Lazy Initialization**: Services initialize required resources on demand to minimize overhead.
7. **Consistent Logging**: All services log important operations and errors for debugging.

## Future Enhancements

Planned enhancements for the service layer include:

1. Service factory for dependency injection
2. Better caching for frequently accessed data
3. Enhanced error reporting and monitoring
4. More comprehensive data translation utilities
5. Service metrics for performance monitoring
