# Django Integration: Service Layer

**Type**: Technical Documentation  
**Last Updated**: 2025-03-19  
**Status**: Implemented (Untested)

## Related Documents

- [Implementation Plan](./implementation_plan.md)
- [Project Structure](./project_structure.md)
- [Database Integration](./database_integration.md)
- [API Layer](./api_layer.md)

## Overview

The service layer is a crucial component of the Django integration, bridging Django's synchronous web framework with the existing asynchronous TITAN Trading System components. This document provides a detailed overview of the service layer architecture, the available services, and how to use them.

## Architecture

The service layer follows a consistent design pattern:

1. **BaseService**: A foundation class that all services inherit from
2. **Async/Sync Bridging**: Utilities for converting between asynchronous and synchronous code
3. **Resource Management**: Consistent patterns for initializing and cleaning up resources
4. **Error Handling**: Comprehensive error handling and logging

### Diagram

```
┌─────────────────┐      ┌───────────────────────┐      ┌───────────────────┐
│                 │      │                       │      │                    │
│  Django Views   │──────▶      Service Layer    │──────▶  TITAN Components  │
│  API Endpoints  │      │                       │      │                    │
│                 │      │  ┌─────────────────┐  │      │  ┌──────────────┐  │
└─────────────────┘      │  │  BaseService    │  │      │  │ DatabaseMgr  │  │
                         │  └─────────────────┘  │      │  └──────────────┘  │
                         │          ▲            │      │         ▲          │
                         │          │            │      │         │          │
                         │  ┌───────┴───────┐    │      │  ┌──────┴───────┐  │
                         │  │ MarketDataSvc │────┼──────┼─▶│CointegTester │  │
                         │  └───────────────┘    │      │  └──────────────┘  │
                         │  ┌───────────────┐    │      │  ┌──────────────┐  │
                         │  │PairAnalysisSvc│────┼──────┼─▶│BacktestEngine│  │
                         │  └───────────────┘    │      │  └──────────────┘  │
                         │  ┌───────────────┐    │      │  ┌──────────────┐  │
                         │  │BacktestService│────┼──────┼─▶│RegimeDetector│  │
                         │  └───────────────┘    │      │  └──────────────┘  │
                         │                       │      │                    │
                         └───────────────────────┘      └───────────────────┘
                                    │                             │
                                    │                             │
                                    │                             │
                                    ▼                             ▼
                         ┌───────────────────────┐      ┌───────────────────┐
                         │                       │      │                    │
                         │     Django Models     │◀─────▶      TimescaleDB   │
                         │                       │      │                    │
                         └───────────────────────┘      └───────────────────┘
```

## Available Services

The service layer includes the following components:

### BaseService

The foundation for all services, providing:
- Async/sync bridging through the `run_async` method and `sync_wrap` decorator
- Resource initialization and cleanup
- Error handling and logging

### MarketDataService

Provides access to market data:
- Fetching price data from TimescaleDB
- Storing new market data
- Converting between Django models and pandas DataFrames

### PairAnalysisService

Handles cointegration testing and pair management:
- Finding cointegrated pairs
- Testing specific symbol pairs
- Calculating pair stability
- Creating TradingPair records

### BacktestService

Manages backtesting operations:
- Running backtests on pairs
- Storing backtest results
- Analyzing backtest performance

### RegimeDetectionService

Handles market regime analysis:
- Detecting market regimes
- Identifying regime shifts
- Managing regime records

### SignalGenerationService

Manages trading signal generation:
- Generating trading signals for pairs
- Managing signal records
- Retrieving active signals

### ParameterService

Handles adaptive parameter management:
- Optimizing strategy parameters
- Performing walk-forward testing
- Managing regime-specific parameters

## Implementation Details

### Async/Sync Bridging

The service layer implements a consistent pattern for bridging asynchronous and synchronous code:

```python
import asyncio
import functools
from typing import TypeVar, Callable, Any

T = TypeVar('T')
AsyncFunc = TypeVar('AsyncFunc', bound=Callable[..., Any])

class BaseService:
    def run_async(self, async_func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run an asynchronous function in a synchronous context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
            
    @staticmethod
    def sync_wrap(async_func: AsyncFunc) -> AsyncFunc:
        """Decorator to create a synchronous wrapper for an async function."""
        @functools.wraps(async_func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            return self.run_async(async_func, self, *args, **kwargs)
        return wrapper
```

### Resource Management

Services properly manage their resources:

```python
class MarketDataService(BaseService):
    def __init__(self):
        super().__init__()
        self.db_manager = None
        
    async def _initialize_resources(self) -> None:
        """Initialize resources if not already initialized."""
        if self.db_manager is None:
            self.db_manager = DatabaseManager(config)
            await self.db_manager.initialize()
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources."""
        if self.db_manager is not None:
            await self.db_manager.close()
            self.db_manager = None
            
    def cleanup(self) -> None:
        """Synchronous method to clean up resources."""
        self.run_async(self._cleanup_resources)
```

### Error Handling

Services implement consistent error handling:

```python
try:
    result = await async_operation()
    return result
except Exception as e:
    self.logger.error(f"Error in operation: {e}")
    self.logger.error(traceback.format_exc())
    raise
```

## Usage

### In Django Views

Services should be used in views to handle business logic:

```python
from trading.services import PairAnalysisService

def analyze_pair(request):
    service = PairAnalysisService()
    try:
        result = service.test_pair_sync(
            symbol1, symbol2, start_date, end_date
        )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        service.cleanup()
```

### In Celery Tasks

For long-running operations, services can be used in Celery tasks:

```python
from celery import shared_task
from trading.services import BacktestService

@shared_task
def run_backtest(pair_id, name, description=None, user_id=None):
    service = BacktestService()
    try:
        # Get pair details and run backtest
        result = service.backtest_pair_sync(...)
        return result
    finally:
        service.cleanup()
```

## Implementation Status

The service layer has been implemented as specified in the architecture documentation, but it has not yet been thoroughly tested. All services are in place with proper async/sync bridging, error handling, and resource management.

### Completed

- BaseService implementation
- Core service implementations (MarketDataService, PairAnalysisService, etc.)
- Async/sync bridging
- Resource management
- Error handling
- Documentation

### Not Yet Tested

- Integration with existing components
- Error handling under various conditions
- Resource cleanup in error cases
- Performance under load
- Celery task integration

## Recommendations for Testing

When testing the service layer, focus on:

1. **Async/Sync Bridging**: Ensure that asynchronous operations work correctly when called from synchronous code.
2. **Resource Management**: Verify that resources are properly initialized and cleaned up, even in error cases.
3. **Error Handling**: Test error conditions to ensure proper handling and reporting.
4. **Data Translation**: Validate that data is correctly translated between Django models and existing data structures.
5. **Performance**: Test performance under load, especially for resource-intensive operations.

## Next Steps

1. Create comprehensive tests for all service components
2. Implement Celery integration for long-running tasks
3. Begin API layer implementation (Phase 3)
4. Update documentation with test results and any required adjustments
