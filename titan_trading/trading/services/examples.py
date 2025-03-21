"""
Example usage of TITAN Trading System services.

This module contains examples of how to use the service layer
in Django views, API endpoints, and Celery tasks.
"""

from datetime import datetime, timedelta
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from celery import shared_task

from .market_data_service import MarketDataService
from .pair_analysis_service import PairAnalysisService
from .backtest_service import BacktestService
from .regime_detection_service import RegimeDetectionService
from .signal_generation_service import SignalGenerationService
from .parameter_service import ParameterService

from ..models.pairs import TradingPair


# Example API view using services
class PairAnalysisView(APIView):
    """API view for analyzing potential trading pairs."""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """
        Analyze a potential trading pair.
        
        Request body should contain:
        - symbol1: First symbol
        - symbol2: Second symbol
        - start_date: Start date for analysis
        - end_date: End date for analysis
        - min_correlation: Minimum correlation (optional)
        """
        # Extract parameters from request
        symbol1 = request.data.get('symbol1')
        symbol2 = request.data.get('symbol2')
        start_date = datetime.fromisoformat(request.data.get('start_date'))
        end_date = datetime.fromisoformat(request.data.get('end_date'))
        min_correlation = float(request.data.get('min_correlation', 0.6))
        
        # Validate parameters
        if not all([symbol1, symbol2, start_date, end_date]):
            return Response({
                'error': 'Missing required parameters'
            }, status=400)
        
        # Use the service
        service = PairAnalysisService()
        try:
            # Test pair cointegration
            result = service.test_pair_sync(
                symbol1, symbol2, start_date, end_date
            )
            
            # Create TradingPair if requested
            if request.data.get('create_pair', False) and result['is_cointegrated']:
                pair = service.create_trading_pair(result, request.user)
                
                # Return result with pair ID
                return Response({
                    'pair_id': pair.id,
                    'analysis': result
                })
            
            return Response(result)
            
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=500)
        finally:
            # Clean up resources
            service.cleanup()


# Example function-based view using services
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_market_data(request):
    """
    Get market data for a symbol.
    
    Query parameters:
    - symbol: Symbol identifier
    - start_date: Start date (ISO format)
    - end_date: End date (ISO format)
    - timeframe: Data timeframe (default: '1d')
    """
    # Extract parameters from request
    symbol = request.query_params.get('symbol')
    start_date = request.query_params.get('start_date')
    end_date = request.query_params.get('end_date')
    timeframe = request.query_params.get('timeframe', '1d')
    
    # Validate parameters
    if not all([symbol, start_date, end_date]):
        return Response({
            'error': 'Missing required parameters'
        }, status=400)
    
    # Parse dates
    try:
        start_date = datetime.fromisoformat(start_date)
        end_date = datetime.fromisoformat(end_date)
    except ValueError:
        return Response({
            'error': 'Invalid date format'
        }, status=400)
    
    # Use the service
    service = MarketDataService()
    try:
        df = service.get_market_data_sync(
            symbol, start_date, end_date, timeframe
        )
        
        # Convert DataFrame to dict
        data = []
        for idx, row in df.iterrows():
            data.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })
            
        return Response(data)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=500)
    finally:
        # Clean up resources
        service.cleanup()


# Example Celery task using services
@shared_task
def run_backtest(pair_id, name, description=None, user_id=None):
    """
    Run a backtest for a trading pair.
    
    Args:
        pair_id: ID of the TradingPair
        name: Name for the backtest
        description: Description of the backtest (optional)
        user_id: ID of the user running the backtest (optional)
    
    Returns:
        Dictionary with backtest results
    """
    # Get TradingPair
    try:
        pair = TradingPair.objects.get(id=pair_id)
    except TradingPair.DoesNotExist:
        return {
            'error': f'Trading pair with ID {pair_id} not found'
        }
    
    # Set up time range (last year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Get parameters
    symbol1 = pair.symbol_1.ticker
    symbol2 = pair.symbol_2.ticker
    hedge_ratio = pair.hedge_ratio
    
    # Use the service
    backtest_service = BacktestService()
    parameter_service = ParameterService()
    
    try:
        # Get optimal parameters
        params = parameter_service.get_optimal_parameters(pair)
        
        # Run backtest
        result = backtest_service.backtest_pair_sync(
            symbol1, symbol2, hedge_ratio, start_date, end_date,
            entry_threshold=params.get('entry_threshold', 2.0),
            exit_threshold=params.get('exit_threshold', 0.0),
            risk_per_trade=params.get('risk_per_trade', 2.0)
        )
        
        # Create backtest record
        backtest_run = backtest_service.create_backtest_run_from_result(
            result, name, description, user_id, pair
        )
        
        return {
            'backtest_id': backtest_run.id,
            'total_return': result.metrics['total_return'],
            'sharpe_ratio': result.metrics['sharpe_ratio'],
            'num_trades': result.metrics['num_trades']
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }
    finally:
        # Clean up resources
        backtest_service.cleanup()
        parameter_service.cleanup()


# Example synchronous function using services
def generate_current_signals():
    """
    Generate signals for all active pairs.
    
    Returns:
        List of current trading signals
    """
    service = SignalGenerationService()
    try:
        signals = service.generate_current_signals()
        return signals
    finally:
        service.cleanup()
