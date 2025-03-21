"""
Signal generation service for TITAN Trading System.

Provides functionality for generating and managing trading signals.
"""
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from django.db import transaction

from .base_service import BaseService
from .market_data_service import MarketDataService
from ..models.symbols import Symbol
from ..models.pairs import TradingPair, PairSpread
from ..models.signals import Signal

from src.market_analysis.mean_reversion import calculate_zscore, generate_mean_reversion_signals


class SignalGenerationService(BaseService):
    """
    Service for signal generation operations.
    
    This service provides methods for:
    - Generating trading signals
    - Managing signal data
    - Retrieving active signals
    """
    
    def __init__(self):
        """Initialize the SignalGenerationService."""
        super().__init__()
        self.market_data_service = MarketDataService()
        
    async def _initialize_resources(self) -> None:
        """Initialize required resources."""
        await self.market_data_service._initialize_resources()
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources."""
        await self.market_data_service._cleanup_resources()
    
    async def generate_pair_signals(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        timeframe: str = '1d',
        zscore_window: int = 20,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signals for a pair.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for signal generation
            end_date: End date for signal generation
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit signals
            timeframe: Data timeframe
            zscore_window: Window size for Z-score calculation
            source: Data source (optional)
            
        Returns:
            Dictionary with signal generation results
        """
        await self._initialize_resources()
        
        self.logger.info(f"Generating signals for {symbol1}/{symbol2} from {start_date} to {end_date}")
        
        # Fetch data for both symbols
        df1 = await self.market_data_service.get_market_data(
            symbol1, start_date, end_date, timeframe, source
        )
        df2 = await self.market_data_service.get_market_data(
            symbol2, start_date, end_date, timeframe, source
        )
        
        # Check if data is available
        if df1.empty or df2.empty:
            self.logger.warning(f"Insufficient data for {symbol1}/{symbol2}")
            return {
                'success': False,
                'error': 'Insufficient data'
            }
        
        # Align both series on the same dates
        joined = pd.DataFrame({
            symbol1: df1['close'],
            symbol2: df2['close']
        }).dropna()
        
        # Calculate spread
        spread = joined[symbol2] - hedge_ratio * joined[symbol1]
        
        # Calculate Z-score
        zscore = calculate_zscore(spread, window=zscore_window)
        
        # Generate signals
        signals = generate_mean_reversion_signals(
            zscore, entry_threshold, exit_threshold
        )
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'time': joined.index,
            'spread': spread.values,
            'zscore': zscore.values,
            'signal': signals.values
        })
        result_df.set_index('time', inplace=True)
        
        # Extract just the signal points
        signal_points = result_df[result_df['signal'] != 0].copy()
        
        # Convert to list of dictionaries
        signal_list = []
        for idx, row in signal_points.iterrows():
            signal_list.append({
                'timestamp': idx,
                'spread': row['spread'],
                'zscore': row['zscore'],
                'signal': int(row['signal'])  # 1 for long, -1 for short
            })
        
        return {
            'success': True,
            'symbol1': symbol1,
            'symbol2': symbol2,
            'hedge_ratio': hedge_ratio,
            'signals': signal_list,
            'summary': {
                'total_signals': len(signal_list),
                'long_signals': sum(1 for s in signal_list if s['signal'] > 0),
                'short_signals': sum(1 for s in signal_list if s['signal'] < 0),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe
            }
        }
    
    @BaseService.sync_wrap
    async def generate_pair_signals_sync(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        timeframe: str = '1d',
        zscore_window: int = 20,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_pair_signals.
        
        Args:
            (Same as generate_pair_signals)
            
        Returns:
            Dictionary with signal generation results
        """
        return await self.generate_pair_signals(
            symbol1, symbol2, hedge_ratio, start_date, end_date,
            entry_threshold, exit_threshold, timeframe, zscore_window, source
        )
    
    def create_signals_from_results(
        self,
        signals_result: Dict[str, Any],
        trading_pair: Optional[TradingPair] = None
    ) -> List[Signal]:
        """
        Create Signal records from signal generation results.
        
        Args:
            signals_result: Signal generation results dictionary
            trading_pair: Associated TradingPair object (optional)
            
        Returns:
            List of created Signal objects
        """
        self.logger.info("Creating signal records from results")
        
        # Extract data
        symbol1 = signals_result['symbol1']
        symbol2 = signals_result['symbol2']
        hedge_ratio = signals_result['hedge_ratio']
        signal_list = signals_result['signals']
        
        # Get or create symbols and trading pair if not provided
        if trading_pair is None:
            # Get or create symbols
            symbol1_obj, _ = Symbol.objects.get_or_create(
                ticker=symbol1,
                defaults={'name': symbol1}
            )
            symbol2_obj, _ = Symbol.objects.get_or_create(
                ticker=symbol2,
                defaults={'name': symbol2}
            )
            
            # Check if trading pair exists
            try:
                trading_pair = TradingPair.objects.get(
                    symbol_1=symbol1_obj,
                    symbol_2=symbol2_obj
                )
            except TradingPair.DoesNotExist:
                # Create new trading pair
                trading_pair = TradingPair(
                    symbol_1=symbol1_obj,
                    symbol_2=symbol2_obj,
                    hedge_ratio=hedge_ratio,
                    cointegration_pvalue=0.05,  # Default value
                    half_life=10.0,  # Default value
                    correlation=0.6,  # Default value
                    is_active=True
                )
                trading_pair.save()
        
        # Create signal records
        signals = []
        with transaction.atomic():
            for signal_data in signal_list:
                signal = Signal(
                    trading_pair=trading_pair,
                    timestamp=signal_data['timestamp'],
                    signal_type='ENTRY' if abs(signal_data['signal']) == 1 else 'EXIT',
                    direction='LONG' if signal_data['signal'] == 1 else 'SHORT',
                    confidence=0.8,  # Default value
                    zscore=signal_data['zscore'],
                    spread_value=signal_data['spread'],
                    status='PENDING'
                )
                signal.save()
                signals.append(signal)
        
        self.logger.info(f"Created {len(signals)} signal records")
        return signals
    
    def get_active_signals(self, limit: int = 10) -> List[Signal]:
        """
        Get recent active trading signals.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of Signal objects
        """
        return list(Signal.objects.filter(
            status='PENDING'
        ).order_by('-timestamp')[:limit])
    
    def get_signals_for_pair(
        self,
        trading_pair: TradingPair,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Signal]:
        """
        Get signals for a specific trading pair.
        
        Args:
            trading_pair: TradingPair object
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            limit: Maximum number of signals to return
            
        Returns:
            List of Signal objects
        """
        # Create query
        query = Signal.objects.filter(trading_pair=trading_pair)
        
        # Apply date filters if provided
        if start_date:
            query = query.filter(timestamp__gte=start_date)
        if end_date:
            query = query.filter(timestamp__lte=end_date)
            
        # Return ordered list with limit
        return list(query.order_by('-timestamp')[:limit])
    
    def update_signal_status(self, signal_id: int, new_status: str) -> bool:
        """
        Update the status of a signal.
        
        Args:
            signal_id: ID of the signal to update
            new_status: New status value ('PENDING', 'EXECUTED', 'CANCELLED')
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            signal = Signal.objects.get(id=signal_id)
            signal.status = new_status
            signal.updated_at = datetime.now()
            signal.save()
            return True
        except Signal.DoesNotExist:
            self.logger.error(f"Signal with ID {signal_id} not found")
            return False
    
    def generate_current_signals(self) -> List[Dict[str, Any]]:
        """
        Generate signals based on current market conditions.
        
        This method checks all active trading pairs and generates
        signals based on the latest market data.
        
        Returns:
            List of signal dictionaries
        """
        self.logger.info("Generating current signals for all active pairs")
        
        # Get all active trading pairs
        active_pairs = TradingPair.objects.filter(is_active=True)
        
        current_signals = []
        for pair in active_pairs:
            # Get parameters for the pair
            symbol1 = pair.symbol_1.ticker
            symbol2 = pair.symbol_2.ticker
            hedge_ratio = pair.hedge_ratio
            
            # Set time range (last 60 days to now)
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=60)
            
            # Generate signals
            try:
                result = self.generate_pair_signals_sync(
                    symbol1, symbol2, hedge_ratio, start_date, end_date,
                    entry_threshold=2.0,  # Default value
                    exit_threshold=0.0,   # Default value
                    timeframe='1d',
                    zscore_window=20
                )
                
                if result['success'] and result['signals']:
                    # Extract only the most recent signal
                    latest_signal = max(result['signals'], key=lambda x: x['timestamp'])
                    
                    # Add pair information
                    latest_signal['pair_id'] = pair.id
                    latest_signal['symbol1'] = symbol1
                    latest_signal['symbol2'] = symbol2
                    
                    current_signals.append(latest_signal)
                    
                    # Update pair's last zscore
                    pair.last_zscore = latest_signal['zscore']
                    pair.last_spread_value = latest_signal['spread']
                    pair.last_updated_spread = latest_signal['timestamp']
                    pair.save()
                    
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol1}/{symbol2}: {e}")
        
        return current_signals
