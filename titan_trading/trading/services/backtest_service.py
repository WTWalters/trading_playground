"""
Backtesting service for TITAN Trading System.

Provides functionality for backtesting trading strategies
and managing backtest results.
"""
import logging
from datetime import datetime
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any, Tuple

from django.db import transaction

from .base_service import BaseService
from .market_data_service import MarketDataService
from ..models.symbols import Symbol
from ..models.pairs import TradingPair
from ..models.backtesting import BacktestRun, BacktestTrade, BacktestResult

from src.market_analysis.backtest import MeanReversionBacktester, BacktestResult as SrcBacktestResult
from src.config.db_config import DatabaseConfig


class BacktestService(BaseService):
    """
    Service for backtesting operations.
    
    This service provides methods for:
    - Running backtests on pairs
    - Storing backtest results
    - Retrieving and analyzing backtest data
    """
    
    def __init__(self):
        """Initialize the BacktestService."""
        super().__init__()
        self.market_data_service = MarketDataService()
        self.backtester = None
        
    async def _initialize_resources(self) -> None:
        """Initialize required resources."""
        await self.market_data_service._initialize_resources()
        
        if self.backtester is None:
            self.logger.info("Initializing MeanReversionBacktester")
            self.backtester = MeanReversionBacktester(
                self.market_data_service.db_manager
            )
    
    async def _cleanup_resources(self) -> None:
        """Clean up resources."""
        await self.market_data_service._cleanup_resources()
    
    async def backtest_pair(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_holding_period: Optional[int] = None,
        risk_per_trade: float = 2.0,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        timeframe: str = '1d',
        zscore_window: Optional[int] = None,
        source: Optional[str] = None
    ) -> SrcBacktestResult:
        """
        Backtest a mean reversion strategy for a pair.
        
        Args:
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            hedge_ratio: Hedge ratio between symbols
            start_date: Start date for backtest
            end_date: End date for backtest
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
            stop_loss_pct: Stop loss percentage (None for no stop loss)
            take_profit_pct: Take profit percentage (None for no take profit)
            max_holding_period: Maximum holding period in days (None for no limit)
            risk_per_trade: Risk percentage per trade
            initial_capital: Initial capital for the backtest
            commission_pct: Commission percentage per trade
            timeframe: Data timeframe
            zscore_window: Window size for Z-score calculation
            source: Data source (optional)
            
        Returns:
            BacktestResult object with backtest results
        """
        await self._initialize_resources()
        
        self.logger.info(f"Running backtest for {symbol1}/{symbol2} from {start_date} to {end_date}")
        return await self.backtester.backtest_pair(
            symbol1, symbol2, hedge_ratio, start_date, end_date,
            entry_threshold, exit_threshold, stop_loss_pct, take_profit_pct,
            max_holding_period, risk_per_trade, initial_capital,
            commission_pct, timeframe, zscore_window, source
        )
    
    @BaseService.sync_wrap
    async def backtest_pair_sync(
        self,
        symbol1: str,
        symbol2: str,
        hedge_ratio: float,
        start_date: datetime,
        end_date: datetime,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_holding_period: Optional[int] = None,
        risk_per_trade: float = 2.0,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        timeframe: str = '1d',
        zscore_window: Optional[int] = None,
        source: Optional[str] = None
    ) -> SrcBacktestResult:
        """
        Synchronous wrapper for backtest_pair.
        
        Args:
            (Same as backtest_pair)
            
        Returns:
            BacktestResult object with backtest results
        """
        return await self.backtest_pair(
            symbol1, symbol2, hedge_ratio, start_date, end_date,
            entry_threshold, exit_threshold, stop_loss_pct, take_profit_pct,
            max_holding_period, risk_per_trade, initial_capital,
            commission_pct, timeframe, zscore_window, source
        )
    
    def create_backtest_run_from_result(
        self,
        backtest_result: SrcBacktestResult,
        name: str,
        description: Optional[str] = None,
        user=None,
        trading_pair: Optional[TradingPair] = None
    ) -> BacktestRun:
        """
        Create a BacktestRun record from a backtest result.
        
        Args:
            backtest_result: BacktestResult object from backtester
            name: Name for the backtest run
            description: Description of the backtest (optional)
            user: User who ran the backtest (optional)
            trading_pair: Associated TradingPair object (optional)
            
        Returns:
            Created BacktestRun object
        """
        self.logger.info(f"Creating backtest run record: {name}")
        
        # Extract pair info from backtest result
        pair_info = backtest_result.pair_info
        symbol1_ticker = pair_info['symbol1']
        symbol2_ticker = pair_info['symbol2']
        
        # Get or create symbols if not provided via trading_pair
        if trading_pair is None:
            symbol1, _ = Symbol.objects.get_or_create(
                ticker=symbol1_ticker,
                defaults={'name': symbol1_ticker}
            )
            symbol2, _ = Symbol.objects.get_or_create(
                ticker=symbol2_ticker,
                defaults={'name': symbol2_ticker}
            )
            
            # Check if trading pair exists
            try:
                trading_pair = TradingPair.objects.get(
                    symbol_1=symbol1, 
                    symbol_2=symbol2
                )
            except TradingPair.DoesNotExist:
                # Create a new trading pair
                trading_pair = TradingPair(
                    symbol_1=symbol1,
                    symbol_2=symbol2,
                    cointegration_pvalue=0.05,  # Default value
                    half_life=10.0,  # Default value
                    correlation=0.6,  # Default value
                    hedge_ratio=pair_info['hedge_ratio'],
                    is_active=True
                )
                trading_pair.save()
        
        # Create backtest run
        with transaction.atomic():
            # Create the backtest run record
            backtest_run = BacktestRun(
                name=name,
                description=description,
                trading_pair=trading_pair,
                user=user,
                start_date=backtest_result.equity_curve.index[0],
                end_date=backtest_result.equity_curve.index[-1],
                initial_capital=pair_info['initial_capital'],
                parameters=json.dumps({
                    'entry_threshold': pair_info['entry_threshold'],
                    'exit_threshold': pair_info['exit_threshold'],
                    'risk_per_trade': pair_info['risk_per_trade'],
                    'timeframe': pair_info['timeframe']
                }),
                status='completed'
            )
            backtest_run.save()
            
            # Create backtest result record
            metrics = backtest_result.metrics
            result = BacktestResult(
                backtest_run=backtest_run,
                total_return=metrics['total_return'],
                annual_return=metrics['annual_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                num_trades=metrics['num_trades'],
                equity_curve=json.dumps(backtest_result.equity_curve.to_dict())
            )
            result.save()
            
            # Create trade records
            trades = []
            for _, trade_data in backtest_result.trades.iterrows():
                # Skip trades without exit time (open trades)
                if pd.isna(trade_data['exit_time']):
                    continue
                    
                trade = BacktestTrade(
                    backtest_run=backtest_run,
                    entry_time=trade_data['entry_time'],
                    exit_time=trade_data['exit_time'],
                    direction=trade_data['direction'],
                    entry_price=trade_data['entry_price_spread'],
                    exit_price=trade_data['exit_price_spread'],
                    entry_score=trade_data['entry_zscore'],
                    exit_score=trade_data['exit_zscore'],
                    pnl=trade_data['pnl'],
                    pnl_pct=trade_data['pnl_pct'],
                    duration_days=trade_data['days_held'],
                    exit_reason=trade_data['exit_reason']
                )
                trades.append(trade)
            
            # Bulk create trades
            if trades:
                BacktestTrade.objects.bulk_create(trades)
        
        return backtest_run
    
    def get_backtest_metrics(self, backtest_run_id: int) -> Dict[str, Any]:
        """
        Get metrics for a backtest run.
        
        Args:
            backtest_run_id: ID of the backtest run
            
        Returns:
            Dictionary with backtest metrics
        """
        try:
            # Get backtest run and result
            result = BacktestResult.objects.get(backtest_run_id=backtest_run_id)
            
            # Return metrics
            return {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'num_trades': result.num_trades
            }
            
        except BacktestResult.DoesNotExist:
            self.logger.error(f"No result found for backtest run ID {backtest_run_id}")
            return {}
    
    def get_backtest_equity_curve(self, backtest_run_id: int) -> pd.Series:
        """
        Get equity curve for a backtest run.
        
        Args:
            backtest_run_id: ID of the backtest run
            
        Returns:
            Pandas Series with equity curve data
        """
        try:
            # Get backtest result
            result = BacktestResult.objects.get(backtest_run_id=backtest_run_id)
            
            # Parse equity curve from JSON
            equity_data = json.loads(result.equity_curve)
            
            # Convert to Series
            equity_curve = pd.Series(equity_data)
            
            # Convert string index to datetime
            equity_curve.index = pd.to_datetime(equity_curve.index)
            
            return equity_curve
            
        except BacktestResult.DoesNotExist:
            self.logger.error(f"No result found for backtest run ID {backtest_run_id}")
            return pd.Series()
    
    def get_backtest_trades(self, backtest_run_id: int) -> pd.DataFrame:
        """
        Get trades for a backtest run.
        
        Args:
            backtest_run_id: ID of the backtest run
            
        Returns:
            DataFrame with trade data
        """
        # Get trades
        trades = BacktestTrade.objects.filter(backtest_run_id=backtest_run_id)
        
        if not trades:
            return pd.DataFrame()
        
        # Convert to DataFrame
        trade_data = []
        for trade in trades:
            trade_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'entry_zscore': trade.entry_score,
                'exit_zscore': trade.exit_score,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'days_held': trade.duration_days,
                'exit_reason': trade.exit_reason
            })
        
        return pd.DataFrame(trade_data)
