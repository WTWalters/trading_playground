"""
Admin configuration for trading app.
"""
from django.contrib import admin
from .models import (
    Symbol, Price, TradingPair, PairSpread, Signal, 
    MarketRegime, RegimeTransition, BacktestRun, 
    BacktestResult, BacktestTrade, WalkForwardTest, WalkForwardWindow
)


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    """Admin configuration for Symbol model."""
    list_display = ('ticker', 'name', 'sector', 'exchange', 'asset_type', 'is_active')
    list_filter = ('is_active', 'asset_type', 'exchange', 'sector')
    search_fields = ('ticker', 'name')
    ordering = ('ticker',)


@admin.register(Price)
class PriceAdmin(admin.ModelAdmin):
    """Admin configuration for Price model."""
    list_display = ('symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'source', 'timeframe')
    list_filter = ('source', 'timeframe')
    search_fields = ('symbol__ticker',)
    date_hierarchy = 'timestamp'
    raw_id_fields = ('symbol',)


@admin.register(TradingPair)
class TradingPairAdmin(admin.ModelAdmin):
    """Admin configuration for TradingPair model."""
    list_display = (
        'id', 'symbol_1', 'symbol_2', 'cointegration_pvalue', 
        'half_life', 'correlation', 'is_active', 'stability_score'
    )
    list_filter = ('is_active',)
    search_fields = ('symbol_1__ticker', 'symbol_2__ticker')
    raw_id_fields = ('symbol_1', 'symbol_2')


@admin.register(PairSpread)
class PairSpreadAdmin(admin.ModelAdmin):
    """Admin configuration for PairSpread model."""
    list_display = ('pair', 'timestamp', 'spread_value', 'z_score', 'mean', 'std_dev')
    list_filter = ('lookback_window',)
    search_fields = ('pair__symbol_1__ticker', 'pair__symbol_2__ticker')
    date_hierarchy = 'timestamp'
    raw_id_fields = ('pair',)


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    """Admin configuration for Signal model."""
    list_display = ('pair', 'signal_type', 'timestamp', 'z_score', 'processed', 'confidence_score')
    list_filter = ('signal_type', 'processed')
    search_fields = ('pair__symbol_1__ticker', 'pair__symbol_2__ticker')
    date_hierarchy = 'timestamp'
    raw_id_fields = ('pair', 'regime')


@admin.register(MarketRegime)
class MarketRegimeAdmin(admin.ModelAdmin):
    """Admin configuration for MarketRegime model."""
    list_display = ('regime_type', 'start_date', 'end_date', 'vix_average', 'volatility_score')
    list_filter = ('regime_type',)
    search_fields = ('description',)
    date_hierarchy = 'start_date'


@admin.register(RegimeTransition)
class RegimeTransitionAdmin(admin.ModelAdmin):
    """Admin configuration for RegimeTransition model."""
    list_display = ('from_regime', 'to_regime', 'transition_date', 'transition_score')
    list_filter = ('from_regime__regime_type', 'to_regime__regime_type')
    date_hierarchy = 'transition_date'
    raw_id_fields = ('from_regime', 'to_regime')


class BacktestResultInline(admin.StackedInline):
    """Inline admin for BacktestResult model."""
    model = BacktestResult
    can_delete = False
    verbose_name_plural = 'Backtest Results'


@admin.register(BacktestRun)
class BacktestRunAdmin(admin.ModelAdmin):
    """Admin configuration for BacktestRun model."""
    list_display = ('name', 'user', 'start_date', 'end_date', 'status', 'created_at')
    list_filter = ('status', 'regime_aware')
    search_fields = ('name', 'description')
    date_hierarchy = 'created_at'
    raw_id_fields = ('user',)
    filter_horizontal = ('pairs', 'regimes')
    inlines = [BacktestResultInline]


@admin.register(BacktestTrade)
class BacktestTradeAdmin(admin.ModelAdmin):
    """Admin configuration for BacktestTrade model."""
    list_display = (
        'backtest', 'pair', 'trade_type', 'entry_date', 
        'exit_date', 'pnl_percent', 'exit_reason'
    )
    list_filter = ('trade_type', 'exit_reason')
    search_fields = ('pair__symbol_1__ticker', 'pair__symbol_2__ticker', 'notes')
    date_hierarchy = 'entry_date'
    raw_id_fields = ('backtest', 'pair', 'entry_signal', 'exit_signal', 'regime')


class WalkForwardWindowInline(admin.TabularInline):
    """Inline admin for WalkForwardWindow model."""
    model = WalkForwardWindow
    extra = 0
    can_delete = False
    verbose_name_plural = 'Windows'
    fields = ('in_sample_start', 'in_sample_end', 'out_of_sample_start', 'out_of_sample_end')
    readonly_fields = ('in_sample_start', 'in_sample_end', 'out_of_sample_start', 'out_of_sample_end')


@admin.register(WalkForwardTest)
class WalkForwardTestAdmin(admin.ModelAdmin):
    """Admin configuration for WalkForwardTest model."""
    list_display = (
        'name', 'user', 'start_date', 'end_date', 
        'in_sample_size', 'out_of_sample_size', 'status'
    )
    list_filter = ('status',)
    search_fields = ('name',)
    date_hierarchy = 'created_at'
    raw_id_fields = ('user',)
    filter_horizontal = ('pairs',)
    inlines = [WalkForwardWindowInline]
