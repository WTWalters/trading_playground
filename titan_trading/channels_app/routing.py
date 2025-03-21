"""
WebSocket URL routing for TITAN Trading System.

This module defines WebSocket URL patterns for the channels app.
"""
from django.urls import path

# Will import consumers once they are implemented
# from .consumers.signals import SignalConsumer
# from .consumers.trading_data import TradingDataConsumer

websocket_urlpatterns = [
    # Will add paths once consumers are implemented
    # path('ws/signals/', SignalConsumer.as_asgi()),
    # path('ws/trading-data/', TradingDataConsumer.as_asgi()),
]
