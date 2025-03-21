"""
App configuration for TITAN Trading System channels app.
"""
from django.apps import AppConfig


class ChannelsAppConfig(AppConfig):
    """
    Configuration for the channels app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'channels_app'
