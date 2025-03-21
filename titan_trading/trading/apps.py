"""
App configuration for TITAN Trading System trading app.
"""
from django.apps import AppConfig


class TradingConfig(AppConfig):
    """
    Configuration for the trading app.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trading'
    
    def ready(self):
        """
        Initialize app when Django starts.
        
        This method is called when Django starts and is a good place to
        set up signal handlers or initialize services.
        """
        pass  # Import signal handlers if needed
