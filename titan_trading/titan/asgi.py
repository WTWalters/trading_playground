"""
ASGI config for TITAN Trading System.

This module exposes the ASGI callable as a module-level variable named ``application``.
"""
import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'titan.settings')

# Initialize Django ASGI application
django_asgi_app = get_asgi_application()

# Import routing after Django setup to avoid import errors
from channels_app.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    # Django's ASGI application for HTTP requests
    'http': django_asgi_app,
    
    # WebSocket handling with authentication support
    'websocket': AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})
