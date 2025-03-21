"""
URL configuration for TITAN Trading API.

This module defines the URL patterns for the API app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

# Create a router for API viewsets
router = DefaultRouter()

# Add API routes here as they are implemented
# Example: router.register(r'symbols', SymbolViewSet)

urlpatterns = [
    # Include router URLs
    path('', include(router.urls)),
    
    # Add other API endpoints here
]
