"""
View functions for users app.

This module contains view functions for user management.
"""
from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin


class ProfileView(LoginRequiredMixin, View):
    """User profile view."""
    
    def get(self, request):
        """Handle GET requests for user profile."""
        return render(request, 'users/profile.html')