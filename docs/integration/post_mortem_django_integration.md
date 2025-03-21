# Trade Post-Mortem Analyzer Django Integration

## Overview

This document outlines the integration of the Trade Post-Mortem Analyzer with the Django web framework to expose its functionality through a web interface and API endpoints.

## Integration Architecture

### 1. Django App Structure

Create a dedicated Django app for post-mortem analysis:

```
titan_trading/
├── trading/
│   ├── ...
├── post_mortem/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── serializers.py
│   ├── services.py
│   ├── urls.py
│   ├── views.py
│   └── templates/
│       └── post_mortem/
│           ├── analysis_detail.html
│           ├── pattern_list.html
│           └── ...
```

### 2. Database Models

Define Django models that mirror the database schema used by the analyzer:

```python
# models.py
from django.db import models
from trading.models import Trade

class TradeAnalysis(models.Model):
    trade = models.OneToOneField(Trade, on_delete=models.CASCADE, related_name='analysis')
    analysis_data = models.JSONField()
    success_score = models.IntegerField()
    primary_factors = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
class AnalysisPattern(models.Model):
    pattern_type = models.CharField(max_length=100)
    description = models.TextField()
    confidence = models.FloatField()
    affected_trades = models.ManyToManyField(Trade)
    category = models.CharField(max_length=50)
    action_items = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
class ImprovementPlan(models.Model):
    focus_areas = models.JSONField()
    improvements = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
class RecommendationTracking(models.Model):
    trade_analysis = models.ForeignKey(TradeAnalysis, on_delete=models.CASCADE)
    recommendation = models.TextField()
    area = models.CharField(max_length=100)
    implementation_status = models.CharField(
        max_length=20,
        choices=[
            ('PENDING', 'Pending'),
            ('IMPLEMENTED', 'Implemented'),
            ('IGNORED', 'Ignored'),
        ],
        default='PENDING'
    )
    implementation_date = models.DateTimeField(null=True, blank=True)
    impact_score = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)
```

### 3. Service Layer

Create a service layer to interface with the core analyzer:

```python
# services.py
from src.llm_integration.trade_analysis.post_mortem_analyzer import TradePostMortemAnalyzer
from src.database.manager import DatabaseManager
from src.llm_integration.config import LLMConfig
from src.llm_integration.clients.factory import LLMClientFactory
from src.llm_integration.trade_analysis.context_collector import TradeContextCollector

class PostMortemService:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.config = LLMConfig()
        self.llm_client = LLMClientFactory(self.config).get_client()
        self.context_collector = TradeContextCollector(self.db_manager)
        self.analyzer = TradePostMortemAnalyzer(
            db_manager=self.db_manager,
            context_collector=self.context_collector,
            llm_client=self.llm_client,
            config=self.config
        )
    
    async def analyze_trade(self, trade_id):
        """Analyze a single trade and store results in Django database"""
        analysis = await self.analyzer.analyze_trade(trade_id)
        # Convert to Django model and save
        return analysis
        
    async def analyze_trade_batch(self, trade_ids, concurrency=3):
        """Analyze multiple trades"""
        results = await self.analyzer.analyze_trade_batch(trade_ids, concurrency=concurrency)
        return results
        
    async def identify_patterns(self, time_period=None, min_trades=10):
        """Identify patterns across trades"""
        patterns = await self.analyzer.identify_patterns(time_period, min_trades)
        return patterns
        
    async def create_improvement_plan(self, patterns):
        """Generate improvement plan"""
        plan = await self.analyzer.create_improvement_plan(patterns)
        return plan
        
    async def track_recommendation_impact(self, recommendation_id, implementation_status):
        """Track implementation status of recommendations"""
        # Implement the tracking system
        pass
```

### 4. API Endpoints

Define REST API endpoints for accessing post-mortem functionality:

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('analyses/', views.TradeAnalysisList.as_view(), name='analysis-list'),
    path('analyses/<str:trade_id>/', views.TradeAnalysisDetail.as_view(), name='analysis-detail'),
    path('analyses/batch/', views.TradeAnalysisBatch.as_view(), name='analysis-batch'),
    path('patterns/', views.PatternList.as_view(), name='pattern-list'),
    path('patterns/<int:pk>/', views.PatternDetail.as_view(), name='pattern-detail'),
    path('improvement-plans/', views.ImprovementPlanList.as_view(), name='improvement-plan-list'),
    path('recommendations/track/', views.TrackRecommendation.as_view(), name='track-recommendation'),
]
```

### 5. Serializers

Create serializers for the API responses:

```python
# serializers.py
from rest_framework import serializers
from .models import TradeAnalysis, AnalysisPattern, ImprovementPlan, RecommendationTracking

class TradeAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = TradeAnalysis
        fields = '__all__'
        
class PatternSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisPattern
        fields = '__all__'
        
class ImprovementPlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImprovementPlan
        fields = '__all__'
        
class RecommendationTrackingSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecommendationTracking
        fields = '__all__'
```

### 6. Views

Implement views for API endpoints and web pages:

```python
# views.py
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import render
from .models import TradeAnalysis, AnalysisPattern, ImprovementPlan
from .serializers import TradeAnalysisSerializer, PatternSerializer, ImprovementPlanSerializer
from .services import PostMortemService
import asyncio

class TradeAnalysisList(generics.ListAPIView):
    queryset = TradeAnalysis.objects.all()
    serializer_class = TradeAnalysisSerializer
    
class TradeAnalysisDetail(APIView):
    def get(self, request, trade_id):
        try:
            analysis = TradeAnalysis.objects.get(trade_id=trade_id)
            serializer = TradeAnalysisSerializer(analysis)
            return Response(serializer.data)
        except TradeAnalysis.DoesNotExist:
            return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)
            
    def post(self, request, trade_id):
        # Create new analysis
        service = PostMortemService()
        loop = asyncio.new_event_loop()
        analysis = loop.run_until_complete(service.analyze_trade(trade_id))
        loop.close()
        
        # Format and return response
        return Response(analysis, status=status.HTTP_201_CREATED)

# Add additional views for other endpoints
```

### 7. Templates and Frontend

Create web templates for viewing analyses and patterns:

```html
<!-- analysis_detail.html -->
{% extends "base.html" %}

{% block content %}
<div class="container">
    <h1>Trade Analysis: {{ analysis.trade.id }}</h1>
    
    <div class="card mb-3">
        <div class="card-header">
            <h3>Performance Analysis</h3>
        </div>
        <div class="card-body">
            <p>{{ analysis.analysis_data.performance_analysis.overall_evaluation }}</p>
            <!-- Additional details -->
        </div>
    </div>
    
    <!-- Other analysis sections -->
    
    <div class="card mb-3">
        <div class="card-header">
            <h3>Improvement Opportunities</h3>
        </div>
        <div class="card-body">
            <ul class="list-group">
                {% for opportunity in analysis.analysis_data.improvement_opportunities %}
                <li class="list-group-item">
                    <h5>{{ opportunity.area }}</h5>
                    <p>{{ opportunity.suggestion }}</p>
                    <p><strong>Expected Impact:</strong> {{ opportunity.expected_impact }}</p>
                    
                    <!-- Recommendation tracking UI -->
                    <div class="form-group">
                        <label>Implementation Status:</label>
                        <select class="form-control recommendation-status" 
                                data-recommendation-id="{{ opportunity.id }}">
                            <option value="PENDING" {% if opportunity.status == 'PENDING' %}selected{% endif %}>Pending</option>
                            <option value="IMPLEMENTED" {% if opportunity.status == 'IMPLEMENTED' %}selected{% endif %}>Implemented</option>
                            <option value="IGNORED" {% if opportunity.status == 'IGNORED' %}selected{% endif %}>Ignored</option>
                        </select>
                    </div>
                </li>
                {% endfor %}
            </ul>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
    // JavaScript for handling recommendation status updates
    $('.recommendation-status').change(function() {
        const recommendationId = $(this).data('recommendation-id');
        const status = $(this).val();
        
        $.ajax({
            url: '/api/post-mortem/recommendations/track/',
            method: 'POST',
            data: {
                recommendation_id: recommendationId,
                status: status
            },
            success: function(response) {
                toastr.success('Status updated successfully');
            },
            error: function(error) {
                toastr.error('Failed to update status');
                console.error(error);
            }
        });
    });
</script>
{% endblock %}
```

## Implementation Plan

### Phase 1: Database Models and Migrations

1. Create the Django app structure
2. Implement database models
3. Create and run migrations
4. Add model admin interfaces

### Phase 2: Service Layer Integration

1. Implement the service layer
2. Create connection and initialization logic
3. Implement asynchronous handling
4. Add error handling and logging

### Phase 3: API Endpoints

1. Create serializers
2. Implement API views
3. Define URL routes
4. Add authentication and permissions

### Phase 4: Frontend Integration

1. Create templates for analysis display
2. Implement recommendation tracking UI
3. Build pattern visualization components
4. Create dashboards for improvement plans

### Phase 5: Testing and Deployment

1. Create unit tests for models and services
2. Implement integration tests
3. Add documentation
4. Deploy to staging and production environments

## Feedback Amplification Loop Integration

To implement the Feedback Amplification Loop, the Django integration will include:

1. **Recommendation Tracking UI**:
   - Status toggle for each recommendation (Pending/Implemented/Ignored)
   - Implementation date tracking
   - Notes field for trader comments

2. **Impact Measurement**:
   - Automatic performance comparison before/after implementation
   - Statistical significance testing
   - ROI calculation for each recommendation type

3. **Prioritization Engine**:
   - Dashboard showing highest-impact recommendations
   - Filtering options for recommendation types
   - Sorting by expected impact

4. **A/B Testing Framework**:
   - Controlled implementation tracking
   - Comparison group handling
   - Results visualization

## Next Steps

1. Start with the database models and basic service layer implementation
2. Implement async handling for the Django views
3. Create minimal API endpoints for core functionality
4. Build recommendation tracking as the first UI component