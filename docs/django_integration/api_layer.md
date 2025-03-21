# API Layer

## API Testing

Each API endpoint will have comprehensive tests:

```python
# api/tests/test_symbols.py

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from trading.models.symbols import Symbol
from users.models import User

class SymbolTests(APITestCase):
    """
    Test the symbol API endpoints
    """
    
    def setUp(self):
        """Set up test data"""
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Create test symbols
        Symbol.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            sector='Technology',
            exchange='NASDAQ',
            is_active=True
        )
        Symbol.objects.create(
            ticker='MSFT',
            name='Microsoft Corporation',
            sector='Technology',
            exchange='NASDAQ',
            is_active=True
        )
        Symbol.objects.create(
            ticker='INACTIVE',
            name='Inactive Company',
            sector='Other',
            exchange='NYSE',
            is_active=False
        )
        
        # Authenticate
        self.client.force_authenticate(user=self.user)
    
    def test_get_symbols_list(self):
        """Test retrieving list of active symbols"""
        url = reverse('symbol-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 2)  # Only active symbols
        
    def test_get_symbol_detail(self):
        """Test retrieving a specific symbol"""
        symbol = Symbol.objects.get(ticker='AAPL')
        url = reverse('symbol-detail', args=[symbol.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['ticker'], 'AAPL')
        
    def test_filter_symbols_by_sector(self):
        """Test filtering symbols by sector"""
        url = reverse('symbol-list')
        response = self.client.get(url, {'sector': 'Technology'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 2)
        
    def test_unauthenticated_access(self):
        """Test unauthenticated access is denied"""
        # Logout
        self.client.force_authenticate(user=None)
        
        url = reverse('symbol-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
```

```python
# api/tests/test_pairs.py

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from trading.models.symbols import Symbol
from trading.models.pairs import TradingPair
from users.models import User
import json

class TradingPairTests(APITestCase):
    """
    Test the trading pair API endpoints
    """
    
    def setUp(self):
        """Set up test data"""
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Create test symbols
        self.symbol1 = Symbol.objects.create(
            ticker='GLD',
            name='SPDR Gold Shares',
            sector='Commodities',
            exchange='NYSE',
            is_active=True
        )
        self.symbol2 = Symbol.objects.create(
            ticker='SLV',
            name='iShares Silver Trust',
            sector='Commodities',
            exchange='NYSE',
            is_active=True
        )
        
        # Create test trading pair
        self.pair = TradingPair.objects.create(
            symbol_1=self.symbol1,
            symbol_2=self.symbol2,
            cointegration_pvalue=0.03,
            half_life=21.5,
            correlation=0.85,
            hedge_ratio=2.1,
            is_active=True,
            stability_score=0.75
        )
        
        # Authenticate
        self.client.force_authenticate(user=self.user)
    
    def test_get_pairs_list(self):
        """Test retrieving list of trading pairs"""
        url = reverse('tradingpair-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
        
    def test_get_pair_detail(self):
        """Test retrieving a specific trading pair"""
        url = reverse('tradingpair-detail', args=[self.pair.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['symbol_1']['ticker'], 'GLD')
        self.assertEqual(response.data['symbol_2']['ticker'], 'SLV')
        
    def test_get_pair_spread(self):
        """Test retrieving spread data for a pair"""
        url = reverse('tradingpair-spread', args=[self.pair.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('timestamps', response.data)
        self.assertIn('spread', response.data)
        self.assertIn('z_score', response.data)
        
    def test_analyze_pair(self):
        """Test analyzing a new pair"""
        url = reverse('tradingpair-analyze')
        data = {
            'symbol_1': 'GLD',
            'symbol_2': 'SLV',
            'start_date': '2022-01-01',
            'end_date': '2022-12-31',
            'lookback_days': 252
        }
        response = self.client.post(url, data, format='json')
        
        # This assumes the PairAnalysisService returns a valid result
        # In a real test, we would mock the service
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_201_CREATED])
```

```python
# api/tests/test_backtesting.py

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from trading.models.symbols import Symbol
from trading.models.pairs import TradingPair
from trading.models.backtesting import BacktestRun, BacktestResult
from users.models import User
from datetime import datetime, timedelta
from django.utils import timezone
import json

class BacktestingTests(APITestCase):
    """
    Test the backtesting API endpoints
    """
    
    def setUp(self):
        """Set up test data"""
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Create test symbols
        self.symbol1 = Symbol.objects.create(
            ticker='GLD',
            name='SPDR Gold Shares',
            sector='Commodities',
            exchange='NYSE',
            is_active=True
        )
        self.symbol2 = Symbol.objects.create(
            ticker='SLV',
            name='iShares Silver Trust',
            sector='Commodities',
            exchange='NYSE',
            is_active=True
        )
        
        # Create test trading pair
        self.pair = TradingPair.objects.create(
            symbol_1=self.symbol1,
            symbol_2=self.symbol2,
            cointegration_pvalue=0.03,
            half_life=21.5,
            correlation=0.85,
            hedge_ratio=2.1,
            is_active=True,
            stability_score=0.75
        )
        
        # Create test backtest
        self.backtest = BacktestRun.objects.create(
            name='Test Backtest',
            user=self.user,
            start_date=timezone.make_aware(datetime(2022, 1, 1)),
            end_date=timezone.make_aware(datetime(2022, 12, 31)),
            parameters={
                'entry_threshold': -2.0,
                'exit_threshold': 0.0,
                'stop_loss': 0.2
            },
            status='COMPLETED'
        )
        self.backtest.pairs.add(self.pair)
        
        # Create test backtest result
        self.backtest_result = BacktestResult.objects.create(
            backtest=self.backtest,
            total_return=0.15,
            annualized_return=0.12,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            max_drawdown=0.08,
            win_rate=0.65,
            profit_factor=2.1,
            trade_count=42,
            detailed_metrics={
                'monthly_returns': [0.02, 0.03, -0.01, 0.04, 0.02, 0.01, 0.02, 0.01, -0.02, 0.01, 0.01, 0.01]
            }
        )
        
        # Authenticate
        self.client.force_authenticate(user=self.user)
    
    def test_get_backtest_list(self):
        """Test retrieving list of backtests"""
        url = reverse('backtestrun-list')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 1)
        
    def test_get_backtest_detail(self):
        """Test retrieving a specific backtest"""
        url = reverse('backtestrun-detail', args=[self.backtest.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['name'], 'Test Backtest')
        
    def test_get_backtest_results(self):
        """Test retrieving backtest results"""
        url = reverse('backtestrun-results', args=[self.backtest.id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['sharpe_ratio'], 1.8)
        self.assertEqual(response.data['win_rate'], 0.65)
        
    def test_create_backtest(self):
        """Test creating a new backtest"""
        url = reverse('backtestrun-list')
        data = {
            'name': 'New Backtest',
            'start_date': '2023-01-01',
            'end_date': '2023-06-30',
            'pair_ids': [self.pair.id],
            'parameters': {
                'entry_threshold': -2.5,
                'exit_threshold': 0.5,
                'stop_loss': 0.15
            }
        }
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(BacktestRun.objects.count(), 2)
```

## Performance Considerations

### Query Optimization

1. **Use select_related and prefetch_related**:
   ```python
   # Good: Efficient query with select_related
   pairs = TradingPair.objects.select_related('symbol_1', 'symbol_2').all()
   
   # Bad: N+1 query problem
   pairs = TradingPair.objects.all()
   for pair in pairs:
       print(f"{pair.symbol_1.ticker}/{pair.symbol_2.ticker}")
   ```

2. **Batch Database Operations**:
   ```python
   # Good: Bulk create
   Signal.objects.bulk_create(signal_objects)
   
   # Bad: Multiple database operations
   for signal_data in signals:
       Signal.objects.create(**signal_data)
   ```

3. **Use Database Functions**:
   ```python
   # Good: Use database functions for calculations
   from django.db.models import Avg, F, ExpressionWrapper, DecimalField
   
   pairs = TradingPair.objects.annotate(
       price_ratio=ExpressionWrapper(
           F('symbol_1__current_price') / F('symbol_2__current_price'),
           output_field=DecimalField()
       )
   )
   
   # Bad: Calculate in Python
   pairs = TradingPair.objects.all()
   for pair in pairs:
       pair.price_ratio = pair.symbol_1.current_price / pair.symbol_2.current_price
   ```

### Response Optimization

1. **Paginate Results**:
   ```python
   # All list endpoints should use pagination
   class BacktestingViewSet(viewsets.ModelViewSet):
       pagination_class = StandardResultsSetPagination
   ```

2. **Minimize Serialized Data**:
   ```python
   # Good: Use field filtering in serializers
   class TradingPairSerializer(serializers.ModelSerializer):
       symbol_1 = SymbolMinimalSerializer(read_only=True)  # Use minimal serializer
       
   # Define minimal serializers for nested objects
   class SymbolMinimalSerializer(serializers.ModelSerializer):
       class Meta:
           model = Symbol
           fields = ['id', 'ticker']  # Only essential fields
   ```

3. **Use Compression Middleware**:
   ```python
   # In settings.py
   MIDDLEWARE = [
       # ...
       'django.middleware.gzip.GZipMiddleware',
   ]
   ```

4. **Use ETags for Caching**:
   ```python
   # Use conditional responses
   @method_decorator(etag(etag_func=lambda request, *args, **kwargs: ...))
   def retrieve(self, request, *args, **kwargs):
       # ...
   ```

### Database Optimization

1. **Use Proper Indexing**:
   ```python
   class Price(TimescaleModel):
       # ...
       class Meta:
           indexes = [
               models.Index(fields=['symbol', 'timestamp']),
           ]
   ```

2. **Use TimescaleDB Features**:
   ```python
   # Using TimescaleDB's time_bucket function
   from django.db import connection
   
   with connection.cursor() as cursor:
       cursor.execute("""
           SELECT time_bucket('1 day', timestamp) AS bucket, 
                  AVG(close) AS avg_close
           FROM trading_price
           WHERE symbol_id = %s
           GROUP BY bucket
           ORDER BY bucket;
       """, [symbol_id])
       result = cursor.fetchall()
   ```

3. **Implement Query Caching**:
   ```python
   from django.core.cache import cache
   
   def get_prices(symbol_id, start_date, end_date):
       cache_key = f"prices_{symbol_id}_{start_date}_{end_date}"
       cached_result = cache.get(cache_key)
       
       if cached_result:
           return cached_result
           
       # Fetch from database
       prices = Price.objects.filter(
           symbol_id=symbol_id,
           timestamp__gte=start_date,
           timestamp__lte=end_date
       ).order_by('timestamp')
       
       # Cache for 30 minutes
       cache.set(cache_key, prices, 60 * 30)
       return prices
   ```

## Security Best Practices

1. **Proper Authentication**:
   - Use JWT with short expiration times
   - Implement refresh token rotation
   - Store tokens securely

2. **Role-Based Access Control**:
   - Implement custom permissions
   - Ensure proper object ownership checks
   - Filter querysets based on user permissions

3. **Input Validation**:
   - Use serializer validation for all inputs
   - Validate query parameters
   - Sanitize inputs to prevent SQL injection

4. **CSRF Protection**:
   - Include proper CSRF protection
   - Use SameSite cookies

5. **Rate Limiting**:
   - Apply rate limiting to all endpoints
   - Use different limits for authenticated vs. anonymous users

6. **Secure Headers**:
   - Set secure headers in responses
   - Implement HSTS, CSP, and X-Content-Type-Options

7. **Audit Logging**:
   - Log all sensitive operations
   - Implement activity logging for user actions

## API Documentation

The API documentation will be available at:

- `/api/docs/` - Swagger UI for interactive documentation
- `/api/redoc/` - ReDoc for more structured documentation
- `/api/schema/` - Raw OpenAPI schema

The documentation will include:

- Endpoint descriptions
- Request/response formats
- Authentication requirements
- Error response formats
- Example requests and responses

## Integration with Frontend

The API will be consumed by the React frontend using:

1. **Axios for HTTP Requests**:
   ```javascript
   import axios from 'axios';

   // Create API instance
   const api = axios.create({
     baseURL: '/api/v1/',
     headers: {
       'Content-Type': 'application/json',
     },
   });

   // Add auth interceptor
   api.interceptors.request.use(
     (config) => {
       const token = localStorage.getItem('access_token');
       if (token) {
         config.headers.Authorization = `Bearer ${token}`;
       }
       return config;
     },
     (error) => Promise.reject(error)
   );

   // Handle token refresh
   api.interceptors.response.use(
     (response) => response,
     async (error) => {
       const originalRequest = error.config;
       
       if (error.response.status === 401 && !originalRequest._retry) {
         originalRequest._retry = true;
         
         try {
           const refreshToken = localStorage.getItem('refresh_token');
           const response = await axios.post('/api/v1/auth/token/refresh/', {
             refresh: refreshToken,
           });
           
           localStorage.setItem('access_token', response.data.access);
           
           originalRequest.headers.Authorization = `Bearer ${response.data.access}`;
           return api(originalRequest);
         } catch (refreshError) {
           // Logout user
           localStorage.removeItem('access_token');
           localStorage.removeItem('refresh_token');
           window.location.href = '/login';
           return Promise.reject(refreshError);
         }
       }
       
       return Promise.reject(error);
     }
   );

   export default api;
   ```

2. **React Query for Data Fetching**:
   ```javascript
   import { useQuery, useMutation, useQueryClient } from 'react-query';
   import api from '../services/api';

   // Fetch trading pairs
   export const useTradingPairs = (filters = {}) => {
     return useQuery(
       ['tradingPairs', filters],
       async () => {
         const response = await api.get('/pairs/', { params: filters });
         return response.data;
       },
       {
         keepPreviousData: true,
         staleTime: 30000, // 30 seconds
       }
     );
   };

   // Fetch pair spread data
   export const usePairSpread = (pairId, startDate, endDate) => {
     return useQuery(
       ['pairSpread', pairId, startDate, endDate],
       async () => {
         const response = await api.get(`/pairs/${pairId}/spread/`, {
           params: { start_date: startDate, end_date: endDate },
         });
         return response.data;
       },
       {
         enabled: !!pairId,
         staleTime: 60000, // 1 minute
       }
     );
   };

   // Create a backtest
   export const useCreateBacktest = () => {
     const queryClient = useQueryClient();
     
     return useMutation(
       async (backtestData) => {
         const response = await api.post('/backtesting/', backtestData);
         return response.data;
       },
       {
         onSuccess: () => {
           queryClient.invalidateQueries('backtests');
         },
       }
     );
   };
   ```

3. **API Context Provider**:
   ```javascript
   import React, { createContext, useContext } from 'react';
   import { useTradingPairs, usePairSpread, useCreateBacktest } from '../hooks/api';

   const APIContext = createContext();

   export const APIProvider = ({ children }) => {
     const api = {
       useTradingPairs,
       usePairSpread,
       useCreateBacktest,
       // Add other API hooks
     };
     
     return <APIContext.Provider value={api}>{children}</APIContext.Provider>;
   };

   export const useAPI = () => {
     const context = useContext(APIContext);
     if (!context) {
       throw new Error('useAPI must be used within an APIProvider');
     }
     return context;
   };
   ```
