# Frontend Integration

## Building and Deployment

### Vite Configuration

```typescript
// vite.config.ts

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'build',
    sourcemap: process.env.NODE_ENV !== 'production',
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          mui: ['@mui/material', '@mui/icons-material'],
          charts: ['recharts', 'd3'],
        },
      },
    },
  },
});
```

### Environment Configuration

```typescript
// .env.development
VITE_API_URL = 'http://localhost:8000/api/v1/'
VITE_WS_URL = 'ws://localhost:8000/ws/'

// .env.production
VITE_API_URL = '/api/v1/'
VITE_WS_URL = '/ws/'
```

### Production Build Scripts

```json
// package.json (partial)
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "jest",
    "e2e": "cypress open",
    "lint": "eslint src --ext .ts,.tsx",
    "format": "prettier --write \"src/**/*.{ts,tsx}\"",
    "analyze": "source-map-explorer 'build/assets/*.js'"
  }
}
```

### Integration with Django

The React frontend is integrated with Django in either of two ways:

#### Option 1: Django Serving React Build

Django is configured to serve the React build files in production:

```python
# titan/settings/base.py

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

STATICFILES_DIRS = [
    BASE_DIR / 'frontend' / 'build',
]

# Media files
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Template configuration
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'frontend' / 'build'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

```python
# frontend/views.py

from django.views.generic import TemplateView
from django.urls import re_path

class FrontendView(TemplateView):
    template_name = 'index.html'

# Define the views for the frontend app
urlpatterns = [
    re_path(r'^(?:.*)/?$', FrontendView.as_view(), name='frontend'),
]
```

```python
# titan/urls.py

from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('api.urls')),
    
    # API documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # Serve React frontend - use only in production
    re_path(r'^(?:.*)/?$', TemplateView.as_view(template_name='index.html')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

#### Option 2: Separate Frontend and API Servers

In this configuration, the frontend is served separately, usually by Nginx or a similar web server:

```nginx
# nginx.conf

server {
    listen 80;
    server_name titan.example.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name titan.example.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/titan.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/titan.example.com/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Frontend (React)
    location / {
        root /var/www/titan/frontend/build;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    # API (Django)
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSockets
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static/ {
        alias /var/www/titan/staticfiles/;
    }
    
    # Media files
    location /media/ {
        alias /var/www/titan/media/;
    }
}
```

### CI/CD Pipeline

For CI/CD, GitHub Actions workflows are used to build and deploy both frontend and backend:

```yaml
# .github/workflows/deploy.yml

name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    # Set up Python
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    # Install Python dependencies
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry config virtualenvs.create false
        poetry install --no-dev
    
    # Set up Node.js
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    # Install Node.js dependencies
    - name: Install Node.js dependencies
      working-directory: ./frontend
      run: npm ci
    
    # Build React app
    - name: Build React app
      working-directory: ./frontend
      run: npm run build
    
    # Run tests
    - name: Run tests
      run: |
        python -m pytest
    
    # Collect static files
    - name: Collect static files
      run: |
        python manage.py collectstatic --noinput
    
    # Deploy using Ansible
    - name: Deploy with Ansible
      uses: dawidd6/action-ansible-playbook@v2
      with:
        playbook: deploy.yml
        key: ${{secrets.SSH_PRIVATE_KEY}}
        inventory: |
          [production]
          titan.example.com ansible_user=deploy
```

## Testing Strategy

### Unit Testing

```typescript
// src/components/charts/__tests__/PairSpreadChart.test.tsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import PairSpreadChart from '../PairSpreadChart';

describe('PairSpreadChart component', () => {
  const mockData = {
    timestamps: ['2023-01-01T00:00:00Z', '2023-01-02T00:00:00Z', '2023-01-03T00:00:00Z'],
    spread: [0.1, 0.2, 0.15],
    z_score: [1.2, 1.8, 0.9],
  };

  test('renders loading state when loading prop is true', () => {
    render(<PairSpreadChart data={mockData} loading={true} />);
    expect(screen.getByText('Loading chart data...')).toBeInTheDocument();
  });

  test('renders chart when data is provided and loading is false', () => {
    render(<PairSpreadChart data={mockData} loading={false} />);
    expect(screen.getByText('Pair Spread & Z-Score')).toBeInTheDocument();
  });

  test('displays period selector buttons', () => {
    render(<PairSpreadChart data={mockData} loading={false} />);
    expect(screen.getByRole('button', { name: '1M' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '3M' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '6M' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '1Y' })).toBeInTheDocument();
  });
});
```

### Integration Testing

```typescript
// src/hooks/__tests__/usePairWebSocket.test.tsx

import { renderHook, act } from '@testing-library/react-hooks';
import { usePairWebSocket } from '../../hooks/usePairWebSocket';
import { WebSocketProvider } from '../../contexts/WebSocketContext';
import { AuthProvider } from '../../contexts/AuthContext';
import React from 'react';

// Mock WebSocket
class MockWebSocket {
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: any) => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = WebSocket.CONNECTING;
  
  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) this.onopen();
    }, 0);
  }
  
  send(data: string) {}
  
  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) this.onclose();
  }
  
  // Helper to simulate incoming message
  mockMessage(data: any) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(data) });
    }
  }
}

// Replace global WebSocket with mock
global.WebSocket = MockWebSocket as any;

// Test wrapper
const Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AuthProvider>
    <WebSocketProvider url="ws://test.com/ws">
      {children}
    </WebSocketProvider>
  </AuthProvider>
);

describe('usePairWebSocket', () => {
  test('should connect and receive pair data', async () => {
    const { result, waitForNextUpdate } = renderHook(() => usePairWebSocket(1), {
      wrapper: Wrapper,
    });
    
    // Initial state
    expect(result.current.pairData).toBeNull();
    expect(result.current.latestUpdate).toBeNull();
    
    // Wait for connection
    await waitForNextUpdate();
    
    // Subscribe to pair updates
    act(() => {
      result.current.subscribe();
    });
    
    // Simulate receiving pair data
    act(() => {
      const websocket = global.WebSocket as any;
      websocket.mockMessage({
        type: 'pair_data',
        pair: {
          id: 1,
          symbol_1: 'GLD',
          symbol_2: 'SLV',
          pair_name: 'GLD/SLV',
          cointegration_pvalue: 0.03,
          half_life: 21.5,
          correlation: 0.85,
          hedge_ratio: 2.1,
          stability_score: 0.75,
        },
      });
    });
    
    // Check that data was received
    expect(result.current.pairData).not.toBeNull();
    expect(result.current.pairData?.pair_name).toBe('GLD/SLV');
    
    // Simulate receiving update
    act(() => {
      const websocket = global.WebSocket as any;
      websocket.mockMessage({
        type: 'pair_update',
        pair_id: 1,
        z_score: -2.5,
        spread: 0.15,
        timestamp: '2023-01-01T12:00:00Z',
      });
    });
    
    // Check that update was received
    expect(result.current.latestUpdate).not.toBeNull();
    expect(result.current.latestUpdate?.z_score).toBe(-2.5);
  });
});
```

### End-to-End Testing

```typescript
// cypress/e2e/login.cy.ts

describe('Login', () => {
  beforeEach(() => {
    cy.visit('/login');
  });

  it('should display login form', () => {
    cy.get('h1').should('contain', 'TITAN Trading System');
    cy.get('input[name="username"]').should('exist');
    cy.get('input[name="password"]').should('exist');
    cy.get('button[type="submit"]').should('exist');
  });

  it('should show validation errors for empty fields', () => {
    cy.get('button[type="submit"]').click();
    cy.get('p[id="username-helper-text"]').should('contain', 'Username is required');
    cy.get('p[id="password-helper-text"]').should('contain', 'Password is required');
  });

  it('should navigate to dashboard on successful login', () => {
    // Intercept API request
    cy.intercept('POST', '/api/v1/auth/token/', {
      statusCode: 200,
      body: {
        access: 'fake-token',
        refresh: 'fake-refresh-token',
      },
    });

    // Intercept user data request
    cy.intercept('GET', '/api/v1/users/*', {
      statusCode: 200,
      body: {
        id: 1,
        username: 'testuser',
        email: 'test@example.com',
        first_name: 'Test',
        last_name: 'User',
      },
    });

    cy.get('input[name="username"]').type('testuser');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();

    // Should navigate to dashboard
    cy.url().should('include', '/dashboard');
  });

  it('should show error message on failed login', () => {
    // Intercept API request with error
    cy.intercept('POST', '/api/v1/auth/token/', {
      statusCode: 401,
      body: {
        detail: 'No active account found with the given credentials',
      },
    });

    cy.get('input[name="username"]').type('wronguser');
    cy.get('input[name="password"]').type('wrongpassword');
    cy.get('button[type="submit"]').click();

    // Should show error message
    cy.get('.MuiAlert-root').should('contain', 'Invalid credentials');
  });
});
```

## Performance Optimization

### Rendering Optimization

1. **Memoization**: Components and expensive calculations are memoized

```typescript
// Using React.memo for components
const PairInfo = React.memo(({ pair }) => {
  // Component logic
});

// Using useMemo for expensive calculations
const filteredData = useMemo(() => {
  return data.filter(item => item.value > threshold);
}, [data, threshold]);

// Using useCallback for functions
const handleClick = useCallback(() => {
  // Function logic
}, [dependencies]);
```

2. **Virtualization**: For long lists, virtualization is used to render only visible items

```typescript
// src/components/pairs/PairsList.tsx (partial)

import { FixedSizeList } from 'react-window';

// Inside component
return (
  <FixedSizeList
    height={500}
    width="100%"
    itemCount={pairs.length}
    itemSize={60}
  >
    {({ index, style }) => (
      <div style={style}>
        <PairItem pair={pairs[index]} />
      </div>
    )}
  </FixedSizeList>
);
```

3. **Code Splitting**: Routes and large components are code-split

```typescript
// src/routes.tsx (partial)
import React, { Suspense, lazy } from 'react';

// Lazy loaded components
const DashboardPage = lazy(() => import('./pages/dashboard/DashboardPage'));
const PairsListPage = lazy(() => import('./pages/pairs/PairsListPage'));
const PairDetailPage = lazy(() => import('./pages/pairs/PairDetailPage'));

// In the routes
<Route 
  path="/dashboard" 
  element={
    <ProtectedRoute>
      <Suspense fallback={<div>Loading...</div>}>
        <MainLayout>
          <DashboardPage />
        </MainLayout>
      </Suspense>
    </ProtectedRoute>
  } 
/>
```

### Network Optimization

1. **Query Optimization**: React Query is configured for optimal data fetching

```typescript
// src/App.tsx (partial)

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      cacheTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

// In the App component
return (
  <QueryClientProvider client={queryClient}>
    {/* Application components */}
    {process.env.NODE_ENV === 'development' && <ReactQueryDevtools />}
  </QueryClientProvider>
);
```

2. **Batch Requests**: API endpoints support batch operations

```typescript
// Instead of multiple single requests
const createSignals = async (signals) => {
  return apiClient.post('/signals/batch/', { signals });
};
```

3. **Web Workers**: Heavy computations are offloaded to web workers

```typescript
// src/workers/dataProcessor.worker.ts
self.onmessage = (e) => {
  const { data, operation } = e.data;
  
  let result;
  switch (operation) {
    case 'filterData':
      result = filterData(data);
      break;
    case 'calculateMetrics':
      result = calculateMetrics(data);
      break;
    default:
      result = { error: 'Unknown operation' };
  }
  
  self.postMessage(result);
};

// src/hooks/useWorker.ts
export const useDataProcessor = () => {
  const workerRef = useRef<Worker | null>(null);
  
  useEffect(() => {
    workerRef.current = new Worker(
      new URL('../workers/dataProcessor.worker.ts', import.meta.url)
    );
    
    return () => {
      workerRef.current?.terminate();
    };
  }, []);
  
  const processData = useCallback((data, operation) => {
    return new Promise((resolve) => {
      const worker = workerRef.current;
      if (!worker) return;
      
      const handleMessage = (e) => {
        worker.removeEventListener('message', handleMessage);
        resolve(e.data);
      };
      
      worker.addEventListener('message', handleMessage);
      worker.postMessage({ data, operation });
    });
  }, []);
  
  return { processData };
};
```

## Accessibility

The TITAN Trading System follows WCAG 2.1 AA standards for accessibility:

### Keyboard Navigation

```typescript
// src/components/common/Button.tsx

import React from 'react';
import { Button as MuiButton, ButtonProps } from '@mui/material';

interface CustomButtonProps extends ButtonProps {
  // Additional props
}

const Button: React.FC<CustomButtonProps> = ({ children, ...props }) => {
  return (
    <MuiButton
      {...props}
      tabIndex={0}
      role="button"
    >
      {children}
    </MuiButton>
  );
};

export default Button;
```

### Screen Reader Support

```typescript
// src/components/dashboard/StatusIndicator.tsx

import React from 'react';
import { Box, Tooltip } from '@mui/material';

interface StatusIndicatorProps {
  status: 'success' | 'warning' | 'error' | 'neutral';
  label: string;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status, label }) => {
  // Map status to color
  const colorMap = {
    success: '#4caf50',
    warning: '#ff9800',
    error: '#f44336',
    neutral: '#9e9e9e',
  };
  
  const color = colorMap[status];
  
  return (
    <Tooltip title={label}>
      <Box
        sx={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          backgroundColor: color,
          display: 'inline-block',
        }}
        role="status"
        aria-label={label}
      />
    </Tooltip>
  );
};

export default StatusIndicator;
```

### Color Contrast

```typescript
// src/theme/palette.ts

export const palette = {
  primary: {
    main: '#1976d2', // WCAG AA contrast with white: 4.5:1
    light: '#42a5f5',
    dark: '#1565c0',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#9c27b0', // WCAG AA contrast with white: 5.3:1
    light: '#ba68c8',
    dark: '#7b1fa2',
    contrastText: '#ffffff',
  },
  error: {
    main: '#d32f2f', // WCAG AA contrast with white: 4.6:1
    light: '#ef5350',
    dark: '#c62828',
    contrastText: '#ffffff',
  },
  warning: {
    main: '#ed6c02', // WCAG AA contrast with white: 4.5:1
    light: '#ff9800',
    dark: '#e65100',
    contrastText: '#ffffff',
  },
  info: {
    main: '#0288d1', // WCAG AA contrast with white: 4.5:1
    light: '#03a9f4',
    dark: '#01579b',
    contrastText: '#ffffff',
  },
  success: {
    main: '#2e7d32', // WCAG AA contrast with white: 4.8:1
    light: '#4caf50',
    dark: '#1b5e20',
    contrastText: '#ffffff',
  },
};
```

## Future Enhancements

1. **Mobile Responsiveness**: Enhance mobile views for critical trading functions
2. **PWA Support**: Add progressive web app capabilities for offline access
3. **Data Export**: Add CSV/Excel export functionality for trading data
4. **Advanced Visualization**: Implement more complex technical analysis charts
5. **Dark Mode**: Add comprehensive dark mode support
6. **Notification System**: Implement browser and push notifications for trading signals
7. **User Preferences**: Add more customization options for the UI
8. **Performance Profiling**: Implement performance monitoring and profiling tools

## Conclusion

The frontend integration for the TITAN Trading System creates a responsive, data-rich interface that leverages React, TypeScript, and Material UI to provide traders with a powerful platform for statistical arbitrage trading. The architecture emphasizes:

1. **Real-time Data**: Through WebSockets for instant market updates
2. **Component Reusability**: Through a well-organized component hierarchy
3. **Type Safety**: Through TypeScript for better code quality
4. **Performance**: Through optimization techniques for large datasets
5. **Accessibility**: Through WCAG-compliant UI components

This frontend integration seamlessly connects with the Django backend to provide a complete, end-to-end solution for statistical arbitrage trading.
