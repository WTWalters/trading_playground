# TITAN Trading System - Django Integration

This is the Django integration for the TITAN Trading System, providing a web interface to the statistical arbitrage trading platform.

## Project Structure

The Django integration follows a modular structure:

- `titan/` - Main Django project
- `common/` - Shared functionality including TimescaleModel base class
- `trading/` - Core trading models and functionality
- `api/` - REST API endpoints
- `users/` - User management and authentication
- `channels_app/` - WebSocket functionality for real-time updates

## Key Features

- Dual database setup with TimescaleDB for time-series data
- JWT-based authentication
- RESTful API for all trading operations
- WebSocket support for real-time data and signals
- Service layer for integrating with existing trading components
- Responsive frontend with React

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL with TimescaleDB extension
- Redis (for Channels and Celery)

### Installation

1. Clone the repository
2. Run the setup script:
   ```
   cd titan_trading
   chmod +x setup.sh
   ./setup.sh
   ```
3. Start the development server:
   ```
   python manage.py runserver
   ```

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Django settings
DJANGO_SECRET_KEY=your_secret_key_here
DJANGO_ENV=development  # or production

# Database settings
DB_NAME=titan_django
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432

# TimescaleDB settings
TIMESCALE_NAME=titan_timescale
TIMESCALE_USER=your_db_user
TIMESCALE_PASSWORD=your_db_password
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432

# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/1

# Celery settings
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Architecture

### Database Structure

The system uses a dual database approach:
- **PostgreSQL**: For user data, configurations, and metadata
- **TimescaleDB**: For time-series market data

TimescaleDB models inherit from the `TimescaleModel` base class and are routed to the TimescaleDB database using the `TimescaleRouter`.

### Core Models

- **Symbol**: Securities with metadata
- **Price**: OHLCV time-series data (TimescaleDB)
- **TradingPair**: Cointegrated security pairs
- **Signal**: Trading signals for entry/exit
- **MarketRegime**: Identified market regimes
- **BacktestRun/Result/Trade**: Backtest configurations and results
- **WalkForwardTest**: Walk-forward optimization

### API Structure

The REST API provides endpoints for:
- Securities and market data
- Pair analysis and cointegration testing
- Trading signals
- Backtesting and performance analysis
- User preferences and authentication

### WebSocket Support

Real-time data is provided via WebSockets for:
- Price updates
- Trading signals
- Backtest progress
- Market regime changes

## Development

### Running Tests

```
python manage.py test
```

### Creating Migrations

```
python manage.py makemigrations
python manage.py migrate --database=default
python manage.py migrate --database=timescale
```

### Running Celery

In a separate terminal:

```
celery -A titan worker -l info
```

## Integration with Existing System

The Django integration uses a service layer to bridge with the existing TITAN Trading System components:

1. **Data Translation**: Converting between Django models and existing data structures
2. **Synchronous Wrappers**: Bridging Django's synchronous views with async trading components
3. **Database Mapping**: Mapping Django ORM operations to the existing database schema

## Deployment

For production deployment:

1. Set `DJANGO_ENV=production` in your environment
2. Configure a production-ready web server (e.g., Nginx + Gunicorn/Daphne)
3. Set up SSL certificates
4. Configure proper database connection pooling
5. Set up monitoring and logging

## License

This project is proprietary and confidential.
