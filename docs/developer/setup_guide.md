# TITAN Trading System Setup Guide

**Type**: Developer Guide  
**Last Updated**: 2025-03-16  

## Related Documents

- [Coding Standards](./coding_standards.md)
- [API Reference](./api_reference.md)
- [Database Schema](../components/data/database_schema.md)

## Overview

This guide provides step-by-step instructions for setting up the development environment for the TITAN Trading System. It covers all required dependencies, configuration steps, and validation procedures to ensure a properly functioning development environment.

## Prerequisites

Before beginning the setup process, ensure you have the following prerequisites installed:

- **Python**: Python 3.12 or higher
- **PostgreSQL**: Version 15 or higher with TimescaleDB extension
- **Git**: Latest version
- **Docker**: Latest version (optional, for containerized development)
- **Poetry**: Latest version (for dependency management)

## Step 1: Clone the Repository

Clone the TITAN Trading System repository from GitHub:

```bash
git clone https://github.com/organization/titan-trading-system.git
cd titan-trading-system
```

## Step 2: Environment Setup

### Using Poetry (Recommended)

Poetry is the recommended dependency management tool for the TITAN Trading System.

1. Install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:

```bash
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

### Using Virtual Environment (Alternative)

Alternatively, you can use a standard Python virtual environment:

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- On Windows:
```bash
venv\Scripts\activate
```

- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Database Setup

### Option 1: Local TimescaleDB Installation

1. Install TimescaleDB following the [official documentation](https://docs.timescale.com/install/latest/self-hosted/).

2. Create a database for the TITAN system:

```bash
createdb titan_trading
```

3. Enable the TimescaleDB extension:

```bash
psql -d titan_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

4. Run the database initialization script:

```bash
python scripts/initialize_database.py
```

### Option 2: Docker Container (Recommended for Development)

1. Start a TimescaleDB container:

```bash
docker run -d --name titan-timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=titan_trading \
  timescale/timescaledb:latest-pg15
```

2. Run the database initialization script:

```bash
python scripts/initialize_database.py --host localhost --port 5432 --user postgres --password postgres --db titan_trading
```

## Step 4: Configuration

1. Create a local configuration file:

```bash
cp config/template.yaml config/local.yaml
```

2. Edit `config/local.yaml` to match your environment settings, particularly:
   - Database connection parameters
   - API keys (if applicable)
   - Logging settings
   - Development mode flags

3. Set the environment variable to use your local configuration:

```bash
export TITAN_CONFIG=local
```

## Step 5: Load Test Data

Load synthetic test data for development and testing:

```bash
python scripts/load_test_data.py
```

This script will:
- Generate synthetic market data
- Create sample cointegrated pairs
- Setup initial strategy parameters
- Populate historical regime data

## Step 6: Run Tests

Verify your setup by running the test suite:

```bash
pytest tests/
```

All tests should pass, confirming that your development environment is properly configured.

## Step 7: Start the Development Server

For components that include a web interface or API, start the development server:

```bash
python scripts/run_dev_server.py
```

The server will be available at `http://localhost:8000`.

## Working with the Codebase

### Project Structure

The TITAN Trading System follows a modular architecture:

```
titan/
├── data/                # Data handling components
│   ├── ingestion/       # Data collection and processing
│   ├── storage/         # Database interface
│   └── validation/      # Data quality checks
├── analysis/            # Analysis components
│   ├── cointegration/   # Pair identification
│   ├── regime/          # Regime detection
│   └── parameters/      # Parameter management
├── backtesting/         # Backtesting engine
│   ├── engine/          # Core backtesting logic
│   ├── metrics/         # Performance metrics
│   └── visualization/   # Results visualization
├── trading/             # Trading components
│   ├── signals/         # Signal generation
│   ├── execution/       # Order execution
│   └── risk/            # Risk management
├── common/              # Shared utilities
│   ├── logging/         # Logging framework
│   ├── config/          # Configuration handling
│   └── exceptions/      # Custom exceptions
└── api/                 # API endpoints (if applicable)
```

### Development Workflow

1. **Branch Management**:
   - `main`: Production-ready code
   - `develop`: Integration branch for feature development
   - Feature branches: Named as `feature/feature-name`
   - Bug fix branches: Named as `fix/bug-description`

2. **Commit Guidelines**:
   - Use descriptive commit messages
   - Include issue ID if applicable (`[TITAN-123] Add feature X`)
   - Keep commits focused on a single logical change

3. **Pull Requests**:
   - Create pull requests against the `develop` branch
   - Include a description of changes and testing performed
   - Request reviews from appropriate team members
   - Ensure all CI checks pass

## Common Development Tasks

### Adding a New Component

1. Create a new module in the appropriate package
2. Implement the component following the coding standards
3. Add tests in the corresponding test directory
4. Update documentation to reflect the new component
5. Create a pull request for review

### Modifying Database Schema

1. Create a new migration script in the `migrations` directory
2. Test the migration locally
3. Update the database schema documentation
4. Create a pull request for review

### Adding a New API Endpoint

1. Create a new endpoint in the appropriate API module
2. Implement request validation and error handling
3. Add tests for the endpoint
4. Update the API documentation
5. Create a pull request for review

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

1. Verify the database is running:
   ```bash
   docker ps | grep timescaledb
   ```

2. Check connection parameters in `config/local.yaml`

3. Run the database diagnostics script:
   ```bash
   python scripts/diagnose_database.py
   ```

### Dependency Issues

If you encounter dependency issues:

1. Update your Poetry dependencies:
   ```bash
   poetry update
   ```

2. Clear the Poetry cache if needed:
   ```bash
   poetry cache clear --all .
   ```

3. Check for conflicting dependencies:
   ```bash
   poetry show --tree
   ```

### Test Failures

If tests are failing:

1. Run tests with increased verbosity:
   ```bash
   pytest tests/ -v
   ```

2. Run specific failing tests:
   ```bash
   pytest tests/path/to/test_file.py::test_function -v
   ```

3. Check logs for detailed error information:
   ```bash
   cat logs/test.log
   ```

## Getting Help

If you encounter issues not covered in this guide:

1. Check the project's issue tracker for similar problems
2. Consult the development team through the project's communication channels
3. Review the related documentation for the component you're working with

## Next Steps

After setting up your development environment, consider:

1. Exploring the [API Reference](./api_reference.md) to understand the available APIs
2. Reviewing the [Coding Standards](./coding_standards.md) to ensure your contributions follow project conventions
3. Examining the [Database Schema](../components/data/database_schema.md) to understand data structures
4. Running through the [Pipeline Integration](./pipeline_integration.md) guide to understand the overall system flow

## See Also

- [Coding Standards](./coding_standards.md) - Coding conventions for the project
- [API Reference](./api_reference.md) - API documentation
- [Pipeline Integration](./pipeline_integration.md) - Guide to integrating components into the pipeline
- [Testing Guidelines](../testing/testing_guidelines.md) - Guide to writing and running tests
