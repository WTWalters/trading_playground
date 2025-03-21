# TITAN Trading System Coding Standards

**Type**: Developer Guide  
**Last Updated**: 2025-03-16  

## Related Documents

- [Setup Guide](./setup_guide.md)
- [API Reference](./api_reference.md)
- [Pipeline Integration](./pipeline_integration.md)

## Overview

This document outlines the coding standards and best practices for the TITAN Trading System. Following these standards ensures code consistency, maintainability, and quality across the project.

## Python Style Guide

The TITAN Trading System follows PEP 8 with some specific modifications and additional requirements.

### Code Formatting

- **Line Length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Formatter**: Black with line length of 100 characters
- **Imports**: Use `isort` for import sorting

Example configuration (pyproject.toml):
```toml
[tool.black]
line-length = 100
target-version = ['py312']

[tool.isort]
profile = "black"
line_length = 100
```

### Naming Conventions

- **Modules**: Lowercase with underscores (e.g., `data_ingestion.py`)
- **Packages**: Lowercase, single word if possible (e.g., `analysis`)
- **Classes**: CapWords/PascalCase (e.g., `DataProcessor`)
- **Functions/Methods**: Lowercase with underscores (e.g., `process_data()`)
- **Variables**: Lowercase with underscores (e.g., `processed_data`)
- **Constants**: Uppercase with underscores (e.g., `MAX_ITERATIONS`)
- **Type Variables**: CapWords with a `_t` suffix (e.g., `DataFrame_t`)

### Documentation

- **Docstrings**: Google-style docstrings for all public modules, classes, and functions
- **Type Hints**: Type hints for all function parameters and return values
- **Comments**: Only for complex or non-obvious code sections

Example:
```python
def calculate_hedge_ratio(prices1: np.ndarray, prices2: np.ndarray) -> float:
    """
    Calculate the hedge ratio between two price series using OLS regression.
    
    Args:
        prices1: First price series as numpy array
        prices2: Second price series as numpy array
        
    Returns:
        float: Calculated hedge ratio
        
    Raises:
        ValueError: If input arrays have different lengths or contain NaN values
    """
    if len(prices1) != len(prices2):
        raise ValueError("Price series must have the same length")
    
    if np.isnan(prices1).any() or np.isnan(prices2).any():
        raise ValueError("Price series contain NaN values")
    
    # For OLS regression, add constant to X (first price series)
    X = sm.add_constant(prices1)
    
    # Fit the model
    model = sm.OLS(prices2, X).fit()
    
    # The hedge ratio is the coefficient of the first price series
    return model.params[1]
```

## Code Organization

### File Structure

Each Python file should follow this structure:

1. Shebang line (if executable)
2. Module docstring
3. Import statements (grouped and sorted)
4. Constants
5. Exception classes
6. Custom type definitions
7. Helper functions and classes
8. Main functions and classes
9. Main execution block (if applicable)

Example:
```python
#!/usr/bin/env python3
"""
Module for cointegration testing of price series pairs.

This module provides functions and classes for testing cointegration
between pairs of financial instruments.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# Project imports
from titan.common.exceptions import DataValidationError
from titan.data.storage import DatabaseClient

# Constants
MAX_LAG = 20
SIGNIFICANCE_LEVEL = 0.05

# Custom exceptions
class CointegrationError(Exception):
    """Exception raised for errors in the cointegration testing process."""
    pass

# Type definitions
PriceSeries_t = Union[List[float], np.ndarray, pd.Series]
CointegrationResult_t = Dict[str, Union[bool, float, np.ndarray]]

# Helper functions
def _validate_price_series(prices1: PriceSeries_t, prices2: PriceSeries_t) -> None:
    """Validate input price series for cointegration testing."""
    # Implementation...

# Main functions
def test_cointegration(prices1: PriceSeries_t, prices2: PriceSeries_t) -> CointegrationResult_t:
    """Test for cointegration between two price series."""
    # Implementation...

# Main execution (if file is executed directly)
if __name__ == "__main__":
    # Example usage code...
```

### Module Structure

Organize related functionality into modules:

- One class per file for major classes
- Group related functions in a single file
- Keep files under 500 lines; split if larger

## Error Handling

### Exceptions

- Define custom exceptions for specific error cases
- Use built-in exceptions for standard error conditions
- Include descriptive error messages
- Handle exceptions at appropriate levels

Example:
```python
# Custom exceptions
class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass

class DatabaseConnectionError(Exception):
    """Exception raised for database connection issues."""
    pass

# Using custom exceptions
def validate_data(data: pd.DataFrame) -> None:
    """Validate the input data."""
    if data.empty:
        raise DataValidationError("Input data is empty")
    
    if data.isnull().any().any():
        raise DataValidationError("Input data contains missing values")

# Handling exceptions
try:
    validate_data(df)
    process_data(df)
except DataValidationError as e:
    logging.error(f"Data validation failed: {e}")
    # Take appropriate action
except Exception as e:
    logging.exception(f"Unexpected error: {e}")
    # Take appropriate action
```

### Logging

- Use the `logging` module for all logging
- Use appropriate log levels:
  - `DEBUG`: Detailed information for debugging
  - `INFO`: Confirmation of expected operation
  - `WARNING`: Unexpected behavior, but operation continues
  - `ERROR`: Operation failed, but program continues
  - `CRITICAL`: Program may not be able to continue
- Include context in log messages

Example:
```python
import logging

# Configuration is handled centrally; just get the logger
logger = logging.getLogger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process the input data."""
    logger.debug(f"Processing data with shape {data.shape}")
    
    try:
        # Processing logic
        result = actual_processing(data)
        logger.info(f"Data processing completed successfully. Output shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
```

## Testing

### Test Structure

- Use pytest for all tests
- Organize tests to mirror the project structure
- One test file per module/class
- Use fixtures for common setup
- Use parametrized tests for testing multiple inputs

Example:
```python
# tests/analysis/test_cointegration.py
import pytest
import numpy as np
from titan.analysis.cointegration import test_cointegration, CointegrationError

@pytest.fixture
def cointegrated_series():
    """Fixture providing a pair of cointegrated series."""
    np.random.seed(42)
    # Create two cointegrated series
    random_walk = np.cumsum(np.random.normal(0, 1, 1000))
    series1 = random_walk
    series2 = 2 * random_walk + np.random.normal(0, 0.5, 1000)
    return series1, series2

@pytest.fixture
def non_cointegrated_series():
    """Fixture providing a pair of non-cointegrated series."""
    np.random.seed(42)
    # Create two independent random walks
    series1 = np.cumsum(np.random.normal(0, 1, 1000))
    series2 = np.cumsum(np.random.normal(0, 1, 1000))
    return series1, series2

def test_cointegration_with_cointegrated_series(cointegrated_series):
    """Test that cointegrated series are correctly identified."""
    series1, series2 = cointegrated_series
    result = test_cointegration(series1, series2)
    assert result["is_cointegrated"] is True
    assert 1.9 < result["hedge_ratio"] < 2.1  # Should be close to 2

def test_cointegration_with_non_cointegrated_series(non_cointegrated_series):
    """Test that non-cointegrated series are correctly identified."""
    series1, series2 = non_cointegrated_series
    result = test_cointegration(series1, series2)
    assert result["is_cointegrated"] is False

@pytest.mark.parametrize("invalid_input", [
    ([], [1, 2, 3]),  # Empty first series
    ([1, 2, 3], []),  # Empty second series
    ([1, 2, 3], [1, 2]),  # Different lengths
])
def test_cointegration_with_invalid_inputs(invalid_input):
    """Test that invalid inputs raise appropriate exceptions."""
    series1, series2 = invalid_input
    with pytest.raises(ValueError):
        test_cointegration(series1, series2)
```

### Test Coverage

- Aim for at least 80% code coverage
- Write tests for both normal operation and edge cases
- Test error handling paths
- Include integration tests for component interactions

## Performance Considerations

### General Guidelines

- Profile code to identify bottlenecks
- Optimize only where needed; prioritize readability
- Use vectorized operations with numpy/pandas
- Consider parallelization for compute-intensive tasks
- Use appropriate data structures for the task

### Database Operations

- Use batch operations for database interactions
- Minimize database roundtrips
- Use efficient query patterns
- Consider indexing strategy
- Use connection pooling

Example:
```python
# Inefficient: Multiple queries
for record in records:
    db.insert_one(record)

# Efficient: Batch insert
db.insert_many(records)
```

### Memory Management

- Be mindful of memory usage for large datasets
- Use generators for large data processing
- Consider chunking for large operations
- Release resources explicitly when done

Example:
```python
# Memory inefficient: Loading entire dataset
def process_large_file(filename):
    data = pd.read_csv(filename)  # Loads entire file into memory
    # Process data
    return result

# Memory efficient: Processing in chunks
def process_large_file(filename, chunk_size=10000):
    results = []
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        # Process chunk
        chunk_result = process_chunk(chunk)
        results.append(chunk_result)
    return pd.concat(results)
```

## Code Reviews

### Review Checklist

All code reviews should check for:

1. **Correctness**: Does the code do what it's supposed to do?
2. **Style Compliance**: Does the code follow the project's style guide?
3. **Testability**: Is the code testable? Are tests included?
4. **Documentation**: Is the code properly documented?
5. **Error Handling**: Are errors handled appropriately?
6. **Performance**: Are there any obvious performance issues?
7. **Security**: Are there any security concerns?
8. **Maintainability**: Is the code easy to understand and maintain?

### Review Process

1. **Automated Reviews**:
   - All pull requests must pass CI/CD checks
   - Code formatting checks (black, isort)
   - Linting (flake8, pylint)
   - Type checking (mypy)
   - Test execution

2. **Manual Reviews**:
   - At least one team member must approve changes
   - Focus on logic, design, and maintainability
   - Provide constructive feedback
   - Use a "praise, question, suggest" approach

## Version Control

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- Feature branches: `feature/feature-name`
- Bug fix branches: `fix/bug-description`
- Release branches: `release/x.y.z`

### Commit Messages

- Use the format: `[Component] Brief description`
- Include issue ID if applicable: `[TITAN-123] Add feature X`
- Use present tense, imperative mood
- Keep messages concise but descriptive
- Include context where necessary

Example:
```
[Analysis] Add Kalman filter for dynamic hedge ratio estimation

Implements a Kalman filter approach for estimating time-varying hedge ratios
in cointegrated pairs. This allows for better adaptation to changing market
conditions compared to the static OLS approach.

Related to: TITAN-456
```

## Tools and Automation

### Required Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **coverage**: Test coverage

### Recommended IDE Configuration

Visual Studio Code settings:
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true
}
```

## Python-Specific Guidelines

### Asynchronous Programming

- Use `asyncio` for I/O-bound tasks
- Follow asyncio best practices
- Be mindful of event loop management
- Use proper exception handling in async code

Example:
```python
import asyncio
from typing import List, Dict, Any

async def fetch_data(symbol: str) -> Dict[str, Any]:
    """Fetch data for a symbol asynchronously."""
    # Async implementation...

async def process_symbols(symbols: List[str]) -> List[Dict[str, Any]]:
    """Process multiple symbols in parallel."""
    tasks = [fetch_data(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)

# Usage
async def main():
    symbols = ["AAPL", "MSFT", "GOOG"]
    results = await process_symbols(symbols)
    # Process results...

if __name__ == "__main__":
    asyncio.run(main())
```

### Type Hints

- Use type hints for all function parameters and return values
- Use `typing` module for complex types
- Consider using Protocol for structural typing
- Use TypeVar for generic functions

Example:
```python
from typing import Dict, List, Optional, TypeVar, Generic, Protocol, Union

T = TypeVar('T')

class DataSource(Protocol):
    """Protocol defining the interface for a data source."""
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data for a symbol within a date range."""
        ...

class DataProcessor(Generic[T]):
    """Generic data processor that works with any data type T."""
    
    def __init__(self, source: DataSource):
        self.source = source
    
    def process(self, data: T) -> T:
        """Process the data."""
        # Implementation...
        return data

def find_pairs(
    symbols: List[str],
    start_date: str,
    end_date: str,
    p_value_threshold: float = 0.05
) -> List[Dict[str, Union[str, float]]]:
    """
    Find cointegrated pairs among the provided symbols.
    
    Args:
        symbols: List of symbols to analyze
        start_date: Start date for the analysis (format: YYYY-MM-DD)
        end_date: End date for the analysis (format: YYYY-MM-DD)
        p_value_threshold: Threshold for cointegration p-value
        
    Returns:
        List of dictionaries containing pair information
    """
    # Implementation...
```

## Forbidden Patterns

The following patterns are discouraged in the TITAN codebase:

### Anti-Patterns

- **Global state**: Avoid using global variables
- **Magic numbers**: Use named constants
- **Nested conditionals**: Refactor to reduce nesting
- **Catch-all exceptions**: Catch specific exceptions
- **Reinventing the wheel**: Use standard libraries

### Code Smells

- Long functions (> 50 lines)
- Complex conditions (consider refactoring)
- Duplicated code (extract to functions)
- Too many parameters (consider refactoring)
- Comments explaining "what" instead of "why"

### Security Concerns

- Hardcoded credentials
- SQL injection vulnerabilities
- Insecure file operations
- Command injection vulnerabilities
- Inadequate input validation

## See Also

- [Setup Guide](./setup_guide.md) - Environment setup instructions
- [API Reference](./api_reference.md) - API documentation
- [Pipeline Integration](./pipeline_integration.md) - Guide to integrating components into the pipeline
- [Testing Guidelines](../testing/testing_guidelines.md) - Guide to writing and running tests
