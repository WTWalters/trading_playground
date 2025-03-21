# CLAUDE.md - Trading Playground Helper

## Essential Commands
- Run tests: `poetry run pytest`
- Run single test: `poetry run pytest tests/path_to_test.py::test_function_name -v`
- Run with asyncio: `poetry run pytest tests/path_to_test.py::test_async_function -v`
- Format code: `poetry run black .`
- Type check: `poetry run mypy src/`
- Run tests with coverage: `poetry run pytest --cov=src`

## Code Style Guidelines
- Type annotations for all functions and methods
- Class names: PascalCase (e.g., `TradeTracker`)
- Functions/variables: snake_case (e.g., `calculate_half_life`)
- Constants: UPPER_CASE (e.g., `MAX_RETRY_ATTEMPTS`)
- Comprehensive docstrings for all modules, classes, and functions
- Import order: standard library, third-party packages, local imports
- Async code should use proper await patterns and error handling
- Use f-strings for string formatting
- Error handling with specific exceptions and descriptive messages
- Use dataclasses where appropriate for data containers
- Test cases should be descriptive and test edge cases