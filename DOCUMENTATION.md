# TITAN Trading System Documentation Index

Welcome to the TITAN Trading System documentation. This index will guide you through all available documentation resources.

## Documentation Structure

The documentation is organized into the following categories:

1. **System Documentation** - Overall system architecture and design
2. **Component Documentation** - Details for specific system components
3. **Development Documentation** - Implementation status and planning
4. **User Guides** - How to use the system
5. **Testing Documentation** - Testing approach and results

## Quick Links

- [System Architecture](docs/architecture/system_architecture.md) - Overview of the entire system
- [Getting Started](docs/user_guides/getting_started.md) - How to set up and run the system
- [Development Status](docs/development/development_status.md) - Current project status
- [Testing Framework](docs/testing/testing_framework.md) - Testing approach

## All Documentation

### System Documentation
- [System Architecture](docs/architecture/system_architecture.md) - Overall system design
- [UI/UX Architecture](docs/architecture/ui_ux_architecture.md) - Frontend design
- [Database Schema](docs/architecture/database_schema.md) - Database design

### Component Documentation
- [Data Ingestion](docs/components/data_ingestion.md) - Data pipeline documentation
- [Market Analysis](docs/components/market_analysis.md) - Market analysis components
- [Parameter Management](docs/components/parameter_management.md) - Adaptive parameter system
- [Backtesting](docs/components/backtesting.md) - Backtesting framework
- [Regime Detection](docs/components/regime_detection.md) - Market regime classification
- [Trade Post-Mortem Analysis](docs/llm_integration/trade_post_mortem.md) - Trade analysis and improvement system

### Development Documentation
- [Development Status](docs/development/development_status.md) - Overall project status
- [Implementation Timeline](docs/development/implementation_timeline.md) - Timeline and roadmap
- [Next Steps](docs/development/next_steps.md) - Planned future work

### User Guides
- [Getting Started](docs/user_guides/getting_started.md) - Setup guide
- [Running Backtests](docs/user_guides/running_backtests.md) - How to run backtests
- [Analyzing Results](docs/user_guides/analyzing_results.md) - How to analyze results

### Testing Documentation
- [Testing Framework](docs/testing/testing_framework.md) - Testing approach
- [Testing Results](docs/testing/testing_results.md) - Summary of test results

### Integration Plans
- [Django Integration](docs/integration/django_integration.md) - Web platform integration
- [Trade Post-Mortem Django Integration](docs/integration/post_mortem_django_integration.md) - Post-mortem analyzer web integration
- [ML Integration](docs/integration/ml_integration.md) - Machine learning integration

## Naming Conventions

To maintain consistency, all documentation files follow these conventions:
- All documentation files use snake_case (underscores between words)
- All documentation files use `.md` extension (Markdown format)
- Directory names use lowercase with underscores for spaces

## References to Legacy Documentation

This index replaces and consolidates several previous documentation files:
- The original README.md remains for project overview
- FIXING_DATA_SOURCE.md is now referenced in the troubleshooting guide
- Previous implementation tracking documents have been consolidated

## Contributing to Documentation

When adding new documentation:
1. Follow the naming conventions above
2. Add a link in this index file
3. Include cross-references to related documentation
4. Place the file in the appropriate category directory
