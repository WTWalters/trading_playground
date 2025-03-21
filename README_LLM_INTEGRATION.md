# LLM Integration for TITAN Trading Platform

## Overview

This repository contains the implementation of Large Language Model (LLM) integration for the TITAN trading platform. The implementation enhances two key areas:

1. **Market Regime Analysis** - Uses LLMs to analyze financial news, Fed statements, and economic reports to enhance the existing statistical regime detection
2. **Trade Post-Mortem Analysis** - Provides detailed analysis of completed trades to identify patterns and improvement opportunities

## Implementation Status

The implementation follows the phased approach outlined in the documentation:

### Phase 1 - Foundation
- ✅ Database schema extensions for news data and trade analysis
- ✅ LLM client architecture with provider abstraction
- ✅ Claude API integration
- ✅ Configuration management with API key security
- ✅ Basic test infrastructure

### Phase 2 - Market Regime Analysis Enhancement
- ✅ LLM-enhanced regime detection framework
- ✅ News data provider framework
- ✅ Alpha Vantage news provider implementation
- ✅ Prompt engineering for regime classification
- ⬜ Integration with existing trading strategies
- ⬜ Performance tracking metrics

### Phase 3 - Trade Post-Mortem Analysis
- ✅ Trade context collection framework
- ⬜ Post-mortem analysis engine
- ⬜ Pattern recognition across trades
- ⬜ Integration with strategy optimization

## Directory Structure

```
/src/llm_integration/                  # Main LLM integration package
  ├── __init__.py                      # Package initialization
  ├── config.py                        # LLM configuration management
  ├── clients/                         # LLM provider clients
  │   ├── __init__.py
  │   ├── base.py                      # Base LLM client interface
  │   ├── claude.py                    # Claude implementation
  │   └── factory.py                   # LLM client factory
  ├── market_analysis/                 # Market regime analysis
  │   ├── __init__.py
  │   └── regime_analyzer.py           # LLM-enhanced regime analyzer
  └── trade_analysis/                  # Trade analysis
      ├── __init__.py
      └── context_collector.py         # Trade context collection

/src/data_ingestion/providers/news/    # News data providers
  ├── __init__.py
  ├── base.py                          # Base news provider interface
  └── alpha_vantage.py                 # Alpha Vantage implementation

/migrations/llm_integration/           # Database migrations
  ├── 001_news_data_schema.sql         # News data schema
  └── 002_trade_analysis_schema.sql    # Trade analysis schema

/prompts/                              # Prompt templates
  ├── regime_analysis/                 # Regime analysis prompts
  │   └── regime_classification.txt    # Regime classification
  └── trade_analysis/                  # Trade analysis prompts
      └── post_mortem.txt              # Post-mortem analysis

/tests/llm_integration/               # Tests for LLM integration
  ├── __init__.py
  ├── test_llm_client.py              # LLM client tests
  ├── test_regime_analyzer.py         # Regime analyzer tests
  └── test_context_collector.py       # Context collector tests

/docs/llm_integration/                # Documentation
  ├── overview.md                     # General overview
  ├── market_regime_analysis.md       # Market regime documentation
  ├── trade_post_mortem.md            # Trade post-mortem documentation
  ├── database_schema.md              # Database schema documentation
  ├── llm_client_configuration.md     # Configuration documentation
  └── performance_metrics.md          # Metrics documentation
```

## API Key Management

API keys for LLM providers are managed securely using environment variables:

1. Copy `.env.example` to `.env`
2. Add your API keys to the `.env` file:
   ```
   CLAUDE_API_KEY=your_claude_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
3. The `.env` file is excluded from version control in `.gitignore`

## Usage Examples

### Market Regime Analysis

```python
from src.database.manager import DatabaseManager
from src.llm_integration.market_analysis import LLMRegimeAnalyzer

# Initialize the database manager and analyzer
db_manager = DatabaseManager(config)
analyzer = LLMRegimeAnalyzer(db_manager)

# Analyze recent news for market regime insights
result = await analyzer.analyze_recent_news(timeframe="7d")
print(f"Detected regime: {result['primary_regime']} with {result['confidence']}% confidence")

# Analyze a Fed statement
fed_statement = "The Federal Open Market Committee decided to maintain the target range..."
fed_analysis = await analyzer.analyze_specific_document(
    document_type="fed_statement",
    document_content=fed_statement,
    document_metadata={"date": statement_date, "source": "Federal Reserve"}
)

# Enhance statistical regime detection with LLM analysis
enhanced_result = await analyzer.enhance_regime_detection(market_data, macro_data)
```

### Trade Context Collection

```python
from src.database.manager import DatabaseManager
from src.llm_integration.trade_analysis import TradeContextCollector

# Initialize the database manager and collector
db_manager = DatabaseManager(config)
collector = TradeContextCollector(db_manager)

# Collect context for a trade
context = await collector.collect_trade_context(trade)
```

## Testing

Run the LLM integration tests:

```bash
python -m pytest tests/llm_integration/
```

## Next Steps

1. Complete the trade post-mortem analyzer implementation
2. Implement pattern recognition across multiple trades
3. Integrate with the existing trading strategies
4. Develop performance tracking metrics dashboard
5. Implement additional LLM providers (DeepSeek, Gemini, Ollama)

## Contributing

When contributing to the LLM integration:

1. Follow the existing architecture patterns
2. Add tests for all new functionality
3. Add appropriate documentation
4. Update the README_LLM_INTEGRATION.md file with new changes
5. Never commit API keys or sensitive information
