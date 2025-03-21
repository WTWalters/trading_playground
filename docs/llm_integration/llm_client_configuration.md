# LLM Client Configuration

## Overview

This document outlines the configuration and setup of LLM clients for the TITAN trading platform. The system supports multiple LLM providers through a consistent interface while managing API keys securely.

## Supported LLM Providers

The system initially supports these LLM providers:

1. **Anthropic Claude** - Primary provider for most analyses
   - Models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
   - Best for: Complex analyses, nuanced financial interpretation, detailed reasoning

2. **DeepSeek** - Alternative for specific analyses
   - Models: deepseek-coder, deepseek-chat
   - Best for: Pattern recognition, quantitative analysis

3. **Google Gemini** - Alternative for specific analyses
   - Models: gemini-pro
   - Best for: Multi-modal analysis (when implemented)

4. **Ollama** - Local deployment option
   - Models: configurable based on local installation
   - Best for: Testing, development, and sensitive data processing

## Configuration Structure

LLM configuration is managed through the existing configuration system with these additions:

```python
class LLMConfig(BaseSettings):
    """LLM configuration settings"""
    # Anthropic Claude settings
    claude_api_key: Optional[str] = None
    claude_default_model: str = "claude-3-opus-20240229"
    claude_timeout: int = 120
    
    # DeepSeek settings
    deepseek_api_key: Optional[str] = None
    deepseek_default_model: str = "deepseek-chat"
    deepseek_timeout: int = 60
    
    # Google Gemini settings
    gemini_api_key: Optional[str] = None
    gemini_default_model: str = "gemini-pro"
    gemini_timeout: int = 60
    
    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_default_model: str = "llama2"
    ollama_timeout: int = 60
    
    # General settings
    default_provider: str = "claude"  # Options: claude, deepseek, gemini, ollama
    max_retries: int = 3
    retry_delay: int = 1
    cache_results: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
```

## API Key Management

### Security Approach

1. **Environment Variables**: All API keys are stored in environment variables
2. **Local .env File**: For development, keys are stored in .env (excluded from git)
3. **Production Secrets**: In production, keys are managed through a secure vault system

### .env File Structure

```
# LLM API Keys
CLAUDE_API_KEY=your_claude_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Other configuration
LLM_DEFAULT_PROVIDER=claude
LLM_CACHE_RESULTS=true
```

## LLM Client Factory

The system uses a factory pattern to create appropriate LLM clients:

```python
class LLMClientFactory:
    """Factory for creating LLM client instances."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def get_client(self, provider: Optional[str] = None) -> LLMClient:
        """
        Get an LLM client instance for the specified provider.
        
        Args:
            provider: Provider ID (claude, deepseek, gemini, ollama)
                     If None, uses default from config
        
        Returns:
            LLMClient instance
        """
        if provider is None:
            provider = self.config.default_provider
            
        if provider == "claude":
            if not self.config.claude_api_key:
                raise ValueError("Claude API key not configured")
            return ClaudeClient(self.config)
            
        elif provider == "deepseek":
            if not self.config.deepseek_api_key:
                raise ValueError("DeepSeek API key not configured")
            return DeepseekClient(self.config)
            
        elif provider == "gemini":
            if not self.config.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            return GeminiClient(self.config)
            
        elif provider == "ollama":
            return OllamaClient(self.config)
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
```

## Usage Example

```python
# Load configuration
config = load_llm_config()

# Create client factory
client_factory = LLMClientFactory(config)

# Get default client (based on configuration)
default_client = client_factory.get_client()

# Get specific provider
claude_client = client_factory.get_client("claude")

# Use the client
result = await claude_client.analyze_text(
    text="Market analysis content...",
    prompt="Analyze the market conditions...",
    model="claude-3-opus-20240229"  # Override default model
)
```

## Testing and Local Development

For testing and local development without API keys:
1. Set up Ollama locally for testing
2. Use the mock LLM client for unit tests
3. Create a .env.example file with placeholders for required keys
4. Always check for key availability before attempting API calls