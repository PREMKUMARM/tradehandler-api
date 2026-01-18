"""
LLM Factory for creating LLM instances based on configuration
"""
from typing import Optional

# Import OpenAI with error handling
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# Import Anthropic with error handling
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

# Import Ollama with error handling
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    try:
        # Fallback for older versions or if community is missing
        from langchain_community.llms import Ollama as ChatOllama
    except ImportError:
        ChatOllama = None

from langchain_core.language_models import BaseChatModel

from .config import AgentConfig, LLMProvider, get_agent_config


def create_llm(config: Optional[AgentConfig] = None) -> BaseChatModel:
    """
    Create LLM instance based on configuration
    
    Args:
        config: Agent configuration (uses global config if not provided)
        
    Returns:
        BaseChatModel instance
    """
    if config is None:
        config = get_agent_config()
    
    if config.llm_provider == LLMProvider.OPENAI:
        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is not installed. "
                "Please install it with: pip install langchain-openai"
            )
        if not config.openai_api_key:
            # Try to get from environment
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or in config.")
        else:
            api_key = config.openai_api_key
            
        return ChatOpenAI(
            model=config.agent_model,
            temperature=config.agent_temperature,
            max_tokens=config.max_tokens,
            api_key=api_key
        )
    
    elif config.llm_provider == LLMProvider.ANTHROPIC:
        if ChatAnthropic is None:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Please install it with: pip install langchain-anthropic"
            )
        if not config.anthropic_api_key:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or in config.")
        else:
            api_key = config.anthropic_api_key
            
        return ChatAnthropic(
            model=config.agent_model,
            temperature=config.agent_temperature,
            max_tokens=config.max_tokens,
            api_key=api_key
        )
    
    elif config.llm_provider == LLMProvider.OLLAMA:
        if ChatOllama is None:
            raise ImportError(
                "langchain-community is not installed. "
                "Please install it with: pip install langchain-community"
            )
        return ChatOllama(
            model=config.agent_model,
            base_url=config.ollama_base_url,
            temperature=config.agent_temperature
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

