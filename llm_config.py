# llm_config.py
"""
LLM Configuration Module
Standalone LLM configuration module that can be imported by multiple files.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from aworld.config.conf import AgentConfig

# Load environment variables
load_dotenv()


# OpenRouter configuration
API_SOURCE = os.getenv("API_SOURCE", "openrouter")
API_KEY = os.getenv("API_KEY", "your_api_key_here")
API_BASE = os.getenv("API_BASE", "https://openrouter.ai/api/v1")


# Model configuration
MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")

TEMPERATURE = 0.7

# Thinking mode control (for models that support it, e.g. Qwen3)
# True = disable thinking mode (recommended, more direct output)
# False = enable thinking mode (outputs <think>...</think> tags)
DISABLE_THINKING = True

# Token limit configuration (prevent context overflow)
# Actual token usage = compressed_context + system_prompt(~10k) + output
MAX_CONTEXT_TOKENS = 35000  # Max tokens passed to Observer (reserves space for system prompt and output)
MAX_OUTPUT_TOKENS = 5000    # Max LLM output tokens

# ================================
# LLM Config Creation
# ================================

def create_llm_config(
    api_source: str = None,
    api_key: str = None,
    api_base: str = None,
    model: str = None,
    temperature: float = None,
    disable_thinking: bool = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> AgentConfig:
    """
    Create an LLM configuration object.

    Args:
        api_source: API source (openrouter/openai), defaults to module config
        api_key: API key, defaults to module config
        api_base: API base URL, defaults to module config
        model: Model name, defaults to module config
        temperature: Temperature parameter, defaults to module config
        disable_thinking: Whether to disable thinking mode, defaults to module config
        extra_params: Additional LLM parameters

    Returns:
        AgentConfig object
    """
    # Use provided parameters or defaults
    _api_source = api_source or API_SOURCE
    _api_key = api_key or API_KEY
    _api_base = api_base or API_BASE
    _model = model or MODEL
    _temperature = temperature if temperature is not None else TEMPERATURE
    _disable_thinking = disable_thinking if disable_thinking is not None else DISABLE_THINKING

    # Create LLM config parameters
    llm_config_params = {
        "llm_provider": "openai",  # Both OpenRouter and OpenAI are compatible with OpenAI API format
        "llm_model_name": _model,
        "llm_api_key": _api_key,
        "llm_temperature": _temperature,
    }

    # Add API base (required for OpenRouter)
    if _api_base:
        llm_config_params["llm_base_url"] = _api_base

    # Handle thinking mode via ext.extra_body
    ext: Dict[str, Any] = {}
    if _disable_thinking:
        ext["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

    # Merge user-provided extra parameters
    if extra_params:
        if "extra_body" in extra_params and "extra_body" in ext:
            ext["extra_body"].update(extra_params["extra_body"])
            extra_params = {k: v for k, v in extra_params.items() if k != "extra_body"}
        ext.update(extra_params)

    # Only add ext when non-empty
    if ext:
        llm_config_params["ext"] = ext

    return AgentConfig(**llm_config_params)


# Default global llm_config object (uses module defaults)
llm_config = create_llm_config()


# Helper function to display config info
def print_config_info():
    """Print current LLM configuration info."""
    print(f"\n{'='*60}")
    print(f"Model: {MODEL}")
    print(f"API Source: {API_SOURCE}")
    if API_BASE:
        print(f"API Base: {API_BASE}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Disable Thinking: {DISABLE_THINKING}")
    print(f"Max Context Tokens: {MAX_CONTEXT_TOKENS}")
    print(f"Max Output Tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Results directory: ./res/{MODEL}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Testing LLM Configuration...")
    print_config_info()
    print(f"\nLLM Config Object: {llm_config}")
    print(f"Model Name: {llm_config.llm_config.llm_model_name}")
    print(f"Temperature: {llm_config.llm_config.llm_temperature}")
    ext = getattr(llm_config, 'ext', {})
    print(f"Ext: {ext}")
    if ext.get('extra_body', {}).get('chat_template_kwargs', {}).get('enable_thinking') is False:
        print("Thinking mode is DISABLED")
