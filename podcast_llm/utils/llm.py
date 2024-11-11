"""
Utilities for working with Language Learning Models (LLMs) in a standardized way.

This module provides utilities for interacting with various LLM providers through
a unified interface. It handles provider-specific implementation details while
exposing a consistent API for the rest of the application.

Key components:
- LLMWrapper: A class that wraps different LLM providers (OpenAI, Google, Anthropic)
  with standardized interfaces for structured output parsing and rate limiting
- Helper functions for configuring and instantiating LLM instances with appropriate
  settings for podcast generation tasks

The module abstracts away provider differences around:
- Message formatting and chunking
- Rate limiting implementation
- Structured output parsing
- Temperature and token limit settings

This allows the rest of the application to work with LLMs in a provider-agnostic way
while still leveraging provider-specific capabilities when beneficial.
"""

import logging
import pydantic
from typing import Any, Optional, Union

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import BaseRateLimiter
from podcast_llm.config import PodcastConfig


logger = logging.getLogger(__name__)


class LLMWrapper(Runnable):
    def __init__(self, 
                 provider: str, 
                 model: str, 
                 temperature: float = 1.0, 
                 max_tokens: int = 8192, 
                 rate_limiter: Union[BaseRateLimiter, None] = None):
        """
        A wrapper class for various LLM providers that standardizes their interfaces.

        This class provides a unified interface for working with different LLM providers
        (OpenAI, Google, Anthropic) while handling provider-specific implementation details.
        It supports structured output parsing and rate limiting across all providers.

        Args:
            provider (str): The LLM provider to use ('openai', 'google', or 'anthropic')
            model (str): The specific model name/identifier for the chosen provider
            temperature (float, optional): Controls randomness in responses. Defaults to 1.0
            max_tokens (int, optional): Maximum tokens in response. Defaults to 8192
            rate_limiter (BaseRateLimiter | None, optional): Rate limiter for API calls. Defaults to None

        Raises:
            ValueError: If an unsupported provider is specified

        The wrapper normalizes differences between providers, particularly around:
        - Structured output parsing
        - Message formatting
        - Rate limiting implementation
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens,
        self.rate_limiter = rate_limiter
        self.parser = StrOutputParser()
        self.schema = None

        provider_to_model = {
            'openai': ChatOpenAI,
            'google': ChatGoogleGenerativeAI,
            'anthropic': ChatAnthropic
        }

        if self.provider not in provider_to_model:
            raise ValueError(f"The LLM provider value '{self.provider}' is not supported.")

        model_class = provider_to_model[self.provider]
        self.llm = model_class(model=self.model, rate_limiter=self.rate_limiter)

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None, 
        **kwargs: Any
    ) -> BaseMessage:
        """
        Invoke the LLM with the given input and configuration.

        This method handles provider-specific invocation details while maintaining a consistent
        interface. It manages structured output formatting and parsing based on the provider's
        capabilities.

        Args:
            input (LanguageModelInput): The input to send to the LLM, typically messages or prompts
            config (Optional[RunnableConfig]): Optional configuration for the invocation
            **kwargs (Any): Additional keyword arguments passed to the underlying LLM

        Returns:
            BaseMessage: The LLM's response message

        The implementation varies by provider:
        - OpenAI/Anthropic: Direct invocation with native structured output support
        - Google: Custom handling for structured output via parser and format instructions
        """
        logger.debug(f"Invoking LLM with prompt:\n{input.to_string()}")

        if self.provider in ('openai', 'anthropic',):
            return self.llm.invoke(input=input, config=config)
        elif self.provider == 'google':
            if self.schema is not None:
                format_instructions = self.parser.get_format_instructions()

                logger.debug(f"LLM provider is {self.provider} and schema is provided. Adding format instructions to prompt:\n{format_instructions}")
                messages = input.to_messages()
                messages[0] = SystemMessage(content=f"{messages[0].content}\n{format_instructions}")
                prompt = ChatPromptValue(messages=messages)
                logger.debug(f"Modified prompt:\n{prompt.to_string()}")

                return self.llm.invoke(input=prompt, config=config)
            else:
                return self.llm.invoke(input=input, config=config)

    def with_structured_output(
        self,
        schema: pydantic.BaseModel 
    ):
        """
        Configure the LLM wrapper to output structured data using a Pydantic schema.

        This method adapts the underlying LLM to output responses conforming to the provided
        Pydantic model schema. The implementation varies by provider:

        - OpenAI/Anthropic: Uses native structured output support
        - Google: Implements structured output via output parser and format instructions

        Args:
            schema (pydantic.BaseModel): The Pydantic model class defining the expected 
                response structure

        Returns:
            LLMWrapper: The wrapper instance configured for structured output
        """
        if self.provider in ('openai', 'anthropic',):
            self.llm = self.llm.with_structured_output(schema)
        elif self.provider == 'google':
            self.schema = schema
            self.parser = PydanticOutputParser(pydantic_object=schema)
            self.llm = self.llm | self.parser
         
        return self


def get_fast_llm(config: PodcastConfig, rate_limiter: BaseRateLimiter | None = None):
    """
    Get a fast LLM model optimized for quick responses.

    Creates and returns an LLM wrapper configured with a fast model variant from the 
    specified provider. Fast models trade some quality for improved response speed.

    Args:
        config (PodcastConfig): Configuration object containing provider settings
        rate_limiter (BaseRateLimiter | None, optional): Rate limiter to control API request 
            frequency. Defaults to None.

    Returns:
        LLMWrapper: Wrapper instance configured with a fast model variant

    Raises:
        ValueError: If the configured fast_llm_provider is not supported

    The function maps providers to their respective fast model variants:
    - OpenAI: gpt-4o-mini
    - Google: gemini-1.5-flash
    - Anthropic: claude-3-5-sonnet
    """
    fast_llm_models = {
        'openai': 'gpt-4o-mini',
        'google': 'gemini-1.5-flash', 
        'anthropic': 'claude-3-5-sonnet-20241022'
    }
    
    if config.fast_llm_provider not in fast_llm_models:
        raise ValueError(f"The fast_llm_provider value '{config.fast_llm_provider}' is not supported.")
        
    return LLMWrapper(config.fast_llm_provider, fast_llm_models[config.fast_llm_provider], rate_limiter=rate_limiter)


def get_long_context_llm(config: PodcastConfig, rate_limiter: BaseRateLimiter | None = None):
    """
    Get a long context LLM model optimized for handling larger prompts.

    Creates and returns an LLM wrapper configured with a model variant that can handle
    longer context windows from the specified provider. These models are optimized for
    processing larger amounts of text at once.

    Args:
        config (PodcastConfig): Configuration object containing provider settings
        rate_limiter (BaseRateLimiter | None, optional): Rate limiter to control API request
            frequency. Defaults to None.

    Returns:
        LLMWrapper: Wrapper instance configured with a long context model variant

    Raises:
        ValueError: If the configured long_context_llm_provider is not supported

    The function maps providers to their respective long context model variants:
    - OpenAI: gpt-4o 
    - Google: gemini-1.5-pro-latest
    - Anthropic: claude-3-5-sonnet
    """
    long_context_llm_models = {
        'openai': 'gpt-4o',
        'google': 'gemini-1.5-pro-latest',
        'anthropic': 'claude-3-5-sonnet-20241022'
    }

    if config.long_context_llm_provider not in long_context_llm_models:
        raise ValueError(f"The long_context_llm_provider value '{config.long_context_llm_provider}' is not supported.")

    return LLMWrapper(config.long_context_llm_provider, long_context_llm_models[config.long_context_llm_provider], rate_limiter=rate_limiter)
