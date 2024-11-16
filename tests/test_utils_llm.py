import pydantic
import pytest
from langchain_core.exceptions import OutputParserException
from podcast_llm.utils.llm import LLMWrapper, get_fast_llm, get_long_context_llm
from podcast_llm.config import PodcastConfig


def test_llm_wrapper_initialization_with_supported_providers():
    """Test that LLMWrapper initializes correctly with supported providers."""
    supported_providers = ['openai', 'google', 'anthropic']
    model_name = 'test-model-name'
    for provider_name in supported_providers:
        llm_wrapper_instance = LLMWrapper(
            provider=provider_name, 
            model=model_name
        )
        assert llm_wrapper_instance.provider == provider_name
        assert llm_wrapper_instance.model == model_name
        assert llm_wrapper_instance.temperature == 1.0
        assert llm_wrapper_instance.max_tokens == 8192
        assert llm_wrapper_instance.rate_limiter is None
        assert llm_wrapper_instance.llm is not None

def test_llm_wrapper_initialization_with_unsupported_provider():
    """Test that LLMWrapper raises ValueError when initialized with an unsupported provider."""
    with pytest.raises(ValueError) as exception_info:
        LLMWrapper(provider='unsupported_provider', model='test-model-name')
    assert "The LLM provider value 'unsupported_provider' is not supported." in str(exception_info.value)

def test_llm_wrapper_with_structured_output_method():
    """Test that LLMWrapper configures structured output correctly."""
    class MockSchema(pydantic.BaseModel):
        example_field: str

    llm_wrapper_instance = LLMWrapper(
        provider='openai', 
        model='test-model-name'
    )
    llm_wrapper_instance = llm_wrapper_instance.with_structured_output(MockSchema)
    assert llm_wrapper_instance.llm is not None
    assert llm_wrapper_instance.schema is None  # For OpenAI provider, schema should not be set
    # Assuming with_structured_output returns self

def test_llm_wrapper_coerce_to_schema():
    """Test that LLMWrapper.coerce_to_schema correctly converts output to schema objects."""
    class Question(pydantic.BaseModel):
        question: str
        
    class Answer(pydantic.BaseModel):
        answer: str
        
    class OtherSchema(pydantic.BaseModel):
        other: str

    llm_wrapper = LLMWrapper(provider='openai', model='test-model')
    
    # Test with Question schema
    llm_wrapper.schema = Question
    question_output = llm_wrapper.coerce_to_schema("What is the meaning of life?")
    assert isinstance(question_output, Question)
    assert question_output.question == "What is the meaning of life?"
    
    # Test with Answer schema
    llm_wrapper.schema = Answer
    answer_output = llm_wrapper.coerce_to_schema("42")
    assert isinstance(answer_output, Answer)
    assert answer_output.answer == "42"
    
    # Test with no schema defined
    llm_wrapper.schema = None
    with pytest.raises(ValueError) as exc_info:
        llm_wrapper.coerce_to_schema("test output")
    assert "Schema is not defined" in str(exc_info.value)
    
    # Test with unsupported schema type
    llm_wrapper.schema = OtherSchema
    with pytest.raises(OutputParserException) as exc_info:
        llm_wrapper.coerce_to_schema("test output")
    assert "Unable to coerce output to schema: OtherSchema" in str(exc_info.value)


def test_get_fast_llm_with_supported_provider():
    """Test that get_fast_llm returns an LLMWrapper with the correct fast model."""
    config_instance = PodcastConfig.load()
    config_instance.fast_llm_provider='openai'
    rate_limiter_instance = None
    fast_llm_instance = get_fast_llm(
        config=config_instance, 
        rate_limiter=rate_limiter_instance
    )
    assert isinstance(fast_llm_instance, LLMWrapper)
    assert fast_llm_instance.provider == 'openai'
    assert fast_llm_instance.model == 'gpt-4o-mini'

def test_get_fast_llm_with_unsupported_provider():
    """Test that get_fast_llm raises ValueError when given an unsupported provider."""
    config_instance = PodcastConfig.load()
    config_instance.fast_llm_provider='unsupported_provider'
    rate_limiter_instance = None
    with pytest.raises(ValueError) as exception_info:
        get_fast_llm(
            config=config_instance, 
            rate_limiter=rate_limiter_instance
        )
    assert "The fast_llm_provider value 'unsupported_provider' is not supported." in str(exception_info.value)

def test_get_long_context_llm_with_supported_provider():
    """Test that get_long_context_llm returns an LLMWrapper with the correct long context model."""
    config_instance = PodcastConfig.load()
    config_instance.long_context_llm_provider='anthropic'
    rate_limiter_instance = None
    long_context_llm_instance = get_long_context_llm(
        config=config_instance, 
        rate_limiter=rate_limiter_instance
    )
    assert isinstance(long_context_llm_instance, LLMWrapper)
    assert long_context_llm_instance.provider == 'anthropic'
    assert long_context_llm_instance.model == 'claude-3-5-sonnet-20241022'

def test_get_long_context_llm_with_unsupported_provider():
    """Test that get_long_context_llm raises ValueError when given an unsupported provider."""
    config_instance = PodcastConfig.load()
    config_instance.long_context_llm_provider='unsupported_provider'
    rate_limiter_instance = None
    with pytest.raises(ValueError) as exception_info:
        get_long_context_llm(
            config=config_instance, 
            rate_limiter=rate_limiter_instance
        )
    assert "The long_context_llm_provider value 'unsupported_provider' is not supported." in str(exception_info.value)


