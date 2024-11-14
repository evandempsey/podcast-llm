import pytest
from unittest.mock import patch
from langchain_openai import OpenAIEmbeddings

from podcast_llm.utils import embeddings
from podcast_llm.config import PodcastConfig


@pytest.fixture
def mock_config():
    """Fixture providing a mock config object"""
    config = PodcastConfig.load()
    config.embeddings_model = 'openai'
    return config


def test_get_embeddings_model_openai(mock_config):
    """Test getting OpenAI embeddings model"""
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
        model = embeddings.get_embeddings_model(mock_config)
        
        assert isinstance(model, OpenAIEmbeddings)


def test_get_embeddings_model_unknown(mock_config):
    """Test getting embeddings model with unknown type defaults to OpenAI"""
    mock_config.embeddings_model = 'unknown_model'
    
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
        model = embeddings.get_embeddings_model(mock_config)
        
        assert isinstance(model, OpenAIEmbeddings)


def test_get_embeddings_model_none(mock_config):
    """Test getting embeddings model with None type defaults to OpenAI"""
    mock_config.embeddings_model = None
    
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
        model = embeddings.get_embeddings_model(mock_config)
        
        assert isinstance(model, OpenAIEmbeddings)
