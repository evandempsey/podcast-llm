"""Utility functions for working with embeddings models.

This module provides functionality for loading and managing embeddings models,
which are used to convert text into vector representations. Currently supports
OpenAI embeddings with potential to expand to other providers.

Functions:
    get_embeddings_model: Returns an initialized embeddings model based on config.
"""


import logging
from langchain_openai import OpenAIEmbeddings

from podcast_llm.config import PodcastConfig


logger = logging.getLogger(__name__)


def get_embeddings_model(config: PodcastConfig):
    """Get the configured embeddings model instance.

    Args:
        config (PodcastConfig): Configuration object containing embeddings settings

    Returns:
        BaseEmbeddings: Initialized embeddings model instance based on config.embeddings_model.
            Currently supports 'openai' which returns OpenAIEmbeddings.
            Defaults to OpenAIEmbeddings if model type not recognized.
    """
    models = {
        'openai': OpenAIEmbeddings
    }

    return models.get(config.embeddings_model, OpenAIEmbeddings)()
