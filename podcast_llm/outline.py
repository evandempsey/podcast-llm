"""
Podcast outline generation module.

This module provides functionality for generating and structuring podcast outlines.
It contains utilities for formatting and manipulating outline structures, as well as
functions for generating complete podcast outlines from topics and research material.

The module leverages LangChain and GPT-4 to intelligently structure podcast content
into a hierarchical outline format. It uses prompts from the LangChain Hub to ensure
consistent and high-quality outline generation.

Functions:
    format_wikipedia_document: Formats Wikipedia content for use in prompts
    outline_episode: Generates a complete podcast outline from a topic and research

Example:
    outline = outline_episode(
        config=podcast_config,
        topic="Artificial Intelligence",
        background_info=research_docs
    )
    print(outline.as_str)
"""


import logging
from langchain import hub
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.llm import get_long_context_llm
from podcast_llm.models import (
    PodcastOutline
)


logger = logging.getLogger(__name__)


def format_wikipedia_document(doc):
    """
    Format a Wikipedia document for use in prompt context.

    Takes a Wikipedia document object and formats its metadata and content into a 
    structured string format suitable for inclusion in LLM prompts. The format
    includes a header with the article title followed by the full article content.

    Args:
        doc: Wikipedia document object containing metadata and page content

    Returns:
        str: Formatted string with article title and content
    """
    return f"### {doc.metadata['title']}\n\n{doc.page_content}"


def outline_episode(config: PodcastConfig, topic: str, background_info: list) -> PodcastOutline:
    """
    Generate a structured outline for a podcast episode.

    Takes a topic and background research information, then uses LangChain and GPT-4 
    to generate a detailed podcast outline with sections and subsections. The outline
    is structured using Pydantic models for type safety and validation.

    Args:
        topic (str): The main topic for the podcast episode
        background_info (list): List of Wikipedia document objects containing research material

    Returns:
        PodcastOutline: Structured outline object containing sections and subsections
    """
    logger.info(f'Generating outline for podcast on: {topic}')
    
    prompthub_path = "evandempsey/podcast_outline:6ceaa688"
    outline_prompt = hub.pull(prompthub_path, )
    logger.info(f"Got prompt from hub: {prompthub_path}")

    outline_llm = get_long_context_llm(config)
    outline_chain = outline_prompt | outline_llm.with_structured_output(
        PodcastOutline
    )

    outline = outline_chain.invoke({
        "episode_structure": config.episode_structure_for_prompt,
        "topic": topic,
        "context_documents": "\n\n".join([format_wikipedia_document(d) for d in background_info])
    })

    logger.info(outline.as_str)
    return outline
