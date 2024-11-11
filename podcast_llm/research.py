"""
Research module for podcast generation.

This module provides functionality to gather background research and information
for podcast episode generation. It handles retrieving content from various sources
like Wikipedia and search engines.

Example:
    >>> from podcast_llm.research import suggest_wikipedia_articles
    >>> from podcast_llm.models import WikipediaPages
    >>> config = PodcastConfig()
    >>> articles: WikipediaPages = suggest_wikipedia_articles(config, "Artificial Intelligence")
    >>> print(articles.pages[0].name)
    'Artificial intelligence'

The research process includes:
- Suggesting relevant Wikipedia articles via LangChain and GPT-4
- Downloading Wikipedia article content
- Performing targeted web searches with Tavily
- Extracting key information from web articles
- Organizing research into structured formats using Pydantic models

The module uses various APIs and services to gather comprehensive background
information while maintaining rate limits and handling errors gracefully.
"""


import logging
from typing import List
from langchain import hub
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from podcast_llm.outline import PodcastOutline
from tavily import TavilyClient
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.llm import get_fast_llm
from podcast_llm.models import (
    SearchQueries,
    WikipediaPages
)
from podcast_llm.extractors.web import WebSourceDocument


logger = logging.getLogger(__name__)


def suggest_wikipedia_articles(config: PodcastConfig, topic: str) -> WikipediaPages:
    """
    Suggest relevant Wikipedia articles for a given topic.

    Uses LangChain and GPT-4 to intelligently suggest Wikipedia articles that would provide good
    background research for a podcast episode on the given topic.

    Args:
        topic (str): The podcast topic to research

    Returns:
        WikipediaPages: A structured list of suggested Wikipedia article titles
    """
    logger.info(f'Suggesting Wikipedia articles for topic: {topic}')

    prompthub_path = "evandempsey/podcast_wikipedia_suggestions:58c92df4"
    wikipedia_prompt = hub.pull(prompthub_path)
    logger.info(f"Got prompt from hub: {prompthub_path}")

    fast_llm = get_fast_llm(config)
    wikipedia_chain = wikipedia_prompt | fast_llm.with_structured_output(
        WikipediaPages
    )
    result = wikipedia_chain.invoke({"topic": topic})
    logger.info(f'Found {len(result.pages)} suggested Wikipedia articles')
    return result


def download_wikipedia_articles(suggestions: WikipediaPages) -> list:
    """
    Download Wikipedia articles based on suggested page titles.

    Takes a structured list of Wikipedia page suggestions and downloads the full content
    of each article using the WikipediaRetriever. Handles errors gracefully if any
    articles fail to download.

    Args:
        suggestions (WikipediaPages): Structured list of suggested Wikipedia page titles

    Returns:
        list: List of retrieved Wikipedia document objects containing page content and metadata
    """
    logger.info('Starting Wikipedia article download')
    retriever = WikipediaRetriever()

    wikipedia_documents = []
    for page in suggestions.pages:
        logger.info(f'Retrieving article: {page.name}')
        try:
            wikipedia_documents.append(retriever.invoke(page.name)[0])
            logger.debug(f'Successfully retrieved article: {page.name}')
        except Exception as e:
            logger.error(f'Failed to retrieve article {page.name}: {str(e)}')

    logger.info(f'Downloaded {len(wikipedia_documents)} Wikipedia articles')
    return wikipedia_documents


def research_background_info(config: PodcastConfig, topic: str) -> list:
    """
    Research background information for a podcast topic.

    Coordinates the research process by first suggesting relevant Wikipedia articles
    based on the topic, then downloading the full content of those articles. Acts as
    the main orchestration function for gathering background research material.

    Args:
        topic (str): The podcast topic to research

    Returns:
        dict: List of retrieved Wikipedia document objects containing article content and metadata
    """
    logger.info(f'Starting research for topic: {topic}')
    
    suggestions = suggest_wikipedia_articles(config, topic)
    wikipedia_content = download_wikipedia_articles(suggestions)

    logger.info('Research completed successfully')
    return wikipedia_content


def perform_tavily_queries(config: PodcastConfig, queries: SearchQueries) -> list:
    """
    Execute search queries using the Tavily API.

    Performs web searches for each provided query using the Tavily search API, filtering out
    certain domains and PDF files. Handles API interaction and result processing to extract
    relevant URLs for further content scraping.

    Args:
        queries (SearchQueries): Structured list of search queries to execute

    Returns:
        list: List of URLs from search results, excluding PDFs and filtered domains
    """
    logger.info("Performing search queries")
    tavily_client = TavilyClient(api_key=config.tavily_api_key)

    exclude_domains = [
        "wikipedia.org", 
        "youtube.com", 
        "books.google.com", 
        "academia.edu", 
        "washingtonpost.com"
    ]

    urls_to_scrape = set()
    for query in queries.queries:
        logger.info(f"Searching for {query.query}")
        response = tavily_client.search(query.query, exclude_domains=exclude_domains, max_results=5)
        urls_to_scrape.update([
            result['url'] for result in response['results'] 
            if not result['url'].endswith(".pdf")])

    return list(urls_to_scrape)


def download_page_content(urls: List[str]) -> List[Document]:
    """
    Download and parse content from a list of URLs.

    Uses the newspaper3k library to download and extract clean text content from web pages.
    Handles errors gracefully and logs success/failure for each URL. Filters out articles
    with no text content.

    Args:
        urls (list): List of URLs to download and parse

    Returns:
        list: List of dictionaries containing the downloaded articles with structure:
            {
                'url': str,      # Original URL
                'title': str,    # Article title
                'text': str      # Cleaned article text content
            }
    """
    logger.info('Downloading page content from URLs.')
    
    downloaded_articles = []
    for url in urls:
        try:
            web_source_doc = WebSourceDocument(url)
            web_source_doc.extract()
            downloaded_articles.append(web_source_doc.as_langchain_document())
        except Exception as e:
            logger.error(f'Unexpected error downloading {url}: {str(e)}')
            
    logger.info(f'Successfully downloaded {len(downloaded_articles)} articles')
    return downloaded_articles


def research_discussion_topics(config: PodcastConfig, topic: str, outline: PodcastOutline) -> list:
    """
    Research in-depth content for podcast discussion topics.

    Takes a podcast topic and outline, then uses LangChain and GPT-4 to generate targeted 
    search queries. These queries are used to find relevant articles via Tavily search.
    The articles are then downloaded and processed to provide detailed research material
    for each section of the podcast.

    Args:
        topic (str): The main topic for the podcast episode
        outline (PodcastOutline): Structured outline containing sections and subsections

    Returns:
        list: List of dictionaries containing downloaded article content with structure:
            {
                'url': str,      # Source URL
                'title': str,    # Article title  
                'text': str      # Article content
            }
    """
    logger.info(f'Suggesting search queries based on podcast outline')
    prompthub_path = "evandempsey/podcast_research_queries:561acf5f"

    search_queries_prompt = hub.pull(prompthub_path)
    logger.info(f"Got prompt from hub: {prompthub_path}")

    fast_llm = get_fast_llm(config)
    search_queries_chain = search_queries_prompt | fast_llm.with_structured_output(
        SearchQueries
    )
    queries = search_queries_chain.invoke({"topic": topic, "podcast_outline": outline.as_str})
    logger.info(f'Got {len(queries.queries)} suggested search queries')

    urls_to_scrape = perform_tavily_queries(config, queries)
    page_content = download_page_content(urls_to_scrape)
    return page_content
