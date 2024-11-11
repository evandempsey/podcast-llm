"""
Web article content extractor for podcast generation.

This module provides functionality to extract article content from web URLs using
the newspaper3k library. It handles downloading and parsing web pages to extract
the main article text while filtering out navigation, ads, and other non-content.

Example:
    >>> from podcast_llm.extractors.web import WebSourceDocument
    >>> extractor = WebSourceDocument('https://example.com/article')
    >>> content = extractor.extract()
    >>> print(content)
    'The main article text content...'

The module supports:
- Downloading and parsing web article content
- Intelligent extraction of main article text
- Filtering out non-content elements like navigation and ads
- Error handling for failed downloads or parsing

The extracted article text is returned as plain text and can be used as source
material for podcast episode generation. The module handles errors gracefully if
articles fail to download or parse properly.
"""

from podcast_llm.extractors.base import BaseSourceDocument
from typing import Optional
from newspaper import Article, ArticleException


class WebSourceDocument(BaseSourceDocument):
    """Extracts text content from web articles using the newspaper3k library.

    This class handles extracting article content from web URLs by downloading and parsing
    the page HTML. It uses newspaper3k to intelligently identify and extract the main
    article text while filtering out navigation, ads, and other non-content elements.

    Example:
        >>> extractor = WebSourceDocument('https://example.com/article')
        >>> content = extractor.extract() 
        >>> print(content)
        'The main article text content...'

    Attributes:
        src (str): The web article URL
        src_type (str): Always 'Website'
        title (str): The extracted article title
        content (Optional[str]): The extracted article text
    """
    def __init__(self, source: str) -> None:
        self.src = source
        self.src_type = 'Website'
        self.title = f"{self.src_type}: {source}"
        self.content: Optional[str] = None

    def extract(self) -> str:
        article = Article(self.src)
        article.download()
        article.parse()

        if not article.text:
            raise ArticleException(f"No website text found for URL: {self.src}")
        
        self.title = article.title
        self.content = article.text
