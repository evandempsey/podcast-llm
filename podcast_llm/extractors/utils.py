"""
Utility functions for extracting content from various source types.

This module provides helper functions for extracting text content from different
source types like YouTube videos, web pages, PDFs, and audio files. It handles
detecting the source type and using the appropriate extractor.

Example:
    >>> from podcast_llm.extractors.utils import extract_content_from_sources
    >>> sources = ['https://youtube.com/watch?v=123', 'article.pdf']
    >>> content = extract_content_from_sources(sources)
    >>> print(len(content))
    2

The module supports:
- Automatic source type detection based on URL/file extension
- Extraction from YouTube videos, web pages, PDFs, and audio files
- Error handling for failed extractions
- Converting extracted content to LangChain document format

The extracted content is returned as a list of LangChain documents that can be
used for further processing. Failed extractions are logged but do not halt
processing of remaining sources.
"""


import logging
from typing import List
from collections import OrderedDict

from .pdf import PDFSourceDocument
from .youtube import YouTubeSourceDocument
from .web import WebSourceDocument
from .audio import AudioSourceDocument
from .plaintext import MarkdownSourceDocument, TextSourceDocument


logger = logging.getLogger(__name__)


def extract_content_from_sources(sources: List) -> List:
    """
    Extract content from a list of source URLs/files.

    Takes a list of source URLs or file paths and extracts text content from each using
    the appropriate extractor based on source type. Supports YouTube videos, web pages,
    PDFs, and audio files.

    Args:
        sources (List): List of source URLs or file paths to extract content from

    Returns:
        List: List of extracted content as LangChain documents

    Raises:
        Exception: If extraction fails for a source, it is logged but processing continues

    Example:
        >>> sources = ['https://youtube.com/watch?v=123', 'article.pdf'] 
        >>> content = extract_content_from_sources(sources)
        >>> print(len(content))
        2
    """
    extracted_content = []
    
    source_type_mapping = OrderedDict([
        ('youtube', (lambda s: 'youtube.com' in s or 'youtu.be' in s, YouTubeSourceDocument)),
        ('web', (lambda s: s.startswith(('http://', 'https://', 'ftp://')), WebSourceDocument)),
        ('pdf', (lambda s: s.lower().endswith('.pdf'), PDFSourceDocument)),
        ('audio', (lambda s: s.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')), AudioSourceDocument)),
        ('markdown', (lambda s: s.lower().endswith(('.md', '.markdown')), MarkdownSourceDocument)),
        ('text', (lambda s: s.lower().endswith(('.txt')), TextSourceDocument))
    ])
    
    for source in sources:
        try:
            logger.info(f"Extracting from source: {source}")
            
            for check_source, source_class in source_type_mapping.values():
                if check_source(source):
                    source_doc = source_class(source=source)
                    source_doc.extract()
                    extracted_content.append(source_doc.as_langchain_document())
                    break

        except Exception:
            logger.info(f"Failed to extract from source: {source}")
    
    return extracted_content
