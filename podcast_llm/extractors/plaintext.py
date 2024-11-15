"""
Plaintext file extraction module.

This module provides functionality for extracting text content from plaintext files
like Markdown documents. It handles loading text files, preserving formatting, and
converting content into a standardized document format.

The module includes:
- MarkdownSourceDocument class for handling Markdown file extraction
- Raw text content preservation including formatting
- Conversion to standardized document format
- UTF-8 encoding support

Example:
    >>> from podcast_llm.extractors.plaintext import MarkdownSourceDocument
    >>> extractor = MarkdownSourceDocument('script.md')
    >>> extractor.extract()
    >>> print(extractor.content)
    '# Title\n\nMarkdown content...'

The extraction process:
1. Opens the text file with UTF-8 encoding
2. Reads the complete file content
3. Preserves original formatting and structure
4. Returns the raw text content

The module integrates with the BaseSourceDocument interface to provide consistent
handling of plaintext files alongside other source types like PDFs and web content.
"""



from podcast_llm.extractors.base import BaseSourceDocument
from typing import Optional


class MarkdownSourceDocument(BaseSourceDocument):
    """
    A document extractor for Markdown files.

    This class handles extracting text content from Markdown files by reading
    the raw text content. It preserves the original Markdown formatting which
    can be used for conversation structure and formatting.

    Attributes:
        src (str): Path to the source Markdown file
        src_type (str): Type of source document ('Markdown File')
        title (str): Title combining source type and filename
        content (Optional[str]): Extracted text content after processing

    Example:
        >>> extractor = MarkdownSourceDocument('script.md')
        >>> extractor.extract()
        >>> print(extractor.content)
        '# Title\n\nMarkdown content...'
    """

    def __init__(self, source: str) -> None:
        """
        Initialize the text extractor.

        Args:
            source: Path to the Markdown file to extract text from
        """
        self.src = source
        self.src_type = 'Markdown File'
        self.title = f"{self.src_type}: {source}"
        self.content: Optional[str] = None

    def extract(self) -> str:
        """
        Extract text content from the Markdown file.

        Returns:
            The extracted text content as a string
        """
        with open(self.src, 'r', encoding='utf-8') as file:
            self.content = file.read()
        return self.content


class TextSourceDocument(BaseSourceDocument):
    """
    A document extractor for plain text files.

    This class handles extracting text content from plain text files by reading
    the raw text content. It provides simple text extraction without any special
    formatting or processing.

    Attributes:
        src (str): Path to the source text file
        src_type (str): Type of source document ('Text File')
        title (str): Title combining source type and filename
        content (Optional[str]): Extracted text content after processing

    Example:
        >>> extractor = TextSourceDocument('document.txt')
        >>> extractor.extract()
        >>> print(extractor.content)
        'Plain text content...'
    """

    def __init__(self, source: str) -> None:
        """
        Initialize the text extractor.

        Args:
            source: Path to the text file to extract text from
        """
        self.src = source
        self.src_type = 'Text File'
        self.title = f"{self.src_type}: {source}"
        self.content: Optional[str] = None

    def extract(self) -> str:
        """
        Extract text content from the text file.

        Returns:
            The extracted text content as a string
        """
        with open(self.src, 'r', encoding='utf-8') as file:
            self.content = file.read()
        return self.content
