"""
PDF file extraction module.

This module provides functionality for extracting text content from PDF files
using LangChain's PyPDFLoader. It handles loading PDFs, extracting text from
each page, and combining the content into a single document.

The module includes:
- PDFSourceDocument class for handling PDF file extraction 
- Page-by-page text extraction
- Combining pages with appropriate spacing
- Conversion to LangChain Document format

Example:
    >>> from podcast_llm.extractors.pdf import PDFSourceDocument
    >>> extractor = PDFSourceDocument('document.pdf')
    >>> extractor.extract()
    >>> print(extractor.content)
    'Text content from PDF pages...'

The extraction process:
1. Loads the PDF using PyPDFLoader
2. Extracts text from each page
3. Combines pages with double newlines between them
4. Returns the complete text content

The module integrates with the BaseSourceDocument interface to provide consistent
handling of PDF files alongside other source types like audio and web content.
"""


from podcast_llm.extractors.base import BaseSourceDocument
from langchain_community.document_loaders import PyPDFLoader
from typing import Optional


class PDFSourceDocument(BaseSourceDocument):
    """
    A document extractor for PDF files.

    This class handles extracting text content from PDF files using the PyPDFLoader
    from LangChain. It loads the PDF, extracts text from each page, and combines
    them into a single document with page breaks.

    Attributes:
        src (str): Path to the source PDF file
        src_type (str): Type of source document ('PDF File')
        title (str): Title combining source type and filename
        content (Optional[str]): Extracted text content after processing

    Example:
        >>> extractor = PDFSourceDocument('document.pdf')
        >>> extractor.extract()
        >>> print(extractor.content)
        'Text content from PDF pages...'
    """

    def __init__(self, source: str) -> None:
        """
        Initialize the PDF extractor.

        Args:
            source: Path to the PDF file to extract text from
        """
        self.src = source
        self.src_type = 'PDF File'
        self.title = f"{self.src_type}: {source}"
        self.content: Optional[str] = None

    def extract(self) -> str:
        """
        Extract text content from the PDF file.

        Returns:
            The extracted text content as a string
        """
        loader = PyPDFLoader(self.src)
        pages = loader.load()
        self.content = '\n\n'.join(page.page_content for page in pages)
        return self.content
