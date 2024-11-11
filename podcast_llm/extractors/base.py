"""
Base content extractor interface for podcast generation.

This module provides the abstract base class that defines the interface for
extracting content from different source types like PDFs, web articles, YouTube
videos, etc. Concrete implementations handle the specifics of extracting from
each source type.

The module defines:
- BaseSourceDocument abstract base class
- Common interface for content extraction
- Conversion to LangChain Document format
- Standard metadata fields

Example:
    >>> class PDFSourceDocument(BaseSourceDocument):
    ...     def extract(self) -> str:
    ...         # PDF-specific extraction logic
    ...         return extracted_text

The base class ensures consistent handling of different source types while allowing
specialized extraction logic in the concrete implementations. This enables modular
addition of new source types while maintaining a uniform interface.
"""


from abc import ABC, abstractmethod
import uuid
from langchain_core.documents import Document


class BaseSourceDocument(ABC):
    """Abstract base class for source document content extractors.

    This class defines the interface for extracting content from different source types
    like PDFs, web articles, YouTube videos, etc. Concrete implementations handle the
    specifics of extracting from each source type.

    Example:
        >>> class PDFSourceDocument(BaseSourceDocument):
        ...     def extract(self) -> str:
        ...         # PDF-specific extraction logic
        ...         return extracted_text

    Attributes:
        src (str): Path or URL to the source media
        src_type (str): Type of source (e.g. 'PDF File', 'Website', 'YouTube video')
        title (str): Title describing the source
        content (Optional[str]): The extracted content text
    """

    @abstractmethod
    def extract(self) -> str:
        """
        Extract content from the source.

        Args:
            source: Path or URL to the source media

        Returns:
            The extracted content as a string
        """
        pass

    def as_langchain_document(self) -> Document:
        """
        Convert the source document to a LangChain Document format.

        Returns:
            Document: A LangChain Document containing the content and metadata
        """
        return Document(
            id=str(uuid.uuid4()),
            page_content=self.content,
            metadata={
                'title': self.title,
                'source': self.src,
                'source_type': self.src_type
            }
        )
