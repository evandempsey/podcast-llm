import pytest
from pathlib import Path
from podcast_llm.extractors.pdf import PDFSourceDocument


@pytest.fixture
def sample_pdf_path(tmp_path) -> Path:
    """Create a temporary PDF file for testing."""
    return Path(__file__).parent / 'test_data' / 'attention.pdf'


def test_pdf_extractor_initialization() -> None:
    """Test that PDFExtractor initializes correctly."""
    pdf_path = 'test.pdf'
    extractor = PDFSourceDocument(pdf_path)
    assert extractor.src == pdf_path
    assert extractor.content is None


def test_pdf_extraction(sample_pdf_path: Path, mocker) -> None:
    """Test PDF content extraction with mocked PyPDFLoader."""
    mock_pages = [
        mocker.Mock(page_content='Page 1 content'),
        mocker.Mock(page_content='Page 2 content'),
        mocker.Mock(page_content='Page 3 content')
    ]
    mocker.patch('podcast_llm.extractors.pdf.PyPDFLoader.load', return_value=mock_pages)
    
    extractor = PDFSourceDocument(str(sample_pdf_path))
    content = extractor.extract()
    
    expected_content = 'Page 1 content\n\nPage 2 content\n\nPage 3 content'
    assert content == expected_content
    assert extractor.content == expected_content


def test_pdf_extraction_failure(sample_pdf_path: Path, mocker) -> None:
    """Test that PDF extraction handles errors gracefully."""
    mocker.patch(
        'podcast_llm.extractors.pdf.PyPDFLoader.load',
        side_effect=Exception('PDF Error')
    )
    
    extractor = PDFSourceDocument(str(sample_pdf_path))
    with pytest.raises(Exception):
        extractor.extract()
