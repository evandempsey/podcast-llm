import pytest
from unittest.mock import Mock, patch
from podcast_llm.extractors import utils
from langchain.schema import Document


@pytest.fixture
def mock_source_docs():
    """Fixture providing mock source document instances"""
    youtube_doc = Mock()
    youtube_doc.as_langchain_document.return_value = Document(page_content='youtube content')
    
    pdf_doc = Mock() 
    pdf_doc.as_langchain_document.return_value = Document(page_content='pdf content')
    
    web_doc = Mock()
    web_doc.as_langchain_document.return_value = Document(page_content='web content')
    
    audio_doc = Mock()
    audio_doc.as_langchain_document.return_value = Document(page_content='audio content')

    word_doc = Mock()
    word_doc.as_langchain_document.return_value = Document(page_content='word content')
    
    return {
        'youtube': youtube_doc,
        'pdf': pdf_doc, 
        'web': web_doc,
        'audio': audio_doc,
        'word': word_doc
    }


def test_extract_content_from_sources_youtube(mock_source_docs):
    """Test extracting content from YouTube source"""
    with patch('podcast_llm.extractors.utils.YouTubeSourceDocument', return_value=mock_source_docs['youtube']):
        sources = ['https://youtube.com/watch?v=123']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 1
        assert result[0].page_content == 'youtube content'
        mock_source_docs['youtube'].extract.assert_called_once()


def test_extract_content_from_sources_pdf(mock_source_docs):
    """Test extracting content from PDF source"""
    with patch('podcast_llm.extractors.utils.PDFSourceDocument', return_value=mock_source_docs['pdf']):
        sources = ['document.pdf']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 1
        assert result[0].page_content == 'pdf content'
        mock_source_docs['pdf'].extract.assert_called_once()


def test_extract_content_from_sources_web(mock_source_docs):
    """Test extracting content from web source"""
    with patch('podcast_llm.extractors.utils.WebSourceDocument', return_value=mock_source_docs['web']):
        sources = ['https://example.com']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 1
        assert result[0].page_content == 'web content'
        mock_source_docs['web'].extract.assert_called_once()


def test_extract_content_from_sources_audio(mock_source_docs):
    """Test extracting content from audio source"""
    with patch('podcast_llm.extractors.utils.AudioSourceDocument', return_value=mock_source_docs['audio']):
        sources = ['audio.mp3']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 1
        assert result[0].page_content == 'audio content'
        mock_source_docs['audio'].extract.assert_called_once()

def test_extract_content_from_sources_docx(mock_source_docs):
    """Test extracting content from Word source"""
    with patch('podcast_llm.extractors.utils.WordSourceDocument', return_value=mock_source_docs['word']):
        sources = ['document.docx']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 1
        assert result[0].page_content == 'word content'
        mock_source_docs['word'].extract.assert_called_once()


def test_extract_content_from_sources_multiple(mock_source_docs):
    """Test extracting content from multiple sources"""
    with patch('podcast_llm.extractors.utils.YouTubeSourceDocument', return_value=mock_source_docs['youtube']), \
         patch('podcast_llm.extractors.utils.PDFSourceDocument', return_value=mock_source_docs['pdf']):
        sources = ['https://youtube.com/watch?v=123', 'document.pdf']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 2
        assert result[0].page_content == 'youtube content'
        assert result[1].page_content == 'pdf content'


def test_extract_content_from_sources_failure():
    """Test handling of extraction failures"""
    with patch('podcast_llm.extractors.utils.YouTubeSourceDocument') as mock_youtube:
        mock_youtube.side_effect = Exception('Extraction failed')
        sources = ['https://youtube.com/watch?v=123']
        result = utils.extract_content_from_sources(sources)
        
        assert len(result) == 0


def test_extract_content_from_sources_empty():
    """Test handling of empty source list"""
    result = utils.extract_content_from_sources([])
    assert len(result) == 0
