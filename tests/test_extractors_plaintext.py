import os
import pytest
from podcast_llm.extractors.plaintext import MarkdownSourceDocument, TextSourceDocument

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

def test_markdown_document_extraction():
    """Test extracting content from a Markdown file"""
    md_path = os.path.join(TEST_DATA_DIR, 'sample.md')
    
    # Create test markdown file
    test_content = '# Test Title\n\nTest markdown content'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    try:
        extractor = MarkdownSourceDocument(md_path)
        content = extractor.extract()
        
        assert content == test_content
        assert extractor.src_type == 'Markdown File'
        assert extractor.title == f'Markdown File: {md_path}'
        assert extractor.content == test_content
    finally:
        # Cleanup test file
        os.remove(md_path)

def test_text_document_extraction():
    """Test extracting content from a plain text file"""
    txt_path = os.path.join(TEST_DATA_DIR, 'sample.txt')
    
    # Create test text file
    test_content = 'Test plain text content'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    try:
        extractor = TextSourceDocument(txt_path)
        content = extractor.extract()
        
        assert content == test_content
        assert extractor.src_type == 'Text File'
        assert extractor.title == f'Text File: {txt_path}'
        assert extractor.content == test_content
    finally:
        # Cleanup test file
        os.remove(txt_path)

def test_file_not_found():
    """Test handling of non-existent files"""
    non_existent_file = os.path.join(TEST_DATA_DIR, 'does_not_exist.md')
    
    with pytest.raises(FileNotFoundError):
        extractor = MarkdownSourceDocument(non_existent_file)
        extractor.extract()

def test_unicode_content():
    """Test handling of Unicode content in files"""
    unicode_path = os.path.join(TEST_DATA_DIR, 'unicode.txt')
    
    # Create test file with Unicode content
    test_content = 'Unicode test: 你好 • ñ • é • 🌟'
    with open(unicode_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    try:
        extractor = TextSourceDocument(unicode_path)
        content = extractor.extract()
        
        assert content == test_content
        assert extractor.content == test_content
    finally:
        # Cleanup test file
        os.remove(unicode_path)
