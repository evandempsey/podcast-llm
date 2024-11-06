import pytest
from podcast_llm.outline import (
    format_wikipedia_document,
    outline_episode
)
from podcast_llm.models import (
    PodcastOutline,
    PodcastSection,
    PodcastSubsection
)
from podcast_llm.config import PodcastConfig
from unittest.mock import Mock, patch

@pytest.fixture
def sample_subsection():
    return PodcastSubsection(title='Test Subsection')

@pytest.fixture
def sample_section():
    return PodcastSection(
        title='Test Section',
        subsections=[
            PodcastSubsection(title='Subsection 1'),
            PodcastSubsection(title='Subsection 2')
        ]
    )

@pytest.fixture
def sample_outline():
    return PodcastOutline(sections=[
        PodcastSection(
            title='Section 1',
            subsections=[
                PodcastSubsection(title='Section 1.1'),
                PodcastSubsection(title='Section 1.2')
            ]
        ),
        PodcastSection(
            title='Section 2', 
            subsections=[
                PodcastSubsection(title='Section 2.1')
            ]
        )
    ])

def test_subsection_as_str(sample_subsection):
    """Test PodcastSubsection string representation"""
    assert sample_subsection.as_str == '-- Test Subsection'

def test_section_as_str(sample_section):
    """Test PodcastSection string representation"""
    expected = 'Test Section\n-- Subsection 1\n-- Subsection 2'
    assert sample_section.as_str == expected

def test_outline_as_str(sample_outline):
    """Test PodcastOutline string representation"""
    expected = 'Section 1\n-- Section 1.1\n-- Section 1.2\nSection 2\n-- Section 2.1'
    assert sample_outline.as_str == expected

def test_format_wikipedia_document():
    """Test Wikipedia document formatting"""
    mock_doc = Mock()
    mock_doc.metadata = {'title': 'Test Article'}
    mock_doc.page_content = 'Test content'
    
    expected = '### Test Article\n\nTest content'
    assert format_wikipedia_document(mock_doc) == expected
