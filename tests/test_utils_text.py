import pytest
from podcast_llm.utils.text import generate_markdown_script
from podcast_llm.models import (
    PodcastOutline, 
    PodcastSection, 
    PodcastSubsection
)

# Test fixtures
@pytest.fixture
def sample_topic():
    return "Understanding Artificial Intelligence"

@pytest.fixture
def sample_outline():
    sections = [
        PodcastSection(
            title="Introduction to AI",
            subsections=[
                PodcastSubsection(title="What is artificial intelligence?"),
                PodcastSubsection(title="Brief history of AI development")
            ]
        ),
        PodcastSection(
            title="Types of AI",
            subsections=[
                PodcastSubsection(title="Narrow AI vs General AI"),
                PodcastSubsection(title="Machine Learning fundamentals")
            ]
        )
    ]
    return PodcastOutline(sections=sections)

@pytest.fixture
def sample_script():
    return [
        {'speaker': 'Interviewer', 'text': 'Welcome to our podcast on AI!'},
        {'speaker': 'Interviewee', 'text': 'Thanks for having me.'},
        {'speaker': 'Interviewer', 'text': "Let's start with the basics."}
    ]

def test_generate_markdown_script_basic(sample_topic, sample_outline, sample_script):
    """Test basic markdown generation with typical inputs"""
    markdown = generate_markdown_script(sample_topic, sample_outline, sample_script)
    
    # Check title
    assert f'# {sample_topic}\n' in markdown
    
    # Check outline sections
    assert '## Outline\n' in markdown
    assert '### Section 1: Introduction to AI\n' in markdown
    assert '### Section 2: Types of AI\n' in markdown
    
    # Check subsections
    assert '- What is artificial intelligence?' in markdown
    assert '- Brief history of AI development' in markdown
    assert '- Narrow AI vs General AI' in markdown
    assert '- Machine Learning fundamentals' in markdown
    
    # Check script formatting
    assert '## Script\n' in markdown
    assert '**Interviewer**: Welcome to our podcast on AI!' in markdown
    assert '**Interviewee**: Thanks for having me.' in markdown

def test_generate_markdown_script_empty_outline():
    """Test markdown generation with empty outline"""
    empty_outline = PodcastOutline(sections=[])
    markdown = generate_markdown_script(
        "Empty Topic",
        empty_outline,
        [{'speaker': 'Interviewer', 'text': 'Hello'}]
    )
    
    assert '# Empty Topic\n' in markdown
    assert '## Outline\n' in markdown
    assert '## Script\n' in markdown
    assert '**Interviewer**: Hello' in markdown

def test_generate_markdown_script_empty_script(sample_topic, sample_outline):
    """Test markdown generation with empty script"""
    markdown = generate_markdown_script(sample_topic, sample_outline, [])
    
    assert f'# {sample_topic}\n' in markdown
    assert '## Outline\n' in markdown
    assert '## Script\n' in markdown
    assert 'Introduction to AI' in markdown
    
def test_generate_markdown_script_special_characters():
    """Test markdown generation with special characters in text"""
    outline = PodcastOutline(sections=[
        PodcastSection(
            title="Special *chars* & symbols",
            subsections=[
                PodcastSubsection(title="Using **bold** & _italic_"),
            ]
        )
    ])
    
    script = [
        {'speaker': 'Interviewer', 'text': 'Testing *markdown* symbols!'},
        {'speaker': 'Interviewee', 'text': 'Using # and ## characters'}
    ]
    
    markdown = generate_markdown_script("Special Topic", outline, script)
    
    # Verify special characters are preserved
    assert 'Special *chars* & symbols' in markdown
    assert 'Using **bold** & _italic_' in markdown
    assert 'Testing *markdown* symbols!' in markdown
    assert 'Using # and ## characters' in markdown

def test_generate_markdown_script_long_content(sample_topic):
    """Test markdown generation with long content"""
    # Create outline with many sections
    sections = [
        PodcastSection(
            title=f"Section {i}",
            subsections=[
                PodcastSubsection(title=f"Subsection {i}.{j}")
                for j in range(5)
            ]
        )
        for i in range(10)
    ]
    long_outline = PodcastOutline(sections=sections)
    
    # Create long script
    long_script = [
        {'speaker': 'Interviewer' if i % 2 == 0 else 'Interviewee',
         'text': f'Line {i} of the conversation'}
        for i in range(50)
    ]
    
    markdown = generate_markdown_script(sample_topic, long_outline, long_script)
    
    # Verify structure is maintained
    assert f'# {sample_topic}\n' in markdown
    assert 'Section 0\n' in markdown
    assert 'Section 9\n' in markdown
    assert 'Subsection 0.0' in markdown
    assert 'Subsection 9.4' in markdown
    assert '**Interviewer**: Line 0' in markdown
    assert '**Interviewee**: Line 49' in markdown

def test_generate_markdown_script_multiline_text():
    """Test markdown generation with multiline text in script"""
    outline = PodcastOutline(sections=[
        PodcastSection(
            title="Multiline Test",
            subsections=[
                PodcastSubsection(title="Test subsection"),
            ]
        )
    ])
    
    script = [
        {'speaker': 'Interviewer', 
         'text': 'This is a\nmultiline\nquestion?'},
        {'speaker': 'Interviewee', 
         'text': 'Here is a\nmultiline\nanswer.'}
    ]
    
    markdown = generate_markdown_script("Multiline Topic", outline, script)
    
    # Verify multiline text is properly formatted
    assert 'This is a\nmultiline\nquestion?' in markdown
    assert 'Here is a\nmultiline\nanswer.' in markdown 