import pytest
from unittest.mock import Mock, patch, mock_open
import os
from pathlib import Path
from pydub import AudioSegment
from google.cloud import texttospeech

from podcast_llm.text_to_speech import (
    clean_text_for_tts,
    merge_audio_files,
    process_line_google,
    generate_audio,
    rate_limit_per_minute,
    combine_consecutive_speaker_chunks
)

# Test data
SAMPLE_LINES = [
    {'speaker': 'Interviewer', 'text': 'Hello *world* with _emphasis_ and—dash'},
    {'speaker': 'Interviewee', 'text': 'Hi there *friend*!'}
]

CLEANED_LINES = [
    {'speaker': 'Interviewer', 'text': 'Hello world with emphasis anddash'},
    {'speaker': 'Interviewee', 'text': 'Hi there friend!'}
]

@pytest.fixture
def mock_audio_segment():
    with patch('podcast_llm.text_to_speech.AudioSegment') as mock:
        mock.empty.return_value = Mock()
        mock.from_file.return_value = Mock()
        yield mock

@pytest.fixture
def mock_tts_client():
    with patch('podcast_llm.text_to_speech.texttospeech.TextToSpeechClient') as mock:
        mock_instance = Mock()
        mock_instance.synthesize_speech.return_value = Mock(audio_content=b'fake_audio')
        mock.return_value = mock_instance
        yield mock_instance


def test_clean_text_for_tts():
    """Test that special characters are properly removed from text"""
    result = clean_text_for_tts(SAMPLE_LINES)
    assert result == CLEANED_LINES

def test_combine_consecutive_speaker_chunks():
    # Test case 1: Empty list
    assert combine_consecutive_speaker_chunks([]) == []
    
    # Test case 2: Single chunk
    single_chunk = [{'speaker': 'Alice', 'text': 'Hello'}]
    assert combine_consecutive_speaker_chunks(single_chunk) == single_chunk
    
    # Test case 3: Different speakers alternating
    different_speakers = [
        {'speaker': 'Alice', 'text': 'Hello'},
        {'speaker': 'Bob', 'text': 'Hi'},
        {'speaker': 'Alice', 'text': 'How are you?'}
    ]
    assert combine_consecutive_speaker_chunks(different_speakers) == different_speakers
    
    # Test case 4: Same speaker consecutive chunks
    same_speaker = [
        {'speaker': 'Alice', 'text': 'Hello'},
        {'speaker': 'Alice', 'text': 'How are you?'},
        {'speaker': 'Bob', 'text': 'Hi'},
        {'speaker': 'Bob', 'text': 'I am good'}
    ]
    expected = [
        {'speaker': 'Alice', 'text': 'Hello How are you?'},
        {'speaker': 'Bob', 'text': 'Hi I am good'}
    ]
    assert combine_consecutive_speaker_chunks(same_speaker) == expected
    
    # Test case 5: Mixed consecutive and non-consecutive
    mixed_chunks = [
        {'speaker': 'Alice', 'text': 'First'},
        {'speaker': 'Alice', 'text': 'Second'},
        {'speaker': 'Bob', 'text': 'Response'},
        {'speaker': 'Alice', 'text': 'Third'}
    ]
    expected_mixed = [
        {'speaker': 'Alice', 'text': 'First Second'},
        {'speaker': 'Bob', 'text': 'Response'},
        {'speaker': 'Alice', 'text': 'Third'}
    ]
    assert combine_consecutive_speaker_chunks(mixed_chunks) == expected_mixed

def test_combine_consecutive_speaker_chunks_preserves_input():
    """Test that the original input is not modified"""
    original = [
        {'speaker': 'Alice', 'text': 'Hello'},
        {'speaker': 'Alice', 'text': 'World'}
    ]
    original_copy = [chunk.copy() for chunk in original]
    
    combine_consecutive_speaker_chunks(original)
    assert original == original_copy

def test_combine_consecutive_speaker_chunks_empty_text():
    """Test handling of empty text fields"""
    chunks = [
        {'speaker': 'Alice', 'text': ''},
        {'speaker': 'Alice', 'text': 'Hello'},
        {'speaker': 'Bob', 'text': ''},
        {'speaker': 'Bob', 'text': ''}
    ]
    expected = [
        {'speaker': 'Alice', 'text': ' Hello'},
        {'speaker': 'Bob', 'text': ' '}
    ]
    assert combine_consecutive_speaker_chunks(chunks) == expected

