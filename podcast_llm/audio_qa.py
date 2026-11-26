"""Check a chapter recording against the dialogue in a saved script."""

import argparse
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
import re
from unicodedata import normalize

from openai import OpenAI, OpenAIError
from pydub import AudioSegment

from podcast_llm.config import PodcastConfig
from podcast_llm.render import ScriptChapter, read_chapter
from podcast_llm.text_to_speech import clean_text_for_tts


WORD = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|[^\W_]+(?:[.'’][^\W_]+)*")
NUMBER_WORDS = {
    word: str(number) for number, word in enumerate(
        'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
        'sixteen seventeen eighteen nineteen twenty'.split()
    )
}
FUNCTION_WORDS = frozenset('a an and but for i in is it of on or so that the then this to we with you'.split())
MAX_UPLOAD_BYTES = 25_000_000
TRANSCRIPTION_CHUNK_MS = 10 * 60 * 1000


def words(text: str) -> list[str]:
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    return WORD.findall(normalize('NFKC', text))


def normalized_word(word: str) -> str:
    cleaned = word.casefold().replace('’', "'").replace(',', '')
    return NUMBER_WORDS.get(cleaned, cleaned)


def compare_transcript(chapter: ScriptChapter, transcript: str) -> list[str]:
    """Report likely omissions and changes by aligning spoken and scripted words."""
    turns = clean_text_for_tts(chapter.turns)
    expected = []
    turn_spans = []
    for turn in turns:
        start = len(expected)
        expected.extend(words(turn['text']))
        turn_spans.append((start, len(expected)))
    heard = words(transcript)
    matcher = SequenceMatcher(a=[normalized_word(word) for word in expected],
                              b=[normalized_word(word) for word in heard], autojunk=False)
    changes = [edit for edit in matcher.get_opcodes() if edit[0] in {'delete', 'replace'}]

    findings = []
    for number, (turn_start, turn_end) in enumerate(turn_spans, 1):
        speaker = turns[number - 1]['speaker']
        for tag, start, end, heard_start, heard_end in changes:
            first, last = max(start, turn_start), min(end, turn_end)
            if first >= last:
                continue
            changed = expected[first:last]
            whole_turn = (first, last) == (turn_start, turn_end)
            key_term = any(word[0].isdigit() or (word[0].isupper() and word.casefold() not in FUNCTION_WORDS)
                           or word.casefold() in NUMBER_WORDS for word in changed)
            if not (whole_turn or key_term or len(changed) >= 3):
                continue
            expected_text = ' '.join(changed)
            if tag == 'delete':
                findings.append(f'Turn {number} ({speaker}) — possibly missing: {expected_text}')
            else:
                heard_text = ' '.join(heard[heard_start:heard_end])
                findings.append(f'Turn {number} ({speaker}) — expected “{expected_text}”, heard “{heard_text}”')
    return findings


def transcribe_audio(audio: Path, config: PodcastConfig) -> str:
    if not config.openai_api_key:
        raise ValueError('Set OPENAI_API_KEY to check chapter audio.')
    try:
        with OpenAI(api_key=config.openai_api_key, timeout=120) as client:
            if audio.stat().st_size <= MAX_UPLOAD_BYTES:
                return client.audio.transcriptions.create(file=audio, model=config.audio_qa_model).text

            # Keep long chapter previews under the upload limit without requiring an MP3 export.
            signal = AudioSegment.from_file(audio).set_channels(1).set_frame_rate(16000).set_sample_width(2)
            transcripts = []
            for chunk in signal[::TRANSCRIPTION_CHUNK_MS]:
                with BytesIO() as buffer:
                    chunk.export(buffer, format='wav')
                    result = client.audio.transcriptions.create(
                        file=('chapter.wav', buffer.getvalue()), model=config.audio_qa_model,
                    )
                    transcripts.append(result.text)
            return '\n'.join(transcripts)
    except OpenAIError as error:
        raise ValueError(f'Audio transcription failed ({type(error).__name__}).') from error


def check_chapter_audio(chapter: ScriptChapter, audio: Path, config: PodcastConfig) -> list[str]:
    return compare_transcript(chapter, transcribe_audio(audio, config))


def format_audio_check(chapter: ScriptChapter, findings: list[str]) -> str:
    header = f'Audio check for {chapter.title}'
    if not findings:
        return f'{header}\nNo likely omissions or changes detected.'
    return '\n'.join([header, *(f'- {finding}' for finding in findings)])


def check_cli(argv: list[str], default_config: str) -> None:
    parser = argparse.ArgumentParser(prog='podcast-llm check', description='Check rendered chapter audio')
    parser.add_argument('script', type=Path, help='Saved Markdown transcript')
    parser.add_argument('--chapter', type=int, required=True, metavar='NUMBER', help='Chapter number (starting at 1)')
    parser.add_argument('--audio', type=Path, help='Chapter audio file (defaults to the chapter preview WAV)')
    parser.add_argument('--config', default=default_config, help='Path to YAML configuration')
    args = parser.parse_args(argv)
    try:
        chapter = read_chapter(args.script, args.chapter)
        audio = args.audio or args.script.with_suffix(f'.chapter-{args.chapter}.wav')
        config = PodcastConfig.load(yaml_path=args.config, require_research_credentials=False)
        findings = check_chapter_audio(chapter, audio, config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(format_audio_check(chapter, findings))
    if findings:
        parser.exit(1)
