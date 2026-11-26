from pathlib import Path
from shutil import which
from subprocess import CalledProcessError

from pydub import AudioSegment
from pydub.generators import Sine
import pytest

from podcast_llm import mastering


TARGETS = mastering.MasteringTargets(-16, -1)
BALANCED = mastering.AudioLevels(-20, -4, 3, -30, 0)


def test_mastering_requires_valid_targets_and_ffmpeg(monkeypatch):
    with pytest.raises(ValueError, match='between -70 and -5 LUFS'):
        mastering.MasteringTargets(-2, -1)
    with pytest.raises(ValueError, match='between -9 and 0 dBTP'):
        mastering.MasteringTargets(-16, 1)
    monkeypatch.setattr(mastering, 'which', lambda name: None)
    with pytest.raises(ValueError, match='FFmpeg on PATH'):
        mastering.require_ffmpeg()


def test_loudness_measurement_parses_ffmpeg_output(tmp_path, mocker):
    source = tmp_path / 'episode.mp3'
    ffmpeg = mocker.patch.object(mastering, 'run')
    ffmpeg.return_value.stderr = ('FFmpeg diagnostics\n{ "input_i": "-16.50", "input_tp": "-1.20",'
                                  ' "input_lra": "3.00", "input_thresh": "-27.00", '
                                  '"target_offset": "0.10" }')

    assert mastering.measure_audio(source, 'ffmpeg') == (-16.5, -1.2, 3, -27, 0.1)
    assert '-f' in ffmpeg.call_args.args[0] and source in ffmpeg.call_args.args[0]

    ffmpeg.return_value.stderr = 'No loudness report'
    with pytest.raises(ValueError, match='Could not read FFmpeg loudness measurements'):
        mastering.measure_audio(source, 'ffmpeg')
    ffmpeg.side_effect = CalledProcessError(1, 'ffmpeg', stderr='Unknown encoder')
    with pytest.raises(ValueError, match='Unknown encoder'):
        mastering.measure_audio(source, 'ffmpeg')


@pytest.mark.parametrize('peak_after_retry', [-1.2, -0.8])
def test_encoder_retries_peak_overshoot_and_keeps_previous_output_on_failure(tmp_path, mocker, peak_after_retry):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    output = tmp_path / 'episode.mp3'
    output.write_bytes(b'previous export')
    mocker.patch.object(mastering, '_balance_chapters', return_value=AudioSegment.silent(duration=1000))
    encode = mocker.patch.object(mastering, '_run_ffmpeg',
                                 side_effect=lambda *args: Path(args[-1]).write_bytes(b'new encode'))
    mocker.patch.object(mastering, 'measure_audio', side_effect=[
        BALANCED, BALANCED._replace(integrated=-16, true_peak=-0.7),
        BALANCED, BALANCED._replace(integrated=-16, true_peak=peak_after_retry),
    ])

    if peak_after_retry > -1:
        with pytest.raises(ValueError, match='above the -1.00 dBTP ceiling'):
            mastering.master_episode([['chapter.wav']], output, work_dir, 'mp3', TARGETS, 'ffmpeg')
        assert output.read_bytes() == b'previous export'
    else:
        final = mastering.master_episode([['chapter.wav']], output, work_dir, 'mp3', TARGETS, 'ffmpeg')
        assert final.true_peak == peak_after_retry
        assert output.read_bytes() == b'new encode'
    assert 'TP=-1.5:' in encode.call_args_list[0].args[4]
    assert 'TP=-2.0:' in encode.call_args_list[1].args[4]
    assert list(tmp_path.glob('*.mp3')) == [output]


@pytest.mark.skipif(which('ffmpeg') is None, reason='FFmpeg is required for mastering integration')
@pytest.mark.parametrize('extension', ['wav', 'mp3'])
def test_mastered_file_balances_chapters_and_meets_targets(tmp_path, extension):
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    chapter_files = []
    for index, db in enumerate((-27, -13)):
        path = tmp_path / f'chapter-{index}.wav'
        Sine(440).to_audio_segment(duration=3000, volume=db).export(path, format='wav').close()
        chapter_files.append([str(path)])
    output = tmp_path / f'episode.{extension}'

    final = mastering.master_episode(chapter_files, output, work_dir, extension,
                                     TARGETS, mastering.require_ffmpeg())

    assert final.integrated == pytest.approx(-16, abs=1)
    assert final.true_peak <= -1
    decoded = AudioSegment.from_file(output)
    assert len(decoded) == pytest.approx(6000, abs=1)
    assert decoded[:3000].dBFS == pytest.approx(decoded[3000:].dBFS, abs=0.2)
