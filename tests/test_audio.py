from pathlib import Path

import numpy as np
import soundfile as sf

from voice_clone_check.audio import (
    prepare_recording,
    suggest_cumulative_boundaries,
    valid_audio_file,
    write_cumulative_clips,
)


def test_prepare_recording_resamples_and_trims(tmp_path: Path):
    sample_rate = 16000
    silence = np.zeros(sample_rate // 2, dtype=np.float32)
    time = np.arange(sample_rate * 3) / sample_rate
    speech = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, np.concatenate([silence, speech, silence]), sample_rate)

    raw = tmp_path / "raw.wav"
    processed = tmp_path / "processed.wav"
    quality, digest = prepare_recording(
        source,
        raw,
        processed,
        24000,
        {
            "min_seconds": 2.0,
            "max_seconds": 10.0,
            "max_clipping_ratio": 0.01,
            "min_snr_db": 0.0,
            "max_silence_ratio": 0.8,
        },
    )

    assert raw.exists()
    assert valid_audio_file(processed)
    assert sf.info(processed).samplerate == 24000
    assert quality.duration < 4.0
    assert len(digest) == 64


def test_duration_boundaries_and_nested_clips_share_the_same_start(tmp_path: Path):
    sample_rate = 24000
    parts = []
    for frequency in (180, 220, 260, 300):
        time = np.arange(sample_rate * 3) / sample_rate
        parts.extend([
            0.2 * np.sin(2 * np.pi * frequency * time),
            np.zeros(sample_rate // 4),
        ])
    source = tmp_path / "passage.wav"
    sf.write(source, np.concatenate(parts), sample_rate)

    boundaries, duration = suggest_cumulative_boundaries(
        source, [3.2, 6.5, 9.7, 13.0], sample_rate=sample_rate
    )
    outputs = [tmp_path / f"clip-{index}.wav" for index in range(4)]
    write_cumulative_clips(source, boundaries, outputs, sample_rate=sample_rate)

    assert duration > 12
    assert boundaries == sorted(boundaries)
    clips = [sf.read(path)[0] for path in outputs]
    assert all(len(left) < len(right) for left, right in zip(clips, clips[1:]))
    assert np.allclose(clips[0], clips[-1][: len(clips[0])], atol=1e-4)
