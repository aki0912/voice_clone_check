from pathlib import Path

import numpy as np
import soundfile as sf

from voice_clone_check.audio import prepare_recording, valid_audio_file


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

