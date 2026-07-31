from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import soundfile as sf

from voice_clone_check.config import ExperimentConfig, load_config
from voice_clone_check.service import ExperimentService


class FakeBackends:
    transcripts: dict[str, str] = {}

    def __init__(self, **kwargs):
        self.sample_rate = kwargs["sample_rate"]

    def synthesize(
        self, text, ref_audio, ref_text, seed, output_path, max_tokens=2048
    ):
        time = np.arange(self.sample_rate) / self.sample_rate
        audio = 0.2 * np.sin(2 * np.pi * (200 + seed % 20) * time)
        sf.write(output_path, audio, self.sample_rate)
        self.transcripts[str(output_path)] = text

    def speaker_embedding(self, audio_path):
        return np.array([1.0, 0.1], dtype=np.float32)

    def utmos(self, audio_path):
        return 4.1

    def transcribe(self, audio_path):
        return self.transcripts[str(audio_path)]

    def release(self):
        pass


def test_smoke_run_is_resumable(tmp_path: Path):
    original = load_config()
    raw = copy.deepcopy(original.raw)
    raw["quality"].update(
        min_seconds=0.5,
        min_snr_db=0.0,
        max_silence_ratio=1.0,
    )
    config = ExperimentConfig(
        raw=raw,
        candidates=original.candidates,
        anchors=original.anchors,
        evaluations=original.evaluations,
    )
    service = ExperimentService(
        root=tmp_path / "experiments",
        config=config,
        backends_factory=FakeBackends,
    )
    experiment_id = service.create_experiment("test")
    sample_rate = 24000
    time = np.arange(sample_rate * 3) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 220 * time)
    source = tmp_path / "voice.wav"
    sf.write(source, audio, sample_rate)

    candidate = config.candidates[0]
    anchor = config.anchors[0]
    service.save_recording(
        experiment_id, "candidate", candidate.id, 1, candidate.text, str(source)
    )
    service.save_recording(
        experiment_id, "anchor", anchor.id, 1, anchor.text, str(source)
    )

    first = service.run(experiment_id, smoke=True)
    second = service.run(experiment_id, smoke=True)
    rows = service.db.generations(experiment_id, complete_only=True)

    assert first == {"completed": 1, "failed": 0, "remaining": 0}
    assert second == {"completed": 0, "failed": 0, "remaining": 0}
    assert len(rows) == 1
    assert rows[0]["cer"] == 0

