from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_clone_check.config import ExperimentConfig, load_config
from voice_clone_check.service import ExperimentService
from voice_clone_check.reports import ReportBuilder


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


class WorkflowBackends(FakeBackends):
    source_transcript = "元音声の正確な台本です。"
    reject_seed = 11

    def synthesize(
        self, text, ref_audio, ref_text, seed, output_path, max_tokens=2048
    ):
        super().synthesize(text, ref_audio, ref_text, seed, output_path, max_tokens)
        self.transcripts[str(output_path)] = "誤った読み" if seed == self.reject_seed else text

    def transcribe(self, audio_path):
        return self.transcripts.get(str(audio_path), self.source_transcript)


def workflow_config() -> ExperimentConfig:
    original = load_config()
    raw = copy.deepcopy(original.raw)
    raw["candidate_prompts"] = raw["candidate_prompts"][:3]
    raw["evaluation_prompts"] = raw["evaluation_prompts"][:1]
    raw["generation"].update(seeds=[21, 22], takes_per_candidate=2)
    raw["candidate_generation"].update(
        seeds=[11, 12, 13, 14],
        takes_per_candidate=2,
        max_attempts=4,
        source_min_seconds=0.5,
        source_max_seconds=10.0,
    )
    raw["quality"].update(
        min_seconds=0.5,
        min_snr_db=0.0,
        max_silence_ratio=1.0,
    )
    from voice_clone_check.config import config_from_raw

    return config_from_raw(raw)


def source_audio(path: Path, frequency: float = 220.0) -> None:
    sample_rate = 24000
    time = np.arange(sample_rate * 3) / sample_rate
    sf.write(path, 0.2 * np.sin(2 * np.pi * frequency * time), sample_rate)


def test_synthetic_screening_and_validation_workflow(tmp_path: Path):
    config = workflow_config()
    WorkflowBackends.transcripts = {}
    service = ExperimentService(
        root=tmp_path / "experiments",
        config=config,
        backends_factory=WorkflowBackends,
    )
    source = tmp_path / "source.wav"
    source_audio(source)
    screening_id = service.create_experiment("screening", mode="synthetic")

    source_result = service.save_source(
        screening_id, str(source), WorkflowBackends.source_transcript
    )
    references = service.generate_candidate_references(screening_id)
    resumed = service.generate_candidate_references(screening_id)
    evaluation = service.run(screening_id)

    assert source_result["quality_ok"]
    assert source_result["cer"] == 0
    assert references == {"accepted": 6, "rejected": 3, "incomplete": 0}
    assert resumed == {"accepted": 0, "rejected": 0, "incomplete": 0}
    assert evaluation == {"completed": 12, "failed": 0, "remaining": 0}
    assert len(service.db.recordings(screening_id, "candidate")) == 6

    validation_id = service.create_validation_experiment(screening_id)
    validation_config = service.experiment_config(validation_id)
    assert len(validation_config.candidates) == 3
    assert service.db.experiment(validation_id)["parent_experiment_id"] == screening_id

    for candidate in validation_config.candidates:
        for take in (1, 2):
            service.save_recording(
                validation_id,
                "candidate",
                candidate.id,
                take,
                candidate.text,
                str(source),
            )
    validation_result = service.run(validation_id)
    assert validation_result == {"completed": 12, "failed": 0, "remaining": 0}

    paths = ReportBuilder(
        service.db,
        validation_config,
        service.experiment_dir(validation_id),
    ).build(validation_id)
    assert "合成事前選定との順位比較" in paths["html"].read_text(encoding="utf-8")


def test_changing_source_invalidates_synthetic_dependents(tmp_path: Path):
    WorkflowBackends.transcripts = {}
    service = ExperimentService(
        root=tmp_path / "experiments",
        config=workflow_config(),
        backends_factory=WorkflowBackends,
    )
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    source_audio(first, 220.0)
    source_audio(second, 330.0)
    experiment_id = service.create_experiment("invalidate", mode="synthetic")
    service.save_source(experiment_id, str(first), WorkflowBackends.source_transcript)
    service.generate_candidate_references(experiment_id, smoke=True)
    service.run(experiment_id, smoke=True)

    service.save_source(experiment_id, str(second), WorkflowBackends.source_transcript)

    assert service.db.recordings(experiment_id, "candidate") == []
    assert service.db.generations(experiment_id) == []


def test_source_transcript_mismatch_blocks_reference_generation(tmp_path: Path):
    WorkflowBackends.transcripts = {}
    service = ExperimentService(
        root=tmp_path / "experiments",
        config=workflow_config(),
        backends_factory=WorkflowBackends,
    )
    source = tmp_path / "source.wav"
    source_audio(source)
    experiment_id = service.create_experiment("mismatch", mode="synthetic")

    result = service.save_source(
        experiment_id,
        str(source),
        "台本とはまったく異なる文章です。",
    )

    assert not result["quality_ok"]
    assert result["cer"] > 0.10
    with pytest.raises(RuntimeError, match="確認済みの元音声"):
        service.generate_candidate_references(experiment_id, smoke=True)
