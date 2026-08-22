from pathlib import Path

from voice_clone_check.db import now_iso
from voice_clone_check.reports import ReportBuilder
from voice_clone_check.service import ExperimentService


def test_report_exports_all_formats_without_absolute_audio_paths(tmp_path: Path):
    service = ExperimentService(root=tmp_path / "experiments")
    experiment_id = service.create_experiment("report test")
    experiment_dir = service.experiment_dir(experiment_id)
    recording_id = service.db.upsert_recording(
        {
            "experiment_id": experiment_id,
            "kind": "candidate",
            "prompt_id": "c01",
            "take": 1,
            "text": service.config.candidates[0].text,
            "raw_path": str(experiment_dir / "recordings/raw/c01.wav"),
            "processed_path": str(experiment_dir / "recordings/processed/c01.wav"),
            "sha256": "0" * 64,
            "duration": 4.0,
            "rms_dbfs": -18.0,
            "peak_dbfs": -3.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 0.1,
            "snr_db": 30.0,
            "quality_ok": 1,
            "warnings_json": "[]",
            "created_at": now_iso(),
        }
    )
    service.db.prepare_generations(
        experiment_id,
        service.db.recordings(experiment_id, "candidate"),
        [{"id": "e01", "text": service.config.evaluations[0].text}],
        [3407],
        service.config.tts_model,
        experiment_dir / "generated",
    )
    generation = service.db.generations(experiment_id)[0]
    service.db.complete_generation(
        generation["id"],
        {
            "elapsed_seconds": 1.0,
            "output_sha256": "1" * 64,
            "similarity": 0.85,
            "utmos": 4.0,
            "cer": 0.0,
            "transcript": service.config.evaluations[0].text,
            "failed": 0,
        },
    )

    paths = ReportBuilder(
        service.db, service.config, experiment_dir
    ).build(experiment_id)

    assert set(paths) == {"csv", "json", "html"}
    assert all(path.exists() for path in paths.values())
    assert str(tmp_path) not in paths["json"].read_text(encoding="utf-8")
    report_html = paths["html"].read_text(encoding="utf-8")
    assert "c01" in report_html
    assert "width:calc(100% - 32px)" in report_html
    assert "width:min(1100px" not in report_html


def test_duration_report_contains_decision_and_confidence_intervals(tmp_path: Path):
    service = ExperimentService(root=tmp_path / "experiments")
    experiment_id = service.create_experiment("duration report", mode="duration")
    config = service.experiment_config(experiment_id)
    experiment_dir = service.experiment_dir(experiment_id)
    for target in config.raw["duration_study"]["targets"]:
        for take in (1, 2, 3):
            recording_id = service.db.upsert_recording({
                "experiment_id": experiment_id,
                "kind": "candidate",
                "prompt_id": target["id"],
                "take": take,
                "text": "累積台本",
                "raw_path": str(experiment_dir / f"{target['id']}-{take}-raw.wav"),
                "processed_path": str(experiment_dir / f"{target['id']}-{take}.wav"),
                "sha256": str(take) * 64,
                "duration": target["seconds"],
                "rms_dbfs": -18.0,
                "peak_dbfs": -3.0,
                "clipping_ratio": 0.0,
                "silence_ratio": 0.1,
                "snr_db": 30.0,
                "quality_ok": 1,
                "warnings_json": "[]",
                "origin": "duration",
                "created_at": now_iso(),
            })
            service.db.prepare_generations(
                experiment_id,
                [service.db.recording(recording_id)],
                [{"id": "e01", "text": "評価文"}],
                [1],
                config.tts_model,
                experiment_dir / "generated",
            )
    for generation in service.db.generations(experiment_id):
        service.db.complete_generation(generation["id"], {
            "elapsed_seconds": 1.0,
            "output_sha256": "1" * 64,
            "similarity": 0.90,
            "utmos": 4.0,
            "cer": 0.01,
            "transcript": "評価文",
            "failed": 0,
        })

    paths = ReportBuilder(service.db, config, experiment_dir).build(experiment_id)
    html = paths["html"].read_text(encoding="utf-8")
    assert "入力音声長の比較" in html
    assert "自動評価上の候補: 約4秒" in html
    assert "15秒条件との差と95%信頼区間" in html
