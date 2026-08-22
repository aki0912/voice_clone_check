from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .audio import (
    inspect_audio,
    prepare_recording,
    sha256_file,
    suggest_cumulative_boundaries,
    valid_audio_file,
    write_cumulative_clips,
)
from .backends import ModelBackends, cosine_similarity
from .config import ExperimentConfig, config_from_raw, load_config
from .db import Database, now_iso
from .paths import experiments_root
from .text_metrics import character_error_rate


ProgressCallback = Callable[[int, int, str], None]


class ExperimentService:
    def __init__(
        self,
        root: Path | None = None,
        config: ExperimentConfig | None = None,
        backends_factory: Callable[..., ModelBackends] = ModelBackends,
    ):
        self.root = root or experiments_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or load_config()
        self.db = Database(self.root / "experiments.sqlite3")
        self.backends_factory = backends_factory
        self._run_lock = threading.Lock()

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def create_experiment(
        self,
        name: str | None = None,
        mode: str = "recorded",
        parent_experiment_id: str | None = None,
        candidate_ids: list[str] | None = None,
    ) -> str:
        if mode not in {"recorded", "synthetic", "validation", "duration"}:
            raise ValueError("実験モードが不正です")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_id = f"{timestamp}-{uuid.uuid4().hex[:6]}"
        default_label = {
            "recorded": "日本語参照セリフ探索",
            "synthetic": "合成音声による事前選定",
            "validation": "上位候補の実録音検証",
            "duration": "入力音声長調査",
        }[mode]
        display_name = (name or "").strip() or f"{default_label} {timestamp}"
        directory = self.experiment_dir(experiment_id)
        for child in (
            "recordings/raw", "recordings/processed", "reference_generated",
            "generated", "reports",
        ):
            (directory / child).mkdir(parents=True, exist_ok=True)
        raw_config = copy.deepcopy(self.config.raw)
        if mode == "duration":
            study = raw_config["duration_study"]
            raw_config["candidate_prompts"] = [
                {
                    "id": str(target["id"]),
                    "category": "duration",
                    "text": f"約{float(target['seconds']):g}秒の累積参照音声",
                }
                for target in study["targets"]
            ]
            raw_config["generation"]["takes_per_candidate"] = len(study["passages"])
        if candidate_ids is not None:
            selected = set(candidate_ids)
            raw_config["candidate_prompts"] = [
                item for item in raw_config["candidate_prompts"]
                if str(item["id"]) in selected
            ]
            if len(raw_config["candidate_prompts"]) != len(selected):
                raise ValueError("候補IDに不明な値が含まれています")
        config_copy = directory / "config.yaml"
        import yaml

        config_copy.write_text(
            yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.db.create_experiment(
            experiment_id,
            display_name,
            raw_config,
            mode=mode,
            parent_experiment_id=parent_experiment_id,
        )
        return experiment_id

    def eligible_duration_anchor_experiments(self) -> list[dict[str, Any]]:
        eligible = []
        for experiment in self.db.list_experiments():
            sources = [
                row for row in self.db.recordings(experiment["id"], "anchor")
                if row["origin"] == "source" and row["quality_ok"]
            ]
            if len(sources) >= 3:
                eligible.append(dict(experiment))
        return eligible

    def create_duration_experiment(
        self, anchor_experiment_id: str | None = None, name: str | None = None
    ) -> str:
        if anchor_experiment_id is None:
            eligible = self.eligible_duration_anchor_experiments()
            if not eligible:
                raise RuntimeError("品質確認済み元音声が3本ある実験が必要です")
            anchor_experiment_id = str(eligible[0]["id"])
        anchors = [
            row for row in self.db.recordings(anchor_experiment_id, "anchor")
            if row["origin"] == "source" and row["quality_ok"]
        ][:3]
        if len(anchors) < 3:
            raise RuntimeError("独立アンカーとして利用できる元音声が3本必要です")
        experiment_id = self.create_experiment(
            name=name,
            mode="duration",
            parent_experiment_id=anchor_experiment_id,
        )
        for index, anchor in enumerate(anchors, start=1):
            self.save_recording(
                experiment_id,
                "anchor",
                f"source{index:02d}",
                1,
                anchor["text"],
                anchor["processed_path"],
                origin="source",
                transcript=anchor["transcript"],
                cer=anchor["cer"],
            )
        return experiment_id

    def duration_study_setup(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.db.experiment(experiment_id)
        if not experiment or experiment["mode"] != "duration":
            raise ValueError("入力音声長調査の実験を選択してください")
        return copy.deepcopy(self.experiment_config(experiment_id).raw["duration_study"])

    def suggest_duration_boundaries(
        self, experiment_id: str, source_path: str
    ) -> dict[str, Any]:
        settings = self.duration_study_setup(experiment_id)
        targets = [float(item["seconds"]) for item in settings["targets"]]
        boundaries, duration = suggest_cumulative_boundaries(
            source_path,
            targets,
            sample_rate=self.experiment_config(experiment_id).sample_rate,
            search_seconds=float(settings.get("boundary_search_seconds", 1.25)),
        )
        return {"boundaries": boundaries, "duration": duration}

    def save_duration_passage(
        self,
        experiment_id: str,
        passage: int,
        source_path: str,
        boundaries: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if self._run_lock.locked():
            raise RuntimeError("評価の実行中は長さ調査音声を変更できません")
        config = self.experiment_config(experiment_id)
        settings = self.duration_study_setup(experiment_id)
        passages = list(settings["passages"])
        targets = list(settings["targets"])
        if passage < 1 or passage > len(passages):
            raise ValueError("調査台本番号が不正です")
        if boundaries is None:
            boundaries = self.suggest_duration_boundaries(
                experiment_id, source_path
            )["boundaries"]
        boundaries = [float(value) for value in boundaries]
        if len(boundaries) != len(targets):
            raise ValueError("4つの累積境界時刻を指定してください")
        clip_dir = self.experiment_dir(experiment_id) / "duration_inputs"
        destinations = [
            clip_dir / f"p{passage:02d}_{target['id']}.wav"
            for target in targets
        ]
        write_cumulative_clips(
            source_path,
            boundaries,
            destinations,
            sample_rate=config.sample_rate,
        )
        backend = self.backends_factory(
            tts_model_id=config.tts_model,
            asr_model_id=config.asr_model,
            speaker_model_id=config.raw["models"]["speaker"],
            utmos_repo=config.raw["models"]["utmos_repo"],
            sample_rate=config.sample_rate,
        )
        results = []
        segments = [str(value) for value in passages[passage - 1]["segments"]]
        try:
            for index, (target, clip_path) in enumerate(zip(targets, destinations)):
                text = "".join(segments[: index + 1])
                quality = dict(config.raw["quality"])
                quality.update(
                    min_seconds=float(target["min_seconds"]),
                    max_seconds=float(target["max_seconds"]),
                )
                result = self.save_recording(
                    experiment_id,
                    "candidate",
                    str(target["id"]),
                    passage,
                    text,
                    str(clip_path),
                    origin="duration",
                    quality_config=quality,
                )
                recording = self.db.recording(int(result["recording_id"]))
                assert recording is not None
                recognized = backend.transcribe(recording["processed_path"])
                cer = character_error_rate(text, recognized)
                warnings = list(result["warnings"])
                max_cer = float(settings.get("max_cer", 0.10))
                if cer > max_cer:
                    warnings.append(
                        f"台本と認識結果が一致しません（CER {cer:.3f} > {max_cer:.3f}）"
                    )
                quality_ok = bool(result["quality_ok"]) and cer <= max_cer
                self.db.update_recording_analysis(
                    int(recording["id"]),
                    transcript=recognized,
                    cer=cer,
                    quality_ok=quality_ok,
                    warnings=warnings,
                )
                results.append({
                    **result,
                    "prompt_id": str(target["id"]),
                    "take": passage,
                    "transcript": recognized,
                    "cer": cer,
                    "quality_ok": quality_ok,
                    "warnings": warnings,
                })
        finally:
            backend.release()
        return results

    def preview_duration_clips(
        self,
        experiment_id: str,
        source_path: str,
        boundaries: list[float],
    ) -> list[str]:
        config = self.experiment_config(experiment_id)
        self.duration_study_setup(experiment_id)
        preview_dir = self.experiment_dir(experiment_id) / "duration_preview"
        destinations = [preview_dir / f"preview_{index}.wav" for index in range(1, 5)]
        paths = write_cumulative_clips(
            source_path,
            [float(value) for value in boundaries],
            destinations,
            sample_rate=config.sample_rate,
        )
        return [str(path) for path in paths]

    def experiment_config(self, experiment_id: str) -> ExperimentConfig:
        experiment = self.db.experiment(experiment_id)
        if not experiment:
            raise ValueError("実験が見つかりません")
        return config_from_raw(json.loads(experiment["config_json"]))

    def source_recording(
        self, experiment_id: str, slot: int
    ) -> dict[str, Any] | None:
        if not self.db.experiment(experiment_id):
            raise ValueError("実験が見つかりません")
        if slot not in {1, 2, 3}:
            raise ValueError("元音声スロットは1〜3で指定してください")
        row = self.db.recording_slot(
            experiment_id, "anchor", f"source{slot:02d}", 1
        )
        if not row or row["origin"] != "source":
            return None
        return dict(row)

    def candidate_reference(
        self, experiment_id: str, prompt_id: str, take: int
    ) -> dict[str, Any] | None:
        config = self.experiment_config(experiment_id)
        if prompt_id not in {item.id for item in config.candidates}:
            raise ValueError("この実験に含まれない候補です")
        if take not in {1, 2}:
            raise ValueError("候補テイクは1または2で指定してください")
        row = self.db.recording_slot(
            experiment_id, "candidate", prompt_id, int(take)
        )
        if not row or row["origin"] != "synthetic":
            return None
        return dict(row)

    def candidate_references(self, experiment_id: str) -> list[dict[str, Any]]:
        self.experiment_config(experiment_id)
        return [
            dict(row)
            for row in self.db.recordings(experiment_id, "candidate")
            if row["origin"] == "synthetic"
        ]

    def save_recording(
        self,
        experiment_id: str,
        kind: str,
        prompt_id: str,
        take: int,
        text: str,
        source_path: str,
        *,
        origin: str = "recorded",
        model_id: str | None = None,
        generation_seed: int | None = None,
        source_sha256: str | None = None,
        transcript: str | None = None,
        cer: float | None = None,
        quality_config: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if not self.db.experiment(experiment_id):
            raise ValueError("実験が見つかりません")
        if kind not in {"candidate", "anchor"}:
            raise ValueError("録音種別が不正です")
        source = Path(source_path)
        old = self.db.recording_slot(experiment_id, kind, prompt_id, int(take))
        extension = source.suffix.lower() or ".wav"
        base = f"{kind}_{prompt_id}_t{take}"
        directory = self.experiment_dir(experiment_id)
        raw_path = directory / "recordings" / "raw" / f"{base}{extension}"
        processed_path = directory / "recordings" / "processed" / f"{base}.wav"
        quality, digest = prepare_recording(
            source,
            raw_path,
            processed_path,
            self.experiment_config(experiment_id).sample_rate,
            quality_config or self.experiment_config(experiment_id).raw["quality"],
        )
        values = {
            "experiment_id": experiment_id,
            "kind": kind,
            "prompt_id": prompt_id,
            "take": int(take),
            "text": text,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "sha256": digest,
            "duration": quality.duration,
            "rms_dbfs": quality.rms_dbfs,
            "peak_dbfs": quality.peak_dbfs,
            "clipping_ratio": quality.clipping_ratio,
            "silence_ratio": quality.silence_ratio,
            "snr_db": quality.snr_db,
            "quality_ok": int(quality.quality_ok),
            "warnings_json": json.dumps(quality.warnings, ensure_ascii=False),
            "origin": origin,
            "model_id": model_id,
            "generation_seed": generation_seed,
            "source_sha256": source_sha256,
            "transcript": transcript,
            "cer": cer,
            "created_at": now_iso(),
        }
        recording_id = self.db.upsert_recording(values)
        changed = not old or old["sha256"] != digest or old["text"] != text
        if changed:
            if kind == "anchor":
                self._invalidate_anchor_scores(experiment_id)
            else:
                self._invalidate_generation_files(recording_id)
        return {"recording_id": recording_id, **quality.to_dict(), "sha256": digest}

    def save_source(
        self,
        experiment_id: str,
        source_path: str,
        transcript: str,
        take: int = 1,
    ) -> dict[str, Any]:
        if self._run_lock.locked():
            raise RuntimeError("生成または評価の実行中は元音声を変更できません")
        experiment = self.db.experiment(experiment_id)
        if not experiment or experiment["mode"] not in {"synthetic", "validation"}:
            raise ValueError("元音声は合成事前選定または検証実験に登録してください")
        text = transcript.strip()
        if not text:
            raise ValueError("元音声の正確な台本を入力してください")
        config = self.experiment_config(experiment_id)
        source_quality = dict(config.raw["quality"])
        settings = config.candidate_generation
        source_quality.update(
            min_seconds=float(settings.get("source_min_seconds", 5.0)),
            max_seconds=float(settings.get("source_max_seconds", 15.0)),
        )
        old = self.db.recording_slot(experiment_id, "anchor", f"source{take:02d}", 1)
        result = self.save_recording(
            experiment_id,
            "anchor",
            f"source{take:02d}",
            1,
            text,
            source_path,
            origin="source",
            quality_config=source_quality,
        )
        recording = self.db.recording(int(result["recording_id"]))
        assert recording is not None
        backend = self.backends_factory(
            tts_model_id=config.tts_model,
            asr_model_id=config.asr_model,
            speaker_model_id=config.raw["models"]["speaker"],
            utmos_repo=config.raw["models"]["utmos_repo"],
            sample_rate=config.sample_rate,
        )
        try:
            recognized = backend.transcribe(recording["processed_path"])
        finally:
            backend.release()
        source_cer = character_error_rate(text, recognized)
        warnings = list(result["warnings"])
        max_cer = float(settings.get("source_max_cer", 0.10))
        if source_cer > max_cer:
            warnings.append(
                f"台本と認識結果が一致しません（CER {source_cer:.3f} > {max_cer:.3f}）"
            )
        quality_ok = bool(result["quality_ok"]) and source_cer <= max_cer
        self.db.update_recording_analysis(
            int(recording["id"]),
            transcript=recognized,
            cer=source_cer,
            quality_ok=quality_ok,
            warnings=warnings,
        )
        changed = (
            not old
            or old["sha256"] != result["sha256"]
            or old["text"] != text
            or bool(old["quality_ok"]) != quality_ok
        )
        if changed and experiment["mode"] == "synthetic":
            for path in self.db.delete_synthetic_recordings(experiment_id):
                Path(path).unlink(missing_ok=True)
        return {
            **result,
            "quality_ok": quality_ok,
            "warnings": warnings,
            "transcript": recognized,
            "cer": source_cer,
        }

    def _invalidate_generation_files(self, recording_id: int) -> None:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT output_path FROM generations WHERE recording_id = ?",
                (recording_id,),
            ).fetchall()
            connection.execute(
                "DELETE FROM generations WHERE recording_id = ?", (recording_id,)
            )
        for row in rows:
            Path(row["output_path"]).unlink(missing_ok=True)

    def _invalidate_anchor_scores(self, experiment_id: str) -> None:
        with self.db.connect() as connection:
            connection.execute(
                """
                UPDATE generations
                SET status='pending', error=NULL, similarity=NULL, utmos=NULL,
                    cer=NULL, transcript=NULL, failed=0, updated_at=?
                WHERE experiment_id=?
                """,
                (now_iso(), experiment_id),
            )

    def recording_progress(self, experiment_id: str) -> dict[str, int]:
        rows = self.db.recordings(experiment_id)
        return {
            "candidate": sum(row["kind"] == "candidate" for row in rows),
            "anchor": sum(row["kind"] == "anchor" for row in rows),
            "quality_ok": sum(bool(row["quality_ok"]) for row in rows),
        }

    def generate_candidate_references(
        self,
        experiment_id: str,
        smoke: bool = False,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, int]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("別の生成または評価が実行中です")
        try:
            return self._generate_candidate_references(
                experiment_id,
                smoke=smoke,
                progress=progress,
                stop_event=stop_event,
            )
        finally:
            self._run_lock.release()

    def _generate_candidate_references(
        self,
        experiment_id: str,
        smoke: bool = False,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, int]:
        experiment = self.db.experiment(experiment_id)
        if not experiment or experiment["mode"] != "synthetic":
            raise ValueError("合成事前選定の実験を選択してください")
        config = self.experiment_config(experiment_id)
        sources = [
            row for row in self.db.recordings(experiment_id, "anchor")
            if row["origin"] == "source" and row["quality_ok"]
        ]
        if not sources:
            raise RuntimeError("品質と台本一致を確認済みの元音声が必要です")
        settings = config.candidate_generation
        seeds = [int(seed) for seed in settings.get("seeds", config.seeds)]
        max_attempts = min(int(settings.get("max_attempts", 4)), len(seeds))
        required_takes = 1 if smoke else int(
            settings.get("takes_per_candidate", config.takes_per_candidate)
        )
        candidates = config.candidates[:1] if smoke else config.candidates
        model_id = config.smoke_tts_model if smoke else config.tts_model
        if smoke and any(
            row["origin"] == "synthetic" and row["model_id"] == config.tts_model
            for row in self.db.recordings(experiment_id, "candidate")
        ):
            return {"accepted": 0, "rejected": 0, "incomplete": 0}
        backend = self.backends_factory(
            tts_model_id=model_id,
            asr_model_id=config.asr_model,
            speaker_model_id=config.raw["models"]["speaker"],
            utmos_repo=config.raw["models"]["utmos_repo"],
            sample_rate=config.sample_rate,
        )
        accepted_total = 0
        rejected_total = 0
        incomplete = 0
        total = len(candidates) * max_attempts
        done = 0
        self.db.set_experiment_status(experiment_id, "generating_references")
        try:
            for candidate in candidates:
                accepted = [
                    row for row in self.db.recordings(experiment_id, "candidate")
                    if row["prompt_id"] == candidate.id
                    and row["origin"] == "synthetic"
                    and row["model_id"] == model_id
                    and row["quality_ok"]
                    and (row["cer"] is None or row["cer"] <= float(settings.get("max_cer", 0.10)))
                ]
                used_seeds = {int(row["generation_seed"]) for row in accepted}
                for seed in seeds[:max_attempts]:
                    if len(accepted) >= required_takes:
                        break
                    if stop_event and stop_event.is_set():
                        self.db.set_experiment_status(experiment_id, "paused")
                        return {
                            "accepted": accepted_total,
                            "rejected": rejected_total,
                            "incomplete": len(candidates),
                        }
                    done += 1
                    source = sources[(done - 1) % len(sources)]
                    if seed in used_seeds:
                        continue
                    existing_attempt = self.db.reference_attempt(
                        experiment_id, candidate.id, int(source["id"]), seed, model_id
                    )
                    if existing_attempt and existing_attempt["status"] == "rejected":
                        rejected_total += 1
                        continue
                    output_path = (
                        self.experiment_dir(experiment_id)
                        / "reference_generated"
                        / f"{candidate.id}_source{source['id']}_s{seed}_{Path(model_id).name}.wav"
                    )
                    if progress:
                        progress(
                            done - 1,
                            total,
                            f"候補参照音声を生成中: {candidate.id} / seed {seed}",
                        )
                    try:
                        if not valid_audio_file(output_path):
                            backend.synthesize(
                                text=candidate.text,
                                ref_audio=source["processed_path"],
                                ref_text=source["text"],
                                seed=seed,
                                output_path=output_path,
                                max_tokens=int(config.raw["generation"]["max_tokens"]),
                            )
                        audio, sample_rate = sf.read(output_path, always_2d=False)
                        quality = inspect_audio(
                            np.asarray(audio, dtype=np.float32),
                            int(sample_rate),
                            config.raw["quality"],
                        )
                        recognized = backend.transcribe(output_path)
                        cer = character_error_rate(candidate.text, recognized)
                        max_cer = float(settings.get("max_cer", 0.10))
                        accepted_ok = quality.quality_ok and cer <= max_cer
                        attempt_values = {
                            "experiment_id": experiment_id,
                            "prompt_id": candidate.id,
                            "source_recording_id": int(source["id"]),
                            "seed": seed,
                            "model_id": model_id,
                            "output_path": str(output_path),
                            "status": "accepted" if accepted_ok else "rejected",
                            "error": None if accepted_ok else "品質またはCERが基準外です",
                            "duration": quality.duration,
                            "quality_ok": int(quality.quality_ok),
                            "transcript": recognized,
                            "cer": cer,
                        }
                        self.db.save_reference_attempt(attempt_values)
                        if not accepted_ok:
                            rejected_total += 1
                            continue
                        take = len(accepted) + 1
                        saved = self.save_recording(
                            experiment_id,
                            "candidate",
                            candidate.id,
                            take,
                            candidate.text,
                            str(output_path),
                            origin="synthetic",
                            model_id=model_id,
                            generation_seed=seed,
                            source_sha256=source["sha256"],
                            transcript=recognized,
                            cer=cer,
                        )
                        accepted_row = self.db.recording(int(saved["recording_id"]))
                        if accepted_row is not None:
                            accepted.append(accepted_row)
                        used_seeds.add(seed)
                        accepted_total += 1
                    except Exception as error:
                        self.db.save_reference_attempt(
                            {
                                "experiment_id": experiment_id,
                                "prompt_id": candidate.id,
                                "source_recording_id": int(source["id"]),
                                "seed": seed,
                                "model_id": model_id,
                                "output_path": str(output_path),
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}"[:2000],
                            }
                        )
                        rejected_total += 1
                if len(accepted) < required_takes:
                    incomplete += 1
            self.db.set_experiment_status(
                experiment_id,
                "references_ready" if incomplete == 0 else "needs_attention",
            )
            if progress:
                progress(done, total, "候補参照音声の生成が終了しました")
            return {
                "accepted": accepted_total,
                "rejected": rejected_total,
                "incomplete": incomplete,
            }
        finally:
            backend.release()

    def _validate_ready(self, experiment_id: str, smoke: bool) -> list[Any]:
        experiment = self.db.experiment(experiment_id)
        if not experiment:
            raise ValueError("実験が見つかりません")
        config = self.experiment_config(experiment_id)
        candidates = self.db.recordings(experiment_id, "candidate")
        anchors = self.db.recordings(experiment_id, "anchor")
        required_candidates = 1 if smoke else len(config.candidates) * config.takes_per_candidate
        required_anchors = 1 if experiment["mode"] in {"synthetic", "validation"} else (
            1 if smoke else len(config.anchors)
        )
        model_id = config.smoke_tts_model if smoke else config.tts_model
        valid_candidates = [
            row for row in candidates
            if row["quality_ok"]
            and (
                experiment["mode"] != "synthetic"
                or (row["origin"] == "synthetic" and row["model_id"] == model_id)
            )
        ]
        valid_anchors = [
            row for row in anchors
            if row["quality_ok"]
            and (
                experiment["mode"] == "recorded"
                or row["origin"] == "source"
            )
        ]
        if len(valid_candidates) < required_candidates:
            raise RuntimeError(
                f"品質確認済み候補が不足しています: {len(valid_candidates)}/{required_candidates}"
            )
        if len(valid_anchors) < required_anchors:
            raise RuntimeError(
                f"品質確認済みアンカーが不足しています: {len(valid_anchors)}/{required_anchors}"
            )
        return valid_candidates[:1] if smoke else valid_candidates

    def prepare_jobs(self, experiment_id: str, smoke: bool = False) -> int:
        config = self.experiment_config(experiment_id)
        candidates = self._validate_ready(experiment_id, smoke)
        evaluations = config.evaluations[:1] if smoke else config.evaluations
        seeds = config.seeds[:1] if smoke else config.seeds
        model_id = config.smoke_tts_model if smoke else config.tts_model
        return self.db.prepare_generations(
            experiment_id,
            candidates,
            [{"id": item.id, "text": item.text} for item in evaluations],
            list(seeds),
            model_id,
            self.experiment_dir(experiment_id) / "generated",
        )

    def _anchor_centroid(self, experiment_id: str, backends: ModelBackends) -> np.ndarray:
        experiment = self.db.experiment(experiment_id)
        if not experiment:
            raise ValueError("実験が見つかりません")
        anchors = [
            row for row in self.db.recordings(experiment_id, "anchor")
            if row["quality_ok"]
            and (experiment["mode"] == "recorded" or row["origin"] == "source")
        ]
        embeddings = [
            backends.speaker_embedding(row["processed_path"]) for row in anchors
        ]
        centroid = np.mean(np.stack(embeddings), axis=0)
        return centroid / np.linalg.norm(centroid)

    def run(
        self,
        experiment_id: str,
        smoke: bool = False,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, int]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("別の実験が実行中です")
        backends: ModelBackends | None = None
        try:
            self.prepare_jobs(experiment_id, smoke)
            experiment = self.db.experiment(experiment_id)
            if not experiment:
                raise ValueError("実験が見つかりません")
            config = self.experiment_config(experiment_id)
            model_id = config.smoke_tts_model if smoke else config.tts_model
            backends = self.backends_factory(
                tts_model_id=model_id,
                asr_model_id=config.asr_model,
                speaker_model_id=config.raw["models"]["speaker"],
                utmos_repo=config.raw["models"]["utmos_repo"],
                sample_rate=config.sample_rate,
            )
            self.db.set_experiment_status(experiment_id, "running")
            pending = self.db.pending_generations(experiment_id, model_id)
            total = len(pending)
            if total == 0:
                self.db.set_experiment_status(experiment_id, "complete")
                return {"completed": 0, "failed": 0, "remaining": 0}
            if progress:
                progress(0, total, "共通アンカーを解析しています")
            anchor_centroid = self._anchor_centroid(experiment_id, backends)
            completed = 0
            failed = 0
            for index, job in enumerate(pending, start=1):
                if stop_event and stop_event.is_set():
                    self.db.set_experiment_status(experiment_id, "paused")
                    break
                label = (
                    f"{job['prompt_id']} / take {job['take']} / "
                    f"{job['eval_id']} / seed {job['seed']}"
                )
                if progress:
                    progress(index - 1, total, f"処理中: {label}")
                started = time.monotonic()
                try:
                    output_path = Path(job["output_path"])
                    if not valid_audio_file(output_path):
                        backends.synthesize(
                            text=job["eval_text"],
                            ref_audio=job["ref_audio"],
                            ref_text=job["ref_text"],
                            seed=int(job["seed"]),
                            output_path=output_path,
                            max_tokens=int(config.raw["generation"]["max_tokens"]),
                        )
                    output_embedding = backends.speaker_embedding(output_path)
                    similarity = cosine_similarity(anchor_centroid, output_embedding)
                    utmos = backends.utmos(output_path)
                    transcript = backends.transcribe(output_path)
                    cer = character_error_rate(job["eval_text"], transcript)
                    self.db.complete_generation(
                        int(job["id"]),
                        {
                            "elapsed_seconds": time.monotonic() - started,
                            "output_sha256": sha256_file(output_path),
                            "similarity": similarity,
                            "utmos": utmos,
                            "cer": cer,
                            "transcript": transcript,
                            "failed": int(
                                cer > float(config.raw["ranking"]["failure_cer"])
                            ),
                        },
                    )
                    completed += 1
                except Exception as error:
                    self.db.fail_generation(int(job["id"]), f"{type(error).__name__}: {error}")
                    failed += 1
            remaining = len(self.db.pending_generations(experiment_id, model_id))
            if remaining == 0:
                self.db.set_experiment_status(experiment_id, "complete")
            elif not (stop_event and stop_event.is_set()):
                self.db.set_experiment_status(experiment_id, "needs_attention")
            if progress:
                progress(total - remaining, total, "実験処理が終了しました")
            return {"completed": completed, "failed": failed, "remaining": remaining}
        finally:
            if backends:
                backends.release()
            self._run_lock.release()

    def create_validation_experiment(
        self,
        screening_experiment_id: str,
        name: str | None = None,
    ) -> str:
        screening = self.db.experiment(screening_experiment_id)
        if not screening or screening["mode"] != "synthetic":
            raise ValueError("合成事前選定の実験を選択してください")
        config = self.experiment_config(screening_experiment_id)
        from .scoring import rank_candidates

        ranking = rank_candidates(
            [
                dict(row)
                for row in self.db.generations(screening_experiment_id, complete_only=True)
            ],
            config.raw["ranking"],
            [dict(row) for row in self.db.votes(screening_experiment_id)],
            bootstrap_samples=int(config.raw["ranking"]["bootstrap_samples"]),
        )
        top_ids = [row["candidate_id"] for row in ranking[:3]]
        if len(top_ids) < 3:
            raise RuntimeError("上位3候補を作るには完了済み候補が3件以上必要です")
        validation_id = self.create_experiment(
            name=name or f"{screening['name']} — 上位3件の実録音検証",
            mode="validation",
            parent_experiment_id=screening_experiment_id,
            candidate_ids=top_ids,
        )
        for source in self.db.recordings(screening_experiment_id, "anchor"):
            if source["origin"] != "source" or not source["quality_ok"]:
                continue
            saved = self.save_recording(
                validation_id,
                "anchor",
                source["prompt_id"],
                1,
                source["text"],
                source["processed_path"],
                origin="source",
                transcript=source["transcript"],
                cer=source["cer"],
            )
            self.db.update_recording_analysis(
                int(saved["recording_id"]),
                transcript=source["transcript"] or source["text"],
                cer=float(source["cer"] or 0.0),
                quality_ok=True,
                warnings=[],
            )
        return validation_id
