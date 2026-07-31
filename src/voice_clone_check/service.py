from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import prepare_recording, sha256_file, valid_audio_file
from .backends import ModelBackends, cosine_similarity
from .config import ExperimentConfig, load_config
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

    def create_experiment(self, name: str | None = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_id = f"{timestamp}-{uuid.uuid4().hex[:6]}"
        display_name = (name or "").strip() or f"日本語参照セリフ探索 {timestamp}"
        directory = self.experiment_dir(experiment_id)
        for child in ("recordings/raw", "recordings/processed", "generated", "reports"):
            (directory / child).mkdir(parents=True, exist_ok=True)
        config_copy = directory / "config.yaml"
        source_config = Path(__file__).resolve().parents[2] / "configs" / "default_ja.yaml"
        shutil.copy2(source_config, config_copy)
        self.db.create_experiment(experiment_id, display_name, self.config.raw)
        return experiment_id

    def save_recording(
        self,
        experiment_id: str,
        kind: str,
        prompt_id: str,
        take: int,
        text: str,
        source_path: str,
    ) -> dict[str, Any]:
        if not self.db.experiment(experiment_id):
            raise ValueError("実験が見つかりません")
        if kind not in {"candidate", "anchor"}:
            raise ValueError("録音種別が不正です")
        source = Path(source_path)
        extension = source.suffix.lower() or ".wav"
        base = f"{kind}_{prompt_id}_t{take}"
        directory = self.experiment_dir(experiment_id)
        raw_path = directory / "recordings" / "raw" / f"{base}{extension}"
        processed_path = directory / "recordings" / "processed" / f"{base}.wav"
        quality, digest = prepare_recording(
            source,
            raw_path,
            processed_path,
            self.config.sample_rate,
            self.config.raw["quality"],
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
            "created_at": now_iso(),
        }
        recording_id = self.db.upsert_recording(values)
        if kind == "anchor":
            self._invalidate_anchor_scores(experiment_id)
        else:
            self._invalidate_generation_files(recording_id)
        return {"recording_id": recording_id, **quality.to_dict(), "sha256": digest}

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

    def _validate_ready(self, experiment_id: str, smoke: bool) -> list[Any]:
        candidates = self.db.recordings(experiment_id, "candidate")
        anchors = self.db.recordings(experiment_id, "anchor")
        required_candidates = 1 if smoke else len(self.config.candidates) * self.config.takes_per_candidate
        required_anchors = 1 if smoke else len(self.config.anchors)
        valid_candidates = [row for row in candidates if row["quality_ok"]]
        valid_anchors = [row for row in anchors if row["quality_ok"]]
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
        candidates = self._validate_ready(experiment_id, smoke)
        evaluations = self.config.evaluations[:1] if smoke else self.config.evaluations
        seeds = self.config.seeds[:1] if smoke else self.config.seeds
        model_id = self.config.smoke_tts_model if smoke else self.config.tts_model
        return self.db.prepare_generations(
            experiment_id,
            candidates,
            [{"id": item.id, "text": item.text} for item in evaluations],
            list(seeds),
            model_id,
            self.experiment_dir(experiment_id) / "generated",
        )

    def _anchor_centroid(self, experiment_id: str, backends: ModelBackends) -> np.ndarray:
        anchors = [
            row for row in self.db.recordings(experiment_id, "anchor") if row["quality_ok"]
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
            model_id = self.config.smoke_tts_model if smoke else self.config.tts_model
            backends = self.backends_factory(
                tts_model_id=model_id,
                asr_model_id=self.config.asr_model,
                speaker_model_id=self.config.raw["models"]["speaker"],
                utmos_repo=self.config.raw["models"]["utmos_repo"],
                sample_rate=self.config.sample_rate,
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
                            max_tokens=int(self.config.raw["generation"]["max_tokens"]),
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
                                cer > float(self.config.raw["ranking"]["failure_cer"])
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
