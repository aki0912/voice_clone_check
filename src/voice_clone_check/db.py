from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'recording',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('candidate', 'anchor')),
    prompt_id TEXT NOT NULL,
    take INTEGER NOT NULL,
    text TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    processed_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    duration REAL NOT NULL,
    rms_dbfs REAL NOT NULL,
    peak_dbfs REAL NOT NULL,
    clipping_ratio REAL NOT NULL,
    silence_ratio REAL NOT NULL,
    snr_db REAL NOT NULL,
    quality_ok INTEGER NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(experiment_id, kind, prompt_id, take)
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL,
    eval_text TEXT NOT NULL,
    seed INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    output_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    elapsed_seconds REAL,
    output_sha256 TEXT,
    similarity REAL,
    utmos REAL,
    cer REAL,
    transcript TEXT,
    failed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(experiment_id, recording_id, eval_id, seed, model_id)
);

CREATE TABLE IF NOT EXISTS listening_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    candidate_a TEXT NOT NULL,
    candidate_b TEXT NOT NULL,
    generation_a INTEGER NOT NULL REFERENCES generations(id),
    generation_b INTEGER NOT NULL REFERENCES generations(id),
    winner TEXT NOT NULL CHECK(winner IN ('a', 'b', 'tie')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recordings_experiment
ON recordings(experiment_id);

CREATE INDEX IF NOT EXISTS idx_generations_status
ON generations(experiment_id, status);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create_experiment(
        self, experiment_id: str, name: str, config: dict[str, Any]
    ) -> None:
        timestamp = now_iso()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments(id, name, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (experiment_id, name, json.dumps(config, ensure_ascii=False), timestamp, timestamp),
            )

    def experiment(self, experiment_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()

    def list_experiments(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC"
                ).fetchall()
            )

    def set_experiment_status(self, experiment_id: str, status: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status, now_iso(), experiment_id),
            )

    def upsert_recording(self, values: dict[str, Any]) -> int:
        columns = (
            "experiment_id", "kind", "prompt_id", "take", "text", "raw_path",
            "processed_path", "sha256", "duration", "rms_dbfs", "peak_dbfs",
            "clipping_ratio", "silence_ratio", "snr_db", "quality_ok",
            "warnings_json", "created_at",
        )
        params = [values[column] for column in columns]
        with self.connect() as connection:
            connection.execute(
                f"""
                INSERT INTO recordings({", ".join(columns)})
                VALUES ({", ".join("?" for _ in columns)})
                ON CONFLICT(experiment_id, kind, prompt_id, take) DO UPDATE SET
                    text=excluded.text,
                    raw_path=excluded.raw_path,
                    processed_path=excluded.processed_path,
                    sha256=excluded.sha256,
                    duration=excluded.duration,
                    rms_dbfs=excluded.rms_dbfs,
                    peak_dbfs=excluded.peak_dbfs,
                    clipping_ratio=excluded.clipping_ratio,
                    silence_ratio=excluded.silence_ratio,
                    snr_db=excluded.snr_db,
                    quality_ok=excluded.quality_ok,
                    warnings_json=excluded.warnings_json,
                    created_at=excluded.created_at
                """,
                params,
            )
            row = connection.execute(
                """
                SELECT id FROM recordings
                WHERE experiment_id = ? AND kind = ? AND prompt_id = ? AND take = ?
                """,
                (
                    values["experiment_id"],
                    values["kind"],
                    values["prompt_id"],
                    values["take"],
                ),
            ).fetchone()
            return int(row["id"])

    def recordings(self, experiment_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM recordings WHERE experiment_id = ?"
        params: list[Any] = [experiment_id]
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY kind, prompt_id, take"
        with self.connect() as connection:
            return list(connection.execute(query, params).fetchall())

    def prepare_generations(
        self,
        experiment_id: str,
        recordings: list[sqlite3.Row],
        evaluations: list[dict[str, str]],
        seeds: list[int],
        model_id: str,
        output_dir: Path,
    ) -> int:
        timestamp = now_iso()
        inserted = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            for recording in recordings:
                for evaluation in evaluations:
                    for seed in seeds:
                        filename = (
                            f"{recording['prompt_id']}_t{recording['take']}_"
                            f"{evaluation['id']}_s{seed}.wav"
                        )
                        cursor = connection.execute(
                            """
                            INSERT OR IGNORE INTO generations(
                                experiment_id, recording_id, eval_id, eval_text,
                                seed, model_id, output_path, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                experiment_id,
                                recording["id"],
                                evaluation["id"],
                                evaluation["text"],
                                seed,
                                model_id,
                                str(output_dir / filename),
                                timestamp,
                                timestamp,
                            ),
                        )
                        inserted += cursor.rowcount
        return inserted

    def pending_generations(
        self, experiment_id: str, model_id: str | None = None
    ) -> list[sqlite3.Row]:
        model_filter = " AND g.model_id = ?" if model_id else ""
        params: tuple[Any, ...] = (
            (experiment_id, model_id) if model_id else (experiment_id,)
        )
        with self.connect() as connection:
            return list(
                connection.execute(
                    f"""
                    SELECT g.*, r.prompt_id, r.take, r.text AS ref_text,
                           r.processed_path AS ref_audio
                    FROM generations g
                    JOIN recordings r ON r.id = g.recording_id
                    WHERE g.experiment_id = ? AND g.status != 'complete'
                    {model_filter}
                    ORDER BY g.id
                    """,
                    params,
                ).fetchall()
            )

    def complete_generation(self, generation_id: int, values: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE generations
                SET status='complete', error=NULL, elapsed_seconds=?,
                    output_sha256=?, similarity=?, utmos=?, cer=?,
                    transcript=?, failed=?, updated_at=?
                WHERE id=?
                """,
                (
                    values["elapsed_seconds"],
                    values["output_sha256"],
                    values["similarity"],
                    values["utmos"],
                    values["cer"],
                    values["transcript"],
                    values["failed"],
                    now_iso(),
                    generation_id,
                ),
            )

    def fail_generation(self, generation_id: int, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE generations
                SET status='failed', error=?, updated_at=? WHERE id=?
                """,
                (error[:2000], now_iso(), generation_id),
            )

    def generations(self, experiment_id: str, complete_only: bool = False) -> list[sqlite3.Row]:
        query = """
            SELECT g.*, r.prompt_id, r.take
            FROM generations g JOIN recordings r ON r.id = g.recording_id
            WHERE g.experiment_id = ?
        """
        if complete_only:
            query += " AND g.status = 'complete'"
        query += " ORDER BY g.id"
        with self.connect() as connection:
            return list(connection.execute(query, (experiment_id,)).fetchall())

    def add_vote(
        self,
        experiment_id: str,
        candidate_a: str,
        candidate_b: str,
        generation_a: int,
        generation_b: int,
        winner: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO listening_votes(
                    experiment_id, candidate_a, candidate_b,
                    generation_a, generation_b, winner, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    candidate_a,
                    candidate_b,
                    generation_a,
                    generation_b,
                    winner,
                    now_iso(),
                ),
            )

    def votes(self, experiment_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM listening_votes WHERE experiment_id = ? ORDER BY id",
                    (experiment_id,),
                ).fetchall()
            )
