import sqlite3
from pathlib import Path

from voice_clone_check.db import Database


def test_database_migrates_legacy_experiment_columns(tmp_path: Path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'recording',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    database = Database(path)

    with database.connect() as migrated:
        experiment_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(experiments)")
        }
        recording_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(recordings)")
        }
    assert {"mode", "parent_experiment_id"} <= experiment_columns
    assert {
        "origin", "model_id", "generation_seed", "source_sha256", "transcript", "cer"
    } <= recording_columns
