from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pykakasi import kakasi

from .paths import default_config_path


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    reading: str
    category: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    candidates: tuple[Prompt, ...]
    anchors: tuple[Prompt, ...]
    evaluations: tuple[Prompt, ...]

    @property
    def tts_model(self) -> str:
        return str(self.raw["models"]["tts"])

    @property
    def smoke_tts_model(self) -> str:
        return str(self.raw["models"]["tts_smoke"])

    @property
    def asr_model(self) -> str:
        return str(self.raw["models"]["asr"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.raw["generation"]["seeds"])

    @property
    def sample_rate(self) -> int:
        return int(self.raw["quality"]["sample_rate"])

    @property
    def takes_per_candidate(self) -> int:
        return int(self.raw["generation"]["takes_per_candidate"])


def _prompts(items: list[dict[str, Any]]) -> tuple[Prompt, ...]:
    converter = kakasi()

    def reading(item: dict[str, Any]) -> str:
        configured = str(item.get("reading", "")).strip()
        if configured:
            return configured
        return "".join(token["hira"] for token in converter.convert(str(item["text"])))

    return tuple(
        Prompt(
            id=str(item["id"]),
            text=str(item["text"]),
            reading=reading(item),
            category=str(item.get("category", "")),
        )
        for item in items
    )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig(
        raw=raw,
        candidates=_prompts(raw["candidate_prompts"]),
        anchors=_prompts(raw["anchor_prompts"]),
        evaluations=_prompts(raw["evaluation_prompts"]),
    )
