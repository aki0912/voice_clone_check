from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def robust_scale(values: np.ndarray, low: float = 5, high: float = 95) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lower, upper = np.nanpercentile(values, [low, high])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.full_like(values, 0.5)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def listening_win_rates(votes: Iterable[dict[str, Any]]) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for vote in votes:
        candidate_a = str(vote["candidate_a"])
        candidate_b = str(vote["candidate_b"])
        winner = str(vote["winner"])
        counts[candidate_a] += 1
        counts[candidate_b] += 1
        if winner == "a":
            scores[candidate_a] += 1
        elif winner == "b":
            scores[candidate_b] += 1
        else:
            scores[candidate_a] += 0.5
            scores[candidate_b] += 0.5
    return {
        candidate: scores[candidate] / count
        for candidate, count in counts.items()
        if count
    }


def rank_candidates(
    rows: Iterable[dict[str, Any]],
    weights: dict[str, float],
    votes: Iterable[dict[str, Any]] = (),
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260731,
) -> list[dict[str, Any]]:
    complete = [
        dict(row)
        for row in rows
        if row.get("similarity") is not None
        and row.get("utmos") is not None
        and row.get("cer") is not None
    ]
    if not complete:
        return []

    similarity = robust_scale(np.array([row["similarity"] for row in complete]))
    utmos = robust_scale(np.array([row["utmos"] for row in complete]))
    intelligibility = robust_scale(np.array([1.0 - row["cer"] for row in complete]))
    metric_weights = np.array(
        [
            float(weights["similarity"]),
            float(weights["utmos"]),
            float(weights["intelligibility"]),
        ]
    )
    if metric_weights.sum() <= 0:
        metric_weights = np.array([0.5, 0.3, 0.2])
    metric_weights = metric_weights / metric_weights.sum()
    automatic = (
        metric_weights[0] * similarity
        + metric_weights[1] * utmos
        + metric_weights[2] * intelligibility
    )
    for index, row in enumerate(complete):
        row["_automatic"] = float(automatic[index])

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        grouped[str(row["prompt_id"])].append(row)

    rng = np.random.default_rng(bootstrap_seed)
    win_rates = listening_win_rates(votes)
    output: list[dict[str, Any]] = []
    for candidate_id, candidate_rows in grouped.items():
        auto_scores = np.array([row["_automatic"] for row in candidate_rows])
        sample_means = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            sample_means[index] = rng.choice(
                auto_scores, size=len(auto_scores), replace=True
            ).mean()
        automatic_mean = float(auto_scores.mean())
        listening = win_rates.get(candidate_id)
        if listening is None:
            final_score = automatic_mean
        else:
            final_score = (
                float(weights["automatic"]) * automatic_mean
                + float(weights["listening"]) * listening
            )
        output.append(
            {
                "candidate_id": candidate_id,
                "samples": len(candidate_rows),
                "similarity": float(np.mean([row["similarity"] for row in candidate_rows])),
                "utmos": float(np.mean([row["utmos"] for row in candidate_rows])),
                "cer": float(np.mean([row["cer"] for row in candidate_rows])),
                "failure_rate": float(np.mean([row.get("failed", 0) for row in candidate_rows])),
                "automatic_score": automatic_mean,
                "ci_low": float(np.percentile(sample_means, 2.5)),
                "ci_high": float(np.percentile(sample_means, 97.5)),
                "listening_win_rate": listening,
                "final_score": final_score,
            }
        )
    output.sort(key=lambda item: item["final_score"], reverse=True)
    for index, row in enumerate(output, start=1):
        row["rank"] = index
    return output
