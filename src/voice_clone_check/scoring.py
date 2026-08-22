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


def analyze_duration_study(
    rows: Iterable[dict[str, Any]],
    settings: dict[str, Any],
    votes: Iterable[dict[str, Any]] = (),
    *,
    bootstrap_seed: int = 20260819,
) -> dict[str, Any]:
    """Analyze matched duration conditions without treating outputs as independent."""
    complete = [
        dict(row) for row in rows
        if row.get("similarity") is not None
        and row.get("utmos") is not None
        and row.get("cer") is not None
    ]
    targets = list(settings.get("targets", ()))
    condition_ids = [str(item["id"]) for item in targets]
    seconds = {str(item["id"]): float(item["seconds"]) for item in targets}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        grouped[str(row["prompt_id"])].append(row)
    summaries = []
    block_summaries = []
    for condition_id in condition_ids:
        values = grouped.get(condition_id, [])
        if not values:
            continue
        summaries.append({
            "condition_id": condition_id,
            "seconds": seconds[condition_id],
            "samples": len(values),
            "similarity": float(np.mean([row["similarity"] for row in values])),
            "utmos": float(np.mean([row["utmos"] for row in values])),
            "cer": float(np.mean([row["cer"] for row in values])),
            "failure_rate": float(np.mean([row.get("failed", 0) for row in values])),
        })
        for take in sorted({int(row["take"]) for row in values}):
            take_values = [row for row in values if int(row["take"]) == take]
            block_summaries.append({
                "condition_id": condition_id,
                "seconds": seconds[condition_id],
                "take": take,
                "samples": len(take_values),
                "similarity": float(np.mean([row["similarity"] for row in take_values])),
                "utmos": float(np.mean([row["utmos"] for row in take_values])),
                "cer": float(np.mean([row["cer"] for row in take_values])),
            })

    by_key = {
        (str(row["prompt_id"]), int(row["take"]), str(row["eval_id"]), int(row["seed"])): row
        for row in complete
    }
    rng = np.random.default_rng(bootstrap_seed)
    samples = int(settings.get("bootstrap_samples", 2000))
    longest = condition_ids[-1] if condition_ids else ""
    comparisons: list[dict[str, Any]] = []
    for condition_id in condition_ids[:-1]:
        take_differences: dict[int, dict[str, list[float]]] = {}
        for key, short in by_key.items():
            prompt_id, take, eval_id, seed = key
            if prompt_id != condition_id:
                continue
            long = by_key.get((longest, take, eval_id, seed))
            if long is None:
                continue
            bucket = take_differences.setdefault(
                take, {"similarity": [], "utmos": [], "cer": []}
            )
            for metric in bucket:
                bucket[metric].append(float(short[metric]) - float(long[metric]))
        take_ids = sorted(take_differences)
        result: dict[str, Any] = {
            "condition_id": condition_id,
            "seconds": seconds.get(condition_id),
            "reference_id": longest,
            "blocks": len(take_differences),
        }
        for metric in ("similarity", "utmos", "cer"):
            per_take = [
                np.asarray(take_differences[take][metric], dtype=np.float64)
                for take in take_ids if take_differences[take][metric]
            ]
            values = np.asarray([items.mean() for items in per_take])
            if values.size == 0:
                result.update({f"{metric}_diff": None, f"{metric}_ci_low": None, f"{metric}_ci_high": None})
                continue
            boot_values = []
            for _ in range(samples):
                selected = rng.integers(0, len(per_take), size=len(per_take))
                nested_means = [
                    rng.choice(per_take[index], size=len(per_take[index]), replace=True).mean()
                    for index in selected
                ]
                boot_values.append(float(np.mean(nested_means)))
            boot = np.asarray(boot_values)
            result.update({
                f"{metric}_diff": float(values.mean()),
                f"{metric}_ci_low": float(np.percentile(boot, 2.5)),
                f"{metric}_ci_high": float(np.percentile(boot, 97.5)),
            })
        comparisons.append(result)

    adjacent_preferences: dict[tuple[str, str], dict[str, Any]] = {}
    vote_rows = [dict(vote) for vote in votes]
    for short_id, long_id in zip(condition_ids, condition_ids[1:]):
        scores = []
        for vote in vote_rows:
            pair = {str(vote["candidate_a"]), str(vote["candidate_b"])}
            if pair != {short_id, long_id}:
                continue
            winner = str(vote["winner"])
            if winner == "tie":
                scores.append(0.5)
            else:
                winner_id = str(vote[f"candidate_{winner}"])
                scores.append(1.0 if winner_id == long_id else 0.0)
        adjacent_preferences[(short_id, long_id)] = {
            "longer_preference": float(np.mean(scores)) if scores else None,
            "votes": len(scores),
        }

    margins = {
        "similarity": float(settings.get("similarity_margin", 0.02)),
        "utmos": float(settings.get("utmos_margin", 0.10)),
        "cer": float(settings.get("cer_margin", 0.02)),
    }
    preference_limit = float(settings.get("listening_longer_preference_limit", 0.60))
    required_votes = int(settings.get("listening_pairs_per_comparison", 12))
    recommendation = None
    status = "incomplete"
    automatic_candidate = None
    for comparison in comparisons:
        if comparison["blocks"] < 3:
            continue
        conclusive = (
            comparison["similarity_ci_low"] is not None
            and comparison["similarity_ci_low"] >= -margins["similarity"]
            and comparison["utmos_ci_low"] >= -margins["utmos"]
            and comparison["cer_ci_high"] <= margins["cer"]
        )
        index = condition_ids.index(comparison["condition_id"])
        relevant = [
            adjacent_preferences[(condition_ids[i], condition_ids[i + 1])]
            for i in range(index, len(condition_ids) - 1)
        ]
        listening_ok = all(
            item["votes"] >= required_votes
            and item["longer_preference"] <= preference_limit
            for item in relevant
        )
        if conclusive:
            if automatic_candidate is None:
                automatic_candidate = comparison["condition_id"]
            if listening_ok:
                recommendation = comparison["condition_id"]
                status = "recommended"
                break
    if recommendation is None and summaries and len(summaries) == len(condition_ids):
        status = "needs_listening" if automatic_candidate else "inconclusive"
    return {
        "status": status,
        "recommended_condition": recommendation,
        "recommended_seconds": seconds.get(recommendation) if recommendation else None,
        "automatic_candidate": automatic_candidate,
        "automatic_candidate_seconds": seconds.get(automatic_candidate) if automatic_candidate else None,
        "summaries": summaries,
        "block_summaries": block_summaries,
        "comparisons": comparisons,
        "adjacent_preferences": [
            {"short_id": short_id, "long_id": long_id, **value}
            for (short_id, long_id), value in adjacent_preferences.items()
        ],
        "margins": margins,
        "required_votes_per_comparison": required_votes,
    }
