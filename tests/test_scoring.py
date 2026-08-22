from voice_clone_check.scoring import (
    analyze_duration_study,
    listening_win_rates,
    rank_candidates,
)


def _row(candidate, similarity, utmos, cer, failed=0):
    return {
        "prompt_id": candidate,
        "similarity": similarity,
        "utmos": utmos,
        "cer": cer,
        "failed": failed,
    }


def test_rank_candidates_prefers_balanced_quality():
    rows = [
        _row("c01", 0.9, 4.2, 0.01),
        _row("c01", 0.88, 4.1, 0.02),
        _row("c02", 0.7, 3.5, 0.15),
        _row("c02", 0.72, 3.6, 0.12),
    ]
    weights = {
        "similarity": 0.5,
        "utmos": 0.3,
        "intelligibility": 0.2,
        "automatic": 0.8,
        "listening": 0.2,
    }
    ranking = rank_candidates(rows, weights, bootstrap_samples=100)
    assert ranking[0]["candidate_id"] == "c01"
    assert ranking[0]["ci_low"] <= ranking[0]["automatic_score"] <= ranking[0]["ci_high"]


def test_listening_win_rates_counts_ties():
    rates = listening_win_rates(
        [
            {"candidate_a": "c01", "candidate_b": "c02", "winner": "a"},
            {"candidate_a": "c01", "candidate_b": "c02", "winner": "tie"},
        ]
    )
    assert rates == {"c01": 0.75, "c02": 0.25}


def _duration_settings():
    return {
        "targets": [
            {"id": "d04", "seconds": 4},
            {"id": "d08", "seconds": 8},
            {"id": "d12", "seconds": 12},
            {"id": "d15", "seconds": 15},
        ],
        "similarity_margin": 0.02,
        "utmos_margin": 0.10,
        "cer_margin": 0.02,
        "bootstrap_samples": 200,
    }


def _duration_rows(similarities):
    return [
        {
            "prompt_id": condition,
            "take": take,
            "eval_id": "e01",
            "seed": 1,
            "similarity": value if not isinstance(value, list) else value[take - 1],
            "utmos": 4.0,
            "cer": 0.01,
            "failed": 0,
        }
        for condition, value in similarities.items()
        for take in (1, 2, 3)
    ]


def _complete_duration_votes():
    return [
        {"candidate_a": short, "candidate_b": long, "winner": "tie"}
        for short, long in (("d04", "d08"), ("d08", "d12"), ("d12", "d15"))
        for _ in range(12)
    ]


def test_duration_study_recommends_shortest_noninferior_condition():
    result = analyze_duration_study(
        _duration_rows({"d04": 0.70, "d08": 0.80, "d12": 0.89, "d15": 0.90}),
        _duration_settings(),
        _complete_duration_votes(),
    )
    assert result["status"] == "recommended"
    assert result["recommended_condition"] == "d12"


def test_duration_study_can_recommend_all_lengths_as_equivalent():
    result = analyze_duration_study(
        _duration_rows({"d04": 0.90, "d08": 0.90, "d12": 0.90, "d15": 0.90}),
        _duration_settings(),
        _complete_duration_votes(),
    )
    assert result["recommended_condition"] == "d04"


def test_duration_study_reports_inconclusive_when_interval_crosses_margin():
    varying = [0.85, 0.90, 0.93]
    result = analyze_duration_study(
        _duration_rows({"d04": varying, "d08": varying, "d12": varying, "d15": 0.90}),
        _duration_settings(),
    )
    assert result["status"] == "inconclusive"
    assert result["recommended_condition"] is None


def test_duration_study_waits_for_blind_listening_before_final_recommendation():
    result = analyze_duration_study(
        _duration_rows({"d04": 0.90, "d08": 0.90, "d12": 0.90, "d15": 0.90}),
        _duration_settings(),
    )
    assert result["status"] == "needs_listening"
    assert result["automatic_candidate"] == "d04"
