from voice_clone_check.scoring import listening_win_rates, rank_candidates


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

