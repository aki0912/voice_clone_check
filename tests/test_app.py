from pathlib import Path

from voice_clone_check.app import build_app, select_listening_pairs
from voice_clone_check.config import load_config
from voice_clone_check.service import ExperimentService


def test_app_builds(tmp_path: Path):
    service = ExperimentService(root=tmp_path / "experiments")
    app = build_app(service)
    assert app is not None


def test_recording_prompts_have_hiragana_readings():
    config = load_config()

    assert config.candidates[0].reading == "けさはあおいそらをみながら、えきまでゆっくりあるきました。"
    assert config.anchors[0].reading


def test_listening_uses_one_generation_per_candidate_pair_and_text():
    rows = [
        {
            "id": index,
            "prompt_id": candidate,
            "eval_id": evaluation,
            "take": take,
            "seed": seed,
        }
        for index, (candidate, evaluation, take, seed) in enumerate(
            (
                (candidate, evaluation, take, seed)
                for candidate in ("c01", "c02", "c03")
                for evaluation in ("e01", "e02")
                for take in (1, 2)
                for seed in (10, 20)
            ),
            start=1,
        )
    ]

    pairs = select_listening_pairs(rows, ["c01", "c02", "c03"], ["e01", "e02"])

    assert len(pairs) == 6
    assert all(left["take"] == right["take"] == 1 for left, right in pairs)
    assert all(left["seed"] == right["seed"] == 10 for left, right in pairs)
