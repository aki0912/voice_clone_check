from pathlib import Path

from voice_clone_check.app import build_app
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
