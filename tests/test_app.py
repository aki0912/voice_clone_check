from pathlib import Path

from voice_clone_check.app import (
    CSS,
    build_app,
    report_preview,
    ruby_markup,
    select_listening_pairs,
    source_recording_status,
)
from voice_clone_check.config import load_config
from voice_clone_check.service import ExperimentService


def test_app_builds(tmp_path: Path):
    service = ExperimentService(root=tmp_path / "experiments")
    app = build_app(service)
    assert app is not None


def test_source_audio_supports_browser_recording(tmp_path: Path):
    service = ExperimentService(root=tmp_path / "experiments")
    app = build_app(service)
    config = app.get_config_file()
    source_components = [
        component
        for component in config["components"]
        if component.get("props", {}).get("label")
        == "新しい音声（ブラウザで録音またはファイルを選択）"
    ]
    saved_components = [
        component
        for component in config["components"]
        if component.get("props", {}).get("label")
        == "現在保存されている元音声"
    ]

    assert len(source_components) == 1
    assert source_components[0]["props"]["sources"] == ["microphone", "upload"]
    assert len(saved_components) == 1
    assert not saved_components[0]["props"]["interactive"]


def test_source_recording_status_describes_empty_saved_and_rerecord_states():
    assert "スロット 2: 未登録" in source_recording_status(None, 2)
    recording = {
        "text": "保存した台本です。",
        "transcript": "保存した台本です。",
        "cer": 0.0,
        "duration": 6.2,
        "snr_db": 28.0,
        "quality_ok": 1,
        "warnings_json": "[]",
    }

    ready = source_recording_status(recording, 1)
    needs_rerecording = source_recording_status(
        {**recording, "quality_ok": 0, "warnings_json": '["背景雑音"]'},
        3,
    )

    assert "登録済み・利用可能" in ready
    assert "再録音する場合" in ready
    assert "保存した台本です。" in ready
    assert "登録済み・要再録" in needs_rerecording
    assert "背景雑音" in needs_rerecording


def test_custom_styles_include_readable_dark_mode_tokens_and_buttons():
    assert "body.dark" in CSS
    assert "--vcc-ink: #f8fafc" in CSS
    assert "--vcc-muted: #cbd5e1" in CSS
    assert "button.record" in CSS
    assert "button.primary" in CSS
    assert "button:focus-visible" in CSS
    assert "background:#fff" not in CSS


def test_recording_prompts_have_hiragana_readings():
    config = load_config()
    prompts = {prompt.id: prompt for prompt in config.candidates}

    assert config.candidates[0].reading == "けさはあおいそらをみながら、えきまでゆっくりあるきました。"
    assert prompts["c02"].text.endswith("時計がはいっています。")
    assert prompts["c02"].reading.endswith("とけいがはいっています。")
    assert prompts["c09"].text.endswith("窓をあけました。")
    assert prompts["c09"].reading.endswith("まどをあけました。")
    assert prompts["c10"].text.startswith("しちがつにじゅうさんにちの")
    assert prompts["c10"].reading.startswith("しちがつにじゅうさんにちの")
    assert "ほそいみち" in prompts["c11"].text
    assert "ほそいみち" in prompts["c11"].reading
    assert config.anchors[0].reading


def test_recording_prompts_render_readings_as_ruby():
    prompt = load_config().candidates[0]

    markup = ruby_markup(prompt)

    assert "<ruby>今朝<rp>（</rp><rt>けさ</rt><rp>）</rp></ruby>" in markup
    assert "<ruby>駅<rp>（</rp><rt>えき</rt><rp>）</rp></ruby>" in markup
    assert "<ruby>青<rp>（</rp><rt>あお</rt><rp>）</rp></ruby>い" in markup
    assert "<ruby>見<rp>（</rp><rt>み</rt><rp>）</rp></ruby>な" in markup
    assert "<ruby>歩<rp>（</rp><rt>ある</rt><rp>）</rp></ruby>き" in markup
    assert "<ruby>見な" not in markup
    assert "<ruby>は" not in markup


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


def test_report_preview_embeds_document_without_leaking_into_parent_page():
    preview = report_preview("<html><body><h1>結果</h1></body></html>")

    assert "<iframe" in preview
    assert "srcdoc=" in preview
    assert "&lt;h1&gt;結果&lt;/h1&gt;" in preview
