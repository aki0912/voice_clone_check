from __future__ import annotations

import html
import itertools
import json
import re
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from .config import Prompt
from .reports import ReportBuilder
from .scoring import rank_candidates
from .service import ExperimentService


CSS = """
:root {
  color-scheme: light dark;
  --vcc-accent: #4856d8;
  --vcc-accent-end: #8b62dd;
  --vcc-ink: #172033;
  --vcc-muted: #59647a;
  --vcc-surface: #ffffff;
  --vcc-border: #cbd3e3;
  --vcc-track: #e5eaf5;
  --vcc-report-surface: #f4f6fb;
  --vcc-hero-start: #f0f2ff;
  --vcc-hero-middle: #f8f5ff;
  --vcc-hero-end: #eef8f5;
  --vcc-focus: #4f46e5;
}
body.dark {
  color-scheme: dark;
  --vcc-accent: #818cf8;
  --vcc-accent-end: #c084fc;
  --vcc-ink: #f8fafc;
  --vcc-muted: #cbd5e1;
  --vcc-surface: #18181b;
  --vcc-border: #52525b;
  --vcc-track: #3f3f46;
  --vcc-report-surface: #09090b;
  --vcc-hero-start: #17172e;
  --vcc-hero-middle: #211a32;
  --vcc-hero-end: #122925;
  --vcc-focus: #a5b4fc;
}
.gradio-container { width:calc(100% - 32px) !important; max-width:1600px !important;
  margin-inline:auto !important;
  color:var(--vcc-ink); }
.vcc-hero { padding: 12px 16px; border-radius: 14px;
  background: linear-gradient(135deg,var(--vcc-hero-start) 0%,
    var(--vcc-hero-middle) 52%,var(--vcc-hero-end) 100%);
  border: 1px solid var(--vcc-border); margin-bottom: 8px; }
.vcc-hero h1 { margin: 0 0 2px; color:var(--vcc-ink);
  font-size: clamp(21px,3vw,28px); letter-spacing: -.025em; }
.vcc-hero p { margin: 0; color: var(--vcc-muted); max-width: 920px; font-size:13px; }
.vcc-experiment-bar { align-items:end; gap:8px; margin-bottom:4px; }
.vcc-experiment-bar button { min-width:112px; }
.vcc-experiment-message:empty { display:none; }
.vcc-prompt { border:1px solid var(--vcc-border); border-radius:14px; padding:14px 16px;
  background:var(--vcc-surface); min-height:112px; }
.vcc-prompt-text { color:var(--vcc-ink); font-size:18px; line-height:2.35;
  overflow-wrap:anywhere; }
.vcc-prompt-text ruby { color:var(--vcc-ink); ruby-position:over; ruby-align:center; }
.vcc-prompt-text rt { color:var(--vcc-muted); font-size:.58em; font-weight:500;
  letter-spacing:.04em; }
.vcc-score-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; }
.vcc-score { border:1px solid var(--vcc-border); border-radius:14px; padding:14px;
  background:var(--vcc-surface); }
.vcc-score strong { display:block; color:var(--vcc-ink); font-size:19px; }
.vcc-score span { color:var(--vcc-muted); font-size:12px; }
.vcc-bar { height:7px; background:var(--vcc-track); border-radius:9px; margin-top:9px; overflow:hidden; }
.vcc-bar i { display:block; height:100%;
  background:linear-gradient(90deg,var(--vcc-accent),var(--vcc-accent-end)); }
.vcc-listening-progress { height:8px; background:var(--vcc-track); border-radius:9px;
  margin:8px 0 12px; overflow:hidden; }
.vcc-listening-progress i { display:block; height:100%;
  background:linear-gradient(90deg,var(--vcc-accent),var(--vcc-accent-end)); }
.vcc-listen-note { border:1px solid var(--vcc-border); border-radius:14px;
  padding:14px 16px; background:var(--vcc-surface); color:var(--vcc-ink); }
.vcc-listen-note p { margin:4px 0; color:var(--vcc-muted); }
.vcc-report-preview { width:100%; }
.vcc-report-preview iframe { width:100%; height:calc(100vh - 260px);
  height:calc(100dvh - 260px); min-height:520px;
  border:1px solid var(--vcc-border); border-radius:14px;
  background:var(--vcc-report-surface); }

/* Keep action labels at AA contrast in both themes. */
.gradio-container button.primary {
  color: #ffffff !important;
  background-color: #c2410c !important;
  border-color: #ea580c !important;
}
.gradio-container button.stop {
  color: #ffffff !important;
  background-color: #b91c1c !important;
  border-color: #ef4444 !important;
}
.gradio-container button.primary:hover { background-color: #9a3412 !important; }
.gradio-container button.stop:hover { background-color: #991b1b !important; }

/* Gradio's audio controls use button variants with their own foreground colors. */
body.dark .gradio-container button.secondary,
body.dark .gradio-container button.record,
body.dark .gradio-container button.upload-button {
  color: #f8fafc !important;
  background-color: #3f3f46 !important;
  border-color: #71717a !important;
}
body.dark .gradio-container button.secondary:hover,
body.dark .gradio-container button.record:hover,
body.dark .gradio-container button.upload-button:hover {
  background-color: #52525b !important;
}
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="tab"]:focus-visible {
  outline: 3px solid var(--vcc-focus) !important;
  outline-offset: 2px;
}
footer { display:none !important; }
"""


KANJI_PATTERN = re.compile(r"[一-龯々〆ヵヶ〇]")
HIRAGANA_PATTERN = re.compile(r"[ぁ-ゖゝゞ]")


def split_okurigana(base: str, reading: str) -> tuple[str, str, str]:
    """Detach a shared trailing hiragana suffix from a ruby annotation."""
    suffix_length = 0
    limit = min(len(base), len(reading))
    while suffix_length < limit:
        base_character = base[-suffix_length - 1]
        reading_character = reading[-suffix_length - 1]
        if (
            base_character != reading_character
            or not HIRAGANA_PATTERN.fullmatch(base_character)
        ):
            break
        suffix_length += 1
    if not suffix_length:
        return base, reading, ""

    ruby_base = base[:-suffix_length]
    ruby_reading = reading[:-suffix_length]
    if not ruby_reading or not KANJI_PATTERN.search(ruby_base):
        return base, reading, ""
    return ruby_base, ruby_reading, base[-suffix_length:]


def ruby_markup(prompt: Prompt) -> str:
    parts = []
    for base, reading in prompt.reading_segments:
        if base != reading and KANJI_PATTERN.search(base):
            ruby_base, ruby_reading, suffix = split_okurigana(base, reading)
            parts.append(
                f"<ruby>{html.escape(ruby_base)}<rp>（</rp>"
                f"<rt>{html.escape(ruby_reading)}</rt><rp>）</rp></ruby>"
                f"{html.escape(suffix)}"
            )
        else:
            parts.append(html.escape(base))
    return "".join(parts)


def select_listening_pairs(
    rows: list[dict[str, Any]],
    candidate_ids: list[str],
    evaluation_ids: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose one deterministic generation for each candidate pair and text."""
    by_candidate_and_evaluation: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["prompt_id"], row["eval_id"])
        by_candidate_and_evaluation.setdefault(key, []).append(row)
    for choices in by_candidate_and_evaluation.values():
        choices.sort(key=lambda row: (row["take"], row["seed"], row["id"]))

    pairs = []
    for left, right in itertools.combinations(candidate_ids, 2):
        for evaluation_id in evaluation_ids:
            left_rows = by_candidate_and_evaluation.get((left, evaluation_id), [])
            right_rows = by_candidate_and_evaluation.get((right, evaluation_id), [])
            if left_rows and right_rows:
                pairs.append((left_rows[0], right_rows[0]))
    return pairs


def select_duration_listening_pairs(
    rows: list[dict[str, Any]],
    condition_ids: list[str],
    evaluation_ids: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build 36 matched adjacent-length comparisons (3 takes x 4 texts x 3 pairs)."""
    selected_evaluations = evaluation_ids[:4]
    indexed = {
        (row["prompt_id"], int(row["take"]), row["eval_id"], int(row["seed"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    if not seeds:
        return []
    seed = seeds[0]
    takes = sorted({int(row["take"]) for row in rows})
    pairs = []
    for short_id, long_id in zip(condition_ids, condition_ids[1:]):
        for take in takes:
            for evaluation_id in selected_evaluations:
                left = indexed.get((short_id, take, evaluation_id, seed))
                right = indexed.get((long_id, take, evaluation_id, seed))
                if left and right:
                    pairs.append((left, right))
    return pairs


def report_preview(document: str) -> str:
    return (
        '<div class="vcc-report-preview">'
        '<iframe title="レポートプレビュー" sandbox="allow-same-origin" '
        f'srcdoc="{html.escape(document, quote=True)}"></iframe></div>'
    )


def source_recording_status(
    recording: dict[str, Any] | None,
    slot: int,
    *,
    enabled: bool = True,
    audio_available: bool = True,
) -> str:
    if not enabled:
        return (
            "### 元音声スロットは使用しません\n\n"
            "この実験は全候補を本人が録音する方式です。候補セリフと共通アンカーを収録してください。"
        )
    if recording is None:
        return (
            f"### ◯ スロット {slot}: 未登録\n\n"
            "下の「新しい音声」で録音またはファイルを選び、台本を入力して保存してください。"
        )

    warnings = json.loads(recording["warnings_json"] or "[]")
    if not audio_available:
        warnings = [*warnings, "保存済み音声ファイルが見つかりません"]
    quality_ok = bool(recording["quality_ok"]) and audio_available
    state = "✅ 登録済み・利用可能" if quality_ok else "⚠️ 登録済み・要再録"
    transcript = html.escape(str(recording.get("transcript") or "—"))
    saved_text = html.escape(str(recording["text"]))
    cer = "—" if recording.get("cer") is None else f"{recording['cer']:.3f}"
    warning_text = "、".join(html.escape(str(item)) for item in warnings) or "なし"
    return (
        f"### {state} — スロット {slot}\n\n"
        "現在保存されている音声は下のプレイヤーで確認できます。"
        "再録音する場合は「新しい音声」で録音し、上書きボタンを押してください。\n\n"
        f"- 保存した台本: {saved_text}\n"
        f"- ASR: {transcript}\n"
        f"- 台本CER: {cer}\n"
        f"- 長さ: {recording['duration']:.2f}秒\n"
        f"- 推定SNR: {recording['snr_db']:.1f} dB\n"
        f"- 注意: {warning_text}"
    )


def candidate_reference_status(
    recording: dict[str, Any] | None,
    prompt_text: str,
    prompt_id: str,
    take: int,
    *,
    audio_available: bool = True,
) -> str:
    text = html.escape(prompt_text)
    if recording is None:
        return (
            '<div class="vcc-listen-note">'
            f"<strong>{html.escape(prompt_id)} / テイク {take}: 未生成</strong>"
            f"<p>{text}</p>"
            "<p>「12候補×2テイクを合成」の完了後に、一覧を更新してください。</p>"
            "</div>"
        )
    warnings = json.loads(recording["warnings_json"] or "[]")
    if not audio_available:
        warnings = [*warnings, "保存済み音声ファイルが見つかりません"]
    ready = bool(recording["quality_ok"]) and audio_available
    state = "利用可能" if ready else "要確認"
    transcript = html.escape(str(recording.get("transcript") or "—"))
    cer = "—" if recording.get("cer") is None else f"{recording['cer']:.3f}"
    seed = recording.get("generation_seed")
    model = html.escape(str(recording.get("model_id") or "—"))
    warning_text = "、".join(html.escape(str(item)) for item in warnings) or "なし"
    return (
        '<div class="vcc-listen-note">'
        f"<strong>{html.escape(prompt_id)} / テイク {take}: {state}</strong>"
        f"<p>{text}</p>"
        f"<p>seed: {seed if seed is not None else '—'} ／ CER: {cer} ／ "
        f"長さ: {recording['duration']:.2f}秒 ／ SNR: {recording['snr_db']:.1f} dB</p>"
        f"<p>ASR: {transcript}</p>"
        f"<p>モデル: {model}</p>"
        f"<p>注意: {warning_text}</p>"
        "</div>"
    )


def build_app(service: ExperimentService | None = None) -> gr.Blocks:
    service = service or ExperimentService()
    config = service.config
    run_state: dict[str, dict[str, Any]] = {}
    state_lock = threading.Lock()

    candidate_text = {item.id: item.text for item in config.candidates}
    anchor_text = {item.id: item.text for item in config.anchors}
    prompts = {item.id: item for item in (*config.candidates, *config.anchors)}

    def prompt_card(prompt_id: str) -> str:
        prompt = prompts.get(prompt_id)
        if prompt is None:
            return (
                '<div class="vcc-prompt"><div class="vcc-prompt-text" lang="ja">'
                f"{html.escape(str(prompt_id))}</div></div>"
            )
        return (
            '<div class="vcc-prompt">'
            f'<div class="vcc-prompt-text" lang="ja">{ruby_markup(prompt)}</div>'
            "</div>"
        )

    def select_candidate(prompt_id: str):
        return prompt_card(prompt_id), gr.update(value=1)

    mode_labels = {
        "recorded": "全候補を実録音",
        "synthetic": "合成音声で事前選定",
        "validation": "上位3件を実録音で検証",
        "duration": "入力音声長を調査",
    }

    def experiment_choices() -> list[tuple[str, str]]:
        return [
            (
                f"{row['name']} · {mode_labels.get(row['mode'], row['mode'])} · "
                f"{row['status']}",
                row["id"],
            )
            for row in service.db.list_experiments()
        ]

    def candidate_choices(experiment_id: str | None) -> list[tuple[str, str]]:
        selected_config = service.experiment_config(experiment_id) if experiment_id else config
        return [
            (f"{item.id} · {item.category}", item.id)
            for item in selected_config.candidates
        ]

    def experiment_candidate_update(experiment_id: str | None):
        choices = candidate_choices(experiment_id)
        selected = choices[0][1] if choices else None
        return gr.update(choices=choices, value=selected), (
            prompt_card(selected) if selected else ""
        )

    def source_slot_updates(experiment_id: str | None, slot: int):
        if not experiment_id:
            return (
                gr.update(value=None),
                gr.update(value=None, interactive=False),
                gr.update(value="", interactive=False),
                source_recording_status(None, int(slot), enabled=False),
                gr.update(value=f"スロット {slot} に保存・台本照合", interactive=False),
            )
        experiment = service.db.experiment(experiment_id)
        enabled = bool(experiment and experiment["mode"] in {"synthetic", "validation"})
        recording = service.source_recording(experiment_id, int(slot)) if enabled else None
        audio_path = None
        audio_available = False
        if recording:
            candidate_path = Path(recording["processed_path"])
            audio_available = candidate_path.exists()
            if audio_available:
                audio_path = str(candidate_path)
        button_text = (
            f"スロット {slot} を再録音して上書き"
            if recording
            else f"スロット {slot} に保存・台本照合"
        )
        return (
            gr.update(value=audio_path),
            gr.update(value=None, interactive=enabled),
            gr.update(
                value=recording["text"] if recording else "",
                interactive=enabled,
            ),
            source_recording_status(
                recording,
                int(slot),
                enabled=enabled,
                audio_available=audio_available,
            ),
            gr.update(value=button_text, interactive=enabled),
        )

    def create_experiment(
        name: str, mode: str, source_slot: int, current_experiment: str | None
    ):
        if mode == "duration":
            eligible_ids = {
                row["id"] for row in service.eligible_duration_anchor_experiments()
            }
            anchor_id = current_experiment if current_experiment in eligible_ids else None
            experiment_id = service.create_duration_experiment(anchor_id, name=name)
        else:
            experiment_id = service.create_experiment(name, mode=mode)
        candidate_update, candidate_card = experiment_candidate_update(experiment_id)
        source_updates = source_slot_updates(experiment_id, int(source_slot))
        reference_updates = candidate_reference_experiment_updates(experiment_id)
        return (
            gr.update(choices=experiment_choices(), value=experiment_id),
            f"実験を作成しました: `{experiment_id}`",
            recording_table(experiment_id),
            readiness(experiment_id),
            candidate_update,
            candidate_card,
            *source_updates,
            *reference_updates,
        )

    def refresh_experiments(current: str | None, source_slot: int):
        choices = experiment_choices()
        values = {value for _, value in choices}
        selected = current if current in values else (choices[0][1] if choices else None)
        candidate_update, candidate_card = experiment_candidate_update(selected)
        source_updates = source_slot_updates(selected, int(source_slot))
        reference_updates = candidate_reference_experiment_updates(selected)
        return (
            gr.update(choices=choices, value=selected),
            recording_table(selected),
            readiness(selected),
            candidate_update,
            candidate_card,
            *source_updates,
            *reference_updates,
        )

    def change_experiment(experiment_id: str | None, source_slot: int):
        candidate_update, candidate_card = experiment_candidate_update(experiment_id)
        source_updates = source_slot_updates(experiment_id, int(source_slot))
        reference_updates = candidate_reference_experiment_updates(experiment_id)
        return (
            recording_table(experiment_id),
            readiness(experiment_id),
            candidate_update,
            candidate_card,
            *source_updates,
            *reference_updates,
        )

    def recording_table(experiment_id: str | None):
        if not experiment_id:
            return []
        rows = service.db.recordings(experiment_id)
        return [
            [
                row["kind"],
                row["prompt_id"],
                row["take"],
                f"{row['duration']:.2f}",
                f"{row['snr_db']:.1f}",
                f"{row['clipping_ratio'] * 100:.3f}%",
                "OK" if row["quality_ok"] else "要再録",
                " / ".join(json.loads(row["warnings_json"])) or "—",
            ]
            for row in rows
        ]

    def reference_candidate_choices(
        experiment_id: str | None,
    ) -> list[tuple[str, str]]:
        selected_config = (
            service.experiment_config(experiment_id) if experiment_id else config
        )
        return [
            (f"{item.id} · {item.category}", item.id)
            for item in selected_config.candidates
        ]

    def candidate_reference_table(experiment_id: str | None):
        if not experiment_id:
            return []
        experiment = service.db.experiment(experiment_id)
        if not experiment or experiment["mode"] != "synthetic":
            return []
        selected_config = service.experiment_config(experiment_id)
        saved = {
            (row["prompt_id"], int(row["take"])): row
            for row in service.candidate_references(experiment_id)
        }
        table = []
        for prompt in selected_config.candidates:
            for take in (1, 2):
                row = saved.get((prompt.id, take))
                if row is None:
                    table.append([prompt.id, take, "未生成", "—", "—", "—", "—", "—"])
                    continue
                warnings = " / ".join(json.loads(row["warnings_json"])) or "—"
                table.append(
                    [
                        prompt.id,
                        take,
                        "利用可能" if row["quality_ok"] else "要確認",
                        row["generation_seed"] if row["generation_seed"] is not None else "—",
                        f"{row['duration']:.2f}",
                        "—" if row["cer"] is None else f"{row['cer']:.3f}",
                        f"{row['snr_db']:.1f}",
                        warnings,
                    ]
                )
        return table

    def candidate_reference_view(
        experiment_id: str | None,
        prompt_id: str | None,
        take: int,
    ):
        if not experiment_id or not prompt_id:
            return "", gr.update(value=None), candidate_reference_status(
                None, "候補を選択してください。", prompt_id or "—", int(take)
            )
        experiment = service.db.experiment(experiment_id)
        selected_config = service.experiment_config(experiment_id)
        prompts_by_id = {item.id: item for item in selected_config.candidates}
        prompt = prompts_by_id.get(prompt_id)
        if prompt is None:
            prompt = selected_config.candidates[0]
            prompt_id = prompt.id
        if not experiment or experiment["mode"] != "synthetic":
            return (
                prompt_card(prompt_id),
                gr.update(value=None),
                '<div class="vcc-listen-note"><strong>合成候補はありません</strong>'
                '<p>この実験方式では合成候補音声を使用しません。</p></div>',
            )
        recording = service.candidate_reference(experiment_id, prompt_id, int(take))
        audio_path = None
        audio_available = False
        if recording:
            path = Path(recording["processed_path"])
            audio_available = path.exists()
            if audio_available:
                audio_path = str(path)
        return (
            prompt_card(prompt_id),
            gr.update(value=audio_path),
            candidate_reference_status(
                recording,
                prompt.text,
                prompt_id,
                int(take),
                audio_available=audio_available,
            ),
        )

    def candidate_reference_experiment_updates(experiment_id: str | None):
        choices = reference_candidate_choices(experiment_id)
        selected = choices[0][1] if choices else None
        prompt_html, audio_update, status = candidate_reference_view(
            experiment_id, selected, 1
        )
        return (
            gr.update(choices=choices, value=selected),
            gr.update(value=1),
            candidate_reference_table(experiment_id),
            prompt_html,
            audio_update,
            status,
        )

    def readiness(experiment_id: str | None) -> str:
        if not experiment_id:
            return "実験を作成または選択してください。"
        selected_config = service.experiment_config(experiment_id)
        experiment = service.db.experiment(experiment_id)
        counts = service.recording_progress(experiment_id)
        candidate_goal = len(selected_config.candidates) * selected_config.takes_per_candidate
        anchor_goal = 1 if experiment["mode"] in {"synthetic", "validation"} else len(selected_config.anchors)
        return (
            f"**方式:** {mode_labels.get(experiment['mode'], experiment['mode'])}　"
            f"**候補:** {counts['candidate']} / {candidate_goal}　"
            f"**アンカー:** {counts['anchor']} / {anchor_goal}　"
            f"**品質OK:** {counts['quality_ok']} / {candidate_goal + anchor_goal}"
        )

    def save_candidate(
        experiment_id: str | None,
        prompt_id: str,
        take: int,
        audio_path: str | None,
    ):
        if not experiment_id:
            raise gr.Error("先に実験を作成してください")
        if not audio_path:
            raise gr.Error("音声を録音または選択してください")
        selected_candidates = {
            item.id: item.text
            for item in service.experiment_config(experiment_id).candidates
        }
        if prompt_id not in selected_candidates:
            raise gr.Error("この実験の候補ではありません")
        result = service.save_recording(
            experiment_id,
            "candidate",
            prompt_id,
            int(take),
            selected_candidates[prompt_id],
            audio_path,
        )
        message = quality_message(result)
        return message, recording_table(experiment_id), readiness(experiment_id)

    def save_source(
        experiment_id: str | None,
        take: int,
        transcript: str,
        audio_path: str | None,
    ):
        if not experiment_id:
            raise gr.Error("先に合成事前選定の実験を作成してください")
        existing = service.source_recording(experiment_id, int(take))
        selected_audio = audio_path
        if not selected_audio and existing:
            selected_audio = existing["processed_path"]
        if not selected_audio:
            raise gr.Error("元音声を録音または選択してください")
        service.save_source(
            experiment_id, selected_audio, transcript, take=int(take)
        )
        source_updates = source_slot_updates(experiment_id, int(take))
        return (
            *source_updates,
            recording_table(experiment_id),
            readiness(experiment_id),
        )

    def save_anchor(
        experiment_id: str | None,
        prompt_id: str,
        audio_path: str | None,
    ):
        if not experiment_id:
            raise gr.Error("先に実験を作成してください")
        if not audio_path:
            raise gr.Error("音声を録音または選択してください")
        experiment = service.db.experiment(experiment_id)
        if experiment["mode"] != "recorded":
            raise gr.Error("この実験では共通アンカーの代わりに元音声を使用します")
        result = service.save_recording(
            experiment_id,
            "anchor",
            prompt_id,
            1,
            anchor_text[prompt_id],
            audio_path,
        )
        message = quality_message(result)
        return message, recording_table(experiment_id), readiness(experiment_id)

    def quality_message(result: dict[str, Any]) -> str:
        status = "✅ 品質チェックを通過しました" if result["quality_ok"] else "⚠️ 再録を推奨します"
        warnings = "、".join(result["warnings"]) or "なし"
        return (
            f"{status}\n\n"
            f"- 長さ: {result['duration']:.2f}秒\n"
            f"- 推定SNR: {result['snr_db']:.1f} dB\n"
            f"- ピーク: {result['peak_dbfs']:.1f} dBFS\n"
            f"- 無音率: {result['silence_ratio'] * 100:.1f}%\n"
            f"- 注意: {warnings}"
        )

    def _progress_callback(experiment_id: str):
        def callback(done: int, total: int, message: str) -> None:
            with state_lock:
                state = run_state.setdefault(experiment_id, {})
                state.update(done=done, total=total, message=message, running=True)
        return callback

    def _run_worker(experiment_id: str, smoke: bool, stop_event: threading.Event):
        try:
            result = service.run(
                experiment_id,
                smoke=smoke,
                progress=_progress_callback(experiment_id),
                stop_event=stop_event,
            )
            with state_lock:
                run_state[experiment_id].update(
                    running=False,
                    message=(
                        f"完了 {result['completed']}件 / 失敗 {result['failed']}件 / "
                        f"残り {result['remaining']}件"
                    ),
                )
        except Exception as error:
            with state_lock:
                run_state.setdefault(experiment_id, {}).update(
                    running=False,
                    message=f"エラー: {type(error).__name__}: {error}",
                )

    def _reference_worker(
        experiment_id: str, smoke: bool, stop_event: threading.Event
    ):
        try:
            result = service.generate_candidate_references(
                experiment_id,
                smoke=smoke,
                progress=_progress_callback(experiment_id),
                stop_event=stop_event,
            )
            with state_lock:
                run_state[experiment_id].update(
                    running=False,
                    message=(
                        f"候補参照音声: 新規採用 {result['accepted']}件 / "
                        f"不採用 {result['rejected']}件 / 未完成候補 {result['incomplete']}件"
                    ),
                )
        except Exception as error:
            with state_lock:
                run_state.setdefault(experiment_id, {}).update(
                    running=False,
                    message=f"エラー: {type(error).__name__}: {error}",
                )

    def start_reference_run(experiment_id: str | None, smoke: bool):
        if not experiment_id:
            raise gr.Error("実験を選択してください")
        with state_lock:
            current = run_state.get(experiment_id)
            if current and current.get("running"):
                return "この実験はすでに実行中です。"
            stop_event = threading.Event()
            run_state[experiment_id] = {
                "running": True,
                "done": 0,
                "total": 0,
                "message": "候補参照音声を準備しています",
                "stop_event": stop_event,
            }
        threading.Thread(
            target=_reference_worker,
            args=(experiment_id, smoke, stop_event),
            daemon=True,
            name=f"vcc-reference-{experiment_id}",
        ).start()
        return "候補参照音声の生成を開始しました。"

    def start_run(experiment_id: str | None, smoke: bool):
        if not experiment_id:
            raise gr.Error("実験を選択してください")
        with state_lock:
            current = run_state.get(experiment_id)
            if current and current.get("running"):
                return "この実験はすでに実行中です。"
            stop_event = threading.Event()
            run_state[experiment_id] = {
                "running": True,
                "done": 0,
                "total": 0,
                "message": "実験を準備しています",
                "stop_event": stop_event,
            }
        worker = threading.Thread(
            target=_run_worker,
            args=(experiment_id, smoke, stop_event),
            daemon=True,
            name=f"vcc-{experiment_id}",
        )
        worker.start()
        return "実験を開始しました。画面を閉じても、このアプリが起動中なら処理は続きます。"

    def stop_run(experiment_id: str | None):
        if not experiment_id:
            return "実験を選択してください。"
        with state_lock:
            state = run_state.get(experiment_id)
            if not state or not state.get("running"):
                return "実行中の処理はありません。"
            state["stop_event"].set()
            state["message"] = "現在の音声が終わり次第、一時停止します"
        return state["message"]

    def poll_run(experiment_id: str | None):
        if not experiment_id:
            return "待機中", [], readiness(experiment_id)
        with state_lock:
            state = dict(run_state.get(experiment_id, {}))
        generations = service.db.generations(experiment_id)
        complete = sum(row["status"] == "complete" for row in generations)
        failed = sum(row["status"] == "failed" for row in generations)
        total = len(generations)
        message = state.get("message") or "未実行"
        status = (
            f"### {message}\n\n"
            f"保存済み: **{complete} / {total}**　処理エラー: **{failed}**"
        )
        errors = [
            [row["prompt_id"], row["eval_id"], row["seed"], row["error"]]
            for row in generations
            if row["status"] == "failed"
        ][-20:]
        return status, errors, readiness(experiment_id)

    def ranking_data(
        experiment_id: str | None,
        similarity_weight: float,
        utmos_weight: float,
        intelligibility_weight: float,
        automatic_weight: float,
    ):
        if not experiment_id:
            return [], "<p>実験を選択してください。</p>"
        selected_config = service.experiment_config(experiment_id)
        selected_text = {item.id: item.text for item in selected_config.candidates}
        rows = [dict(row) for row in service.db.generations(experiment_id, complete_only=True)]
        votes = [dict(row) for row in service.db.votes(experiment_id)]
        weights = dict(selected_config.raw["ranking"])
        weights.update(
            similarity=similarity_weight,
            utmos=utmos_weight,
            intelligibility=intelligibility_weight,
            automatic=automatic_weight,
            listening=1.0 - automatic_weight,
        )
        ranking = rank_candidates(
            rows,
            weights,
            votes,
            bootstrap_samples=int(weights["bootstrap_samples"]),
        )
        table = [
            [
                row["rank"],
                row["candidate_id"],
                selected_text.get(row["candidate_id"], ""),
                round(row["final_score"], 4),
                f"{row['ci_low']:.3f}–{row['ci_high']:.3f}",
                round(row["similarity"], 4),
                round(row["utmos"], 3),
                round(row["cer"], 4),
                f"{row['failure_rate'] * 100:.1f}%",
                "—" if row["listening_win_rate"] is None else f"{row['listening_win_rate'] * 100:.1f}%",
            ]
            for row in ranking
        ]
        cards = "".join(
            (
                f'<div class="vcc-score"><span>#{row["rank"]} {html.escape(row["candidate_id"])}</span>'
                f'<strong>{row["final_score"]:.3f}</strong>'
                f'<span>{html.escape(selected_text.get(row["candidate_id"], ""))}</span>'
                f'<div class="vcc-bar"><i style="width:{row["final_score"] * 100:.1f}%"></i></div></div>'
            )
            for row in ranking[:3]
        )
        return table, f'<div class="vcc-score-grid">{cards}</div>' if cards else "<p>完了データがありません。</p>"

    def next_pair(experiment_id: str | None):
        if not experiment_id:
            raise gr.Error("実験を選択してください")
        selected_config = service.experiment_config(experiment_id)
        rows = [dict(row) for row in service.db.generations(experiment_id, complete_only=True)]
        experiment = service.db.experiment(experiment_id)
        previous = {
            frozenset((row["generation_a"], row["generation_b"]))
            for row in service.db.votes(experiment_id)
        }
        available_rows = [row for row in rows if Path(row["output_path"]).exists()]
        if experiment and experiment["mode"] == "duration":
            condition_ids = [
                str(item["id"])
                for item in selected_config.raw["duration_study"]["targets"]
            ]
            comparisons = select_duration_listening_pairs(
                available_rows,
                condition_ids,
                [evaluation.id for evaluation in selected_config.evaluations],
            )
        else:
            weights = selected_config.raw["ranking"]
            ranking = rank_candidates(rows, weights, (), bootstrap_samples=200)
            top = [row["candidate_id"] for row in ranking[:3]]
            if len(top) < 2:
                raise gr.Error("A/B試聴には完了済み候補が2つ以上必要です")
            available_rows = [row for row in available_rows if row["prompt_id"] in top]
            comparisons = select_listening_pairs(
                available_rows,
                top,
                [evaluation.id for evaluation in selected_config.evaluations],
            )
        comparison_keys = [frozenset((a["id"], b["id"])) for a, b in comparisons]
        completed = sum(key in previous for key in comparison_keys)
        total = len(comparisons)
        for a, b in comparisons:
            key = frozenset((a["id"], b["id"]))
            if key in previous:
                continue
            evaluation = next(
                item for item in selected_config.evaluations
                if item.id == a["eval_id"]
            )
            swap = (completed + int(a["id"])) % 2 == 1
            if swap:
                a, b = b, a
            pair = {
                "candidate_a": a["prompt_id"],
                "candidate_b": b["prompt_id"],
                "generation_a": a["id"],
                "generation_b": b["id"],
            }
            current = completed + 1
            percent = current / total * 100 if total else 0
            message = (
                f"### 比較 {current} / {total}\n\n"
                f'<div class="vcc-listening-progress" role="progressbar" '
                f'aria-valuenow="{current}" aria-valuemin="0" aria-valuemax="{total}">'
                f'<i style="width:{percent:.1f}%"></i></div>\n\n'
                "同じ読み上げ文を、**異なる参照音声から1回ずつ生成**した比較です。"
                "乱数やテイク違いによる同内容の比較は繰り返しません。\n\n"
                f"**評価文:** {evaluation.text}"
            )
            return a["output_path"], b["output_path"], pair, message
        if total:
            return (
                None,
                None,
                {},
                f"### 試聴完了 {total} / {total}\n\n"
                "すべての比較が終わりました。結果タブで順位を確認してください。",
            )
        return None, None, {}, "比較できる音声がありません。合成・評価の完了状況を確認してください。"

    def vote(experiment_id: str | None, pair: dict[str, Any], winner: str):
        if not experiment_id or not pair:
            raise gr.Error("先に比較音声を読み込んでください")
        service.db.add_vote(
            experiment_id,
            pair["candidate_a"],
            pair["candidate_b"],
            int(pair["generation_a"]),
            int(pair["generation_b"]),
            winner,
        )
        return next_pair(experiment_id)

    def export_reports(
        experiment_id: str | None,
        similarity_weight: float,
        utmos_weight: float,
        intelligibility_weight: float,
        automatic_weight: float,
    ):
        if not experiment_id:
            raise gr.Error("実験を選択してください")
        weights = {
            "similarity": similarity_weight,
            "utmos": utmos_weight,
            "intelligibility": intelligibility_weight,
            "automatic": automatic_weight,
            "listening": 1.0 - automatic_weight,
        }
        selected_config = service.experiment_config(experiment_id)
        paths = ReportBuilder(
            service.db,
            selected_config,
            service.experiment_dir(experiment_id),
        ).build(experiment_id, weights)
        return (
            report_preview(paths["html"].read_text(encoding="utf-8")),
            str(paths["html"]),
            str(paths["csv"]),
            str(paths["json"]),
            "レポートを更新しました。下の画面で確認できます。",
        )

    def create_validation(experiment_id: str | None, source_slot: int):
        if not experiment_id:
            raise gr.Error("合成事前選定の実験を選択してください")
        validation_id = service.create_validation_experiment(experiment_id)
        candidate_update, candidate_card = experiment_candidate_update(validation_id)
        source_updates = source_slot_updates(validation_id, int(source_slot))
        reference_updates = candidate_reference_experiment_updates(validation_id)
        return (
            gr.update(choices=experiment_choices(), value=validation_id),
            f"上位3件の実録音検証を作成しました: `{validation_id}`",
            recording_table(validation_id),
            readiness(validation_id),
            candidate_update,
            candidate_card,
            *source_updates,
            *reference_updates,
        )

    def duration_passage_text(experiment_id: str | None, passage: int) -> str:
        selected = service.experiment_config(experiment_id) if experiment_id else config
        settings = selected.raw.get("duration_study", {})
        passages = settings.get("passages", [])
        if not passages:
            return "調査台本が設定されていません。"
        item = passages[int(passage) - 1]
        lines = [f"### 調査台本 {passage}", "各区切りで短く自然に間を置いてください。"]
        lines.extend(
            f"{index}. {segment}" for index, segment in enumerate(item["segments"], start=1)
        )
        return "\n\n".join(lines)

    def suggest_duration(
        experiment_id: str | None, audio_path: str | None
    ):
        if not experiment_id or not audio_path:
            raise gr.Error("長さ調査実験と録音を選択してください")
        result = service.suggest_duration_boundaries(experiment_id, audio_path)
        boundaries = result["boundaries"]
        return (*boundaries, f"録音長 {result['duration']:.2f}秒。無音位置から境界を提案しました。試聴して必要なら調整してください。")

    def save_duration(
        experiment_id: str | None,
        passage: int,
        audio_path: str | None,
        b1: float,
        b2: float,
        b3: float,
        b4: float,
    ):
        if not experiment_id or not audio_path:
            raise gr.Error("長さ調査実験と録音を選択してください")
        results = service.save_duration_passage(
            experiment_id,
            int(passage),
            audio_path,
            [b1, b2, b3, b4],
        )
        lines = [f"### 台本 {passage} を4条件に切り出しました"]
        for result in results:
            state = "OK" if result["quality_ok"] else "要再録"
            lines.append(
                f"- {result['prompt_id']}: {result['duration']:.2f}秒 / "
                f"CER {result['cer']:.3f} / {state}"
            )
        return "\n".join(lines), recording_table(experiment_id), readiness(experiment_id)

    def preview_duration(
        experiment_id: str | None,
        audio_path: str | None,
        b1: float,
        b2: float,
        b3: float,
        b4: float,
    ):
        if not experiment_id or not audio_path:
            raise gr.Error("長さ調査実験と録音を選択してください")
        return tuple(service.preview_duration_clips(
            experiment_id, audio_path, [b1, b2, b3, b4]
        ))

    with gr.Blocks(title="Voice Clone Check") as app:
        gr.HTML(
            """
            <section class="vcc-hero">
              <h1>Voice Clone Check</h1>
              <p>12種類の日本語セリフを同じ条件で比べ、本人らしさ・自然さ・読みの正確さから、Qwen3-TTSに最適な参照音声を探します。</p>
            </section>
            """
        )
        with gr.Row(elem_classes="vcc-experiment-bar"):
            experiment_select = gr.Dropdown(
                label="実験",
                choices=experiment_choices(),
                interactive=True,
                scale=3,
            )
            refresh_button = gr.Button("一覧を更新", scale=1)
            experiment_name = gr.Textbox(
                label="新しい実験名",
                placeholder="例: 内蔵マイク・自然な会話調",
                scale=3,
            )
            experiment_mode = gr.Dropdown(
                label="方式",
                choices=[
                    ("サンプル音声から事前選定", "synthetic"),
                    ("全候補を本人が録音", "recorded"),
                    ("入力音声長を調査", "duration"),
                ],
                value="synthetic",
                scale=2,
            )
            create_button = gr.Button("新しい実験を作成", variant="primary", scale=1)
        experiment_message = gr.Markdown(elem_classes="vcc-experiment-message")

        with gr.Tabs():
            with gr.Tab("1. 収録"):
                recording_readiness = gr.Markdown("実験を作成または選択してください。")
                gr.Markdown(
                    "合成事前選定では、最初に元音声と正確な台本を登録します。実録音方式と検証実験では、候補を各2テイク録音します。"
                )
                with gr.Accordion("合成事前選定: 元音声を登録", open=True):
                    gr.Markdown(
                        "本人の許諾を得た、1人だけが話す5〜15秒の音声を使用してください。ASR結果と台本のCERが0.10以下の場合に利用できます。"
                    )
                    source_take = gr.Radio(
                        label="元音声スロット", choices=[1, 2, 3], value=1
                    )
                    source_quality = gr.Markdown(
                        source_recording_status(None, 1, enabled=False)
                    )
                    saved_source_audio = gr.Audio(
                        label="現在保存されている元音声",
                        interactive=False,
                    )
                    source_transcript = gr.Textbox(
                        label="元音声の正確な台本",
                        lines=3,
                        placeholder="音声で実際に話している内容を一字一句入力",
                        interactive=False,
                    )
                    source_audio = gr.Audio(
                        label="新しい音声（ブラウザで録音またはファイルを選択）",
                        sources=["microphone", "upload"],
                        type="filepath",
                        format="wav",
                        interactive=False,
                    )
                    save_source_button = gr.Button(
                        "スロット 1 に保存・台本照合",
                        variant="primary",
                        interactive=False,
                    )
                with gr.Accordion("入力音声長調査: 3本の連続音声を登録", open=True):
                    gr.Markdown(
                        "長さ調査方式で使用します。15秒前後を連続録音し、各句の終わりの境界を確認して保存してください。"
                    )
                    duration_passage = gr.Radio(
                        label="調査台本", choices=[1, 2, 3], value=1
                    )
                    duration_prompt = gr.Markdown(duration_passage_text(None, 1))
                    duration_audio = gr.Audio(
                        label="長さ調査用の連続音声",
                        sources=["microphone", "upload"],
                        type="filepath",
                        format="wav",
                    )
                    suggest_duration_button = gr.Button("無音位置から境界を提案")
                    with gr.Row():
                        duration_b1 = gr.Number(label="句1 終了秒", value=4.0)
                        duration_b2 = gr.Number(label="句2 終了秒", value=8.0)
                        duration_b3 = gr.Number(label="句3 終了秒", value=12.0)
                        duration_b4 = gr.Number(label="句4 終了秒", value=15.0)
                    preview_duration_button = gr.Button("境界ごとの累積音声を試聴")
                    with gr.Row():
                        duration_preview1 = gr.Audio(label="約4秒", interactive=False)
                        duration_preview2 = gr.Audio(label="約8秒", interactive=False)
                        duration_preview3 = gr.Audio(label="約12秒", interactive=False)
                        duration_preview4 = gr.Audio(label="約15秒", interactive=False)
                    save_duration_button = gr.Button(
                        "4つの累積参照音声を保存・台本照合", variant="primary"
                    )
                    duration_status = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 候補セリフ")
                        candidate_select = gr.Dropdown(
                            label="候補",
                            choices=[
                                (f"{item.id} · {item.category}", item.id)
                                for item in config.candidates
                            ],
                            value=config.candidates[0].id,
                        )
                        candidate_prompt = gr.HTML(
                            prompt_card(config.candidates[0].id)
                        )
                        candidate_take = gr.Radio(
                            label="テイク", choices=[1, 2], value=1
                        )
                        candidate_audio = gr.Audio(
                            label="録音またはWAVを選択",
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                        )
                        save_candidate_button = gr.Button(
                            "候補音声を保存・検査", variant="primary"
                        )
                        candidate_quality = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### 共通アンカー")
                        anchor_select = gr.Dropdown(
                            label="アンカー",
                            choices=[
                                (item.id, item.id)
                                for item in config.anchors
                            ],
                            value=config.anchors[0].id,
                        )
                        anchor_prompt = gr.HTML(prompt_card(config.anchors[0].id))
                        anchor_audio = gr.Audio(
                            label="録音またはWAVを選択",
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                        )
                        save_anchor_button = gr.Button("アンカー音声を保存・検査")
                        anchor_quality = gr.Markdown()
                recordings_frame = gr.Dataframe(
                    headers=[
                        "種別", "ID", "テイク", "秒", "SNR", "clip",
                        "品質", "注意",
                    ],
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("2. 合成・評価"):
                gr.Markdown(
                    "合成事前選定では、先に候補参照音声を生成してください。その後、評価音声を生成します。初回はモデルを取得します。"
                )
                with gr.Row():
                    reference_smoke_button = gr.Button("候補生成スモーク")
                    reference_full_button = gr.Button(
                        "12候補×2テイクを合成", variant="primary"
                    )
                with gr.Row():
                    smoke_button = gr.Button("評価スモークテスト")
                    full_button = gr.Button("全件の評価を開始", variant="primary")
                    stop_button = gr.Button("一時停止", variant="stop")
                run_status = gr.Markdown("未実行")
                error_table = gr.Dataframe(
                    headers=["候補", "評価文", "seed", "エラー"],
                    interactive=False,
                    wrap=True,
                )
                timer = gr.Timer(2.0, active=True)

            with gr.Tab("3. 合成候補の試聴") as reference_listening_tab:
                gr.Markdown(
                    "品質検査を通過し、実際の評価に使用される合成候補を試聴できます。候補生成の完了後、このタブを開くか一覧を更新してください。"
                )
                reference_refresh_button = gr.Button(
                    "合成候補の一覧を更新", variant="primary"
                )
                reference_table = gr.Dataframe(
                    headers=[
                        "候補", "テイク", "状態", "seed", "秒", "CER", "SNR", "注意",
                    ],
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    reference_candidate_select = gr.Dropdown(
                        label="試聴する候補",
                        choices=[
                            (f"{item.id} · {item.category}", item.id)
                            for item in config.candidates
                        ],
                        value=config.candidates[0].id,
                    )
                    reference_take = gr.Radio(
                        label="テイク", choices=[1, 2], value=1
                    )
                reference_prompt = gr.HTML(prompt_card(config.candidates[0].id))
                reference_audio = gr.Audio(
                    label="選択中の合成候補音声",
                    interactive=False,
                )
                reference_status = gr.HTML(
                    candidate_reference_status(
                        None,
                        config.candidates[0].text,
                        config.candidates[0].id,
                        1,
                    )
                )

            with gr.Tab("4. 結果"):
                ranking_cards = gr.HTML("<p>完了データがありません。</p>")
                with gr.Accordion("ランキング重み", open=False):
                    similarity_weight = gr.Slider(
                        0, 1, value=0.50, step=0.05, label="話者類似度"
                    )
                    utmos_weight = gr.Slider(
                        0, 1, value=0.30, step=0.05, label="UTMOS"
                    )
                    intelligibility_weight = gr.Slider(
                        0, 1, value=0.20, step=0.05, label="読みの正確性"
                    )
                    automatic_weight = gr.Slider(
                        0, 1, value=0.80, step=0.05, label="自動評価の比率"
                    )
                ranking_button = gr.Button("ランキングを更新", variant="primary")
                validation_button = gr.Button("上位3件の実録音検証を作成")
                ranking_table = gr.Dataframe(
                    headers=[
                        "順位", "候補", "セリフ", "総合", "95% CI",
                        "類似度", "UTMOS", "CER", "失敗率", "試聴勝率",
                    ],
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("5. ブラインド試聴"):
                listening_message = gr.Markdown(
                    "自動評価の上位3候補を、各評価文につき1回ずつ比較します。"
                )
                pair_state = gr.State({})
                load_pair_button = gr.Button("比較を開始 / 次へ", variant="primary")
                with gr.Row():
                    audio_a = gr.Audio(label="音声 A", interactive=False)
                    audio_b = gr.Audio(label="音声 B", interactive=False)
                with gr.Row():
                    vote_a = gr.Button("A が良い")
                    vote_tie = gr.Button("同程度")
                    vote_b = gr.Button("B が良い")

            with gr.Tab("6. レポート"):
                gr.Markdown(
                    "現在の重みと試聴結果でレポートを更新し、この画面に表示します。必要な場合だけ各形式をダウンロードできます。"
                )
                export_button = gr.Button("レポートを更新・表示", variant="primary")
                export_status = gr.Markdown()
                report_frame = gr.HTML(
                    "<p>「レポートを更新・表示」を押すと、ここに結果が表示されます。</p>"
                )
                with gr.Accordion("ダウンロード", open=False):
                    html_file = gr.File(label="HTMLレポート")
                    csv_file = gr.File(label="全生成データ CSV")
                    json_file = gr.File(label="集計 JSON")

        create_button.click(
            create_experiment,
            inputs=[experiment_name, experiment_mode, source_take, experiment_select],
            outputs=[
                experiment_select,
                experiment_message,
                recordings_frame,
                recording_readiness,
                candidate_select,
                candidate_prompt,
                saved_source_audio,
                source_audio,
                source_transcript,
                source_quality,
                save_source_button,
                reference_candidate_select,
                reference_take,
                reference_table,
                reference_prompt,
                reference_audio,
                reference_status,
            ],
        )
        refresh_button.click(
            refresh_experiments,
            inputs=[experiment_select, source_take],
            outputs=[
                experiment_select,
                recordings_frame,
                recording_readiness,
                candidate_select,
                candidate_prompt,
                saved_source_audio,
                source_audio,
                source_transcript,
                source_quality,
                save_source_button,
                reference_candidate_select,
                reference_take,
                reference_table,
                reference_prompt,
                reference_audio,
                reference_status,
            ],
        )
        experiment_select.change(
            change_experiment,
            inputs=[experiment_select, source_take],
            outputs=[
                recordings_frame,
                recording_readiness,
                candidate_select,
                candidate_prompt,
                saved_source_audio,
                source_audio,
                source_transcript,
                source_quality,
                save_source_button,
                reference_candidate_select,
                reference_take,
                reference_table,
                reference_prompt,
                reference_audio,
                reference_status,
            ],
        )
        source_take.change(
            source_slot_updates,
            inputs=[experiment_select, source_take],
            outputs=[
                saved_source_audio,
                source_audio,
                source_transcript,
                source_quality,
                save_source_button,
            ],
        )
        duration_passage.change(
            duration_passage_text,
            inputs=[experiment_select, duration_passage],
            outputs=[duration_prompt],
        )
        suggest_duration_button.click(
            suggest_duration,
            inputs=[experiment_select, duration_audio],
            outputs=[duration_b1, duration_b2, duration_b3, duration_b4, duration_status],
        )
        save_duration_button.click(
            save_duration,
            inputs=[
                experiment_select, duration_passage, duration_audio,
                duration_b1, duration_b2, duration_b3, duration_b4,
            ],
            outputs=[duration_status, recordings_frame, recording_readiness],
        )
        preview_duration_button.click(
            preview_duration,
            inputs=[
                experiment_select, duration_audio,
                duration_b1, duration_b2, duration_b3, duration_b4,
            ],
            outputs=[
                duration_preview1, duration_preview2,
                duration_preview3, duration_preview4,
            ],
        )
        reference_listening_tab.select(
            candidate_reference_experiment_updates,
            inputs=[experiment_select],
            outputs=[
                reference_candidate_select,
                reference_take,
                reference_table,
                reference_prompt,
                reference_audio,
                reference_status,
            ],
        )
        reference_refresh_button.click(
            candidate_reference_experiment_updates,
            inputs=[experiment_select],
            outputs=[
                reference_candidate_select,
                reference_take,
                reference_table,
                reference_prompt,
                reference_audio,
                reference_status,
            ],
        )
        for component in (reference_candidate_select, reference_take):
            component.change(
                candidate_reference_view,
                inputs=[
                    experiment_select,
                    reference_candidate_select,
                    reference_take,
                ],
                outputs=[reference_prompt, reference_audio, reference_status],
            )
        candidate_select.change(
            select_candidate,
            inputs=[candidate_select],
            outputs=[candidate_prompt, candidate_take],
        )
        anchor_select.change(
            prompt_card,
            inputs=[anchor_select],
            outputs=[anchor_prompt],
        )
        save_candidate_button.click(
            save_candidate,
            inputs=[
                experiment_select,
                candidate_select,
                candidate_take,
                candidate_audio,
            ],
            outputs=[candidate_quality, recordings_frame, recording_readiness],
        )
        save_source_button.click(
            save_source,
            inputs=[
                experiment_select,
                source_take,
                source_transcript,
                source_audio,
            ],
            outputs=[
                saved_source_audio,
                source_audio,
                source_transcript,
                source_quality,
                save_source_button,
                recordings_frame,
                recording_readiness,
            ],
        )
        save_anchor_button.click(
            save_anchor,
            inputs=[experiment_select, anchor_select, anchor_audio],
            outputs=[anchor_quality, recordings_frame, recording_readiness],
        )
        reference_smoke_button.click(
            lambda experiment_id: start_reference_run(experiment_id, True),
            inputs=[experiment_select],
            outputs=[run_status],
        )
        reference_full_button.click(
            lambda experiment_id: start_reference_run(experiment_id, False),
            inputs=[experiment_select],
            outputs=[run_status],
        )
        smoke_button.click(
            lambda experiment_id: start_run(experiment_id, True),
            inputs=[experiment_select],
            outputs=[run_status],
        )
        full_button.click(
            lambda experiment_id: start_run(experiment_id, False),
            inputs=[experiment_select],
            outputs=[run_status],
        )
        stop_button.click(
            stop_run, inputs=[experiment_select], outputs=[run_status]
        )
        timer.tick(
            poll_run,
            inputs=[experiment_select],
            outputs=[run_status, error_table, recording_readiness],
        )
        ranking_button.click(
            ranking_data,
            inputs=[
                experiment_select,
                similarity_weight,
                utmos_weight,
                intelligibility_weight,
                automatic_weight,
            ],
            outputs=[ranking_table, ranking_cards],
        )
        validation_button.click(
            create_validation,
            inputs=[experiment_select, source_take],
            outputs=[
                experiment_select,
                experiment_message,
                recordings_frame,
                recording_readiness,
                candidate_select,
                candidate_prompt,
                saved_source_audio,
                source_audio,
                source_transcript,
                source_quality,
                save_source_button,
                reference_candidate_select,
                reference_take,
                reference_table,
                reference_prompt,
                reference_audio,
                reference_status,
            ],
        )
        load_pair_button.click(
            next_pair,
            inputs=[experiment_select],
            outputs=[audio_a, audio_b, pair_state, listening_message],
        )
        for button, winner in ((vote_a, "a"), (vote_tie, "tie"), (vote_b, "b")):
            button.click(
                lambda experiment_id, pair, choice=winner: vote(
                    experiment_id, pair, choice
                ),
                inputs=[experiment_select, pair_state],
                outputs=[audio_a, audio_b, pair_state, listening_message],
            )
        export_button.click(
            export_reports,
            inputs=[
                experiment_select,
                similarity_weight,
                utmos_weight,
                intelligibility_weight,
                automatic_weight,
            ],
            outputs=[report_frame, html_file, csv_file, json_file, export_status],
        )
    return app
