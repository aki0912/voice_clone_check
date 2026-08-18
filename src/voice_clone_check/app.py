from __future__ import annotations

import html
import itertools
import json
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from .reports import ReportBuilder
from .scoring import rank_candidates
from .service import ExperimentService


CSS = """
:root { --vcc-accent: #4856d8; --vcc-ink: #172033; --vcc-muted: #68738a; }
.gradio-container { max-width: 1220px !important; color: var(--vcc-ink); }
.vcc-hero { padding: 12px 16px; border-radius: 14px;
  background: linear-gradient(135deg,#f0f2ff 0%,#f8f5ff 52%,#eef8f5 100%);
  border: 1px solid #dfe3f2; margin-bottom: 8px; }
.vcc-hero h1 { margin: 0 0 2px; font-size: clamp(21px,3vw,28px); letter-spacing: -.025em; }
.vcc-hero p { margin: 0; color: var(--vcc-muted); max-width: 920px; font-size:13px; }
.vcc-experiment-bar { align-items:end; gap:8px; margin-bottom:4px; }
.vcc-experiment-bar button { min-width:112px; }
.vcc-experiment-message:empty { display:none; }
.vcc-prompt { border:1px solid #dfe3ee; border-radius:14px; padding:14px 16px;
  background:#fff; min-height:112px; }
.vcc-prompt-text { font-size:17px; line-height:1.75; overflow-wrap:anywhere; }
.vcc-prompt-reading { color:var(--vcc-muted); font-size:14px; line-height:1.7;
  margin-top:8px; overflow-wrap:anywhere; }
.vcc-prompt-reading strong { color:var(--vcc-ink); }
.vcc-score-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; }
.vcc-score { border:1px solid #dfe3ee; border-radius:14px; padding:14px; background:white; }
.vcc-score strong { display:block; font-size:19px; }
.vcc-score span { color:var(--vcc-muted); font-size:12px; }
.vcc-bar { height:7px; background:#e9ecf5; border-radius:9px; margin-top:9px; overflow:hidden; }
.vcc-bar i { display:block; height:100%; background:linear-gradient(90deg,#4856d8,#8b62dd); }
footer { display:none !important; }
"""


def build_app(service: ExperimentService | None = None) -> gr.Blocks:
    service = service or ExperimentService()
    config = service.config
    run_state: dict[str, dict[str, Any]] = {}
    state_lock = threading.Lock()

    candidate_text = {item.id: item.text for item in config.candidates}
    anchor_text = {item.id: item.text for item in config.anchors}
    prompts = {item.id: item for item in (*config.candidates, *config.anchors)}

    def prompt_card(prompt_id: str) -> str:
        prompt = prompts[prompt_id]
        return (
            '<div class="vcc-prompt">'
            f'<div class="vcc-prompt-text">{html.escape(prompt.text)}</div>'
            f'<div class="vcc-prompt-reading"><strong>よみ：</strong>'
            f'{html.escape(prompt.reading)}</div></div>'
        )

    def select_candidate(prompt_id: str):
        return prompt_card(prompt_id), gr.update(value=1)

    def experiment_choices() -> list[tuple[str, str]]:
        return [
            (f"{row['name']} · {row['status']}", row["id"])
            for row in service.db.list_experiments()
        ]

    def create_experiment(name: str):
        experiment_id = service.create_experiment(name)
        return (
            gr.update(choices=experiment_choices(), value=experiment_id),
            f"実験を作成しました: `{experiment_id}`",
            recording_table(experiment_id),
            readiness(experiment_id),
        )

    def refresh_experiments(current: str | None):
        choices = experiment_choices()
        values = {value for _, value in choices}
        selected = current if current in values else (choices[0][1] if choices else None)
        return (
            gr.update(choices=choices, value=selected),
            recording_table(selected),
            readiness(selected),
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

    def readiness(experiment_id: str | None) -> str:
        if not experiment_id:
            return "実験を作成または選択してください。"
        counts = service.recording_progress(experiment_id)
        candidate_goal = len(config.candidates) * config.takes_per_candidate
        anchor_goal = len(config.anchors)
        return (
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
        result = service.save_recording(
            experiment_id,
            "candidate",
            prompt_id,
            int(take),
            candidate_text[prompt_id],
            audio_path,
        )
        message = quality_message(result)
        return message, recording_table(experiment_id), readiness(experiment_id)

    def save_anchor(
        experiment_id: str | None,
        prompt_id: str,
        audio_path: str | None,
    ):
        if not experiment_id:
            raise gr.Error("先に実験を作成してください")
        if not audio_path:
            raise gr.Error("音声を録音または選択してください")
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
        rows = [dict(row) for row in service.db.generations(experiment_id, complete_only=True)]
        votes = [dict(row) for row in service.db.votes(experiment_id)]
        weights = dict(config.raw["ranking"])
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
                candidate_text.get(row["candidate_id"], ""),
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
                f'<span>{html.escape(candidate_text.get(row["candidate_id"], ""))}</span>'
                f'<div class="vcc-bar"><i style="width:{row["final_score"] * 100:.1f}%"></i></div></div>'
            )
            for row in ranking[:3]
        )
        return table, f'<div class="vcc-score-grid">{cards}</div>' if cards else "<p>完了データがありません。</p>"

    def next_pair(experiment_id: str | None):
        if not experiment_id:
            raise gr.Error("実験を選択してください")
        rows = [dict(row) for row in service.db.generations(experiment_id, complete_only=True)]
        weights = config.raw["ranking"]
        ranking = rank_candidates(rows, weights, (), bootstrap_samples=200)
        top = [row["candidate_id"] for row in ranking[:3]]
        if len(top) < 2:
            raise gr.Error("A/B試聴には完了済み候補が2つ以上必要です")
        previous = {
            frozenset((row["generation_a"], row["generation_b"]))
            for row in service.db.votes(experiment_id)
        }
        by_key = {
            (row["prompt_id"], row["take"], row["eval_id"], row["seed"]): row
            for row in rows
            if row["prompt_id"] in top and Path(row["output_path"]).exists()
        }
        for left, right in itertools.combinations(top, 2):
            for take in range(1, config.takes_per_candidate + 1):
                for evaluation in config.evaluations:
                    for seed in config.seeds:
                        a = by_key.get((left, take, evaluation.id, seed))
                        b = by_key.get((right, take, evaluation.id, seed))
                        if not a or not b:
                            continue
                        key = frozenset((a["id"], b["id"]))
                        if key in previous:
                            continue
                        swap = (len(previous) + int(a["id"])) % 2 == 1
                        if swap:
                            a, b = b, a
                        pair = {
                            "candidate_a": a["prompt_id"],
                            "candidate_b": b["prompt_id"],
                            "generation_a": a["id"],
                            "generation_b": b["id"],
                        }
                        return (
                            a["output_path"],
                            b["output_path"],
                            pair,
                            f"同じ文を読み上げた2音声です。本人らしさと自然さを合わせて選んでください。\n\n**評価文:** {evaluation.text}",
                        )
        return None, None, {}, "未評価の組み合わせがありません。結果タブで順位を確認してください。"

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
        paths = ReportBuilder(
            service.db,
            config,
            service.experiment_dir(experiment_id),
        ).build(experiment_id, weights)
        return (
            str(paths["html"]),
            str(paths["csv"]),
            str(paths["json"]),
            "レポートを更新しました。",
        )

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
            create_button = gr.Button("新しい実験を作成", variant="primary", scale=1)
        experiment_message = gr.Markdown(elem_classes="vcc-experiment-message")

        with gr.Tabs():
            with gr.Tab("1. 収録"):
                recording_readiness = gr.Markdown("実験を作成または選択してください。")
                gr.Markdown(
                    "マイクとの距離、部屋、声量を揃え、表示文だけを自然な会話調で読んでください。候補は各2テイク、アンカーは各1テイクです。"
                )
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
                    "初回はモデルを取得します。フル探索は384音声を処理するため長時間かかります。小型モデルによるスモークテストで先に一連の動作を確認できます。"
                )
                with gr.Row():
                    smoke_button = gr.Button("スモークテスト")
                    full_button = gr.Button("フル探索を開始", variant="primary")
                    stop_button = gr.Button("一時停止", variant="stop")
                run_status = gr.Markdown("未実行")
                error_table = gr.Dataframe(
                    headers=["候補", "評価文", "seed", "エラー"],
                    interactive=False,
                    wrap=True,
                )
                timer = gr.Timer(2.0, active=True)

            with gr.Tab("3. 結果"):
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
                ranking_table = gr.Dataframe(
                    headers=[
                        "順位", "候補", "セリフ", "総合", "95% CI",
                        "類似度", "UTMOS", "CER", "失敗率", "試聴勝率",
                    ],
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("4. ブラインド試聴"):
                listening_message = gr.Markdown(
                    "自動評価の上位3候補が決まったら比較を開始してください。"
                )
                pair_state = gr.State({})
                load_pair_button = gr.Button("次の比較を読み込む", variant="primary")
                with gr.Row():
                    audio_a = gr.Audio(label="音声 A", interactive=False)
                    audio_b = gr.Audio(label="音声 B", interactive=False)
                with gr.Row():
                    vote_a = gr.Button("A が良い")
                    vote_tie = gr.Button("同程度")
                    vote_b = gr.Button("B が良い")

            with gr.Tab("5. レポート"):
                gr.Markdown(
                    "現在の重みと試聴結果で、HTML・CSV・JSONをまとめて更新します。HTMLには上位候補の代表音声も埋め込まれます。"
                )
                export_button = gr.Button("レポートを書き出す", variant="primary")
                export_status = gr.Markdown()
                html_file = gr.File(label="HTMLレポート")
                csv_file = gr.File(label="全生成データ CSV")
                json_file = gr.File(label="集計 JSON")

        create_button.click(
            create_experiment,
            inputs=[experiment_name],
            outputs=[
                experiment_select,
                experiment_message,
                recordings_frame,
                recording_readiness,
            ],
        )
        refresh_button.click(
            refresh_experiments,
            inputs=[experiment_select],
            outputs=[experiment_select, recordings_frame, recording_readiness],
        )
        experiment_select.change(
            lambda value: (recording_table(value), readiness(value)),
            inputs=[experiment_select],
            outputs=[recordings_frame, recording_readiness],
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
        save_anchor_button.click(
            save_anchor,
            inputs=[experiment_select, anchor_select, anchor_audio],
            outputs=[anchor_quality, recordings_frame, recording_readiness],
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
            outputs=[html_file, csv_file, json_file, export_status],
        )
    return app
