from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from .config import ExperimentConfig, config_from_raw
from .db import Database
from .scoring import analyze_duration_study, rank_candidates


REPORT_TEMPLATE = Template(
    """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ name }} — Voice Clone Check</title>
  <style>
    :root { color-scheme: light; --ink:#18202f; --muted:#657087; --line:#dde3ee;
      --paper:#fff; --wash:#f4f6fb; --accent:#4555d8; --good:#157a62; }
    * { box-sizing:border-box } body { margin:0; color:var(--ink); background:var(--wash);
      font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP",sans-serif; }
    main { width:calc(100% - 32px); margin:36px auto 72px; }
    header,.card { background:var(--paper); border:1px solid var(--line); border-radius:18px;
      box-shadow:0 10px 35px rgba(37,49,91,.06); }
    header { padding:30px 34px; margin-bottom:18px; }
    h1 { margin:0 0 6px; font-size:clamp(24px,4vw,38px); letter-spacing:-.03em; }
    h2 { margin:0 0 14px; font-size:20px; } p { margin:5px 0; color:var(--muted); }
    .card { padding:24px; margin:16px 0; overflow:auto; }
    table { width:100%; border-collapse:collapse; white-space:nowrap; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; }
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    .rank { font-weight:750; color:var(--accent); } .winner { background:#f1f3ff; }
    .meter { width:150px; height:7px; border-radius:8px; background:#e8ebf4; overflow:hidden; }
    .meter span { display:block; height:100%; background:linear-gradient(90deg,var(--accent),#8d65df); }
    .clips { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
    .clip { padding:16px; border:1px solid var(--line); border-radius:12px; }
    audio { width:100%; margin-top:8px; }
    footer { text-align:center; color:var(--muted); margin-top:22px; font-size:12px; }
  </style>
</head>
<body><main>
  <header>
    <h1>{{ name }}</h1>
    <p>Qwen3-TTS 日本語参照セリフ探索結果 — {{ mode_label }}</p>
    <p>完了音声 {{ completed }}件 ／ 失敗判定 {{ failed }}件</p>
  </header>
  {% if duration_analysis %}
  <section class="card">
    <h2>入力音声長の比較</h2>
    {% if duration_analysis.status == 'recommended' %}
      <p><strong>推奨最短長: 約{{ '%.0f'|format(duration_analysis.recommended_seconds) }}秒</strong></p>
    {% elif duration_analysis.status == 'needs_listening' %}
      <p><strong>自動評価上の候補: 約{{ '%.0f'|format(duration_analysis.automatic_candidate_seconds) }}秒</strong> — 最終判定には36件のブラインド試聴を完了してください。</p>
    {% elif duration_analysis.status == 'inconclusive' %}
      <p><strong>判定不能:</strong> 信頼区間が判定幅をまたいでいます。録音を追加してください。</p>
    {% else %}<p>評価データがまだ不足しています。</p>{% endif %}
    <svg viewBox="0 0 720 230" role="img" aria-label="長さ別話者類似度" style="width:100%;max-width:900px">
      <line x1="55" y1="190" x2="690" y2="190" stroke="#9aa4b8"/>
      {% for point in duration_chart %}
        {% if not loop.first %}<line x1="{{ loop.previtem.x }}" y1="{{ loop.previtem.y }}" x2="{{ point.x }}" y2="{{ point.y }}" stroke="#4555d8" stroke-width="3"/>{% endif %}
        <circle cx="{{ point.x }}" cy="{{ point.y }}" r="6" fill="#4555d8"/>
        <text x="{{ point.x }}" y="215" text-anchor="middle">{{ point.seconds|int }}秒</text>
        <text x="{{ point.x }}" y="{{ point.y - 12 }}" text-anchor="middle">{{ '%.3f'|format(point.value) }}</text>
      {% endfor %}
    </svg>
    <table><thead><tr><th>条件</th><th>件数</th><th>類似度</th><th>UTMOS</th><th>CER</th><th>失敗率</th></tr></thead><tbody>
      {% for row in duration_analysis.summaries %}<tr>
        <td>{{ row.condition_id }}（約{{ row.seconds|int }}秒）</td><td>{{ row.samples }}</td>
        <td>{{ '%.3f'|format(row.similarity) }}</td><td>{{ '%.3f'|format(row.utmos) }}</td>
        <td>{{ '%.3f'|format(row.cer) }}</td><td>{{ '%.1f%%'|format(row.failure_rate * 100) }}</td>
      </tr>{% endfor %}
    </tbody></table>
    <h2 style="margin-top:24px">録音別の推移</h2>
    <table><thead><tr><th>録音</th><th>条件</th><th>件数</th><th>類似度</th><th>UTMOS</th><th>CER</th></tr></thead><tbody>
      {% for row in duration_analysis.block_summaries %}<tr>
        <td>台本 {{ row.take }}</td><td>{{ row.condition_id }}（約{{ row.seconds|int }}秒）</td><td>{{ row.samples }}</td>
        <td>{{ '%.3f'|format(row.similarity) }}</td><td>{{ '%.3f'|format(row.utmos) }}</td><td>{{ '%.3f'|format(row.cer) }}</td>
      </tr>{% endfor %}
    </tbody></table>
  </section>
  <section class="card">
    <h2>15秒条件との差と95%信頼区間</h2>
    <table><thead><tr><th>条件</th><th>録音ブロック</th><th>類似度差</th><th>UTMOS差</th><th>CER差</th></tr></thead><tbody>
      {% for row in duration_analysis.comparisons %}<tr>
        <td>{{ row.condition_id }}</td><td>{{ row.blocks }}</td>
        <td>{{ '%.3f'|format(row.similarity_diff) }} [{{ '%.3f'|format(row.similarity_ci_low) }}, {{ '%.3f'|format(row.similarity_ci_high) }}]</td>
        <td>{{ '%.3f'|format(row.utmos_diff) }} [{{ '%.3f'|format(row.utmos_ci_low) }}, {{ '%.3f'|format(row.utmos_ci_high) }}]</td>
        <td>{{ '%.3f'|format(row.cer_diff) }} [{{ '%.3f'|format(row.cer_ci_low) }}, {{ '%.3f'|format(row.cer_ci_high) }}]</td>
      </tr>{% endfor %}
    </tbody></table>
    <h2 style="margin-top:24px">隣接長のブラインド試聴</h2>
    <table><thead><tr><th>比較</th><th>回答数</th><th>長い条件の選好率</th></tr></thead><tbody>
      {% for row in duration_analysis.adjacent_preferences %}<tr>
        <td>{{ row.short_id }} 対 {{ row.long_id }}</td><td>{{ row.votes }} / {{ duration_analysis.required_votes_per_comparison }}</td>
        <td>{{ '未実施' if row.longer_preference is none else '%.1f%%'|format(row.longer_preference * 100) }}</td>
      </tr>{% endfor %}
    </tbody></table>
  </section>
  {% else %}
  <section class="card">
    <h2>候補ランキング</h2>
    <table><thead><tr>
      <th>順位</th><th>候補</th><th>総合</th><th>95% CI</th>
      <th>類似度</th><th>UTMOS</th><th>CER</th><th>失敗率</th><th>試聴勝率</th>
    </tr></thead><tbody>
    {% for row in ranking %}
      <tr class="{{ 'winner' if row.rank == 1 else '' }}">
        <td class="rank">{{ row.rank }}</td><td>{{ row.candidate_id }} — {{ texts[row.candidate_id] }}</td>
        <td>{{ '%.3f'|format(row.final_score) }}<div class="meter"><span style="width:{{ row.final_score * 100 }}%"></span></div></td>
        <td>{{ '%.3f'|format(row.ci_low) }}–{{ '%.3f'|format(row.ci_high) }}</td>
        <td>{{ '%.3f'|format(row.similarity) }}</td><td>{{ '%.3f'|format(row.utmos) }}</td>
        <td>{{ '%.3f'|format(row.cer) }}</td><td>{{ '%.1f%%'|format(row.failure_rate * 100) }}</td>
        <td>{{ '未実施' if row.listening_win_rate is none else '%.1f%%'|format(row.listening_win_rate * 100) }}</td>
      </tr>
    {% endfor %}
    </tbody></table>
  </section>
  {% endif %}
  {% if comparison %}
  <section class="card">
    <h2>合成事前選定との順位比較</h2>
    <p>最終判断では、実録音検証の順位とブラインド試聴を優先してください。</p>
    <table><thead><tr>
      <th>候補</th><th>事前選定</th><th>実録音検証</th><th>順位変動</th>
      <th>事前スコア</th><th>検証スコア</th>
    </tr></thead><tbody>
    {% for row in comparison %}<tr>
      <td>{{ row.candidate_id }} — {{ texts[row.candidate_id] }}</td>
      <td>{{ row.screening_rank }}</td><td>{{ row.validation_rank }}</td>
      <td>{{ '%+d'|format(row.screening_rank - row.validation_rank) }}</td>
      <td>{{ '%.3f'|format(row.screening_score) }}</td>
      <td>{{ '%.3f'|format(row.validation_score) }}</td>
    </tr>{% endfor %}
    </tbody></table>
  </section>
  {% endif %}
  {% if clips %}
  <section class="card"><h2>上位候補のサンプル</h2><div class="clips">
    {% for clip in clips %}<div class="clip">
      <strong>{{ clip.candidate_id }} / {{ clip.eval_id }}</strong>
      <p>{{ clip.text }}</p><audio controls preload="none" src="{{ clip.data_uri }}"></audio>
    </div>{% endfor %}
  </div></section>
  {% endif %}
  <footer>Voice Clone Check — 自動指標だけでなく、必ず実際の試聴結果も確認してください。</footer>
</main></body></html>"""
)


class ReportBuilder:
    def __init__(self, db: Database, config: ExperimentConfig, experiment_dir: Path):
        self.db = db
        self.config = config
        self.experiment_dir = experiment_dir

    def build(
        self,
        experiment_id: str,
        ranking_weights: dict[str, float] | None = None,
    ) -> dict[str, Path]:
        experiment = self.db.experiment(experiment_id)
        if not experiment:
            raise ValueError("実験が見つかりません")
        generations = [dict(row) for row in self.db.generations(experiment_id)]
        recordings = [dict(row) for row in self.db.recordings(experiment_id)]
        exported_recordings = []
        for row in recordings:
            exported = dict(row)
            for field in ("raw_path", "processed_path"):
                path = Path(str(exported.get(field, "")))
                try:
                    exported[field] = str(path.relative_to(self.experiment_dir))
                except ValueError:
                    exported[field] = path.name
            exported_recordings.append(exported)
        exported_generations = []
        for row in generations:
            exported = dict(row)
            output = Path(str(exported.get("output_path", "")))
            try:
                exported["output_path"] = str(output.relative_to(self.experiment_dir))
            except ValueError:
                exported["output_path"] = output.name
            exported_generations.append(exported)
        votes = [dict(row) for row in self.db.votes(experiment_id)]
        weights = dict(self.config.raw["ranking"])
        if ranking_weights:
            weights.update(ranking_weights)
        ranking = rank_candidates(
            generations,
            weights,
            votes,
            bootstrap_samples=int(weights["bootstrap_samples"]),
        )
        duration_analysis = None
        duration_chart: list[dict[str, Any]] = []
        if experiment["mode"] == "duration":
            duration_analysis = analyze_duration_study(
                generations,
                self.config.raw["duration_study"],
                votes,
            )
            points = duration_analysis["summaries"]
            if points:
                values = [float(point["similarity"]) for point in points]
                low, high = min(values), max(values)
                spread = max(high - low, 0.02)
                for index, point in enumerate(points):
                    duration_chart.append({
                        "x": 80 + index * (580 / max(1, len(points) - 1)),
                        "y": 180 - (float(point["similarity"]) - low) / spread * 120,
                        "seconds": point["seconds"],
                        "value": point["similarity"],
                    })
        comparison: list[dict[str, Any]] = []
        if experiment["parent_experiment_id"] and experiment["mode"] != "duration":
            parent = self.db.experiment(experiment["parent_experiment_id"])
            if parent:
                parent_config = config_from_raw(json.loads(parent["config_json"]))
                parent_ranking = rank_candidates(
                    [
                        dict(row)
                        for row in self.db.generations(parent["id"], complete_only=True)
                    ],
                    parent_config.raw["ranking"],
                    [dict(row) for row in self.db.votes(parent["id"])],
                    bootstrap_samples=int(
                        parent_config.raw["ranking"]["bootstrap_samples"]
                    ),
                )
                parent_by_id = {row["candidate_id"]: row for row in parent_ranking}
                for current in ranking:
                    previous = parent_by_id.get(current["candidate_id"])
                    if previous:
                        comparison.append(
                            {
                                "candidate_id": current["candidate_id"],
                                "screening_rank": previous["rank"],
                                "validation_rank": current["rank"],
                                "screening_score": previous["final_score"],
                                "validation_score": current["final_score"],
                            }
                        )
        report_dir = self.experiment_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        csv_path = report_dir / "generations.csv"
        fields = list(exported_generations[0].keys()) if exported_generations else [
            "id", "prompt_id", "take", "eval_id", "seed", "status"
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(exported_generations)

        json_path = report_dir / "summary.json"
        payload = {
            "experiment": {
                "id": experiment["id"],
                "name": experiment["name"],
                "status": experiment["status"],
            },
            "weights": weights,
            "ranking": ranking,
            "recordings": exported_recordings,
            "generations": exported_generations,
            "votes": votes,
            "comparison": comparison,
            "duration_analysis": duration_analysis,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        text_map = {prompt.id: prompt.text for prompt in self.config.candidates}
        clips: list[dict[str, Any]] = []
        for ranked in ranking[:3]:
            candidates = [
                row
                for row in generations
                if row["prompt_id"] == ranked["candidate_id"]
                and row["status"] == "complete"
                and Path(row["output_path"]).exists()
            ]
            if not candidates:
                continue
            selected = max(candidates, key=lambda row: row.get("similarity") or -1)
            audio = Path(selected["output_path"]).read_bytes()
            clips.append(
                {
                    "candidate_id": selected["prompt_id"],
                    "eval_id": selected["eval_id"],
                    "text": selected["eval_text"],
                    "data_uri": "data:audio/wav;base64,"
                    + base64.b64encode(audio).decode("ascii"),
                }
            )

        html_path = report_dir / "report.html"
        html_path.write_text(
            REPORT_TEMPLATE.render(
                name=experiment["name"],
                mode_label={
                    "recorded": "全候補を実録音",
                    "synthetic": "合成音声による事前選定",
                    "validation": "上位候補の実録音検証",
                    "duration": "入力音声長調査",
                }.get(experiment["mode"], experiment["mode"]),
                completed=sum(row["status"] == "complete" for row in generations),
                failed=sum(bool(row["failed"]) for row in generations),
                ranking=ranking,
                texts=text_map,
                clips=clips,
                comparison=comparison,
                duration_analysis=duration_analysis,
                duration_chart=duration_chart,
            ),
            encoding="utf-8",
        )
        return {"csv": csv_path, "json": json_path, "html": html_path}
