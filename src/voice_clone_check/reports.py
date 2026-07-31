from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

from .config import ExperimentConfig
from .db import Database
from .scoring import rank_candidates


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
    main { width:min(1100px,calc(100% - 32px)); margin:36px auto 72px; }
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
    <p>Qwen3-TTS 日本語参照セリフ探索結果</p>
    <p>完了音声 {{ completed }}件 ／ 失敗判定 {{ failed }}件</p>
  </header>
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
            "generations": exported_generations,
            "votes": votes,
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
                completed=sum(row["status"] == "complete" for row in generations),
                failed=sum(bool(row["failed"]) for row in generations),
                ranking=ranking,
                texts=text_map,
                clips=clips,
            ),
            encoding="utf-8",
        )
        return {"csv": csv_path, "json": json_path, "html": html_path}
