from __future__ import annotations

import argparse
import sys

from .app import CSS, build_app
from .reports import ReportBuilder
from .service import ExperimentService


def _progress(done: int, total: int, message: str) -> None:
    print(f"[{done}/{total}] {message}", flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone-check",
        description="Qwen3-TTS 日本語参照セリフ探索ツール",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="起動時にブラウザを自動で開かない",
    )
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="新しい実験を作成")
    create_parser.add_argument("--name")
    create_parser.add_argument(
        "--mode",
        choices=("synthetic", "recorded", "duration"),
        default="synthetic",
    )
    create_parser.add_argument(
        "--anchor-experiment",
        help="長さ調査で独立アンカー3本を引き継ぐ実験ID",
    )

    run_parser = subparsers.add_parser("run", help="保存済み実験を実行・再開")
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--smoke", action="store_true")

    reference_parser = subparsers.add_parser(
        "generate-references",
        help="元音声から候補参照音声を生成・再開",
    )
    reference_parser.add_argument("--experiment", required=True)
    reference_parser.add_argument("--smoke", action="store_true")

    source_parser = subparsers.add_parser(
        "add-source",
        help="合成事前選定へ元音声と正確な台本を登録",
    )
    source_parser.add_argument("--experiment", required=True)
    source_parser.add_argument("--audio", required=True)
    source_parser.add_argument("--transcript", required=True)
    source_parser.add_argument("--take", type=int, choices=(1, 2, 3), default=1)

    candidate_parser = subparsers.add_parser(
        "add-candidate",
        help="実録音候補を登録",
    )
    candidate_parser.add_argument("--experiment", required=True)
    candidate_parser.add_argument("--prompt", required=True)
    candidate_parser.add_argument("--take", type=int, choices=(1, 2), required=True)
    candidate_parser.add_argument("--audio", required=True)

    duration_parser = subparsers.add_parser(
        "add-duration-passage",
        help="長さ調査用の連続音声を4条件へ切り出して登録",
    )
    duration_parser.add_argument("--experiment", required=True)
    duration_parser.add_argument("--passage", type=int, choices=(1, 2, 3), required=True)
    duration_parser.add_argument("--audio", required=True)
    duration_parser.add_argument(
        "--boundaries",
        type=float,
        nargs=4,
        metavar=("B1", "B2", "B3", "B4"),
        help="各句の累積終了秒。省略時は無音位置から推定",
    )

    validation_parser = subparsers.add_parser(
        "create-validation",
        help="合成事前選定の上位3件から実録音検証を作成",
    )
    validation_parser.add_argument("--experiment", required=True)
    validation_parser.add_argument("--name")

    report_parser = subparsers.add_parser("report", help="レポートを再生成")
    report_parser.add_argument("--experiment", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    service = ExperimentService()
    if args.command == "create":
        if args.mode == "duration":
            experiment_id = service.create_duration_experiment(
                args.anchor_experiment, name=args.name
            )
        else:
            experiment_id = service.create_experiment(args.name, mode=args.mode)
        print(f"実験を作成しました: {experiment_id}")
        return 0
    if args.command == "run":
        result = service.run(args.experiment, smoke=args.smoke, progress=_progress)
        print(
            f"完了={result['completed']} 失敗={result['failed']} 残り={result['remaining']}"
        )
        return 0 if result["remaining"] == 0 else 1
    if args.command == "generate-references":
        result = service.generate_candidate_references(
            args.experiment,
            smoke=args.smoke,
            progress=_progress,
        )
        print(
            f"新規採用={result['accepted']} 不採用={result['rejected']} "
            f"未完成候補={result['incomplete']}"
        )
        return 0 if result["incomplete"] == 0 else 1
    if args.command == "add-source":
        result = service.save_source(
            args.experiment,
            args.audio,
            args.transcript,
            take=args.take,
        )
        print(
            f"品質={'OK' if result['quality_ok'] else '要確認'} "
            f"CER={result['cer']:.3f} ASR={result['transcript']}"
        )
        return 0 if result["quality_ok"] else 1
    if args.command == "add-duration-passage":
        results = service.save_duration_passage(
            args.experiment,
            args.passage,
            args.audio,
            list(args.boundaries) if args.boundaries else None,
        )
        for result in results:
            print(
                f"{result['prompt_id']}: "
                f"品質={'OK' if result['quality_ok'] else '要確認'} "
                f"長さ={result['duration']:.2f}秒 CER={result['cer']:.3f}"
            )
        return 0 if all(result["quality_ok"] for result in results) else 1
    if args.command == "add-candidate":
        config = service.experiment_config(args.experiment)
        prompts = {item.id: item for item in config.candidates}
        if args.prompt not in prompts:
            parser.error(f"この実験に候補 {args.prompt} はありません")
        result = service.save_recording(
            args.experiment,
            "candidate",
            args.prompt,
            args.take,
            prompts[args.prompt].text,
            args.audio,
        )
        print(
            f"品質={'OK' if result['quality_ok'] else '要確認'} "
            f"長さ={result['duration']:.2f}秒"
        )
        return 0 if result["quality_ok"] else 1
    if args.command == "create-validation":
        experiment_id = service.create_validation_experiment(
            args.experiment, name=args.name
        )
        print(f"検証実験を作成しました: {experiment_id}")
        return 0
    if args.command == "report":
        config = service.experiment_config(args.experiment)
        paths = ReportBuilder(
            service.db,
            config,
            service.experiment_dir(args.experiment),
        ).build(args.experiment)
        for label, path in paths.items():
            print(f"{label}: {path.relative_to(service.root.parent.parent)}")
        return 0

    app = build_app(service)
    app.queue(default_concurrency_limit=4)
    app.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=not args.no_browser,
        show_error=True,
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
