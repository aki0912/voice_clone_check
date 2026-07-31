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

    run_parser = subparsers.add_parser("run", help="保存済み実験を実行・再開")
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--smoke", action="store_true")

    report_parser = subparsers.add_parser("report", help="レポートを再生成")
    report_parser.add_argument("--experiment", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    service = ExperimentService()
    if args.command == "run":
        result = service.run(args.experiment, smoke=args.smoke, progress=_progress)
        print(
            f"完了={result['completed']} 失敗={result['failed']} 残り={result['remaining']}"
        )
        return 0 if result["remaining"] == 0 else 1
    if args.command == "report":
        paths = ReportBuilder(
            service.db,
            service.config,
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
