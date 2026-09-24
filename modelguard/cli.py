"""Small first-review command-line presentation of Modules 1 and 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelguard.review_output import render_html, render_terminal, report_dict
from modelguard.scan import scan_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelguard", description="Static GGUF and safetensors structural inspection")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="inspect one model file without executing it")
    scan.add_argument("file", type=Path)
    scan.add_argument("--json", dest="json_path", type=Path, help="write machine-readable report")
    scan.add_argument("--html", dest="html_path", type=Path, help="write a self-contained review report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = scan_file(args.file)
    print(render_terminal(result))
    try:
        if args.json_path:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(report_dict(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"JSON report: {args.json_path}")
        if args.html_path:
            args.html_path.parent.mkdir(parents=True, exist_ok=True)
            args.html_path.write_text(render_html(result), encoding="utf-8")
            print(f"HTML report: {args.html_path}")
    except OSError as exc:
        print(f"Could not write report: {exc}")
        return 2
    return 1 if any(f.severity in ("ERROR", "CRITICAL") for f in result.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
