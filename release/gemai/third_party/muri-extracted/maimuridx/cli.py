"""Noninteractive entry point, usable by absolute path from any directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .api import DetectionError, detect_file
else:
    from api import DetectionError, detect_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MaiMuriDX headless detector: raw Simai or maidata.txt")
    parser.add_argument("input", nargs="?", type=Path, help="raw Simai file, maidata.txt, or its directory")
    parser.add_argument("-f", "--file", dest="file", type=Path, help="alternative to positional input")
    parser.add_argument("-d", "--difficulty", help="numeric suffix of maidata &inote_<n>")
    parser.add_argument("--format", choices=("auto", "simai", "maidata"), default="auto")
    parser.add_argument("--first", type=float, default=None, help="override chart offset, in seconds")
    parser.add_argument("--config", type=Path, help="JSON object of partial upstream configuration overrides")
    parser.add_argument("--hand-radius", type=float, help="override hand_radius_normal (default: 60px / 1080 canvas)")
    parser.add_argument("--timeout", type=float, default=120.0, help="maximum detection seconds (default: 120)")
    parser.add_argument("-j", "--json", type=Path, help="write complete JSON result; otherwise print JSON to stdout")
    parser.add_argument("-o", "--report", "--output", dest="report", type=Path, help="write readable upstream report")
    args = parser.parse_args(argv)
    if (args.input is None) == (args.file is None):
        parser.error("supply exactly one input path, positionally or with --file")
    if args.json and args.report and args.json.resolve() == args.report.resolve():
        parser.error("JSON and report output paths must differ")
    try:
        source = (args.input or args.file).resolve()
        if source.is_dir():
            source /= "maidata.txt"
        protected = {source}
        if args.config:
            protected.add(args.config.resolve())
        module_root = Path(__file__).resolve().parent
        for output in (args.json, args.report):
            if output is None:
                continue
            target = output.resolve()
            if target in protected or (module_root / "upstream") in target.parents or (
                module_root in target.parents and target.suffix == ".py"
            ):
                raise ValueError("output must not overwrite the chart, configuration, or detector source")
        overrides = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("--config must contain a JSON object")
        result = detect_file(
            args.input or args.file,
            difficulty=args.difficulty,
            first=args.first,
            format=args.format,
            config=overrides,
            hand_radius=args.hand_radius,
            timeout=args.timeout,
        )
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.json:
            args.json.write_text(serialized, encoding="utf-8")
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            sys.stdout.write(serialized)
        if args.report:
            args.report.write_text(result["report"], encoding="utf-8")
    except (DetectionError, OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
