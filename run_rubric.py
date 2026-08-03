"""Compatibility shim for the unified in-process rubric evaluator."""

from __future__ import annotations

import argparse

from eval_tool.cli import main as cli_main
from eval_tool.rubrics import RUBRIC_VERSIONS


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="evaluation or pipeline config")
    parser.add_argument("--rubric", required=True, choices=RUBRIC_VERSIONS)
    parser.add_argument(
        "--out-root",
        help="parent for the per-rubric output directory",
    )
    parser.add_argument("--no-pairwise", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    translated = [
        "eval",
        "--config",
        args.config,
        "--rubric",
        args.rubric,
    ]
    if args.out_root:
        translated.extend(["--out-root", args.out_root])
    if args.no_pairwise:
        translated.append("--no-pairwise")
    if args.dry_run:
        translated.append("--dry-run")
    return cli_main(translated)


if __name__ == "__main__":
    main()
