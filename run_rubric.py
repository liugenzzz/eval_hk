"""Run one rubric version without hand-editing config.json.

Two mistakes are very easy to make when running several rubric versions in a row, and
both are silent:

  1. Forgetting to change ``out_dir`` -- report.py writes judge_detail_all.xlsx to a fixed
     name inside it, so the next run overwrites the previous one and you lose the arm you
     just paid for.
  2. Forgetting to change the pointwise prompt -- you get two runs of the same rubric under
     two different labels, which looks like a real comparison right up until the numbers
     come out suspiciously identical.

This derives a config per version, so neither can happen. It prints the judge fingerprint
before starting: a different fingerprint is the proof the prompt actually changed.

    python run_rubric.py --config config.json --rubric v3b
    python run_rubric.py --config config.json --rubric v4b --no-pairwise
    python run_rubric.py --config config.json --rubric v3b --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUBRICS = ("v1", "v2", "v3", "v3b", "v4", "v4b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="your existing config.json")
    ap.add_argument("--rubric", required=True, choices=RUBRICS)
    ap.add_argument("--out-root", default=None,
                    help="parent for the per-rubric out_dir (default: alongside the original)")
    ap.add_argument("--no-pairwise", action="store_true",
                    help="skip pairwise judging -- it roughly halves runtime and is not "
                         "part of the pointwise rubric comparison")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the derived config and print the command, do not run")
    args = ap.parse_args()

    base_path = Path(args.config)
    raw = json.loads(base_path.read_text(encoding="utf-8"))

    prompt_rel = f"prompts/judge_equip_pointwise_{args.rubric}.txt"
    if not Path(prompt_rel).exists():
        raise SystemExit(f"prompt not found: {prompt_rel}\n"
                         f"(unzip the prompts/ folder into the project root first)")

    raw.setdefault("judge", {}).setdefault("prompt_files", {})["pointwise"] = prompt_rel

    original_out = Path(raw.get("out_dir", "eval_report"))
    root = Path(args.out_root) if args.out_root else original_out.parent
    raw["out_dir"] = str(root / f"{original_out.name}_{args.rubric}")

    if args.no_pairwise:
        raw["do_pairwise"] = False

    derived = base_path.with_name(f"{base_path.stem}_{args.rubric}.json")
    derived.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rubric      {args.rubric}")
    print(f"prompt      {prompt_rel}")
    print(f"out_dir     {raw['out_dir']}")
    print(f"do_pairwise {raw.get('do_pairwise', True)}")
    print(f"config      {derived}")

    # Show the fingerprint up front: if it matches the previous run's, the prompt did not
    # actually change and the cache will replay the old verdicts.
    try:
        sys.path.insert(0, str(Path.cwd()))
        from eval_tool.config import load_config
        print(f"fingerprint {load_config(derived).judge.fingerprint}")
    except Exception as exc:
        print(f"fingerprint <unavailable: {type(exc).__name__}: {exc}>")

    cmd = [sys.executable, "-m", "eval_tool", "--config", str(derived)]
    print(f"\n$ {' '.join(cmd)}\n")
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
