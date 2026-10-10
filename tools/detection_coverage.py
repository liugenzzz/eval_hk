"""Audit detection JSONL coverage against a scoring profile without running inference."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "eval_tool/profiles/grounding_zh_v1.json"


def audit(data_path: Path, profile_path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    turn_counts: Counter[tuple[str, int]] = Counter()
    total = 0
    with data_path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_path}:{line_no}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{data_path}:{line_no}: expected object")
            metadata = record.get("metadata") or {}
            task = metadata.get("task_type") or record.get("task_type") or "<missing>"
            if not isinstance(task, str):
                raise ValueError(f"{data_path}:{line_no}: invalid task_type")
            messages = record.get("conversations") or record.get("messages") or []
            if not isinstance(messages, list):
                raise ValueError(f"{data_path}:{line_no}: invalid conversations/messages")
            # A scored turn requires a question and a gold answer.
            human_roles = {"human", "user"}
            answer_roles = {"gpt", "assistant", "model"}
            turns = sum(
                str(messages[i].get("from", messages[i].get("role", ""))) in human_roles
                and str(messages[i + 1].get("from", messages[i + 1].get("role", ""))) in answer_roles
                for i in range(len(messages) - 1)
                if isinstance(messages[i], dict) and isinstance(messages[i + 1], dict)
            )
            counts[task] += 1
            total += 1
            for turn in range(1, turns + 1):
                turn_counts[(task, turn)] += 1

    weights = profile.get("report", {}).get("dataset_weights", {})
    derived = {entry["dataset"] for entry in profile.get("derive", [])}
    slices = {}
    for name in profile.get("enabled_datasets", []):
        if name in derived:
            slices[name] = {"rows": None, "available": None,
                            "weight": float(weights.get(name, 0)), "source": "derived_at_runtime"}
            continue
        config = profile["datasets"][name]
        selections = config.get("params", {}).get("select", [])
        matched = set()
        for selection in selections:
            tasks = selection.get("task_type", list(counts))
            turns = [selection["turn"]] if "turn" in selection else range(1, 4)
            matched.update((task, turn) for task in tasks for turn in turns)
        n = sum(turn_counts[pair] for pair in matched)
        slices[name] = {"rows": n, "available": n > 0,
                        "weight": float(weights.get(name, 0)), "source": "evaluation_jsonl"}

    weighted_total = sum(item["weight"] for item in slices.values())
    weighted_available = sum(item["weight"] for item in slices.values()
                             if item["available"] is True)
    return {
        "records": total,
        "task_counts": dict(sorted(counts.items())),
        "scoring_slices": slices,
        "acceptance_weight_available": weighted_available,
        "acceptance_weight_total": weighted_total,
        "acceptance_coverage": weighted_available / weighted_total if weighted_total else None,
        "note": "Coverage reports source availability, not model performance. Derived slices are evaluated at runtime.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    print(json.dumps(audit(args.jsonl, args.profile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
