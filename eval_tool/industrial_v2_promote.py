"""Freeze a fully human-reviewed industrial candidate into a scored-only artifact."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

from .industrial_mapping import file_sha256, iter_jsonl
from .industrial_v2 import REVIEWED_NORMAL, UNSPECIFIED, defect_bucket
from .industrial_v2_review import RUNS, collect_review_rows


DECISIONS = frozenset({"usable", "conflict", "insufficient"})
HUMAN_FIELDS = frozenset({
    "reviewer_id_1", "reviewer_1_decision", "reviewer_id_2", "reviewer_2_decision",
    "human_reference_validity", "adjudicator_id", "adjudication_reason",
    "human_defect_type", "review_notes",
})


def _nonempty(value: str) -> str:
    return value.strip()


def promote(source: Path, candidate: Path, details_dir: Path, decisions_csv: Path, out: Path) -> dict:
    source, candidate, details_dir = Path(source), Path(candidate), Path(details_dir)
    decisions_csv, out = Path(decisions_csv), Path(out)
    audit_path = candidate.with_suffix(".audit.json")
    manifest_path = out.with_suffix(".audit.json")
    if out.exists() or manifest_path.exists():
        raise FileExistsError("released artifact or audit already exists")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("source_sha256") != file_sha256(source) or audit.get("output_sha256") != file_sha256(candidate):
        raise ValueError("source/candidate SHA-256 differs from candidate audit")
    original = [row for _, row in iter_jsonl(source)]
    edited = [row for _, row in iter_jsonl(candidate)]
    if len(original) != len(edited) or len(original) % 2:
        raise ValueError("candidate/source row counts differ")
    source_by_image = {}
    for old, new in zip(original, edited):
        expected = copy.deepcopy(old)
        meta = expected["metadata"]
        dtype = meta["defect_type"]
        if dtype in REVIEWED_NORMAL:
            if meta["defect_status"] != "abnormal":
                raise ValueError("reviewed normal correction no longer applies")
            meta["defect_status"] = "normal"
            meta["status_decision_source"] = "frozen_industrial_mapping_review"
        meta["defect_bucket"] = defect_bucket(meta["defect_status"], dtype)
        meta["defect_type_specified"] = dtype not in UNSPECIFIED
        if expected != new:
            raise ValueError(f"candidate differs from deterministic v2 transform: {old.get('id')}")
        key = meta["source_image_key"]
        source_by_image.setdefault(key, []).append(new)
    if len(source_by_image) * 2 != len(edited):
        raise ValueError("candidate is not exactly paired by image")

    expected_rows, _ = collect_review_rows(source, details_dir, include_all=True)
    expected = {row["source_image_key"]: row for row in expected_rows}
    if len(expected) != len(source_by_image) or set(expected) != set(source_by_image):
        raise ValueError("review template does not cover every source image")
    with decisions_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or set(reader.fieldnames) != set(expected_rows[0]):
            raise ValueError("review CSV columns differ from full template")
        decisions = list(reader)
    keys = [row["source_image_key"] for row in decisions]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("review CSV has missing, duplicate, or unknown image keys")
    frozen = {}
    for row in decisions:
        key = row["source_image_key"]
        template = expected[key]
        for field in set(template) - HUMAN_FIELDS:
            if str(row[field]) != str(template[field]):
                raise ValueError(f"review evidence changed: {key}/{field}")
        reviewer1, reviewer2 = _nonempty(row["reviewer_id_1"]), _nonempty(row["reviewer_id_2"])
        first, second = row["reviewer_1_decision"], row["reviewer_2_decision"]
        final = row["human_reference_validity"]
        if not reviewer1 or not reviewer2 or reviewer1 == reviewer2 or {first, second, final} - DECISIONS:
            raise ValueError(f"incomplete/invalid independent reviews: {key}")
        adjudicator = _nonempty(row["adjudicator_id"])
        reason = _nonempty(row["adjudication_reason"])
        if first == second:
            if final != first:
                raise ValueError(f"final decision differs from agreeing reviewers: {key}")
        elif not adjudicator or adjudicator in {reviewer1, reviewer2} or not reason:
            raise ValueError(f"independent adjudication is required: {key}")
        frozen[key] = {field: row[field] for field in HUMAN_FIELDS}

    released = []
    for row in edited:
        key = row["metadata"]["source_image_key"]
        if frozen[key]["human_reference_validity"] != "usable":
            continue
        result = copy.deepcopy(row)
        result["metadata"]["reference_validity"] = "usable"
        result["metadata"]["reference_validity_source"] = "two_reviewer_frozen_review"
        released.append(result)
    counts = Counter(value["human_reference_validity"] for value in frozen.values())
    manifest = {
        "schema_version": 1, "chain": "industrial", "candidate_version": "industrial_eval_v2",
        "source_sha256": file_sha256(source), "candidate_sha256": file_sha256(candidate),
        "decision_csv_sha256": file_sha256(decisions_csv),
        "detail_sha256": {run: file_sha256(details_dir / run / "judge_detail_all.xlsx") for run in RUNS},
        "images_reviewed": len(frozen), "decisions": dict(sorted(counts.items())),
        "released_rows": len(released), "released_images": len(released) // 2,
        "excluded_images": {key: value for key, value in sorted(frozen.items())
                            if value["human_reference_validity"] != "usable"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        for row in released:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["released_sha256"] = file_sha256(out)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "candidate", "details-dir", "decisions-csv", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = promote(args.source, args.candidate, args.details_dir, args.decisions_csv, args.out)
    print(json.dumps({key: result[key] for key in ("images_reviewed", "decisions", "released_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
