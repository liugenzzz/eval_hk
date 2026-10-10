"""Build a read-only, image-level human reference review queue from past runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .industrial_mapping import iter_jsonl
from .industrial_v2 import REVIEWED_NORMAL, UNSPECIFIED


RUNS = ("industrial_full_base32b_raw", "industrial_full_base32b_newprompt",
        "industrial_full_ckpt5334_raw", "industrial_full_ckpt5334_newprompt")
REVIEW_STATUSES = {"conflict", "insufficient"}


def collect_review_rows(source: Path, details_dir: Path, *, include_all: bool = False) -> tuple[list[dict], dict[str, int]]:
    source, details_dir = Path(source), Path(details_dir)
    images = defaultdict(dict)
    id_to_image = {}
    for _, record in iter_jsonl(source):
        meta = record["metadata"]
        key, lang = meta["source_image_key"], meta["language"]
        if lang in images[key] or lang not in {"zh", "en"} or record["id"] in id_to_image:
            raise ValueError("duplicate or invalid source image/language/id")
        images[key][lang] = record
        id_to_image[record["id"] + "__t1"] = key
    if any(set(pair) != {"zh", "en"} for pair in images.values()):
        raise ValueError("source images must have zh/en pairs")

    statuses = defaultdict(dict)
    for run in RUNS:
        path = details_dir / run / "judge_detail_all.xlsx"
        frame = pd.read_excel(path)
        required = {"dataset", "index", "industrial_reference_status"}
        if not required <= set(frame):
            raise ValueError(f"missing required columns in {path}")
        if len(frame) != len(id_to_image) or frame["index"].duplicated().any():
            raise ValueError(f"row count or duplicate index mismatch in {path}")
        for _, result in frame.iterrows():
            idx = result["index"]
            if idx not in id_to_image:
                raise ValueError(f"unknown sample id in {path}: {idx}")
            lang = idx.split("_", 1)[0]
            if result["dataset"] != f"industrial_{lang}":
                raise ValueError(f"dataset/language mismatch in {path}: {idx}")
            value = result["industrial_reference_status"]
            statuses[id_to_image[idx]][f"{run}:{lang}"] = "missing" if pd.isna(value) else str(value)
        if len(statuses) != len(images):
            raise ValueError(f"incomplete run: {path}")

    rows = []
    trigger_counts = defaultdict(int)
    for key, pair in sorted(images.items()):
        zh, en = pair["zh"], pair["en"]
        zm, em = zh["metadata"], en["metadata"]
        if (zm["defect_type"], zm["defect_status"], zm["source_domain"]) != (
            em["defect_type"], em["defect_status"], em["source_domain"]
        ):
            raise ValueError(f"inconsistent paired labels: {key}")
        dtype = zm["defect_type"]
        run_values = statuses[key]
        triggers = []
        if dtype in REVIEWED_NORMAL:
            triggers.append("status_corrected")
        if dtype in UNSPECIFIED:
            triggers.append("type_unspecified")
        if any(value in REVIEW_STATUSES for value in run_values.values()):
            triggers.append("judge_reference_questioned")
        if not triggers and not include_all:
            continue
        for trigger in triggers:
            trigger_counts[trigger] += 1
        rows.append({
            "source_image_key": key, "image_path": zh["images"][0],
            "source_domain": zm["source_domain"], "defect_type": dtype,
            "v1_defect_status": zm["defect_status"],
            "candidate_defect_status": "normal" if dtype in REVIEWED_NORMAL else zm["defect_status"],
            "zh_sample_id": zh["id"], "en_sample_id": en["id"],
            "zh_reference": zh["conversations"][1]["value"],
            "en_reference": en["conversations"][1]["value"],
            "review_triggers": ",".join(triggers),
            "review_priority": "priority" if triggers else "routine",
            "judge_statuses_by_run": json.dumps(run_values, ensure_ascii=False, sort_keys=True),
            "judge_status_disagreement": len(set(run_values.values())) > 1,
            "reviewer_id_1": "", "reviewer_1_decision": "",
            "reviewer_id_2": "", "reviewer_2_decision": "",
            "human_reference_validity": "", "adjudicator_id": "",
            "adjudication_reason": "", "human_defect_type": "", "review_notes": "",
        })
    return rows, dict(trigger_counts)


def build_review_queue(source: Path, details_dir: Path, out: Path) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows, trigger_counts = collect_review_rows(source, details_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["source_image_key"])
        writer.writeheader()
        writer.writerows(rows)
    return {"images": len(rows), "triggers": trigger_counts, "output": str(out)}


def build_full_review_template(source: Path, details_dir: Path, out: Path) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows, trigger_counts = collect_review_rows(source, details_dir, include_all=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"images": len(rows), "priority_images": sum(bool(r["review_triggers"]) for r in rows),
            "triggers": trigger_counts, "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="Include every image for fixed-denominator review")
    args = parser.parse_args()
    func = build_full_review_template if args.full else build_review_queue
    print(json.dumps(func(args.source, args.details_dir, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
