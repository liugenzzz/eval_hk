"""Convert the held-out Aircraft_Vqa split to eval_hk's ShareGPT input.

Only test and validation inputs are accepted. Train records must never become
evaluation examples, even when an alternate input path is supplied.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path("/mnt/si003010kcx0/mmdata/mmcode/Aircraft_Vqa/data/vqa_five_class_nosystem")
DEFAULT_OUTPUT = Path("/mnt/si003010kcx0/mmdata/mmcode/eval_hk/data/aircraft_vqa")
TASKS = frozenset({
    "discrimination", "classification_open", "classification_mc",
    "grounding_single", "grounding_all", "grounding_negative",
    "grounding_counterfactual", "region_word",
})
DEFECTS = frozenset({"fastener_missing", "corrosion", "crack", "dent", "scratch"})
REAL_DATASETS = frozenset({
    "aircraft_skin_defects", "uts_aircraft_defect",
    "mm_general_aircraft_image_level", "sutd_aircraft_ai_v4",
    "priority_aircraft_curated",
})


def domain_of(dataset: str) -> str:
    if dataset in REAL_DATASETS:
        return "aircraft_real"
    if dataset.startswith("synthetic_"):
        return "synthetic"
    return "transfer"


def size_bucket_of(answer: str) -> str | None:
    """Bucket the largest reference box by fraction of image area."""
    try:
        boxes = json.loads(answer)
    except (TypeError, ValueError):
        return None
    if not isinstance(boxes, list):
        return None
    areas = []
    for box in boxes:
        if not isinstance(box, dict):
            return None
        coords = box.get("bbox_2d")
        if not isinstance(coords, list) or len(coords) != 4:
            return None
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in coords):
            return None
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return None
        areas.append((x2 - x1) * (y2 - y1) / 1_000_000)
    if not areas:
        return None
    area = max(areas)
    return "small" if area < .01 else "medium" if area < .1 else "large"


def convert_record(row: dict[str, Any], domain_filter: str = "aircraft_real") -> dict[str, Any] | None:
    task = row.get("task")
    if task not in TASKS:
        return None
    dataset = row.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        return None
    domain = domain_of(dataset)
    if domain_filter == "aircraft_real" and domain != "aircraft_real":
        return None
    if domain_filter not in {"aircraft_real", "all"}:
        raise ValueError(f"Unsupported domain filter: {domain_filter}")

    status = row.get("image_status")
    defects = row.get("image_defect_types")
    if status not in {"normal", "anomalous"} or not isinstance(defects, list):
        return None
    if any(defect not in DEFECTS for defect in defects):
        return None
    if (status == "normal" and defects) or (status == "anomalous" and not defects):
        return None
    target = row.get("target_type") or row.get("absent_type")
    if target is not None and target not in DEFECTS:
        return None
    # Positive class questions need an unambiguous ground-truth class.
    if task in {"classification_open", "classification_mc", "grounding_single"} and not target:
        return None
    if task in {"classification_open", "classification_mc", "grounding_single", "grounding_all", "region_word"} and status != "anomalous":
        return None
    if task == "grounding_counterfactual" and (not target or target in defects):
        return None
    if task == "grounding_negative" and status != "normal":
        return None
    qa_id, image = row.get("qa_id"), row.get("image")
    question, answer = row.get("question"), row.get("answer")
    image_hw = row.get("image_hw")
    if not all(isinstance(value, str) and value for value in (qa_id, image, question, answer)):
        return None
    if not (isinstance(image_hw, list) and len(image_hw) == 2 and
            all(isinstance(value, int) and value > 0 for value in image_hw)):
        return None

    metadata: dict[str, Any] = {
        "task_type": task,
        "defect_type": target or (defects[0] if len(defects) == 1 else "multi_defect" if defects else "normal"),
        "image_status": status,
        "source_dataset": dataset,
        "source_image": row.get("sample_id") or image,
        "domain": domain,
        "bbox_scale": 1000,
        "image_hw": image_hw,
        "target_type": target,
    }
    if task.startswith("grounding_"):
        if row.get("coord_mode") != "norm1000":
            return None
        bucket = size_bucket_of(answer)
        if answer != "[]" and bucket is None:
            return None
        if bucket is not None:
            metadata["size_bucket"] = bucket
    return {
        "id": qa_id,
        "images": [image],
        "conversations": [
            {"from": "human", "value": "<image>" + question},
            {"from": "gpt", "value": answer},
        ],
        "metadata": metadata,
    }


def convert_split(source: Path, destination: Path, domain_filter: str = "aircraft_real") -> Counter[str]:
    if source.name not in {"test.raw.jsonl", "val.raw.jsonl"}:
        raise ValueError("Only held-out test.raw.jsonl and val.raw.jsonl are allowed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            converted = convert_record(row, domain_filter)
            if converted is None:
                counts["skipped"] += 1
                continue
            if converted["id"] in seen:
                raise ValueError(f"Duplicate qa_id in {source}:{line_number}: {converted['id']}")
            seen.add(converted["id"])
            writer.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts[converted["metadata"]["task_type"]] += 1
    return counts


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", choices=("test", "val", "both"), default="both")
    parser.add_argument("--domain", choices=("aircraft_real", "all"), default="aircraft_real")
    args = parser.parse_args(argv)
    splits = ("test", "val") if args.splits == "both" else (args.splits,)
    for split in splits:
        source = args.source_dir / f"{split}.raw.jsonl"
        destination = args.output_dir / f"{split}.{args.domain}.jsonl"
        counts = convert_split(source, destination, args.domain)
        print(f"{destination}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
