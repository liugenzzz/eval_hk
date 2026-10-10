"""Create an auditable v2 copy of an existing paired industrial evaluation set.

This changes only the reviewed status labels. Reference/image validity requires a
separate human decision; it is never inferred from a model or judge response.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .industrial_mapping import file_sha256, iter_jsonl


REVIEWED_NORMAL = frozenset({
    "good_weld", "polycrystalline_good", "monocrystalline_good", "normal_coding",
})
UNSPECIFIED = frozenset({
    "unknown", "none", "defect", "defective", "combined", "random", "local",
})
AIRCRAFT_FIVE = frozenset({
    "crack", "monocrystalline_crack", "polycrystalline_crack", "fissure", "crazing",
    "scratch", "small_scratches", "scratch_head", "scratch_neck", "dent",
    "pitted_surface", "pitting", "surface_dents_or_pitting", "concavity",
    "tooth_bottom_oxidation_layer", "tooth_surface_oxidation_layer_tooth_bottom_oxidation_layer",
    "surface_dents_or_pitting_tooth_bottom_oxidation_layer", "paint_off",
})


def defect_bucket(status: str, defect_type: str) -> str:
    if status == "normal":
        return "normal"
    if defect_type in UNSPECIFIED:
        return "unspecified"
    return "aircraft_five" if defect_type in AIRCRAFT_FIVE else "outside_aircraft_five"


def build_v2(source: Path, out: Path) -> dict:
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    records = [record for _, record in iter_jsonl(source)]
    by_image = defaultdict(list)
    ids = set()
    for record in records:
        metadata = record.get("metadata", {})
        key = metadata.get("source_image_key")
        language = metadata.get("language")
        identity = record.get("id")
        if not key or language not in {"zh", "en"} or not identity or identity in ids:
            raise ValueError("source requires unique ids and paired image/language metadata")
        ids.add(identity)
        by_image[key].append(record)
    for key, pair in by_image.items():
        if {r["metadata"]["language"] for r in pair} != {"zh", "en"} or len(pair) != 2:
            raise ValueError(f"image {key!r} is not a zh/en pair")
        if len({(r["metadata"].get("defect_type"), r["metadata"].get("defect_status")) for r in pair}) != 1:
            raise ValueError(f"image {key!r} has inconsistent labels")
    corrected = []
    unspecified = []
    counts = Counter()
    for key, pair in sorted(by_image.items()):
        dtype = pair[0]["metadata"]["defect_type"]
        if dtype in REVIEWED_NORMAL:
            if pair[0]["metadata"]["defect_status"] != "abnormal":
                raise ValueError(f"reviewed correction is no longer applicable: {key}")
            corrected.append(key)
        if dtype in UNSPECIFIED:
            unspecified.append(key)
        for record in pair:
            meta = record["metadata"]
            if dtype in REVIEWED_NORMAL:
                meta["defect_status"] = "normal"
                meta["status_decision_source"] = "frozen_industrial_mapping_review"
            meta["defect_bucket"] = defect_bucket(meta["defect_status"], dtype)
            meta["defect_type_specified"] = dtype not in UNSPECIFIED
            counts[(meta["language"], meta["defect_bucket"])] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit = {
        "version": "industrial_eval_v2", "source": str(source),
        "source_sha256": file_sha256(source), "output_sha256": file_sha256(out),
        "rows": len(records), "images": len(by_image),
        "corrected_normal_images": corrected,
        "unspecified_type_images": unspecified,
        "strata": [{"language": lang, "bucket": bucket, "rows": n}
                   for (lang, bucket), n in sorted(counts.items())],
        "reference_validity": "pending_independent_human_review",
    }
    out.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = build_v2(args.source, args.out)
    print(json.dumps({k: audit[k] for k in ("rows", "images", "reference_validity")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
