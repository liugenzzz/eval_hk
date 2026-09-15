"""Freeze matched, single-turn Chinese/English industrial evaluation samples.

Run ``python -m eval_tool.industrial_sample --help`` for the independent CLI.
No translation, image access, or model calls are performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .eval_set import clean_question, load_records, normalize_conversations, sha256_of, split_turns

STRATA_FIELDS = ("object_category", "defect_type", "source_domain")
OUTPUTS = ("industrial_zh.jsonl", "industrial_en.jsonl", "manifest.json")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _images(record: dict[str, Any]) -> list[str]:
    values = record.get("images") or record.get("image") or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Each record requires nonempty images/image paths")
    return values


def _index(path: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for line, record in enumerate(load_records(path), 1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {line} must be an object")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: record {line} metadata must be an object")
        identity = metadata.get("pair_id", record.get("id"))
        if isinstance(identity, bool) or not isinstance(identity, (str, int)) or not str(identity).strip():
            raise ValueError(f"{path}: record {line} missing stable id / metadata.pair_id")
        identity = str(identity)
        if identity in indexed:
            raise ValueError(f"{path}: duplicate pair id {identity!r}")
        conversations = normalize_conversations(record)
        turns = split_turns(conversations)
        if len(conversations) != 2 or len(turns) != 1:
            raise ValueError(f"{path}: {identity!r} must contain exactly one question/answer turn")
        question, answer = turns[0]
        if not clean_question(question) or not clean_question(answer):
            raise ValueError(f"{path}: {identity!r} has empty question or answer")
        for field in STRATA_FIELDS:
            value = metadata.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{path}: {identity!r} metadata.{field} must be a nonempty string or null")
        if not metadata.get("object_category"):
            raise ValueError(f"{path}: {identity!r} metadata.object_category is required")
        if "object_category" in record and record["object_category"] != metadata["object_category"]:
            raise ValueError(f"{path}: {identity!r} conflicting top-level object_category")
        if "id" in record and record["id"] != identity and "original_id" in metadata and metadata["original_id"] != record["id"]:
            raise ValueError(f"{path}: {identity!r} conflicting metadata.original_id")
        images = _images(record)
        image_tags = question.count("<image>")
        if image_tags not in (0, len(images)):
            raise ValueError(f"{path}: {identity!r} image marker count must be zero or match image paths")
        indexed[identity] = record
    return indexed


def sample_paired(zh: str | Path, en: str | Path, out_dir: str | Path,
                  sample_n: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Write equal-sized paired samples; return their reproducibility manifest.

    Missing stratum labels remain null; present labels must use identical canonical
    values across translations. Round-robin draws balance object categories first,
    then defect/source strata within each object. Small strata are exhausted first.
    Hash ordering makes selection independent of input order.
    """
    if isinstance(sample_n, bool) or not isinstance(sample_n, int) or sample_n <= 0:
        raise ValueError("sample_n must be a positive integer")
    zh, en, out_dir = Path(zh), Path(en), Path(out_dir)
    if any((out_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError(f"Refusing to overwrite sample artifacts in {out_dir}")
    indices = {"zh": _index(zh), "en": _index(en)}
    if indices["zh"].keys() != indices["en"].keys():
        raise ValueError("Chinese and English pair id sets differ")
    if sample_n > len(indices["zh"]):
        raise ValueError(f"Requested {sample_n} pairs but only {len(indices['zh'])} available")
    groups = defaultdict(list)
    for identity, record in indices["zh"].items():
        labels = tuple(record.get("metadata", {}).get(field) for field in STRATA_FIELDS)
        other = tuple(indices["en"][identity].get("metadata", {}).get(field) for field in STRATA_FIELDS)
        if labels != other:
            raise ValueError(f"Pair {identity!r} has inconsistent stratum labels")
        if _images(record) != _images(indices["en"][identity]):
            raise ValueError(f"Pair {identity!r} has inconsistent image paths")
        groups[labels].append(identity)

    def rank(value: Any) -> str:
        return hashlib.sha256(_json([seed, value]).encode("utf-8")).hexdigest()

    ordered = sorted(groups, key=lambda labels: (rank(labels), _json(labels)))
    queues = {labels: deque(sorted(groups[labels], key=lambda identity: (rank(identity), identity)))
              for labels in ordered}
    objects = sorted({labels[0] for labels in ordered}, key=lambda name: (rank(name), name))
    object_strata = {name: deque(labels for labels in ordered if labels[0] == name) for name in objects}
    selected = []
    while len(selected) < sample_n:
        for name in objects:
            strata = object_strata[name]
            while strata and not queues[strata[0]]:
                strata.popleft()
            if not strata:
                continue
            selected.append(queues[strata[0]].popleft())
            strata.rotate(-1)
            if len(selected) == sample_n:
                break
    selected_set = set(selected)
    distribution = [{**dict(zip(STRATA_FIELDS, labels)), "available": len(groups[labels]),
                     "selected": sum(identity in selected_set for identity in groups[labels])}
                    for labels in ordered]
    manifest = {
        "schema_version": 1, "seed": seed, "sample_n_per_language": sample_n,
        "available_pairs": len(indices["zh"]), "sample_ids": selected,
        "sampling": "equal object-category round-robin, then defect/source round-robin; SHA-256 ordered",
        "strata_fields": list(STRATA_FIELDS), "distribution": distribution,
        "sources": {lang: {"path": str(path.resolve()), "sha256": sha256_of(path)}
                    for lang, path in (("zh", zh), ("en", en))},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in ("zh", "en"):
        with (out_dir / f"industrial_{lang}.jsonl").open("x", encoding="utf-8") as handle:
            for identity in selected:
                record = deepcopy(indices[lang][identity])
                record.setdefault("metadata", {}).update(language=lang, pair_id=identity)
                if "id" in record and record["id"] != identity:
                    record["metadata"]["original_id"] = record["id"]
                record["id"] = identity
                handle.write(_json(record) + "\n")
    with (out_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh", required=True, type=Path)
    parser.add_argument("--en", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        sample_paired(args.zh, args.en, args.out_dir, args.sample_n, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"industrial_sample: {exc}\n")


if __name__ == "__main__":
    main()
