"""Extract independent bilingual training-regression samples into one JSONL."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .industrial_labels import build_label_mapping, extract_labels, image_identity

OUTPUTS = ("industrial_eval.jsonl", "manifest.json", "mapping_audit.json")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _read(path, language, rejects):
    records = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        offset = 0
        for line, raw in enumerate(handle, 1):
            current_offset = offset
            offset += len(raw)
            digest.update(raw)
            try:
                record = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                conv = record.get("conversations")
                if (not isinstance(conv, list) or len(conv) != 2
                        or not all(isinstance(c, dict) for c in conv)
                        or conv[0].get("from") != "human" or conv[1].get("from") != "gpt"):
                    raise ValueError("expected one human/gpt conversation pair")
                if any(not isinstance(c.get("value"), str) or not c["value"].replace("<image>", "").strip() for c in conv):
                    raise ValueError("empty/non-string question or answer")
                if conv[0]["value"].count("<image>") > 1:
                    raise ValueError("multiple image markers")
                if not isinstance(record.get("metadata", {}), dict):
                    raise ValueError("metadata must be an object")
                image_key, _ = image_identity(record)
                obj, defect = extract_labels(record)
                compact = {"images": record.get("images", record.get("image")),
                           "conversations": [{"from": "human", "value": "<image>question"},
                                             {"from": "gpt", "value": f"Category: {obj}; Anomaly type: {defect}; Description:"}],
                           "_offset": current_offset,
                           "_sample_id": language + "_" + _hash([image_key, conv])}
                records.append((line, compact))
            except (ValueError, TypeError, KeyError) as exc:
                rejects.append({"language": language, "source_line": line, "reason": str(exc)})
    return records, digest.hexdigest()


def _select(records, count, seed):
    # Keep one deterministic question per image before balancing classes.
    by_image = defaultdict(list)
    for record in records:
        by_image[record["metadata"]["source_id"]].append(record)
    candidates = [min(group, key=lambda r: (_hash([seed, r["id"]]), r["id"])) for group in by_image.values()]
    groups = defaultdict(list)
    for record in candidates:
        meta = record["metadata"]
        groups[(meta["object_category"], meta["defect_type"])].append(record)
    queues = {key: deque(sorted(rows, key=lambda r: (_hash([seed, r["id"]]), r["id"]))) for key, rows in groups.items()}
    objects = sorted({key[0] for key in queues}, key=lambda v: (_hash([seed, v]), v))
    strata = {obj: deque(sorted((key for key in queues if key[0] == obj), key=lambda v: (_hash([seed, v]), v))) for obj in objects}
    selected = []
    while len(selected) < min(count, len(candidates)):
        for obj in objects:
            while strata[obj] and not queues[strata[obj][0]]:
                strata[obj].popleft()
            if strata[obj]:
                selected.append(queues[strata[obj][0]].popleft())
                strata[obj].rotate(-1)
                if len(selected) == count:
                    break
    return selected, len(candidates)


def extract(zh, en, out_dir, zh_n=500, en_n=500, seed=42, mapping=None):
    """Write auditable samples; insufficient pools emit audit/manifest and raise."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (zh_n, en_n)):
        raise ValueError("zh_n and en_n must be positive integers")
    out_dir = Path(out_dir)
    if any((out_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError(f"Refusing to overwrite artifacts in {out_dir}")
    paths = {"zh": Path(zh), "en": Path(en)}
    rejects, sources, inputs, signatures = [], {}, {}, {}
    for lang, path in paths.items():
        signatures[lang] = _signature(path)
        inputs[lang], digest = _read(path, lang, rejects)
        if _signature(path) != signatures[lang]:
            raise ValueError(f"Source changed during scan: {path}")
        sources[lang] = {"path": str(path.resolve()), "sha256": digest, "valid_records": len(inputs[lang])}
    overrides = json.loads(Path(mapping).read_text(encoding="utf-8-sig")) if mapping else None
    labels = build_label_mapping([r for _, r in inputs["en"]], [r for _, r in inputs["zh"]], overrides)
    resolved = {"zh": [], "en": []}
    for lang, rows in inputs.items():
        for line, original in rows:
            result = labels.resolve(original, lang)
            if result["status"] != "resolved":
                rejects.append({"language": lang, "source_line": line, **result})
                continue
            record = deepcopy(original)
            image_key = result["image_key"]
            identity = original["_sample_id"]
            meta = record.setdefault("metadata", {})
            prior = {key: record.pop(key) for key in ("category", "object_category", "defect_type", "language") if key in record}
            if prior:
                meta["original_top_level_labels"] = prior
            if "id" in record:
                meta["original_id"] = record["id"]
            meta.update({key: result[key] for key in ("object_category", "defect_type", "raw_object_category", "raw_defect_type", "source_domain", "mapping_source")})
            meta.update(language=lang, source_id=_hash(image_key), source_image_key=image_key, source_image=image_key,
                        source_jsonl=str(paths[lang].resolve()), source_line=line,
                        selection_source="training_regression")
            # Loader category precedence must not reuse legacy translated metadata.
            meta["category"] = meta["object_category"]
            record["id"] = identity
            resolved[lang].append(record)
    selections, available = {}, {}
    for lang, count in (("zh", zh_n), ("en", en_n)):
        selections[lang], available[lang] = _select(resolved[lang], count, seed)
    complete = available["zh"] >= zh_n and available["en"] >= en_n
    chosen = selections["zh"] + selections["en"]
    counts = Counter((r["metadata"]["language"], r["metadata"]["object_category"], r["metadata"]["defect_type"]) for r in chosen)
    image_counts = Counter(r["metadata"]["source_id"] for r in chosen)
    manifest = {"schema_version": 1, "status": "complete" if complete else "insufficient_samples",
                "selection_source": "training_regression", "seed": seed,
                "requested": {"zh": zh_n, "en": en_n}, "available_unique_images": available,
                "sources": sources, "selected_ids": [r["id"] for r in chosen] if complete else [],
                "selected_count": len(chosen) if complete else 0,
                "unique_images": len(image_counts) if complete else 0,
                "shared_bilingual_images": sum(n > 1 for n in image_counts.values()) if complete else 0,
                "distribution": [{"language": k[0], "object_category": k[1], "defect_type": k[2], "count": v} for k, v in sorted(counts.items())] if complete else [],
                "rejected_records": len(rejects),
                "sampling": "independent languages; one question per image per language; equal object round-robin then defect round-robin; seeded SHA256 ordering"}
    if mapping:
        manifest["mapping_override"] = {"path": str(Path(mapping).resolve()), "sha256": hashlib.sha256(Path(mapping).read_bytes()).hexdigest()}
    prepared = []
    if complete:
        for lang, path in paths.items():
            if _signature(path) != signatures[lang]:
                raise ValueError(f"Source changed before extraction: {path}")
        for record in chosen:
            lang = record["metadata"]["language"]
            with paths[lang].open("rb") as source:
                source.seek(record["_offset"])
                original = json.loads(source.readline().decode("utf-8-sig"))
            image_key, _ = image_identity(original)
            if lang + "_" + _hash([image_key, original.get("conversations")]) != record["id"]:
                raise ValueError(f"Source changed at selected record: {paths[lang]}:{record['metadata']['source_line']}")
            prior = {key: original.pop(key) for key in ("category", "object_category", "defect_type", "language") if key in original}
            meta = original.setdefault("metadata", {})
            if prior:
                meta["original_top_level_labels"] = prior
            if "id" in original:
                meta["original_id"] = original["id"]
            meta.update(record["metadata"])
            original["id"] = record["id"]
            prepared.append(original)
        for lang, path in paths.items():
            if _signature(path) != signatures[lang]:
                raise ValueError(f"Source changed during extraction: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in (("mapping_audit.json", {"mapping": labels.audit(), "rejected_records": rejects}), ("manifest.json", manifest)):
        with (out_dir / filename).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    if not complete:
        raise ValueError(f"Insufficient resolved unique images: {available}; review {out_dir / 'mapping_audit.json'} and rerun in a new output directory")
    with (out_dir / "industrial_eval.jsonl").open("x", encoding="utf-8") as handle:
        for original in prepared:
            handle.write(_json(original) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("zh", "en", "out-dir"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--zh-n", type=int, default=500)
    parser.add_argument("--en-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping", type=Path)
    args = parser.parse_args()
    try:
        result = extract(args.zh, args.en, args.out_dir, args.zh_n, args.en_n, args.seed, args.mapping)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(_json({"output": str(args.out_dir / "industrial_eval.jsonl"),
                 "selected_per_language": result["requested"],
                 "selected_count": result["selected_count"],
                 "unique_images": result["unique_images"],
                 "shared_bilingual_images": result["shared_bilingual_images"],
                 "rejected_records": result["rejected_records"],
                 "manifest": str(args.out_dir / "manifest.json"),
                 "mapping_audit": str(args.out_dir / "mapping_audit.json")}))


if __name__ == "__main__":
    main()
