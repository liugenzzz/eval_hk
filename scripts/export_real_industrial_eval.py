"""Export 500 paired real industrial images and 1000 evaluable conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from build_real_industrial_mapping import PARTS, canonical, image_key, labels, normalize, records


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build(base: Path, mapping_path: Path, output: Path, target_root: Path, count=500, seed=42):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    object_map = {(r["source_domain"], r["language"], r["alias"]): r["canonical"]
                  for r in mapping["objects"]}
    defect_map = {(r["source_domain"], r["language"], r["object_category"], r["alias"]): r["canonical"]
                  for r in mapping["defects"]}
    invalid = Counter()
    candidates = {}

    for archive_name, en_member, zh_member in PARTS:
        with zipfile.ZipFile(base / archive_name) as archive:
            english = {}
            conflicted = set()
            for record in records(archive, en_member, invalid):
                try:
                    key, domain = image_key(record)
                    obj, defect = labels(record)
                    signature = (canonical(obj, object_label=True), canonical(defect))
                    if None in signature or not all(signature):
                        raise ValueError("non_ascii_english_label")
                except (TypeError, ValueError) as error:
                    invalid[str(error)] += 1
                    continue
                if key in english and english[key][0] != signature:
                    conflicted.add(key)
                else:
                    english.setdefault(key, (signature, record))

            for record in records(archive, zh_member, invalid):
                try:
                    key, domain = image_key(record)
                    if key in conflicted or key not in english:
                        raise ValueError("unpaired_or_conflicting_image")
                    zh_obj, zh_defect = labels(record)
                    en_signature, en_record = english[key]
                    obj = object_map.get((domain, "zh", normalize(zh_obj)))
                    defect = defect_map.get((domain, "zh", obj, normalize(zh_defect)))
                    if (obj, defect) != en_signature:
                        raise ValueError("ambiguous_or_conflicting_mapping")
                    for conversation in (record.get("conversations"), en_record.get("conversations")):
                        if (not isinstance(conversation, list) or len(conversation) != 2 or
                            [turn.get("from") for turn in conversation] != ["human", "gpt"] or
                            any(not isinstance(turn.get("value"), str) or not turn["value"].strip()
                                for turn in conversation)):
                            raise ValueError("invalid_conversation")
                    # Keep one paired conversation per image, deterministically.
                    pair = (obj, defect, record, en_record, archive_name)
                    old = candidates.get(key)
                    if old is None or digest(json.dumps(record, ensure_ascii=False, sort_keys=True)) < digest(
                            json.dumps(old[2], ensure_ascii=False, sort_keys=True)):
                        candidates[key] = pair
                except (TypeError, ValueError) as error:
                    invalid[str(error)] += 1

    with zipfile.ZipFile(base / "defect-tetect-images.zip") as source_images:
        members = {}
        for name in source_images.namelist():
            if name.startswith("defect-tetect-images/"):
                members.setdefault(name.casefold(), name)
    available_candidates = {
        key: value for key, value in candidates.items()
        if ("defect-tetect-images/" + key).casefold() in members
    }
    invalid["image_missing_from_archive"] = len(candidates) - len(available_candidates)
    candidates = available_candidates
    ranked = sorted(candidates, key=lambda key: digest(f"{seed}:{key}"))
    by_object = defaultdict(list)
    for key in ranked:
        by_object[candidates[key][0]].append(key)
    selected = []
    while len(selected) < count and any(by_object.values()):
        for obj in sorted(by_object):
            if by_object[obj] and len(selected) < count:
                selected.append(by_object[obj].pop(0))
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} paired images available, need {count}")

    output.mkdir(parents=True, exist_ok=True)
    image_archive = output / "selected_images.zip"
    dataset = output / "industrial_eval.jsonl"
    image_hashes = {}
    selected_ids = []
    missing = []
    with zipfile.ZipFile(base / "defect-tetect-images.zip") as source_images, \
            zipfile.ZipFile(image_archive, "w", compression=zipfile.ZIP_STORED) as images, \
            dataset.open("w", encoding="utf-8") as stream:
        for key in selected:
            member = members[("defect-tetect-images/" + key).casefold()]
            actual_key = member.removeprefix("defect-tetect-images/")
            image_bytes = source_images.read(member)
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            images.writestr(member, image_bytes)
            obj, defect, zh_record, en_record, source_archive = candidates[key]
            for lang, original in (("zh", zh_record), ("en", en_record)):
                record = dict(original)
                record_id = lang + "_" + digest(f"{key}:{lang}")
                image_hashes[record_id] = image_sha
                selected_ids.append(record_id)
                record["id"] = record_id
                record["images"] = [str(target_root / actual_key)]
                record.pop("image", None)
                metadata = dict(record.get("metadata") or {})
                metadata.update(language=lang, category=obj, object_category=obj,
                                defect_type=defect, source_domain=key.split("/", 1)[0],
                                defect_status="normal" if defect in {"normal", "good", "ok"} else "abnormal",
                                source_image_key=actual_key, source_archive=source_archive,
                                selection_source="training_regression")
                record["metadata"] = metadata
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "status": "complete", "selection_source": "training_regression",
        "requested": {"zh": count, "en": count}, "selected_images": count,
        "selected_ids": selected_ids, "image_hashes": image_hashes,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "target_image_root": str(target_root),
        "skipped": dict(sorted(invalid.items())),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(image_archive) as images:
        if images.testzip() is not None or len(images.namelist()) != count:
            raise ValueError("Image archive verification failed")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build(args.base, args.mapping, args.output, args.target_root,
                     count=args.count, seed=args.seed)
    print(json.dumps({"selected_images": manifest["selected_images"],
                      "records": len(manifest["selected_ids"]),
                      "dataset_sha256": manifest["dataset_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
