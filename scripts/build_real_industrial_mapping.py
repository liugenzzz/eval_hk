"""Build a portable bilingual industrial label mapping from paired ZIP datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import yaml


PARTS = (
    ("outputs_part0.zip", "outputs_part0/sharegpt/train_en.json", "outputs_part0/sharegpt/train_zh.json"),
    ("outputs_part1.zip", "outputs_part1/sharegpt/train_en.json", "outputs_part1/sharegpt/train_zh.json"),
)


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def canonical(value: str, *, object_label: bool = False) -> str | None:
    value = normalize(value)
    if object_label:
        value = {"airplane": "aircraft", "aeroplane": "aircraft"}.get(value, value)
    if not value.isascii():
        return None
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_") or None


def labels(record: dict) -> tuple[str, str]:
    answers = [turn.get("value", "") for turn in record.get("conversations", [])
               if str(turn.get("from", "")).lower() in {"gpt", "assistant"}]
    if len(answers) != 1:
        raise ValueError("expected one assistant answer")
    match = re.match(
        r"^\s*(?:Category|类别)\s*[:：]\s*(.*?)\s*[;；]\s*"
        r"(?:Anomaly\s+type|异常类型)\s*[:：]\s*(.*?)\s*[;；]",
        answers[0], re.I | re.S,
    )
    if not match or not all(part.strip() for part in match.groups()):
        raise ValueError("missing category/anomaly prefix")
    return tuple(part.strip() for part in match.groups())


def image_key(record: dict) -> tuple[str, str]:
    images = record.get("images", record.get("image"))
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("expected one image")
    value = str(images[0]).replace("\\", "/").strip()
    for marker in ("defect-tetect-images/", "defect-detect-images/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = str(PurePosixPath(value))
    return value, value.split("/", 1)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(archive: zipfile.ZipFile, member: str, invalid: Counter):
    with archive.open(member) as stream:
        for raw in stream:
            raw = raw.strip()
            if raw in {b"", b"[", b"]"}:
                continue
            if raw.startswith(b"["):
                raw = raw[1:].lstrip()
            if raw.endswith(b"]"):
                raw = raw[:-1].rstrip()
            raw = raw.rstrip(b",").rstrip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid[f"invalid_json_line:{member}"] += 1
                continue
            if isinstance(value, dict):
                yield value
            else:
                invalid[f"non_object_line:{member}"] += 1


def build(base: Path, output: Path) -> dict:
    objects: dict[tuple[str, str], Counter] = defaultdict(Counter)
    defects: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    english_objects: dict[tuple[str, str], Counter] = defaultdict(Counter)
    english_defects: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    invalid = Counter()
    totals = Counter()

    for archive_name, en_member, zh_member in PARTS:
        archive_path = base / archive_name
        with zipfile.ZipFile(archive_path) as archive:
            english_by_image = defaultdict(set)
            for en_record in records(archive, en_member, invalid):
                totals[f"{archive_name}:en"] += 1
                try:
                    en_image, en_domain = image_key(en_record)
                    en_object, en_defect = labels(en_record)
                    object_id = canonical(en_object, object_label=True)
                    defect_id = canonical(en_defect)
                    if not object_id or not defect_id:
                        raise ValueError("non_ascii_english_label")
                except (TypeError, ValueError) as error:
                    invalid[str(error)] += 1
                    continue
                english_by_image[(en_image, en_domain)].add((object_id, defect_id))
                english_objects[(en_domain, normalize(en_object))][object_id] += 1
                english_defects[(en_domain, object_id, normalize(en_defect))][defect_id] += 1

            for zh_record in records(archive, zh_member, invalid):
                totals[f"{archive_name}:zh"] += 1
                try:
                    zh_image, zh_domain = image_key(zh_record)
                    zh_object, zh_defect = labels(zh_record)
                    pairs = english_by_image.get((zh_image, zh_domain), set())
                    if len(pairs) != 1:
                        raise ValueError("english_image_missing_or_ambiguous")
                    object_id, defect_id = next(iter(pairs))
                except (TypeError, ValueError) as error:
                    invalid[str(error)] += 1
                    continue
                domain = zh_domain
                objects[(domain, normalize(zh_object))][object_id] += 1
                defects[(domain, object_id, normalize(zh_defect))][defect_id] += 1

    def resolved_rows(mapping, language):
        rows, conflicts = [], []
        for key, counts in sorted(mapping.items()):
            values = sorted(counts)
            row = {"source_domain": key[0], "language": language, "alias": key[-1],
                   "canonical_candidates": values, "samples": sum(counts.values())}
            if len(key) == 3:
                row["object_category"] = key[1]
            if len(values) == 1:
                row["canonical"] = values[0]
                rows.append(row)
            else:
                row["candidate_counts"] = dict(sorted(counts.items()))
                conflicts.append(row)
        return rows, conflicts

    zh_objects, object_conflicts = resolved_rows(objects, "zh")
    zh_defects, defect_conflicts = resolved_rows(defects, "zh")
    en_objects, en_object_conflicts = resolved_rows(english_objects, "en")
    en_defects, en_defect_conflicts = resolved_rows(english_defects, "en")
    document = {
        "schema_version": 1,
        "version": "industrial_bilingual_real_v1",
        "status": "portable",
        "path_policy": "Image and dataset roots are intentionally omitted; source_domain is derived after either defect-tetect-images/ or defect-detect-images/.",
        "sources": [{"archive": name, "sha256": sha256(base / name),
                     "en_records": totals[f"{name}:en"], "zh_records": totals[f"{name}:zh"]}
                    for name, _, _ in PARTS],
        "objects": en_objects + zh_objects,
        "defects": en_defects + zh_defects,
        "conflicts": object_conflicts + defect_conflicts + en_object_conflicts + en_defect_conflicts,
        "audit": {"records": dict(sorted(totals.items())), "invalid_records": dict(sorted(invalid.items())),
                  "resolved_objects": len(en_objects) + len(zh_objects),
                  "resolved_defects": len(en_defects) + len(zh_defects)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.base, args.output)
    print(yaml.safe_dump(result["audit"], allow_unicode=True, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
