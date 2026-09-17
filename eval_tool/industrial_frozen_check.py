"""Opt-in content checks before reusing a frozen industrial evaluation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .industrial_mapping import file_sha256


def verify_frozen_dataset(dataset, manifest=None, image_root=None):
    dataset = Path(dataset)
    manifest = Path(manifest) if manifest else dataset.parent / "manifest.json"
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("Industrial manifest is not a completed build")
    if report.get("dataset_sha256") != file_sha256(dataset):
        raise ValueError("Frozen industrial evaluation JSONL changed or is not bound to manifest")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    identities = [r.get("id") for r in rows]
    if len(identities) != len(set(identities)) or identities != report.get("selected_ids"):
        raise ValueError("Frozen sample IDs/order differ from manifest")
    hashes = report.get("image_hashes", {})
    seen = {}
    for record in rows:
        images = record.get("images", record.get("image"))
        images = [images] if isinstance(images, str) else images
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError("Frozen industrial records require a single image")
        image = Path(image_root or dataset.parent) / images[0]
        if image not in seen:
            seen[image] = file_sha256(image)
        if seen[image] != hashes.get(record["id"]):
            raise ValueError(f"Frozen industrial image changed: {record['id']}")
    return {"status": "verified", "records": len(rows), "image_paths": len(seen),
            "dataset_sha256": report["dataset_sha256"]}


def verify_config_datasets(tsv_dir, datasets, dataset_params):
    checked = set()
    for key, name in datasets.items():
        params = dataset_params.get(key, {})
        if not params.get("verify_frozen_manifest"):
            continue
        path = Path(tsv_dir) / f"{name}.jsonl"
        identity = (path, params.get("image_root"))
        if identity not in checked:
            verify_frozen_dataset(path, image_root=params.get("image_root"))
            checked.add(identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    print(json.dumps(verify_frozen_dataset(**vars(parser.parse_args(argv)))))


if __name__ == "__main__":
    main()
