"""Relocate an exported industrial eval set into an isolated image directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


OLD_PREFIX = "defect-tetect-images/"
NEW_PREFIX = "eval_defect-tetect-images/"
NEW_ROOT = "/mnt/data3/code/data_process/eval_defect-tetect-images/"
CONTAINER_ROOT = "/code/data_process/eval_defect-tetect-images/"


def relocate(folder: Path) -> None:
    dataset = folder / "industrial_eval.jsonl"
    archive = folder / "selected_images.zip"
    manifest_path = folder / "manifest.json"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_paths = {OLD_PREFIX + row["images"][0].split("/defect-tetect-images/", 1)[1]
                 for row in rows}
    if len(old_paths) != manifest["selected_images"]:
        raise ValueError("JSONL image count differs from manifest")
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != old_paths or source.testzip() is not None:
            raise ValueError("Image archive does not match JSONL")
        for row in rows:
            old_member = OLD_PREFIX + row["images"][0].split("/defect-tetect-images/", 1)[1]
            if hashlib.sha256(source.read(old_member)).hexdigest() != manifest["image_hashes"][row["id"]]:
                raise ValueError(f"Image hash mismatch for {row['id']}")

        new_archive = archive.with_suffix(".zip.pending")
        with zipfile.ZipFile(new_archive, "w", compression=zipfile.ZIP_STORED) as target:
            for member in source.namelist():
                target.writestr(NEW_PREFIX + member.removeprefix(OLD_PREFIX), source.read(member))

    for row in rows:
        relative = row["images"][0].split("/defect-tetect-images/", 1)[1]
        row["images"] = [NEW_ROOT + relative]
    new_dataset = dataset.with_suffix(".jsonl.pending")
    new_dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
    manifest["dataset_sha256"] = hashlib.sha256(new_dataset.read_bytes()).hexdigest()
    manifest["target_image_root"] = NEW_ROOT.rstrip("/")
    new_manifest = manifest_path.with_suffix(".json.pending")
    new_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    new_archive.replace(archive)
    new_dataset.replace(dataset)
    new_manifest.replace(manifest_path)


def set_container_root(folder: Path) -> None:
    dataset = folder / "industrial_eval.jsonl"
    manifest_path = folder / "manifest.json"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in rows:
        current = row["images"][0]
        if not current.startswith(NEW_ROOT):
            raise ValueError(f"Unexpected image root: {current}")
        row["images"] = [CONTAINER_ROOT + current[len(NEW_ROOT):]]
    pending = dataset.with_suffix(".jsonl.pending")
    pending.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                       encoding="utf-8")
    manifest["dataset_sha256"] = hashlib.sha256(pending.read_bytes()).hexdigest()
    manifest["target_image_root"] = CONTAINER_ROOT.rstrip("/")
    pending_manifest = manifest_path.with_suffix(".json.pending")
    pending_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(dataset)
    pending_manifest.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--container-root", action="store_true")
    args = parser.parse_args()
    (set_container_root if args.container_root else relocate)(args.folder)


if __name__ == "__main__":
    main()
