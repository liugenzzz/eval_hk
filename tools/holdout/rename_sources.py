"""Rename the qa / vqa files of an existing export to the readable source names in
export_train_data.SOURCE_NAMES (e.g. aircraft_vqa_v1 -> aircraft_defect), and fix manifest.json to match.
Row ids and the manifest's batch keys keep the DB batch_tag. Safe to rerun: already-renamed files are skipped.

  python rename_sources.py <export dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_train_data import SOURCE_NAMES  # noqa: E402


def renamed(name):
    head, base = os.path.split(name)
    batch, rest = base.split(".", 1)
    return os.path.join(head, f"{SOURCE_NAMES[batch]}.{rest}") if batch in SOURCE_NAMES else name


def main():
    out = sys.argv[1]
    path = os.path.join(out, "manifest.json")
    man = json.load(open(path, encoding="utf-8"))
    moves = {n: renamed(n) for n in man["files"] if renamed(n) != n}
    for old, new in moves.items():
        if os.path.exists(os.path.join(out, new)):
            sys.exit(f"{new} already exists")
    for old, new in moves.items():
        os.rename(os.path.join(out, old), os.path.join(out, new))
        print(f"  {old} -> {new}")
    man["files"] = dict(sorted((moves.get(n, n), e) for n, e in man["files"].items()))
    for t in man["batches"].values():
        t["written"] = {moves.get(n, n): v for n, v in t["written"].items()}
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    print(f"renamed {len(moves)} files; manifest updated")


if __name__ == "__main__":
    main()
