"""Swap re-exported batches into an existing export_train_data.py output.

Used when data or exclusions change for some batches: re-export only those batches into a scratch dir with
--only, then splice them in. Files merge several sources (one file per task type), so every file that holds rows
of those batches is rewritten: their old rows dropped (matched by the "<batch>:" id prefix), the new rows appended.
Rejects, manifest entries and dataset_info (if the export has one) follow. Rows of other batches are untouched.

  python export_train_data.py --out <scratch> --only b1,b2 --exclude-media <list>
  python splice_export.py <out> <scratch>
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_train_data import dataset_info  # noqa: E402


def batch_of_line(line):
    # every row starts with {"id": "<batch>:<n>", ...
    return line[8:line.index('"', 8)].rpartition(":")[0]


def rewrite(dest, drop, new_src):
    """dest minus rows of the batches in drop, plus every row of new_src. Returns rows written."""
    tmp = dest + ".splice_tmp"
    rows = 0
    with open(tmp, "w", encoding="utf-8") as w:
        if os.path.exists(dest):
            with open(dest, encoding="utf-8") as r:
                for line in r:
                    if batch_of_line(line) not in drop:
                        w.write(line)
                        rows += 1
        if new_src:
            with open(new_src, encoding="utf-8") as r:
                for line in r:
                    w.write(line)
                    rows += 1
    os.replace(tmp, dest)
    return rows


def main():
    out, redo = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    new = json.load(open(os.path.join(redo, "manifest.json"), encoding="utf-8"))
    batches = set(new["batches"])
    touched = {n for n, e in man["files"].items() if batches & set(e["batches"])} | set(new["files"])
    for name in sorted(touched):
        old_e, new_e = man["files"].get(name), new["files"].get(name)
        dest = os.path.join(out, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        rows = rewrite(dest, batches, os.path.join(redo, name) if new_e else None)
        if rows == 0:
            os.remove(dest)
            man["files"].pop(name, None)
            continue
        kept = [b for b in (old_e or {}).get("batches", []) if b not in batches]
        e = dict(old_e or new_e)
        e.update(rows=rows, bytes=os.path.getsize(dest),
                 batches=sorted(set(kept) | set((new_e or {}).get("batches", []))))
        man["files"][name] = e
    for b in batches:
        src, dest = os.path.join(redo, "_rejects", f"{b}.jsonl"), os.path.join(out, "_rejects", f"{b}.jsonl")
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest + ".splice_tmp")
            os.replace(dest + ".splice_tmp", dest)
        elif os.path.exists(dest):
            os.remove(dest)
    man["files"] = dict(sorted(man["files"].items()))
    man["batches"].update(new["batches"])
    man.setdefault("reexports", []).append({"created": new["created"], "batches": sorted(batches),
                                            "exclude_media_file": new["exclude_media_file"],
                                            "elapsed_minutes": new["elapsed_minutes"]})
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    if os.path.exists(os.path.join(out, "dataset_info.json")):
        with open(os.path.join(out, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info(out, man["files"]), f, ensure_ascii=False, indent=2)
    print(f"spliced {sorted(batches)} into {len(touched)} files (scratch {redo} can be removed)")


if __name__ == "__main__":
    main()
