"""Swap re-exported batches into an existing export_train_data.py output.

Used when the exclusions change after a full export (e.g. more eval copies found): re-export only the
affected batches into a scratch dir with --only, then splice them in. qa / vqa files of those batches are
replaced; the merged <type>/pt/<type>.pt.jsonl files are rewritten without the old rows of those batches
(matched by the "<batch>:" id prefix) and with the new ones appended. Rejects, manifest entries and
dataset_info (if the export has one) follow. Every other batch is left untouched.

  python export_train_data.py --out <scratch> --only b1,b2 --exclude-media <new list>
  python splice_export.py <out> <scratch>
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_train_data import dataset_info  # noqa: E402


def move_into(src, dest):
    """Copy next to dest, then rename: works when the scratch dir is on another filesystem."""
    tmp = dest + ".splice_tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    os.remove(src)


def rewrite_pt(dest, drop, new_src):
    """dest minus rows of the batches in drop, plus every row of new_src (if any). Returns rows kept + added."""
    tmp = dest + ".splice_tmp"
    rows = 0
    with open(tmp, "w", encoding="utf-8") as w:
        if os.path.exists(dest):
            with open(dest, encoding="utf-8") as r:
                for line in r:
                    if line[8:line.index('"', 8)].rpartition(":")[0] in drop:   # {"id": "<batch>:<n>", ...
                        continue
                    w.write(line)
                    rows += 1
        if new_src:
            with open(new_src, encoding="utf-8") as r:
                for line in r:
                    w.write(line)
                    rows += 1
    os.replace(tmp, dest)
    if new_src:
        os.remove(new_src)
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
        kind = (new_e or old_e)["kind"]
        if kind == "pt":
            rows = rewrite_pt(dest, batches, os.path.join(redo, name) if new_e else None)
            others = [b for b in (old_e or {}).get("batches", []) if b not in batches]
            e = dict(old_e or new_e)
            e.update(rows=rows, bytes=os.path.getsize(dest),
                     batches=sorted(set(others) | set((new_e or {}).get("batches", []))))
            man["files"][name] = e
        elif new_e:
            move_into(os.path.join(redo, name), dest)
            man["files"][name] = new_e
        else:
            os.remove(dest)
            man["files"].pop(name)
    for b in batches:
        src, dest = os.path.join(redo, "_rejects", f"{b}.jsonl"), os.path.join(out, "_rejects", f"{b}.jsonl")
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            move_into(src, dest)
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
    print(f"spliced {sorted(batches)} into {len(touched)} files "
          f"(scratch {redo} now holds only its manifest; remove it when done)")


if __name__ == "__main__":
    main()
