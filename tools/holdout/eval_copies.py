"""List training media paths whose content is identical (md5) to an eval image, using the md5 cache that
register_eval_sets.py writes for the training pools (defect_tetect, aircraft_vqa_v1, mbjc_v1). Eval images
missing from the cache are hashed here. The list feeds export_train_data.py --exclude-media for eval
copies the DB does not flag. Reads only files, never the DB.

  python eval_copies.py <eval.jsonl> [<eval.jsonl> ...] > exclude_media.txt
"""
import collections
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

CACHE = os.environ.get("MD5_CACHE", "/tmp/register_eval_sets.md5.json")


def md5(path):
    try:
        with open(path, "rb") as f:
            return path, hashlib.md5(f.read()).hexdigest()
    except OSError:
        return path, None


def main():
    cache = json.load(open(CACHE, encoding="utf-8"))
    ev = set()
    for path in sys.argv[1:]:
        for line in open(path, encoding="utf-8"):
            imgs = json.loads(line).get("images") or []
            ev.update([imgs] if isinstance(imgs, str) else imgs)
    with ThreadPoolExecutor(32) as ex:
        ev_hash = dict(ex.map(md5, sorted(ev)))
    want = {h for h in ev_hash.values() if h}
    hits = sorted(p for p, h in cache.items() if h in want and p not in ev)
    pools = collections.Counter(p.split("/corpus_media/")[-1].split("/")[0] for p in hits)
    print(f"eval images {len(ev)}, readable {len(want)} distinct contents, training copies {len(hits)} {dict(pools)}",
          file=sys.stderr)
    for p in hits:
        print(p)


if __name__ == "__main__":
    main()
