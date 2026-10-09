"""Find training media that are byte-identical copies of eval media, across every media pool in the DB.

holdout_setup.py and register_eval_sets.py only compared each eval set with the batch it came from; the
first cross-pool check found 25 aircraft eval images in defect_tetect and 9 park eval clips under other
paths. This compares every eval set with every image / video in corpus_media and defect_tetect_media:
candidates share the file name and the file size with an eval file, md5 decides.

Output TSV (feeds register_eval_sets.py --copies): eval_set, key_type, train_path, batch_tag, eval_path
Reads the DB and files only.

  python find_copies.py > copies.tsv
"""
import collections
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pymysql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_train_data import SOURCE_NAMES  # noqa: E402
from fielddb import connect  # noqa: E402

R = "/mnt/si003010kcx0/mmdata/eval_runs"
EVAL_SETS = {
    "aircraft_eval_v2": "/mnt/si003010kcx0/mmdata/mmcode/eval_hk/data/aircraft_vqa/aircraft_eval_v2.jsonl",
    "book_eval_v1": f"{R}/book/data/book_vqa.jsonl",
    "journal_eval_v1": f"{R}/journal/data/journal_vqa.jsonl",
    "park_video_v1": f"{R}/park/data/park_video.jsonl",
    "industrial_eval_v1": f"{R}/industrial/data/industrial_eval.jsonl",
    "mbjc_test_v1": f"{R}/det/data/mbjc_test_v1.jsonl",
    "journal_gi82_eval_v1": f"{R}/journal_gi82/data/journal_gi82_vqa.jsonl",
}


# readable names for the summary only; the TSV keeps the registered eval_set (register_eval_sets.py --copies uses it)
DISPLAY = {"aircraft_eval_v2": "aircraft_defect", "park_video_v1": "video_park_abnormal", "mbjc_test_v1": "detection",
           "industrial_eval_v1": "industrial", "book_eval_v1": "book", "journal_eval_v1": "journal",
           "journal_gi82_eval_v1": "journal_gi82"}


def eval_media(path):
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for key in ("images", "videos"):
                v = rec.get(key) or []
                out.update([v] if isinstance(v, str) else v)
    return out


def size(p):
    try:
        return p, os.stat(p).st_size
    except OSError:
        return p, None


def md5(p):
    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
    except OSError:
        return p, None
    return p, h.hexdigest()


def main():
    ev = {name: eval_media(path) for name, path in EVAL_SETS.items()}
    names = {os.path.basename(p) for paths in ev.values() for p in paths}
    db = connect(read_timeout=1800)
    cur = db.cursor(pymysql.cursors.SSCursor)
    pool = collections.defaultdict(list)          # basename -> [(path, media_type, batch)]
    for sql in ("""SELECT m.new_path, m.media_type, s.batch_tag FROM corpus_media m
                     JOIN corpus_samples s ON s.id = m.sample_id""",
                """SELECT m.new_path, m.media_type, s.batch_tag FROM defect_tetect_media m
                     JOIN defect_tetect_samples s ON s.id = m.sample_id"""):
        cur.execute(sql)
        for path, mtype, batch in cur:
            b = os.path.basename(path)
            if b in names:
                pool[b].append((path, mtype, batch))
        cur.close()
        cur = db.cursor(pymysql.cursors.SSCursor)
    db.close()
    for b in pool:
        pool[b] = sorted(set(pool[b]))
    cand = {p for lst in pool.values() for p, _, _ in lst} | {p for paths in ev.values() for p in paths}
    with ThreadPoolExecutor(48) as ex:
        sizes = dict(ex.map(size, sorted(cand), chunksize=256))
    pairs = []
    for name, paths in ev.items():
        for ep in paths:
            for tp, mtype, batch in pool.get(os.path.basename(ep), ()):
                if sizes.get(ep) is not None and sizes.get(ep) == sizes.get(tp):
                    pairs.append((name, mtype, tp, batch, ep))
    with ThreadPoolExecutor(32) as ex:
        hashes = dict(ex.map(md5, sorted({p for pr in pairs for p in (pr[2], pr[4])})))
    hits = sorted({pr for pr in pairs if hashes.get(pr[2]) and hashes.get(pr[2]) == hashes.get(pr[4])})
    stats = collections.Counter((h[0], h[3], "same path" if h[2] == h[4] else "copy") for h in hits)
    for k, n in sorted(stats.items()):
        name = f"{DISPLAY.get(k[0], k[0])} ({k[0]})"
        print(f"{name:44s} {SOURCE_NAMES.get(k[1], k[1]):28s} {k[2]:9s} {n:6d}", file=sys.stderr)
    print(f"candidate pairs {len(pairs):,}, identical {len(hits):,}", file=sys.stderr)
    for h in hits:
        print("\t".join(h))


if __name__ == "__main__":
    main()
