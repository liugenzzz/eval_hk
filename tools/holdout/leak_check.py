"""Do any eval-set items appear in a set of training files? Run it on every export before training.

Streams each training row once against small key sets built from the eval sets:
  text   sha1 of the whitespace-stripped turn (first 400 chars), turns of >= 30 chars only
  media  exact path; then basename, confirmed by md5 (eval copies keep the original file name)
  id     eval items that carry a DB id (<batch>:<id>, or metadata.corpus_id) vs the "id" column of
         the export
Templated prompts and short answers collide across different images, so for the image sets (aircraft,
det, industrial) only media hits mean leakage; for book / journal the text hits do.

  python leak_check.py --train <export dir or jsonl files> [--out /tmp/leak_check]
Eval sets are named like the training data (aircraft_defect, video_park_abnormal, detection, ...); the name
registered in field.eval_holdout is shown next to it.
2026-10-09 on the 32B training files (train_data/*.jsonl): book 1200/1200 answers verbatim, journal
828/829, park all 180 clips, aircraft 1713/1984 images md5-identical; det and industrial clean.
"""
import argparse
import ast
import collections
import glob
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

R = "/mnt/si003010kcx0/mmdata/eval_runs"
# display name (same vocabulary as the training data) -> (eval_set registered in field.eval_holdout, eval file)
EVALS = {
    "book": ("book_eval_v1", f"{R}/book/data/book_vqa.jsonl"),
    "journal": ("journal_eval_v1", f"{R}/journal/data/journal_vqa.jsonl"),
    "journal_gi82": ("journal_gi82_eval_v1", f"{R}/journal_gi82/data/journal_gi82_vqa.jsonl"),
    "video_park_abnormal": ("park_video_v1", f"{R}/park/data/park_video.jsonl"),
    "detection": ("mbjc_test_v1", f"{R}/det/data/mbjc_test_v1.jsonl"),
    "industrial": ("industrial_eval_v1", f"{R}/industrial/data/industrial_eval.jsonl"),
    "aircraft_defect": ("aircraft_eval_v2",
                        "/mnt/si003010kcx0/mmdata/mmcode/eval_hk/data/aircraft_vqa/aircraft_eval_v2.jsonl"),
}
PARK_BATCH = "video_abnormal_behavior_v1"
MIN_TEXT = 30
WS = re.compile(r"\s+")
ID_RE = re.compile(r"^([A-Za-z0-9_]+_v\d+):(\d+)")


def tkey(s):
    if not isinstance(s, str):
        return None
    n = WS.sub("", s.replace("<image>", "").replace("<video>", ""))
    return hashlib.sha1(n[:400].encode("utf-8")).digest() if len(n) >= MIN_TEXT else None


def parse_maybe(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return ast.literal_eval(v)
    return v


def turns(rec):
    conv = parse_maybe(rec.get("conversations"))
    if conv is None:
        conv = parse_maybe(rec.get("messages"))
    users, assts = [], []
    for t in conv or []:
        role = t.get("from", t.get("role"))
        text = t.get("value", t.get("content"))
        (assts if role in ("gpt", "assistant") else users).append(text)
    if "instruction" in rec:
        ins, inp = rec.get("instruction") or "", rec.get("input") or ""
        users += [ins, inp, ins + inp]
        assts.append(rec.get("output"))
    return users, assts


def media(rec):
    out = []
    for k in ("images", "videos"):
        v = parse_maybe(rec.get(k))
        out += [v] if isinstance(v, str) else [p for p in (v or []) if isinstance(p, str)]
    return out


def eval_db_id(name, rec):
    m = ID_RE.match(str(rec.get("id", "")))
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    meta = parse_maybe(rec.get("metadata")) or {}
    if name == "video_park_abnormal" and isinstance(meta, dict) and meta.get("corpus_id"):
        return f"{PARK_BATCH}:{meta['corpus_id']}"
    return None


def md5(path):
    try:
        with open(path, "rb") as f:
            return path, hashlib.md5(f.read()).hexdigest()
    except OSError:
        return path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/leak_check")
    a = ap.parse_args()
    files = []
    for t in a.train:
        if os.path.isdir(t):  # an export dir: every qa / vqa file (pt has no QA pairs to match)
            files += sorted(glob.glob(os.path.join(t, "**", "qa", "*.jsonl"), recursive=True)
                            + glob.glob(os.path.join(t, "**", "vqa", "*.jsonl"), recursive=True))
        else:
            files.append(t)
    os.makedirs(a.out, exist_ok=True)

    akey, ukey, pkey, bkey, idkey = {}, {}, {}, {}, {}
    n_items = collections.Counter()
    for name, (_, path) in EVALS.items():
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                iid = rec.get("id", i)
                n_items[name] += 1
                us, as_ = turns(rec)
                for t in as_:
                    k = tkey(t)
                    if k:
                        akey.setdefault(k, []).append((name, iid))
                for t in us:
                    k = tkey(t)
                    if k:
                        ukey.setdefault(k, []).append((name, iid))
                for p in media(rec):
                    pkey.setdefault(p, []).append((name, iid))
                    bkey.setdefault(os.path.basename(p), []).append((name, iid, p))
                did = eval_db_id(name, rec)
                if did:
                    idkey.setdefault(did, []).append((name, iid))
    print("eval items:", dict(n_items), "| eval items with a DB id:", len(idkey), flush=True)

    hit = {k: collections.defaultdict(set) for k in ("assistant", "user", "path", "id")}
    pairs = set()
    rows = 0
    for fn in files:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                rows += 1
                rec = json.loads(line)
                if rec.get("id") in idkey:
                    for name, iid in idkey[rec["id"]]:
                        hit["id"][name].add(iid)
                us, as_ = turns(rec)
                for t in as_:
                    k = tkey(t)
                    if k in akey:
                        for name, iid in akey[k]:
                            hit["assistant"][name].add(iid)
                for t in us:
                    k = tkey(t)
                    if k in ukey:
                        for name, iid in ukey[k]:
                            hit["user"][name].add(iid)
                for p in media(rec):
                    if p in pkey:
                        for name, iid in pkey[p]:
                            hit["path"][name].add(iid)
                    for name, iid, ep in bkey.get(os.path.basename(p), ()):
                        pairs.add((name, iid, ep, p))
        print(f"  scanned {os.path.basename(fn)}  (total {rows:,} rows)", flush=True)

    paths = sorted({p for pr in pairs for p in pr[2:]})
    with ThreadPoolExecutor(32) as ex:
        h = dict(ex.map(md5, paths))
    md5_hit = collections.defaultdict(set)
    for name, iid, ep, tp in pairs:
        if h.get(ep) and h.get(ep) == h.get(tp):
            md5_hit[name].add(iid)

    print(f"\ntraining files {len(files)}, rows {rows:,}")
    print(f"{'eval set (registered as)':44s} {'items':>6s} {'answer':>7s} {'prompt':>7s} {'path':>6s} {'md5':>6s} {'db id':>6s}")
    summary = {}
    for name in EVALS:
        row = {"eval_set": EVALS[name][0], "items": n_items[name], "answer_text": len(hit["assistant"][name]),
               "prompt_text": len(hit["user"][name]),
               "media_path": len(hit["path"][name]), "media_md5": len(md5_hit[name]), "db_id": len(hit["id"][name])}
        summary[name] = row
        print(f"{name + ' (' + row['eval_set'] + ')':44s} {row['items']:6d} {row['answer_text']:7d} {row['prompt_text']:7d} {row['media_path']:6d} "
              f"{row['media_md5']:6d} {row['db_id']:6d}")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"files": files, "rows": rows, "summary": summary,
                   "hits": {k: {n: sorted(map(str, v)) for n, v in d.items()} for k, d in hit.items()},
                   "md5_hits": {n: sorted(map(str, v)) for n, v in md5_hit.items()}}, f, ensure_ascii=False)
    print("written", os.path.join(a.out, "summary.json"))


if __name__ == "__main__":
    main()
