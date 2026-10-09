"""Check an export_train_data.py output before anyone trains on it.

  every file    parses, one key set per file, rows pass the LlamaFactory rules, <image>/<video>
                placeholders match the media lists, ids unique, row count == manifest
  accounting    per batch: rows read == rows in the DB, and written + held out + excluded + rejected == read
  media         a random sample of media paths exists on disk
  dataset_info  if present: every file is registered, and nothing else

  python verify_export.py <export dir> [--sample 20000]
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_train_data import check_openai, check_sharegpt  # noqa: E402


def check_row(fmt, rec):
    if fmt == "pt":
        return None if isinstance(rec.get("text"), str) and rec["text"].strip() else "text 为空"
    if fmt == "sharegpt":
        why = check_sharegpt(rec.get("conversations"))
        texts = [m["value"] for m in rec["conversations"]] if not why else []
    elif fmt == "openai":
        why = check_openai(rec.get("messages"))
        texts = [m["content"] for m in rec["messages"]] if not why else []
    else:
        why = None if str(rec.get("instruction", "")).strip() and str(rec.get("output", "")).strip() else "alpaca 空"
        texts = [rec.get("instruction", ""), rec.get("input", ""), rec.get("output", "")]
    if why:
        return why
    joined = "".join(texts)
    if joined.count("<image>") != len(rec.get("images", [])):
        return "<image> 数与 images 不符"
    if joined.count("<video>") != len(rec.get("videos", [])):
        return "<video> 数与 videos 不符"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sample", type=int, default=20000)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.out, "manifest.json"), encoding="utf-8"))
    info_path = os.path.join(a.out, "dataset_info.json")
    info = json.load(open(info_path, encoding="utf-8")) if os.path.exists(info_path) else None
    problems = []
    ids = collections.defaultdict(set)
    reservoir, seen_media = [], 0
    rng = random.Random(0)
    for name, meta in man["files"].items():
        fmt = meta["format"]
        keysets, bad, n = collections.Counter(), collections.Counter(), 0
        with open(os.path.join(a.out, name), encoding="utf-8") as f:
            for line in f:
                n += 1
                rec = json.loads(line)
                keysets[tuple(rec)] += 1
                why = check_row(fmt, rec)
                if why:
                    bad[why] += 1
                batch, _, rid = rec["id"].rpartition(":")
                rid = int(rid)
                if rid in ids[batch]:
                    bad["id 重复"] += 1
                ids[batch].add(rid)
                for p in rec.get("images", []) + rec.get("videos", []):
                    seen_media += 1
                    if len(reservoir) < a.sample:
                        reservoir.append(p)
                    elif rng.random() < a.sample / seen_media:
                        reservoir[rng.randrange(a.sample)] = p
        status = "ok"
        if n != meta["rows"]:
            problems.append(f"{name}: {n} 行，manifest 记 {meta['rows']}")
            status = "行数不符"
        if len(keysets) != 1:
            problems.append(f"{name}: 字段组合 {len(keysets)} 种 {list(keysets)[:3]}")
            status = "字段不一"
        if bad:
            problems.append(f"{name}: {dict(bad)}")
            status = "有坏行"
        print(f"  {status:6s} {name:62s} {n:>10,} 行  {meta['bytes'] / 1e9:6.2f} GB  字段 {list(keysets)[0] if keysets else ''}",
              flush=True)

    print("\n=== 批次对账 ===")
    for b, t in man["batches"].items():
        written = sum(t["written"].values())
        rejected = sum(t["rejected"].values())
        acc = written + t["holdout"] + t["excluded_media"] + rejected
        ok = t["rows_read"] == t["rows_in_db"] and acc == t["rows_read"] and len(ids[b]) == written
        if not ok:
            problems.append(f"{b}: 库 {t['rows_in_db']} 读 {t['rows_read']} 写+留出+剔除+拒收 {acc} 唯一 id {len(ids[b])}")
        print(f"  {'ok' if ok else 'BAD':3s} {b:28s} 库 {t['rows_in_db']:>10,}  写出 {written:>10,}  留出 {t['holdout']:>7,}"
              f"  剔除 {t['excluded_media']:>4,}  拒收 {rejected:>6,} {dict(t['rejected']) if rejected else ''}")

    missing = [p for p in reservoir if not os.path.exists(p)]
    print(f"\n=== 媒体抽查 === {len(reservoir):,} / {seen_media:,} 个引用，缺失 {len(missing)}")
    if missing:
        problems.append(f"媒体缺失 {len(missing)}/{len(reservoir)}，如 {missing[:3]}")

    if info is not None:  # optional (export_train_data.py --dataset-info)
        registered = {os.path.relpath(v["file_name"], a.out) for v in info.values()}
        if registered != set(man["files"]):
            problems.append(f"dataset_info 与文件不一致：多 {registered - set(man['files'])} 少 {set(man['files']) - registered}")
        print(f"=== dataset_info === {len(info)} 个注册，{'一致' if registered == set(man['files']) else '不一致'}")

    print("\n结果：" + ("全部通过" if not problems else f"{len(problems)} 个问题"))
    for p in problems:
        print("  - " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
