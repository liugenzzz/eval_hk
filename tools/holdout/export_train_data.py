"""Export the field DB as LlamaFactory training files, organised by corpus type and pt / qa / vqa, with
every eval hold-out left out.

What goes out
  corpus_samples         every batch except --exclude (regexes on batch_tag; none by default)
  defect_tetect_samples  defect_part0_v1 / defect_part1_v1
Left out
  rows with is_holdout=1 (the eval sets registered in eval_holdout), and rows using any media path listed
  in --exclude-media (one absolute path per line; a safety net for eval copies not flagged in the DB).

Layout (--out)
  <corpus_type>/pt/<corpus_type>.pt.jsonl    {"id", "text"}: all pretraining text of that corpus type in one
                                             file (data_format pt, plus the question-less pipeline_export
                                             rows of aircraft_book_v1)
  <corpus_type>/qa/<source>.<format>.jsonl   text-only QA                 format: sharegpt | openai | alpaca
  <corpus_type>/vqa/<source>.<format>.jsonl  QA with images or videos
  (<source> is the batch_tag, or its readable name from SOURCE_NAMES, e.g. aircraft_vqa_v1 -> aircraft_defect)
  (a .image / .video / .mixed suffix appears only if one batch+format holds both images and videos)
  dataset_info.json                          only with --dataset-info: LlamaFactory registration of every file
  manifest.json                              per file: rows, bytes, format, modality, batches; per batch: rows in the
                                             DB, written per file, skipped per reason
  _rejects/<batch>.jsonl                     {"id", "data_format", "task_type", "reason"} of every row not written
qa / vqa is decided per row by whether it carries media, not by task_kind_map, so that every file has one
column set (datasets raises CastError otherwise) and vqa never contains a row without its image.

Every training row starts with "id": "<batch>:<db id>", the key the eval sets use, so a later leak
check is an id lookup. Media paths are the absolute new_path from corpus_media / defect_tetect_media in
media_idx order. Rows are checked with the rules LlamaFactory enforces (as in the 9-30 cleaning,
clean_invalid_rows.py). When a row has media but no placeholder at all, one <image>/<video> per item is
prepended to the first user turn (the 9-30 repair did this for 5,392 alpaca rows); any other
placeholder/media mismatch is rejected. A row with both images and videos goes to its own .mixed file
(ZeRO-3 deadlocks when one micro-batch mixes them).

  python export_train_data.py --out DIR --bench              # time one chunk of every batch, then estimate
  python export_train_data.py --out DIR --workers 4          # full export
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time

import pymysql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fielddb import connect  # noqa: E402

CHUNK = 20000
# File names use these readable source names; ids and the manifest keep the DB batch_tag, so every row
# still maps back to the DB.
SOURCE_NAMES = {
    "video_v1": "video_bilibili",                         # B站视频
    "video_abnormal_behavior_v1": "video_park_abnormal",  # 园区异常行为
    "aircraft_vqa_v1": "aircraft_defect",                 # 飞机缺陷问答
}
SHAREGPT_TAGS = {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                 "assistant_tag": "gpt", "system_tag": "system"}
OPENAI_TAGS = {"role_tag": "role", "content_tag": "content", "user_tag": "user",
               "assistant_tag": "assistant", "system_tag": "system"}
TABLES = {
    # pipeline_export rows (aircraft_book_v1) are ~10x larger than what is used: fetch only those fields
    "corpus_samples": ("""SELECT id, data_format, task_type, is_holdout, media_count,
                                 IF(is_holdout = 1 OR data_format = 'pt', NULL,
                                    IF(data_format = 'pipeline_export',
                                       JSON_OBJECT('images', content->'$.images', 'question', content->'$.question',
                                                   'answer', content->'$.answer', 'text', content->'$.text'),
                                       content)),
                                 IF(is_holdout = 0 AND data_format = 'pt', content->>'$.text', NULL),
                                 IF(is_holdout = 0 AND data_format = 'pipeline_export', train_conversations, NULL)
                            FROM corpus_samples WHERE batch_tag = %s AND id BETWEEN %s AND %s ORDER BY id""",
                       "corpus_media"),
    "defect_tetect_samples": ("""SELECT id, data_format, NULL, is_holdout, NULL, IF(is_holdout = 1, NULL, content),
                                        NULL, NULL
                                   FROM defect_tetect_samples WHERE batch_tag = %s AND id BETWEEN %s AND %s
                                  ORDER BY id""", "defect_tetect_media"),
}

G = {}  # per-worker state: db connection, args


# ---------------------------------------------------------------- row checks (LlamaFactory rules)

def check_sharegpt(conv):
    if not isinstance(conv, list) or not conv:
        return "conversations 缺失或为空"
    msgs = conv[1:] if isinstance(conv[0], dict) and conv[0].get("from") == "system" else conv
    if len(msgs) < 2:
        return "去掉 system 后不足 2 轮"
    if len(msgs) % 2:
        return "轮数为奇数"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            return "轮次不是对象"
        if m.get("from") != ("human" if i % 2 == 0 else "gpt"):
            return "角色顺序不是 human/gpt 交替"
        if not str(m.get("value") or "").strip():
            return "有空轮次"
    return None


def check_openai(msgs):
    if not isinstance(msgs, list) or len(msgs) < 2:
        return "messages 缺失或不足 2 条"
    body = [m for m in msgs if isinstance(m, dict) and m.get("role") != "system"]
    if len(body) % 2:
        return "去掉 system 后轮数为奇数"
    for i, m in enumerate(body):
        if m.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return "角色顺序不是 user/assistant 交替"
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return "有空轮次"
    return None


def fix_placeholders(texts, first_user_idx, images, videos):
    """texts: list of strings (all turns). Returns (texts, repaired, reason)."""
    joined = "".join(texts)
    repaired = False
    for tok, items in (("<image>", images), ("<video>", videos)):
        n = joined.count(tok)
        if n == len(items):
            continue
        if n == 0 and items:
            texts[first_user_idx] = tok * len(items) + texts[first_user_idx]
            joined = "".join(texts)
            repaired = True
            continue
        return texts, repaired, f"{tok} 占位符数与媒体数不符"
    return texts, repaired, None


def declared_media(c, images, videos):
    """The sample's own image/video list must not be longer than its media rows."""
    for key, have in (("images", images), ("videos", videos)):
        want = c.get(key)
        if isinstance(want, str):
            want = [want]
        if isinstance(want, list) and len(want) > len(have):
            raise ValueError(f"媒体行缺失（{key} {len(have)}/{len(want)}）")


def convert(fmt, content, text, tconv, images, videos):
    """-> (out_format, record, repaired) or raises ValueError(reason)."""
    if fmt == "pt":
        if not text or not text.strip():
            raise ValueError("pt 文本为空")
        if images or videos:
            raise ValueError("pt 行带媒体")
        return "pt", {"text": text}, False
    c = json.loads(content)
    declared_media(c, images, videos)
    if fmt in ("sharegpt", "pipeline_export"):
        if fmt == "pipeline_export":
            if tconv is None:
                # question "" and the body in answer/text: pretraining text, like domain_knowledge_corpus
                # elsewhere (fix_empty_conversations.py cleared their train_conversations on purpose)
                body = c.get("text") or c.get("answer")
                if not str(c.get("question") or "").strip() and isinstance(body, str) and body.strip():
                    if images or videos:
                        raise ValueError("无问题的语料行带媒体")
                    return "pt", {"text": body}, False
                raise ValueError("pipeline_export 无 train_conversations")
            conv = json.loads(tconv)
        else:
            conv = c.get("conversations")
        why = check_sharegpt(conv)
        if why:
            raise ValueError(why)
        conv = [dict(m) for m in conv]
        first = next(i for i, m in enumerate(conv) if m["from"] == "human")
        texts, repaired, why = fix_placeholders([m["value"] for m in conv], first, images, videos)
        if why:
            raise ValueError(why)
        for m, t in zip(conv, texts):
            m["value"] = t
        return "sharegpt", {"conversations": conv}, repaired
    if fmt == "openai_messages":
        msgs = c.get("messages")
        why = check_openai(msgs)
        if why:
            raise ValueError(why)
        msgs = [dict(m) for m in msgs]
        first = next(i for i, m in enumerate(msgs) if m["role"] == "user")
        texts, repaired, why = fix_placeholders([m["content"] for m in msgs], first, images, videos)
        if why:
            raise ValueError(why)
        for m, t in zip(msgs, texts):
            m["content"] = t
        return "openai", {"messages": msgs}, repaired
    if fmt == "alpaca":
        ins, inp, out = (c.get("instruction") or ""), (c.get("input") or ""), (c.get("output") or "")
        if not str(ins).strip():
            raise ValueError("instruction 为空")
        if not str(out).strip():
            raise ValueError("output 为空")
        texts, repaired, why = fix_placeholders([ins, inp, out], 0, images, videos)
        if why:
            raise ValueError(why)
        return "alpaca", {"instruction": texts[0], "input": texts[1], "output": texts[2]}, repaired
    raise ValueError(f"未知 data_format {fmt}")


# ---------------------------------------------------------------- worker

def init_worker(args):
    G["args"] = args
    G["db"] = connect(read_timeout=1800, write_timeout=1800, autocommit=True)
    cur = G["db"].cursor()
    cur.execute("SET SESSION net_write_timeout = 1800, net_read_timeout = 1800")
    G["exclude_media"] = set()
    if args["exclude_media"]:
        with open(args["exclude_media"], encoding="utf-8") as f:
            G["exclude_media"] = {line.strip() for line in f if line.strip()}


def run_unit(unit):
    seq, table, batch, ctype, lo, hi = unit
    args = G["args"]
    t0 = time.time()
    db = G["db"]
    row_sql, media_table = TABLES[table]
    cur = db.cursor()
    cur.execute(f"""SELECT sample_id, media_type, new_path FROM {media_table}
                     WHERE sample_id BETWEEN %s AND %s ORDER BY sample_id, media_idx""", (lo, hi))
    media = collections.defaultdict(list)
    for sid, mtype, path in cur.fetchall():
        media[sid].append((mtype, path))
    stats = {"rows": 0, "holdout": 0, "excluded_media": 0, "repaired": 0, "bytes": 0,
             "written": collections.Counter(), "rejected": collections.Counter(), "media_short": 0}
    outs, rej = {}, None
    part_dir = os.path.join(args["out"], "_parts")
    ss = db.cursor(pymysql.cursors.SSCursor)
    ss.execute(row_sql, (batch, lo, hi))
    for rid, fmt, task, hold, media_count, content, text, tconv in ss:
        stats["rows"] += 1
        key = f"{batch}:{rid}"
        if hold:
            stats["holdout"] += 1
            continue
        items = media.get(rid, [])
        images = [p for t, p in items if t == "image"]
        videos = [p for t, p in items if t == "video"]
        reason = None
        if G["exclude_media"] and any(p in G["exclude_media"] for _, p in items):
            stats["excluded_media"] += 1
            continue
        if media_count is not None and len(items) < media_count:
            reason = f"媒体行缺失（{len(items)}/{media_count}）"
        else:
            try:
                out_fmt, rec, repaired = convert(fmt, content, text, tconv, images, videos)
            except (ValueError, json.JSONDecodeError, TypeError, AttributeError, StopIteration) as e:
                reason = str(e) or e.__class__.__name__
        if reason:
            stats["rejected"][reason] += 1
            if rej is None:
                rej = open(os.path.join(part_dir, f"{seq:06d}.rejects.{batch}.jsonl"), "w", encoding="utf-8")
            rej.write(json.dumps({"id": key, "data_format": fmt, "task_type": task, "reason": reason},
                                 ensure_ascii=False) + "\n")
            continue
        if repaired:
            stats["repaired"] += 1
        if out_fmt == "pt":
            name = f"{ctype}/pt/{ctype}.pt.jsonl"          # one pt file per corpus type
            line = {"id": key, **rec}
        else:
            modality = "mixed" if images and videos else "image" if images else "video" if videos else "text"
            kind = "qa" if modality == "text" else "vqa"
            name = f"{ctype}/{kind}/{SOURCE_NAMES.get(batch, batch)}.{out_fmt}.{modality}.jsonl"  # suffix dropped at merge if unique
            line = {"id": key, **rec}
            if images:
                line["images"] = images
            if videos:
                line["videos"] = videos
        s = json.dumps(line, ensure_ascii=False) + "\n"
        f = outs.get(name)
        if f is None:
            f = outs[name] = open(os.path.join(part_dir, f"{seq:06d}." + name.replace("/", "__")), "w",
                                  encoding="utf-8")
        f.write(s)
        stats["written"][name] += 1
        stats["bytes"] += len(s.encode("utf-8"))
    ss.close()
    for f in outs.values():
        f.close()
    if rej:
        rej.close()
    stats["seconds"] = time.time() - t0
    return seq, table, batch, stats


# ---------------------------------------------------------------- driver

def batches(cur, exclude):
    out = []
    for table in TABLES:
        cur.execute(f"SELECT DISTINCT batch_tag FROM {table}")
        for (b,) in cur.fetchall():
            if any(re.search(p, b) for p in exclude):
                continue
            cur.execute(f"SELECT MIN(id), MAX(id), COUNT(*) FROM {table} WHERE batch_tag = %s", (b,))
            lo, hi, n = cur.fetchone()
            cur.execute(f"SELECT corpus_type FROM {table} WHERE batch_tag = %s LIMIT 1", (b,))
            out.append((table, b, cur.fetchone()[0], lo, hi, n))
    return out


def units_for(plan, bench):
    seq, units = 0, []
    for table, b, ctype, lo, hi, n in plan:
        starts = list(range(lo, hi + 1, CHUNK))
        if bench:  # one chunk from the middle of each batch
            starts = [starts[len(starts) // 2]]
        for s in starts:
            units.append((seq, table, b, ctype, s, min(s + CHUNK - 1, hi)))
            seq += 1
    return units


def describe(name):
    """Full part name -> (corpus_type, kind, batch or None, format, modality)."""
    ctype, kind, base = name.split("/")
    if kind == "pt":
        return ctype, kind, None, "pt", "text"
    batch, fmt, modality, _ = base.split(".")
    return ctype, kind, batch, fmt, modality


def final_names(names):
    """Drop the modality suffix wherever a batch+format+kind has only one modality."""
    groups = collections.defaultdict(set)
    for n in names:
        ctype, kind, batch, fmt, modality = describe(n)
        if kind != "pt":
            groups[(ctype, kind, batch, fmt)].add(modality)
    out = {}
    for n in names:
        ctype, kind, batch, fmt, modality = describe(n)
        out[n] = n if kind == "pt" or len(groups[(ctype, kind, batch, fmt)]) > 1 else f"{ctype}/{kind}/{batch}.{fmt}.jsonl"
    return out


def merge(out, final):
    parts = sorted(os.listdir(os.path.join(out, "_parts")))
    by_name = collections.defaultdict(list)
    for p in parts:
        seq, rest = p.split(".", 1)
        by_name[rest].append(p)
    for rest, files in by_name.items():
        if rest.startswith("rejects."):
            dest = os.path.join(out, "_rejects", rest[len("rejects."):])
        else:
            dest = os.path.join(out, final[rest.replace("__", "/")])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".tmp", "wb") as w:
            for p in sorted(files):
                with open(os.path.join(out, "_parts", p), "rb") as r:
                    shutil.copyfileobj(r, w, 16 << 20)
        os.replace(dest + ".tmp", dest)
    shutil.rmtree(os.path.join(out, "_parts"))


def dataset_info(out, files):
    """files: final name -> manifest entry (corpus_type, kind, format, modality, ...)."""
    info = {}
    for name in sorted(files):
        meta = files[name]
        path = os.path.join(out, name)
        if meta["kind"] == "pt":
            info[f"field_{meta['corpus_type']}_pt"] = {"file_name": path, "columns": {"prompt": "text"}}
            continue
        fmt, modality = meta["format"], meta["modality"]
        entry = {"file_name": path}
        if fmt == "alpaca":
            entry["formatting"] = "alpaca"
            entry["columns"] = {"prompt": "instruction", "query": "input", "response": "output"}
        else:
            entry["formatting"] = "sharegpt"
            entry["columns"] = {"messages": "conversations" if fmt == "sharegpt" else "messages"}
            entry["tags"] = SHAREGPT_TAGS if fmt == "sharegpt" else OPENAI_TAGS
        if modality in ("image", "mixed"):
            entry["columns"]["images"] = "images"
        if modality in ("video", "mixed"):
            entry["columns"]["videos"] = "videos"
        stem = os.path.basename(name)[:-len(".jsonl")].replace(".", "_")
        info[f"field_{meta['corpus_type']}_{meta['kind']}_{stem}"] = entry
    return info


def file_entries(out, totals, final):
    files = {}
    for b, t in totals.items():
        for full, n in t["written"].items():
            ctype, kind, _, fmt, modality = describe(full)
            e = files.setdefault(final[full], {"rows": 0, "corpus_type": ctype, "kind": kind, "format": fmt,
                                               "modality": modality, "batches": []})
            e["rows"] += n
            e["batches"].append(b)
    for name, e in files.items():
        e["bytes"] = os.path.getsize(os.path.join(out, name))
        e["batches"] = sorted(set(e["batches"]))
    return dict(sorted(files.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exclude", default="", help="comma-separated regexes on batch_tag")
    ap.add_argument("--exclude-media", default="", help="file of media paths whose rows are left out")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated batch_tags (testing, or re-export for splice_export.py)")
    ap.add_argument("--dataset-info", action="store_true", help="also write a LlamaFactory dataset_info.json")
    a = ap.parse_args()
    exclude = [p for p in a.exclude.split(",") if p]
    if os.path.exists(os.path.join(a.out, "manifest.json")) and not a.bench:
        sys.exit(f"{a.out} already has a manifest; use a new --out")
    os.makedirs(os.path.join(a.out, "_parts"), exist_ok=True)
    t0 = time.time()
    db = connect(read_timeout=600)
    plan = batches(db.cursor(), exclude)
    db.close()
    if a.only:
        keep = set(a.only.split(","))
        plan = [p for p in plan if p[1] in keep]
    units = units_for(plan, a.bench)
    print(f"{len(plan)} 个批次，{sum(p[5] for p in plan):,} 行，{len(units)} 个分块，{a.workers} 个进程  "
          f"(列批次 {time.time() - t0:.0f}s)", flush=True)
    for table, b, ctype, lo, hi, n in plan:
        print(f"  {table:22s} {ctype:10s} {b:28s} {n:>10,} 行  id {lo}-{hi}")

    totals = {b: {"table": t, "corpus_type": ctype, "rows_in_db": n, "rows_read": 0, "holdout": 0,
                  "excluded_media": 0, "repaired_placeholders": 0, "written": collections.Counter(),
                  "rejected": collections.Counter(), "seconds": 0.0, "bytes": 0} for t, b, ctype, lo, hi, n in plan}
    done_rows = 0
    total_rows = sum(p[5] for p in plan)
    args = {"out": a.out, "exclude_media": a.exclude_media}
    with mp.Pool(a.workers, initializer=init_worker, initargs=(args,)) as pool:
        for i, (seq, table, batch, st) in enumerate(pool.imap_unordered(run_unit, units), 1):
            t = totals[batch]
            t["rows_read"] += st["rows"]
            t["holdout"] += st["holdout"]
            t["excluded_media"] += st["excluded_media"]
            t["repaired_placeholders"] += st["repaired"]
            t["written"].update(st["written"])
            t["rejected"].update(st["rejected"])
            t["seconds"] += st["seconds"]
            t["bytes"] += st["bytes"]
            done_rows += st["rows"]
            if a.bench or i % 20 == 0 or i == len(units):
                el = time.time() - t0
                rate = done_rows / el if el else 0
                eta = (total_rows - done_rows) / rate / 60 if rate and not a.bench else 0
                print(f"[{i}/{len(units)}] {batch} {st['rows']:,} 行 {st['seconds']:.1f}s "
                      f"({st['rows'] / max(st['seconds'], 1e-6):,.0f} 行/s, {st['bytes'] / 1e6 / max(st['seconds'], 1e-6):.1f} MB/s)"
                      f" | 累计 {done_rows:,} 行 {el / 60:.1f}m" + (f" 预计剩余 {eta:.0f}m" if eta else ""),
                      flush=True)

    if a.bench:
        print("\n=== 估算（按每个批次一个分块的实测速度，单进程；N 个进程并行时约除以 N，以库的读速为上限）===")
        est = 0
        for b, t in totals.items():
            if not t["rows_read"]:
                continue
            per_row = t["seconds"] / t["rows_read"]
            est += per_row * t["rows_in_db"]
            print(f"  {b:28s} {t['rows_in_db']:>10,} 行 × {per_row * 1000:.2f} ms = {per_row * t['rows_in_db'] / 60:6.1f} 进程·分钟"
                  f"   样本 {t['rows_read']:,} 行 → 写出 {sum(t['written'].values()):,}，留出 {t['holdout']:,}，"
                  f"剔除 {t['excluded_media']:,}，拒收 {dict(t['rejected'])}，补占位符 {t['repaired_placeholders']:,}")
        print(f"  合计 {est / 60:.1f} 进程·分钟；{a.workers} 进程约 {est / 60 / a.workers:.0f} 分钟（未计合并 / 校验）")
        shutil.rmtree(os.path.join(a.out, "_parts"), ignore_errors=True)
        return 0

    t_merge = time.time()
    names = set()
    for t in totals.values():
        names.update(t["written"])
    final = final_names(names)
    merge(a.out, final)
    files = file_entries(a.out, totals, final)
    if a.dataset_info:
        with open(os.path.join(a.out, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info(a.out, files), f, ensure_ascii=False, indent=2)
    written = {b: collections.Counter() for b in totals}
    for b, t in totals.items():
        for full, n in t["written"].items():
            written[b][final[full]] += n
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exclude_batches": exclude, "exclude_media_file": a.exclude_media,
        "elapsed_minutes": {"export": round((t_merge - t0) / 60, 1), "merge": round((time.time() - t_merge) / 60, 1)},
        "batches": {b: {**{k: v for k, v in t.items() if k not in ("written", "rejected")},
                        "written": dict(written[b]), "rejected": dict(t["rejected"])} for b, t in totals.items()},
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n完成：{len(files)} 个文件，{sum(v['rows'] for v in files.values()):,} 行，"
          f"{sum(v['bytes'] for v in files.values()) / 1e9:.1f} GB；导出 {(t_merge - t0) / 60:.1f}m，合并 {(time.time() - t_merge) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
