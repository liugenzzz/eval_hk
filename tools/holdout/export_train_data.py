"""Export the field DB as training files, one file per task type, with every eval hold-out left out.

Layout (--out)
  book/<领域>/<pt|qa|vqa>/<task_type>.jsonl        领域 = domain_name of the 七大类 (domain_code 01-07)
  book/其他/<标签>/<pt|qa|vqa>/<task_type>.jsonl   books without a 七大类 label: the domain_code when it is a real
                                                    one (304), else the source (航空书籍 / 军事武器 / GI82书籍), and
                                                    未分类 for CPT chunks the classifier left unlabelled
  <journal|video|detection>/<pt|qa|vqa>/<task_type>.jsonl
  industrial/<飞机缺陷|工业质检>/vqa/<task_type>.jsonl
  exceptions (user, 2026-10-09): park video's 24 LLaVA-Video subsets merge into park_caption /
  park_multiple_choice_qa / park_open_ended_qa; detection goes one file per scene category (domain_name);
  defect_tetect has no task type: industrial/工业质检/vqa/defect_cn.jsonl / defect_en.jsonl (language column)
  manifest.json                                    per file: rows, bytes, kind, task, batches; per batch: rows in the DB,
                                                    written, held out, excluded, rejected
  _rejects/<batch>.jsonl                           {"id", "data_format", "task_type", "reason"} of every row not written
  dataset_info.json                                only with --dataset-info
Rows of one task type are merged across sources, so everything except pt is converted to sharegpt:
  pt   {"id", "text"}
  else {"id", "conversations", "images" | "videos"}   openai roles user/assistant -> human/gpt; alpaca
       instruction (+ "\n" + input) -> human, output -> gpt; pipeline_export via train_conversations
pt / qa / vqa follows task_kind_map; a row carrying media always goes to vqa. Every vqa row carries its media key
("videos" under video/, "images" elsewhere; an empty list when the row has none), so a file has one column set.
A row with both images and videos, or a media type foreign to its corpus type, gets a .mixed / .image / .video
suffix (ZeRO-3 deadlocks when one micro-batch mixes images and videos).

Every row starts with "id": "<batch_tag>:<db id>" (the key the eval sets use). Media paths are the absolute
new_path from corpus_media / defect_tetect_media in media_idx order. Rows are checked with the rules LlamaFactory
enforces (clean_invalid_rows.py); a row with media but no placeholder at all gets one <image>/<video> per item
prepended to the first user turn; any other placeholder/media mismatch is rejected.

  python export_train_data.py --out DIR --bench              # time one chunk of every batch, then estimate
  python export_train_data.py --out DIR --workers 8          # full export
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
# Readable names of sources, used in docs and summaries; row ids and the manifest keep the DB batch_tag.
SOURCE_NAMES = {
    "video_v1": "video_bilibili",                         # B站视频
    "video_abnormal_behavior_v1": "video_park_abnormal",  # 园区异常行为
    "aircraft_vqa_v1": "aircraft_defect",                 # 飞机缺陷问答
}
SEVEN = {"01", "02", "03", "04", "05", "06", "07"}         # 七大类 domain codes (names from domain_name)
BOOK_OTHER = {"aircraft_book_v1": "航空书籍", "weapon_book_v1": "军事武器", "book_gi82_v1": "GI82书籍"}
KINDS = ("pt", "qa", "vqa")
SHAREGPT_TAGS = {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                 "assistant_tag": "gpt", "system_tag": "system"}
ROLE = {"user": "human", "assistant": "gpt", "system": "system"}
TABLES = {
    # pipeline_export rows (aircraft_book_v1) are ~10x larger than what is used: fetch only those fields
    "corpus_samples": ("""SELECT id, data_format, task_type, is_holdout, media_count, domain_code, domain_name,
                                 IF(is_holdout = 1 OR data_format = 'pt', NULL,
                                    IF(data_format = 'pipeline_export',
                                       JSON_OBJECT('images', content->'$.images', 'question', content->'$.question',
                                                   'answer', content->'$.answer', 'text', content->'$.text'),
                                       content)),
                                 IF(is_holdout = 0 AND data_format = 'pt', content->>'$.text', NULL),
                                 IF(is_holdout = 0 AND data_format = 'pipeline_export', train_conversations, NULL)
                            FROM corpus_samples WHERE batch_tag = %s AND id BETWEEN %s AND %s ORDER BY id""",
                       "corpus_media"),
    # defect_tetect has no task type: its language (generated column cn / en) names the file, part0 and part1 merged
    "defect_tetect_samples": ("""SELECT id, data_format, CONCAT('defect_', language), is_holdout, NULL, domain_code, domain_name,
                                        IF(is_holdout = 1, NULL, content), NULL, NULL
                                   FROM defect_tetect_samples WHERE batch_tag = %s AND id BETWEEN %s AND %s
                                  ORDER BY id""", "defect_tetect_media"),
}

G = {}  # per-worker state: db connection, args, task kinds


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


def to_sharegpt(fmt, c, tconv):
    """Conversation turns of a non-pt row as sharegpt [{from, value}]."""
    if fmt == "sharegpt":
        return c.get("conversations")
    if fmt == "pipeline_export":
        return json.loads(tconv)
    if fmt == "openai_messages":
        msgs = c.get("messages")
        why = check_openai(msgs)
        if why:
            raise ValueError(why)
        return [{"from": ROLE.get(m["role"], m["role"]), "value": m["content"]} for m in msgs]
    if fmt == "alpaca":
        ins, inp, out = (c.get("instruction") or ""), (c.get("input") or ""), (c.get("output") or "")
        if not str(ins).strip():
            raise ValueError("instruction 为空")
        if not str(out).strip():
            raise ValueError("output 为空")
        return [{"from": "human", "value": ins + ("\n" + inp if str(inp).strip() else "")},
                {"from": "gpt", "value": out}]
    raise ValueError(f"未知 data_format {fmt}")


def convert(fmt, content, text, tconv, images, videos):
    """-> ("pt" | "sft", record, repaired) or raises ValueError(reason)."""
    if fmt == "pt":
        if not text or not text.strip():
            raise ValueError("pt 文本为空")
        if images or videos:
            raise ValueError("pt 行带媒体")
        return "pt", {"text": text}, False
    c = json.loads(content)
    declared_media(c, images, videos)
    if fmt == "pipeline_export" and tconv is None:
        # question "" and the body in answer/text: pretraining text, like domain_knowledge_corpus elsewhere
        # (fix_empty_conversations.py cleared their train_conversations on purpose)
        body = c.get("text") or c.get("answer")
        if not str(c.get("question") or "").strip() and isinstance(body, str) and body.strip():
            if images or videos:
                raise ValueError("无问题的语料行带媒体")
            return "pt", {"text": body}, False
        raise ValueError("pipeline_export 无 train_conversations")
    conv = to_sharegpt(fmt, c, tconv)
    why = check_sharegpt(conv)
    if why:
        raise ValueError(why)
    conv = [{"from": m["from"], "value": m["value"]} for m in conv]
    first = next(i for i, m in enumerate(conv) if m["from"] == "human")
    texts, repaired, why = fix_placeholders([m["value"] for m in conv], first, images, videos)
    if why:
        raise ValueError(why)
    for m, t in zip(conv, texts):
        m["value"] = t
    return "sft", {"conversations": conv}, repaired


PARK_KINDS = ("caption", "multiple_choice_qa", "open_ended_qa")


def task_file(ctype, batch, task, dname):
    """File name (without .jsonl) a row goes to; the task type unless the user asked otherwise (2026-10-09)."""
    if batch == "video_abnormal_behavior_v1":         # 24 LLaVA-Video subsets (duration x source x kind) -> 3 kinds
        for k in PARK_KINDS:
            if task.endswith(k):
                return f"park_{k}"
    if ctype == "detection" and dname:                 # by scene category: 通用目标 / GQA场景 / 航拍 / 遥感
        return dname
    return task or batch


def location(ctype, batch, dcode, dname):
    """Directory under the export root before <pt|qa|vqa>: book by 七大类 domain, industrial by
    飞机缺陷 / 工业质检, the rest flat."""
    if ctype == "industrial" and dname:
        return f"industrial/{dname}"
    if ctype != "book":
        return ctype
    if dcode in SEVEN and dname:
        return f"book/{dname}"
    if dcode and dcode != "unknown":
        return f"book/其他/{dcode}"
    return f"book/其他/{BOOK_OTHER.get(batch, '未分类')}"


# ---------------------------------------------------------------- worker

def init_worker(args):
    G["args"] = args
    G["db"] = connect(read_timeout=1800, write_timeout=1800, autocommit=True)
    cur = G["db"].cursor()
    cur.execute("SET SESSION net_write_timeout = 1800, net_read_timeout = 1800")
    cur.execute("SELECT task_type, task_kind FROM task_kind_map")
    G["kind"] = dict(cur.fetchall())
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
             "written": collections.Counter(), "rejected": collections.Counter()}
    outs, rej = {}, None
    part_dir = os.path.join(args["out"], "_parts")
    default_media = "video" if ctype == "video" else "image"
    ss = db.cursor(pymysql.cursors.SSCursor)
    ss.execute(row_sql, (batch, lo, hi))
    for rid, fmt, task, hold, media_count, dcode, dname, content, text, tconv in ss:
        stats["rows"] += 1
        key = f"{batch}:{rid}"
        if hold:
            stats["holdout"] += 1
            continue
        items = media.get(rid, [])
        images = [p for t, p in items if t == "image"]
        videos = [p for t, p in items if t == "video"]
        if G["exclude_media"] and any(p in G["exclude_media"] for _, p in items):
            stats["excluded_media"] += 1
            continue
        reason = None
        if media_count is not None and len(items) < media_count:
            reason = f"媒体行缺失（{len(items)}/{media_count}）"
        else:
            try:
                out, rec, repaired = convert(fmt, content, text, tconv, images, videos)
            except (ValueError, json.JSONDecodeError, TypeError, AttributeError, KeyError, StopIteration) as e:
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
        loc = location(ctype, batch, dcode, dname)
        task_name = task_file(ctype, batch, task, dname)
        line = {"id": key, **rec}
        if out == "pt":
            name = f"{loc}/pt/{task_name}.jsonl"
        elif not images and not videos and G["kind"].get(task) != "vqa":
            name = f"{loc}/qa/{task_name}.jsonl"
        else:
            mtype = "mixed" if images and videos else "video" if videos else "image" if images else default_media
            name = f"{loc}/vqa/{task_name}{'' if mtype == default_media else '.' + mtype}.jsonl"
            if mtype in ("image", "mixed"):
                line["images"] = images
            if mtype in ("video", "mixed"):
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
    """Relative file name -> manifest fields."""
    parts = name.split("/")
    kind = parts[-2]
    base = parts[-1][:-len(".jsonl")]
    task, _, suffix = base.partition(".")
    ctype = parts[0]
    domain = "/".join(parts[1:-2])
    if kind == "pt":
        modality = "text"
    elif kind == "qa":
        modality = "text"
    else:
        modality = suffix or ("video" if ctype == "video" else "image")
    return {"corpus_type": ctype, "domain": domain, "kind": kind, "task_type": task,
            "format": "pt" if kind == "pt" else "sharegpt", "modality": modality}


def merge(out):
    parts = sorted(os.listdir(os.path.join(out, "_parts")))
    by_name = collections.defaultdict(list)
    for p in parts:
        seq, rest = p.split(".", 1)
        by_name[rest].append(p)
    for rest, files in by_name.items():
        if rest.startswith("rejects."):
            dest = os.path.join(out, "_rejects", rest[len("rejects."):])
        else:
            dest = os.path.join(out, rest.replace("__", "/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".tmp", "wb") as w:
            for p in sorted(files):
                with open(os.path.join(out, "_parts", p), "rb") as r:
                    shutil.copyfileobj(r, w, 16 << 20)
        os.replace(dest + ".tmp", dest)
    shutil.rmtree(os.path.join(out, "_parts"))


def file_entries(out, totals):
    files = {}
    for b, t in totals.items():
        for name, n in t["written"].items():
            e = files.setdefault(name, {"rows": 0, **describe(name), "batches": []})
            e["rows"] += n
            e["batches"].append(b)
    for name, e in files.items():
        e["bytes"] = os.path.getsize(os.path.join(out, name))
        e["batches"] = sorted(set(e["batches"]))
    return dict(sorted(files.items()))


def dataset_info(out, files):
    """Optional LlamaFactory registration (--dataset-info). files: name -> manifest entry."""
    info = {}
    for name in sorted(files):
        meta = files[name]
        path = os.path.join(out, name)
        stem = name[:-len(".jsonl")].replace("/", "_").replace(".", "_")
        if meta["kind"] == "pt":
            info[f"field_{stem}"] = {"file_name": path, "columns": {"prompt": "text"}}
            continue
        entry = {"file_name": path, "formatting": "sharegpt", "columns": {"messages": "conversations"},
                 "tags": SHAREGPT_TAGS}
        if meta["modality"] in ("image", "mixed"):
            entry["columns"]["images"] = "images"
        if meta["modality"] in ("video", "mixed"):
            entry["columns"]["videos"] = "videos"
        info[f"field_{stem}"] = entry
    return info


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
    merge(a.out)
    files = file_entries(a.out, totals)
    if a.dataset_info:
        with open(os.path.join(a.out, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_info(a.out, files), f, ensure_ascii=False, indent=2)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layout": "book/<领域>/<kind>/<task>.jsonl; <type>/<kind>/<task>.jsonl",
        "exclude_batches": exclude, "exclude_media_file": a.exclude_media,
        "elapsed_minutes": {"export": round((t_merge - t0) / 60, 1), "merge": round((time.time() - t_merge) / 60, 1)},
        "batches": {b: {**{k: v for k, v in t.items() if k not in ("written", "rejected")},
                        "written": dict(t["written"]), "rejected": dict(t["rejected"])} for b, t in totals.items()},
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n完成：{len(files)} 个文件，{sum(v['rows'] for v in files.values()):,} 行，"
          f"{sum(v['bytes'] for v in files.values()) / 1e9:.1f} GB；导出 {(t_merge - t0) / 60:.1f}m，合并 {(time.time() - t_merge) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
