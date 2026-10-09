"""Build a journal eval set from one journal batch: whole issues held out, items reported per task type.

Same unit as journal_eval_v1 (journal_eval_build.py): in the journal batches source_name is one issue, and
every row of a chosen issue is held out. Differences, for a magazine that spans decades (GI82 航空知识
1958-2025):
  - issues are sampled one per stratum of the year order, so the set covers the whole time range
  - an issue qualifies if it has every *common* evaluable task type (present in at least half the issues);
    requiring every type would leave almost nothing when one type is rare (85 rows in GI82)
  - no size cap by default (--size-cap 0.75 reproduces journal_eval_v1's "drop the largest quarter")
Items: vqa / qa rows (task_kind_map), at most --per-task per task type, converted to sharegpt
conversations with one <image> per image, LaTeX escapes that JSON turned into control characters put back.

Writes the eval file plus <out>.issues.txt; registering and flagging the issues is
register_eval_sets.py --books <set> <batch> <out>.issues.txt. Reads the DB only.

  python build_journal_eval.py --batch journal_gi82_v1 --set journal_gi82_eval_v1 \
      --out /mnt/si003010kcx0/mmdata/eval_runs/journal_gi82/data/journal_gi82_vqa.jsonl
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fielddb import connect  # noqa: E402

YEAR = re.compile(r"(19[5-9]\d|20[0-4]\d)")


def year_of(issue):
    m = YEAR.search(issue)
    return int(m.group(1)) if m else 9999


def fix_text(s):
    # "\beta" / "\frac" written unescaped into JSON decode to backspace / form feed
    return s.replace("\x08", "\\b").replace("\x0c", "\\f") if isinstance(s, str) else s


def conversation(fmt, content, images):
    c = json.loads(content)
    if fmt == "openai_messages":
        conv = []
        for m in c.get("messages") or []:
            if m.get("role") not in ("user", "assistant"):
                continue
            text = m.get("content")
            if isinstance(text, list):  # content parts
                text = "".join("<image>" if p.get("type") in ("image", "image_url") else p.get("text", "")
                               for p in text if isinstance(p, dict))
            conv.append({"from": "human" if m["role"] == "user" else "gpt", "value": fix_text(text or "")})
    elif fmt == "alpaca":
        human = c.get("instruction") or ""
        if c.get("input"):
            human += "\n" + c["input"]
        conv = [{"from": "human", "value": fix_text(human)}, {"from": "gpt", "value": fix_text(c.get("output") or "")}]
    else:
        return None
    if len(conv) < 2 or any(not str(t["value"]).strip() for t in conv):
        return None
    n = sum(t["value"].count("<image>") for t in conv)
    if n == 0 and images:
        conv[0]["value"] = "<image>" * len(images) + conv[0]["value"]
    elif n != len(images):
        return None
    return conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--issues", type=int, default=20)
    ap.add_argument("--per-task", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size-cap", type=float, default=1.0,
                    help="keep issues up to this size quantile (journal_eval_v1 used 0.75; GI82's recent issues "
                         "are the largest, so 0.75 left 2019-2025 out)")
    a = ap.parse_args()
    db = connect(read_timeout=1800)
    cur = db.cursor()
    cur.execute("""SELECT s.source_name, COALESCE(k.task_kind, 'vqa'), s.task_type, COUNT(*)
                     FROM corpus_samples s LEFT JOIN task_kind_map k ON k.task_type = s.task_type
                    WHERE s.batch_tag = %s GROUP BY 1, 2, 3""", (a.batch,))
    per_issue = collections.defaultdict(collections.Counter)
    all_rows = collections.Counter()
    for issue, kind, task, n in cur.fetchall():
        all_rows[issue] += n
        if kind in ("vqa", "qa"):
            per_issue[issue][task] += n
    presence = collections.Counter(t for c in per_issue.values() for t in c)
    common = sorted(t for t, n in presence.items() if n >= len(per_issue) / 2)
    ok = [i for i, c in per_issue.items() if all(t in c for t in common)]
    sizes = sorted(sum(per_issue[i].values()) for i in ok)
    cap = sizes[min(len(sizes) - 1, int(len(sizes) * a.size_cap))] if sizes else 0
    ok = sorted((i for i in ok if sum(per_issue[i].values()) <= cap), key=lambda i: (year_of(i), i))
    decades = collections.Counter(f"{year_of(i) // 10 * 10}s" for i in ok)
    print("合格期按年代：", dict(sorted(decades.items())), flush=True)
    rng = random.Random(a.seed)
    k = min(a.issues, len(ok))
    picked = sorted(rng.choice(ok[s * len(ok) // k:(s + 1) * len(ok) // k]) for s in range(k))
    print(f"{len(per_issue)} 期；常见可评估任务 {len(common)} 种 {common}；合格 {len(ok)} 期；"
          f"按年代分 {k} 段各抽 1 期，年份 {[year_of(i) for i in picked]}；留出 {sum(all_rows[i] for i in picked):,} 行",
          flush=True)

    holes = ",".join(["%s"] * len(picked))
    cur.execute(f"""SELECT s.id, s.source_name, s.task_type, COALESCE(k.task_kind, 'vqa'), s.data_format, s.content
                      FROM corpus_samples s LEFT JOIN task_kind_map k ON k.task_type = s.task_type
                     WHERE s.batch_tag = %s AND s.source_name IN ({holes})""", [a.batch, *picked])
    by_task = collections.defaultdict(list)
    for row in cur.fetchall():
        if row[3] in ("vqa", "qa"):
            by_task[row[2]].append(row)
    chosen = []
    for task in sorted(by_task):
        pool = sorted(by_task[task])
        rng.shuffle(pool)
        chosen.extend(pool[:a.per_task])
    media = collections.defaultdict(list)
    ids = [r[0] for r in chosen]
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        cur.execute(f"""SELECT sample_id, new_path FROM corpus_media WHERE sample_id IN ({",".join(["%s"] * len(part))})
                         ORDER BY sample_id, media_idx""", part)
        for sid, path in cur.fetchall():
            media[sid].append(path)
    db.close()
    records, skipped = [], collections.Counter()
    for sid, issue, task, kind, fmt, content in chosen:
        images = media.get(sid, [])
        conv = conversation(fmt, content, images)
        if conv is None:
            skipped["无法转成一问一答 / 占位符与图片数不符"] += 1
            continue
        if any(not os.path.exists(p) for p in images):
            skipped["图片缺失"] += 1
            continue
        records.append({"id": f"{a.batch}:{sid}", "images": images, "conversations": conv, "category": task,
                        "metadata": {"batch_tag": a.batch, "issue": issue, "task_type": task, "task_kind": kind,
                                     "corpus_id": sid, "eval_set": a.set}})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out + ".issues.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(picked) + "\n")
    print(f"写出 {a.out}: {len(records)} 条，按任务 {dict(collections.Counter(r['category'] for r in records))}")
    print(f"  有图 {sum(1 for r in records if r['images'])} 条；跳过 {dict(skipped)}；期名清单 {a.out}.issues.txt")


if __name__ == "__main__":
    main()
