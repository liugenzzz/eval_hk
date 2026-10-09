"""Register the eval sets that were never in field.eval_holdout, flag their training copies,
and re-check the flags of the sets that already are.

  industrial_eval_v1  eval_runs/industrial/data/industrial_eval.jsonl. Drawn from the
                      defect_tetect training data by design (selection_source=training_regression),
                      so every image normally has training copies.
  mbjc_test_v1        eval_runs/det/data/mbjc_test_v1.jsonl, the mbjc test split. Only the train
                      split was ingested (mbjc_v1).

Matching is by md5 of the file content, as in holdout_setup.py (file names repeat across datasets
and renamed copies slip through). Candidates are every image of defect_tetect, aircraft_vqa_v1 and
mbjc_v1. A training image identical to an eval image is registered with key_value = its training
new_path, and every row using it gets is_holdout=1. An eval image with no training copy is
registered by its eval-side path (note says so), so the registry lists every eval set in use.

defect_tetect_samples has no is_holdout column: --apply adds it (ALGORITHM=INSTANT, after the
table's metadata lock is free) plus v_train_defect_tetect_samples, mirroring corpus_samples and
v_train_samples.

  python register_eval_sets.py            # audit, changes nothing
  python register_eval_sets.py --apply

Two more modes (audit unless --apply):
  --copies <copies.tsv>                    find_copies.py output: register every training copy that still has
                                           unflagged rows under its eval set, and flag those rows
  --books <eval_set> <batch> <issues.txt>  whole books / journal issues (source_name) held out, e.g. the
                                           build_journal_eval.py issue list
"""
import ast
import collections
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fielddb import connect  # noqa: E402

R = "/mnt/si003010kcx0/mmdata/eval_runs"
SETS = {
    "industrial_eval_v1": {"file": f"{R}/industrial/data/industrial_eval.jsonl",
                           "pools": ["defect_tetect", "aircraft_vqa_v1"]},
    "mbjc_test_v1": {"file": f"{R}/det/data/mbjc_test_v1.jsonl", "pools": ["mbjc_v1"]},
}
CACHE = os.environ.get("MD5_CACHE", "/tmp/register_eval_sets.md5.json")
FLAG_DDL = ("ALTER TABLE defect_tetect_samples ADD COLUMN is_holdout tinyint NOT NULL DEFAULT 0 "
            "COMMENT '1 = 已被评估集留出，训练导出时排除（来源见 eval_holdout）', ALGORITHM=INSTANT")
VIEW = ("CREATE OR REPLACE VIEW v_train_defect_tetect_samples AS "
        "SELECT * FROM defect_tetect_samples WHERE is_holdout = 0")


def md5(path):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return path, None
    return path, h.hexdigest()


def hash_all(paths, cache):
    todo = [p for p in paths if p not in cache]
    with ThreadPoolExecutor(48) as ex:
        for p, h in ex.map(md5, todo, chunksize=64):
            cache[p] = h
    return {p: cache[p] for p in paths}


def eval_images(path):
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            imgs = rec.get("images") or []
            if isinstance(imgs, str):
                imgs = ast.literal_eval(imgs) if imgs.startswith("[") else [imgs]
            out.update(imgs)
    return out


def pool_paths(cur, pool):
    if pool == "defect_tetect":
        cur.execute("SELECT DISTINCT new_path FROM defect_tetect_media WHERE media_type='image'")
    else:
        cur.execute("""SELECT DISTINCT m.new_path FROM corpus_media m
                         JOIN corpus_samples s ON s.id = m.sample_id
                        WHERE s.batch_tag = %s AND m.media_type = 'image'""", (pool,))
    return [r[0] for r in cur.fetchall()]


def check_existing(cur):
    """Rows whose registered image/video/book is not flagged. Should be 0 for every set."""
    print("\n=== 已登记评估集的标记核对 ===")
    cur.execute("""SELECT h.eval_set, COUNT(DISTINCT s.id),
                          COUNT(DISTINCT CASE WHEN s.is_holdout = 0 THEN s.id END)
                     FROM eval_holdout h
                     JOIN corpus_media m ON m.new_path = h.key_value
                     JOIN corpus_samples s ON s.id = m.sample_id
                    WHERE h.key_type IN ('image', 'video')
                    GROUP BY h.eval_set""")
    bad = 0
    for name, n, unflagged in cur.fetchall():
        print(f"  {name:22s} 用到登记图/视频的训练行 {n:>7,}  未标记 {unflagged:,}")
        bad += unflagged
    cur.execute("""SELECT h.eval_set, COUNT(*), SUM(s.is_holdout = 0)
                     FROM eval_holdout h JOIN corpus_samples s ON s.source_name = h.key_value
                    WHERE h.key_type = 'book' GROUP BY h.eval_set""")
    for name, n, unflagged in cur.fetchall():
        print(f"  {name:22s} 登记书名的训练行     {n:>7,}  未标记 {int(unflagged or 0):,}")
        bad += int(unflagged or 0)
    return bad


def tables_of(batch):
    if batch.startswith("defect_part"):
        return "defect_tetect_media", "defect_tetect_samples"
    return "corpus_media", "corpus_samples"


def register_copies(db, tsv, apply):
    cur = db.cursor()
    rows = [line.rstrip("\n").split("\t") for line in open(tsv, encoding="utf-8") if line.strip()]
    todo = []
    for eval_set, mtype, train_path, batch, eval_path in rows:
        if train_path == eval_path:      # the eval item's own file: its rows are held out with the eval set
            continue
        media_t, samp_t = tables_of(batch)
        cur.execute(f"""SELECT COUNT(DISTINCT s.id) FROM {media_t} m JOIN {samp_t} s ON s.id = m.sample_id
                         WHERE m.new_path = %s AND s.is_holdout = 0""", (train_path,))
        n = cur.fetchone()[0]
        if n:
            todo.append((eval_set, mtype, train_path, batch, ("md5 与评估文件一致: " + eval_path)[:500], n))
    summary = collections.Counter()
    for t in todo:
        summary[(t[0], t[3])] += 1
    print(f"副本 {len(rows):,} 对里仍有未标记训练行的：{len(todo):,} 个训练路径，共 {sum(t[5] for t in todo):,} 行")
    for (s, b), n in sorted(summary.items()):
        print(f"  {s:22s} {b:28s} {n:5d} 个路径")
    if not apply:
        print("(审计模式；加 --apply 登记并标记)")
        return 0
    flagged = 0
    for eval_set, mtype, path, batch, note, _ in todo:
        cur.execute("""INSERT IGNORE INTO eval_holdout (eval_set, key_type, key_value, batch_tag, note)
                       VALUES (%s, %s, %s, %s, %s)""", (eval_set, mtype, path, batch, note))
        media_t, samp_t = tables_of(batch)
        cur.execute(f"""UPDATE {samp_t} s JOIN (SELECT DISTINCT sample_id FROM {media_t} WHERE new_path = %s) x
                          ON x.sample_id = s.id SET s.is_holdout = 1 WHERE s.is_holdout = 0""", (path,))
        flagged += cur.rowcount
    db.commit()
    print(f"已登记 {len(todo):,} 个路径，新标记 {flagged:,} 行")
    return 0


def register_books(db, eval_set, batch, issues_file, apply):
    cur = db.cursor()
    names = [line.strip() for line in open(issues_file, encoding="utf-8") if line.strip()]
    holes = ",".join(["%s"] * len(names))
    cur.execute(f"""SELECT COUNT(*), SUM(is_holdout = 0), COUNT(DISTINCT source_name) FROM corpus_samples
                     WHERE batch_tag = %s AND source_name IN ({holes})""", [batch, *names])
    n, unflagged, found = cur.fetchone()
    print(f"{eval_set}: {len(names)} 个书名/期名，库里找到 {found}，共 {n:,} 行，未标记 {int(unflagged or 0):,} 行")
    if found != len(names):
        print("  有书名/期名在库里找不到，停止")
        return 1
    if not apply:
        print("(审计模式；加 --apply 登记并标记)")
        return 0
    cur.executemany("""INSERT IGNORE INTO eval_holdout (eval_set, key_type, key_value, batch_tag, note)
                       VALUES (%s, 'book', %s, %s, '期刊/书籍评估留出（整期或整本）')""",
                    [(eval_set, x, batch) for x in names])
    reg = cur.rowcount
    cur.execute(f"""UPDATE corpus_samples SET is_holdout = 1
                     WHERE batch_tag = %s AND source_name IN ({holes}) AND is_holdout = 0""", [batch, *names])
    db.commit()
    print(f"已登记 {reg} 个，新标记 {cur.rowcount:,} 行")
    return 0


def main():
    apply = "--apply" in sys.argv
    if "--copies" in sys.argv:
        db = connect()
        try:
            return register_copies(db, sys.argv[sys.argv.index("--copies") + 1], apply)
        finally:
            db.close()
    if "--books" in sys.argv:
        i = sys.argv.index("--books")
        db = connect()
        try:
            return register_books(db, *sys.argv[i + 1:i + 4], apply)
        finally:
            db.close()
    t0 = time.time()
    db = connect()
    cur = db.cursor()
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

    plan = {}
    for name, spec in SETS.items():
        ev = sorted(eval_images(spec["file"]))
        ev_hash = hash_all(ev, cache)
        by_md5 = collections.defaultdict(list)
        for p, h in ev_hash.items():
            if h:
                by_md5[h].append(p)
        print(f"\n[{name}] 评估图 {len(ev):,} 张（可读 {sum(1 for h in ev_hash.values() if h):,}）"
              f"  {(time.time() - t0) / 60:.1f}m", flush=True)
        regs, matched_eval = [], set()
        for pool in spec["pools"]:
            paths = pool_paths(cur, pool)
            tr_hash = hash_all(paths, cache)
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
            hits = [(p, h) for p, h in tr_hash.items() if h and h in by_md5]
            for p, h in hits:
                matched_eval.update(by_md5[h])
                regs.append((name, "image", p, pool if pool != "defect_tetect" else None,
                             ("md5 与评估图一致: " + by_md5[h][0])[:500]))
            print(f"  候选池 {pool:16s} 训练图 {len(paths):>8,} 张，与评估图内容相同 {len(hits):,} 张"
                  f"  {(time.time() - t0) / 60:.1f}m", flush=True)
        unmatched = [p for p in ev if p not in matched_eval]
        for p in unmatched:
            regs.append((name, "image", p, spec["pools"][0] if spec["pools"][0] != "defect_tetect" else None,
                         "评估侧路径；训练库按 md5 比对无相同图（登记备查）"))
        print(f"  有训练副本的评估图 {len(matched_eval):,} 张，无副本 {len(unmatched):,} 张 → 登记 {len(regs):,} 行")
        plan[name] = regs

    # defect_tetect rows carry no batch_tag in the registry rows above; fill it from the media table
    for name, regs in plan.items():
        fixed = []
        for r in regs:
            if r[3] is None and "/defect-tetect/" in r[2]:
                cur.execute("""SELECT s.batch_tag FROM defect_tetect_media m
                                 JOIN defect_tetect_samples s ON s.id = m.sample_id
                                WHERE m.new_path = %s LIMIT 1""", (r[2],))
                row = cur.fetchone()
                r = (r[0], r[1], r[2], row[0] if row else None, r[4])
            fixed.append(r)
        plan[name] = fixed

    # how many training rows each set will flag
    for name, regs in plan.items():
        keys = [r[2] for r in regs]
        n_cs = n_dt = 0
        for i in range(0, len(keys), 1000):
            part = keys[i:i + 1000]
            holes = ",".join(["%s"] * len(part))
            cur.execute(f"SELECT COUNT(DISTINCT sample_id) FROM corpus_media WHERE new_path IN ({holes})", part)
            n_cs += cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(DISTINCT sample_id) FROM defect_tetect_media WHERE new_path IN ({holes})", part)
            n_dt += cur.fetchone()[0]
        print(f"[{name}] 将标记留出：corpus_samples {n_cs:,} 行，defect_tetect_samples {n_dt:,} 行")

    bad = check_existing(cur)
    if not apply:
        print(f"\n(审计模式，未改动任何东西；加 --apply 执行)  {(time.time() - t0) / 60:.1f}m")
        db.close()
        return 0

    from mdl import alter_when_free, column_exists  # field_tools/mdl.py
    print("\n=== 写入 ===", flush=True)
    for name, regs in plan.items():
        cur.executemany("""INSERT IGNORE INTO eval_holdout (eval_set, key_type, key_value, batch_tag, note)
                           VALUES (%s, %s, %s, %s, %s)""", regs)
        db.commit()
        print(f"  eval_holdout 登记 {name}: {cur.rowcount:,} 行")

    if column_exists(cur, "defect_tetect_samples", "is_holdout"):
        print("  defect_tetect_samples.is_holdout: 已存在")
    elif not alter_when_free(cur, "defect_tetect_samples", FLAG_DDL, "is_holdout"):
        return 1
    cur.execute(VIEW)
    db.commit()

    for name in plan:
        cur.execute("""UPDATE corpus_samples s
                         JOIN (SELECT DISTINCT m.sample_id FROM corpus_media m
                                 JOIN eval_holdout h ON h.key_type IN ('image', 'video')
                                                    AND h.key_value = m.new_path
                                WHERE h.eval_set = %s) x ON x.sample_id = s.id
                          SET s.is_holdout = 1 WHERE s.is_holdout = 0""", (name,))
        n1 = cur.rowcount
        cur.execute("""UPDATE defect_tetect_samples s
                         JOIN (SELECT DISTINCT m.sample_id FROM defect_tetect_media m
                                 JOIN eval_holdout h ON h.key_type = 'image' AND h.key_value = m.new_path
                                WHERE h.eval_set = %s) x ON x.sample_id = s.id
                          SET s.is_holdout = 1 WHERE s.is_holdout = 0""", (name,))
        n2 = cur.rowcount
        db.commit()
        print(f"  {name}: 新标记 corpus_samples {n1:,} 行，defect_tetect_samples {n2:,} 行")

    if bad:
        print(f"\n  已登记评估集有 {bad:,} 行未标记，补标：")
        cur.execute("""UPDATE corpus_samples s
                         JOIN (SELECT DISTINCT m.sample_id FROM corpus_media m
                                 JOIN eval_holdout h ON h.key_type IN ('image', 'video')
                                                    AND h.key_value = m.new_path) x ON x.sample_id = s.id
                          SET s.is_holdout = 1 WHERE s.is_holdout = 0""")
        print(f"    图/视频 {cur.rowcount:,} 行")
        cur.execute("""UPDATE corpus_samples s JOIN eval_holdout h
                           ON h.key_type = 'book' AND h.key_value = s.source_name
                          SET s.is_holdout = 1 WHERE s.is_holdout = 0""")
        print(f"    书名 {cur.rowcount:,} 行")
        db.commit()
    check_existing(cur)
    cur.execute("SELECT eval_set, key_type, COUNT(*) FROM eval_holdout GROUP BY 1, 2")
    print("\n登记表：", cur.fetchall())
    db.close()
    print(f"\n完成  {(time.time() - t0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
