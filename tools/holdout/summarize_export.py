"""Write <out>/README.md for the local full copy of the field DB (an export_train_data.py output).

The README states what the directory is (the full training data, eval hold-outs excluded), the rules for
using and updating it (extract from here; every data change goes to the DB and to this copy in the same
round), totals per corpus type and pt / qa / vqa, every file, per-source accounting, the leak check and the
update history. Rerun it after every update; it rebuilds the whole file from manifest.json.

  python summarize_export.py <out> [--leak <out>/_logs/leak_check/summary.json]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_train_data import SOURCE_NAMES  # noqa: E402

ORDER = ["book", "journal", "video", "detection", "industrial"]
TYPE_NAMES = {"book": "书籍", "journal": "期刊", "video": "视频", "detection": "目标检测", "industrial": "工业"}
TOOLS = "/mnt/si003010kcx0/mmdata/mmcode/eval_hk/tools/holdout"
# leak_check.py summaries written before 2026-10-09 used short keys
OLD_EVAL_KEYS = {"aircraft_v2": ("aircraft_defect", "aircraft_eval_v2"), "park": ("video_park_abnormal", "park_video_v1"),
                 "det": ("detection", "mbjc_test_v1"), "industrial": ("industrial", "industrial_eval_v1"),
                 "book": ("book", "book_eval_v1"), "journal": ("journal", "journal_eval_v1"),
                 "journal_gi82": ("journal_gi82", "journal_gi82_eval_v1")}
SOURCE_LABELS = {"video_bilibili": "B站视频", "video_park_abnormal": "园区异常行为视频", "aircraft_defect": "飞机缺陷问答"}


def gb(n):
    return f"{n / 1e9:.2f}"


def type_key(t):
    return (ORDER.index(t) if t in ORDER else 99, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--leak", default="")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    files, batches = man["files"], man["batches"]
    types = sorted({e["corpus_type"] for e in files.values()}, key=type_key)
    n_rows = sum(e["rows"] for e in files.values())
    n_bytes = sum(e["bytes"] for e in files.values())
    held = sum(t["holdout"] for t in batches.values())
    in_db = sum(t["rows_in_db"] for t in batches.values())

    L = [f"# 领域训练数据 · 全量本地副本（`{os.path.basename(out)}`）", ""]
    L += [f"**这是 field 库的全量训练数据**：库里全部 {len(batches)} 个来源批次、{in_db:,} 行，"
          f"去掉评估留出 {held:,} 行后，共 {n_rows:,} 行、{gb(n_bytes)} GB。**不含任何评估数据。**", "",
          f"- 位置：`{out}`",
          f"- 数据版本：库快照 {man['created']}（最近一次更新见文末「更新记录」）",
          f"- 生成和更新工具：`{TOOLS}/`（git 仓库 eval_hk 内，说明见同目录 README.md）", ""]

    L += ["## 怎么用", "",
          "- **后续要训练数据，直接从这里抽**：按语料类型、领域和 pt / qa / vqa 挑任务文件，或者从文件里抽子集，不用再查库。",
          "- 书籍：`book/<七大类>/<pt|qa|vqa>/<任务类型>.jsonl`；没有七大类标签的放进 `book/其他/<304|航空书籍|军事武器|GI82书籍|未分类>/`。同一领域、同一任务的不同来源合成一个文件。",
          "- CPT 按领域放进 `pt/books_combined.jsonl`，沿用原名，不再细分。",
          "- 期刊：`journal/<pt|qa|vqa>/<任务类型>.jsonl`，两批来源合并为 11 种任务。",
          "- 视频：`video/vqa/` 下保留 B站 3 个任务；园区合成 `park_caption.jsonl`、`park_multiple_choice_qa.jsonl`、`park_open_ended_qa.jsonl`。",
          "- 检测：`detection/vqa/` 按通用目标、GQA场景、航拍、遥感四种场景各一个文件。",
          "- 工业：`industrial/飞机缺陷/vqa/<任务类型>.jsonl`（16 种任务）；`industrial/工业质检/vqa/defect_cn.jsonl` 和 `defect_en.jsonl`（part0 / part1 按语言合并）。",
          "- 格式：pt 每行含 `id` 和 `text`；qa / vqa 统一为 sharegpt 的 `conversations`。vqa 带 `images` 或 `videos`，媒体为绝对路径；没有媒体的样本写空列表，保证字段一致。",
          "- pt / qa / vqa 按库里 `task_kind_map` 定；带图或视频的行一律进 vqa。",
          '- 每行第一个字段是 `"id": "<来源批次>:<库 id>"`，可回库追溯；合并文件仍能靠 id 分出来源。',
          "- 来源显示名对照如下（文件名按任务命名，行内 id 和 manifest 仍是库里的批次名）：",
          *[f"  - `{v}` = 库里 `{k}`（{SOURCE_LABELS.get(v, '')}）" for k, v in SOURCE_NAMES.items()],
          "- 评估集**不在这里**：文件在 `eval_runs/<评估集>/data/`（飞机的在 `eval_hk/data/aircraft_vqa/`），"
          "登记在库表 `field.eval_holdout`。导出时已按登记排除，文末的泄漏检查可以确认。", ""]

    L += ["## 更新规范：库一版、本地一版，同一轮完成", "",
          "**数据有任何变动（新批次入库、改数据、新增评估集），库和这个目录必须在同一轮里一起更新。"
          "只改库不改本地，或者只改本地不改库，都不允许。**", "",
          "1. **先改库**：新数据按 `mm_env/INGEST_SPEC.md` 入库。新评估集先登记，"
          "命令是 `register_eval_sets.py`，整期或整本的用 `--books`，媒体副本用 `--copies`。",
          "2. **再改本地**：只重导受影响的批次，然后拼回本目录：",
          "   ```bash",
          f"   PY=/mnt/si003010kcx0/mmdata/mm_env/mysql/bin/python",
          '   REDO=$(mktemp -d /tmp/field_redo.XXXXXX)',
          f'   $PY {TOOLS}/export_train_data.py --out "$REDO" --only <批次1,批次2> --workers 8',
          f'   $PY {TOOLS}/splice_export.py {out} "$REDO"',
          "   ```",
          "   要整体重建时，先导到新目录，校验通过后再替换本目录。耗时以本次日志为准；长任务放 tmux / nohup 中后台运行，日志写文件，避免 SSH 断线中止。",
          "3. **检查**：两项都通过才算更新完成。",
          "   ```bash",
          f"   $PY {TOOLS}/verify_export.py {out}",
          f"   /opt/conda/bin/python {TOOLS}/leak_check.py --train {out} --out {out}/_logs/leak_check",
          "   ```",
          f"4. **重写本说明**：`$PY {TOOLS}/summarize_export.py {out} --leak {out}/_logs/leak_check/summary.json`。"
          "更新记录会自动带上。", ""]

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for e in files.values():
        k = agg[(e["corpus_type"], e["kind"])]
        k[0] += 1
        k[1] += e["rows"]
        k[2] += e["bytes"]
    L += ["## 总览", "", "| 类型 | pt 行数 | qa 行数 | vqa 行数 | 合计行数 | 文件数 | 体积 GB |", "|---|---|---|---|---|---|---|"]
    tot = [0] * 6
    for t in types:
        row = [agg[(t, k)][1] for k in ("pt", "qa", "vqa")]
        nf = sum(agg[(t, k)][0] for k in ("pt", "qa", "vqa"))
        size = sum(agg[(t, k)][2] for k in ("pt", "qa", "vqa"))
        L.append(f"| {t}（{TYPE_NAMES.get(t, t)}） | {row[0]:,} | {row[1]:,} | {row[2]:,} | {sum(row):,} | {nf} | {gb(size)} |")
        for i, v in enumerate(row + [sum(row), nf, size]):
            tot[i] += v
    L.append(f"| **合计** | {tot[0]:,} | {tot[1]:,} | {tot[2]:,} | {tot[3]:,} | {tot[4]} | {gb(tot[5])} |")
    L += ["", "pt / qa / vqa 按库里 `task_kind_map` 定；带图或视频的行一律进 vqa。", ""]

    # Domain metadata is recorded by v3; the path fallback also handles older manifests.
    for corpus, label in (("book", "书籍领域"), ("industrial", "工业子领域")):
        grouped = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
        for name, e in files.items():
            if e["corpus_type"] != corpus:
                continue
            domain = e.get("domain", "/".join(name.split("/")[1:-2])) or "未分类"
            values = grouped[domain]
            values[("pt", "qa", "vqa").index(e["kind"])] += e["rows"]
            values[3] += 1
            values[4] += e["bytes"]
        if grouped:
            L += [f"### {label}", "", "| 领域 | pt 行数 | qa 行数 | vqa 行数 | 合计行数 | 文件数 | 体积 GB |",
                  "|---|---|---|---|---|---|---|"]
            preferred = ["共性基础", "研发设计", "生产制造", "试验测试", "维修保障", "运营管理", "作战应用",
                         "飞机缺陷", "工业质检"]
            for domain, values in sorted(grouped.items(), key=lambda item: (
                    preferred.index(item[0]) if item[0] in preferred else len(preferred), item[0])):
                pt, qa, vqa, count, size = values
                L.append(f"| {domain} | {pt:,} | {qa:,} | {vqa:,} | {pt + qa + vqa:,} | {count} | {gb(size)} |")
            L.append("")

    L += ["## 文件明细", ""]
    by_directory = collections.defaultdict(list)
    for name, e in files.items():
        by_directory[name.rsplit("/", 1)[0]].append((name, e))
    for directory in sorted(by_directory, key=lambda d: (type_key(d.split("/")[0]), d)):
        L += [f"### `{directory}/`", "", "| 文件 | 格式 | 行数 | GB | 来源批次 |", "|---|---|---|---|---|"]
        for name, e in sorted(by_directory[directory]):
            src = ", ".join(e["batches"]) if len(e["batches"]) <= 3 else f"{len(e['batches'])} 个批次"
            L.append(f"| `{name}` | {e['format']} | {e['rows']:,} | {gb(e['bytes'])} | {src} |")
        L.append("")

    L += ["## 每个来源的对账", "",
          "| 类型 | 来源批次 | 库里行数 | 写出 | 评估留出 | 剔除（副本） | 拒收 |", "|---|---|---|---|---|---|---|"]
    for b, t in sorted(batches.items(), key=lambda x: (type_key(x[1]["corpus_type"]), x[0])):
        rej = sum(t["rejected"].values())
        why = f"（{'；'.join(f'{k} {v:,}' for k, v in t['rejected'].items())}）" if rej else ""
        L.append(f"| {t['corpus_type']} | {b} | {t['rows_in_db']:,} | {sum(t['written'].values()):,} | {t['holdout']:,} | "
                 f"{t['excluded_media']:,} | {rej:,}{why} |")
    tw = sum(sum(t["written"].values()) for t in batches.values())
    tx = sum(t["excluded_media"] for t in batches.values())
    tr = sum(sum(t["rejected"].values()) for t in batches.values())
    L.append(f"| | **合计** | {in_db:,} | {tw:,} | {held:,} | {tx:,} | {tr:,} |")
    L += ["", "库里行数 = 写出 + 评估留出 + 剔除 + 拒收。拒收的行逐条列在 `_rejects/<批次>.jsonl`（id + 原因）。", ""]

    if a.leak and os.path.exists(a.leak):
        s = json.load(open(a.leak, encoding="utf-8"))["summary"]
        L += ["## 泄漏检查（评估集 vs 本目录的 qa / vqa 文件）", "",
              "| 评估集（库里登记名） | 条数 | 答案原文命中 | 题面命中 | 媒体路径命中 | 媒体 md5 命中 | 库 id 命中 |",
              "|---|---|---|---|---|---|---|"]
        for k, v in s.items():
            name, reg = (k, v["eval_set"]) if "eval_set" in v else OLD_EVAL_KEYS.get(k, (k, ""))
            L.append(f"| {name}（{reg}） | {v['items']:,} | {v['answer_text']} | {v['prompt_text']} | {v['media_path']} | "
                     f"{v['media_md5']} | {v['db_id']} |")
        L += ["", "后三列为 0，说明评估集的图、视频和样本都没有进本目录。"
              "aircraft_defect、video_park_abnormal、detection、industrial 这几个的文本命中，"
              "是同一句模板化问法或答案用在了别的图片或视频上，不算泄漏。", ""]

    L += ["## 更新记录", "", "| 时间 | 内容 | 耗时 |", "|---|---|---|",
          f"| {man['created']} | 全量导出：{len(batches)} 个来源批次（含 GI82，去掉其中 20 期评估留出）"
          f" | 导出 {man['elapsed_minutes']['export']} 分钟 + 合并 {man['elapsed_minutes']['merge']} 分钟 |"]
    for r in man.get("reexports", []):
        L.append(f"| {r['created']} | 重导并拼回：{', '.join(r['batches'])} | 导出 {r['elapsed_minutes']['export']} 分钟 |")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"written {os.path.join(out, 'README.md')} ({len(L)} lines)")


if __name__ == "__main__":
    main()
