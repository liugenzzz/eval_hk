"""Write <out>/README.md for an export_train_data.py output: totals per corpus type and pt / qa / vqa, every file,
per-source accounting (rows in the DB, written, held out for eval, excluded, rejected) and timings.

  python summarize_export.py <out> [--leak <out>/_logs/leak_check/summary.json]
"""
import argparse
import collections
import json
import os

ORDER = ["book", "journal", "video", "detection", "industrial"]


def gb(n):
    return f"{n / 1e9:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--leak", default="")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.out, "manifest.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(a.out, "dataset_info.json"), encoding="utf-8"))
    reg_name = {os.path.relpath(v["file_name"], a.out): k for k, v in info.items()}
    files, batches = man["files"], man["batches"]
    types = sorted({e["corpus_type"] for e in files.values()}, key=lambda t: (ORDER.index(t) if t in ORDER else 99, t))

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for e in files.values():
        k = agg[(e["corpus_type"], e["kind"])]
        k[0] += 1
        k[1] += e["rows"]
        k[2] += e["bytes"]
    L = [f"# 训练数据导出：{os.path.basename(a.out.rstrip('/'))}", ""]
    L += [f"生成时间 {man['created']}；导出 {man['elapsed_minutes']['export']} 分钟 + 合并 {man['elapsed_minutes']['merge']} 分钟"
          + "".join(f"；之后局部重导 {r['batches']}" for r in man.get("reexports", [])), ""]
    L += ["来源：field 库 `corpus_samples` + `defect_tetect_samples`，**已排除全部评估留出（`is_holdout=1`）**。"
          "生成方式和规则见 `mmcode/eval_hk/tools/holdout/README.md`。", ""]
    L += ["## 总览（第一级：语料类型；第二级：pt / qa / vqa）", "",
          "| 类型 | pt 行数 | qa 行数 | vqa 行数 | 合计行数 | 文件数 | 体积 GB |", "|---|---|---|---|---|---|---|"]
    tot = [0, 0, 0, 0, 0, 0]
    for t in types:
        row = [agg[(t, k)][1] for k in ("pt", "qa", "vqa")]
        nf = sum(agg[(t, k)][0] for k in ("pt", "qa", "vqa"))
        size = sum(agg[(t, k)][2] for k in ("pt", "qa", "vqa"))
        L.append(f"| {t} | {row[0]:,} | {row[1]:,} | {row[2]:,} | {sum(row):,} | {nf} | {gb(size)} |")
        for i, v in enumerate(row + [sum(row), nf]):
            tot[i] += v
        tot[5] += size
    L.append(f"| **合计** | {tot[0]:,} | {tot[1]:,} | {tot[2]:,} | {tot[3]:,} | {tot[4]} | {gb(tot[5])} |")
    L += ["", "qa 和 vqa 按这一行**带不带图或视频**来分：带媒体的是 vqa，纯文本问答是 qa。"
          "同一来源同一格式只有一种模态时，文件名不带后缀。", ""]

    L += ["## 文件明细", "", "| 文件 | LlamaFactory 注册名 | 格式 | 行数 | GB | 来源批次 |", "|---|---|---|---|---|---|"]
    for name, e in files.items():
        src = ", ".join(e["batches"]) if len(e["batches"]) <= 3 else f"{len(e['batches'])} 个批次"
        L.append(f"| `{name}` | `{reg_name.get(name, '')}` | {e['format']} | {e['rows']:,} | {gb(e['bytes'])} | {src} |")
    L += ["", "使用：LlamaFactory 的 `dataset_dir` 指向本目录，`dataset:` 写上表的注册名（可以逗号连接多个）。"
          "同一目录下的 `dataset_info.json` 已注册全部文件。pt 文件给 `stage: pt` 用，qa/vqa 文件给 `stage: sft` 用。", ""]

    L += ["## 每个来源的对账", "",
          "| 类型 | 批次 | 库里行数 | 写出 | 评估留出 | 剔除（副本） | 拒收 | 拒收原因 |", "|---|---|---|---|---|---|---|---|"]
    for b, t in sorted(batches.items(), key=lambda x: (ORDER.index(x[1]["corpus_type"]) if x[1]["corpus_type"] in ORDER else 99, x[0])):
        rej = sum(t["rejected"].values())
        why = "；".join(f"{k} {v:,}" for k, v in t["rejected"].items()) if rej else ""
        L.append(f"| {t['corpus_type']} | {b} | {t['rows_in_db']:,} | {sum(t['written'].values()):,} | {t['holdout']:,} | "
                 f"{t['excluded_media']:,} | {rej:,} | {why} |")
    tb = {k: sum(t[k] for t in batches.values()) for k in ("rows_in_db", "holdout", "excluded_media")}
    tw = sum(sum(t["written"].values()) for t in batches.values())
    tr = sum(sum(t["rejected"].values()) for t in batches.values())
    L.append(f"| | **合计** | {tb['rows_in_db']:,} | {tw:,} | {tb['holdout']:,} | {tb['excluded_media']:,} | {tr:,} | |")
    L += ["", "被拒收的行逐条列在 `_rejects/<批次>.jsonl`（id + 原因），可以回库里查原始样本。", ""]

    if a.leak and os.path.exists(a.leak):
        s = json.load(open(a.leak, encoding="utf-8"))["summary"]
        L += ["## 泄漏检查（`leak_check.py`）", "",
              "| 评估集 | 条数 | 答案原文命中 | 题面命中 | 媒体路径命中 | 媒体 md5 命中 | 库 id 命中 |", "|---|---|---|---|---|---|---|"]
        for k, v in s.items():
            L.append(f"| {k} | {v['items']:,} | {v['answer_text']} | {v['prompt_text']} | {v['media_path']} | {v['media_md5']} | {v['db_id']} |")
        L += ["", "看后三列，它们为 0 就说明评估集没有进训练。图片类评估集的文本命中，是同一句模板化问法或答案用在了别的图片上，不算泄漏。", ""]
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:20 + len(types)]))


if __name__ == "__main__":
    main()
