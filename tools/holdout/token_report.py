"""Markdown report from token_stats.json: tokens per corpus type / kind, book by domain, bytes per token.

  python token_report.py <token_stats.json> <out.md>
"""
import collections
import json
import sys

ORDER = ["book", "journal", "video", "detection", "industrial"]
NAMES = {"book": "书籍", "journal": "期刊", "video": "视频", "detection": "目标检测", "industrial": "工业"}


def fmt(n):
    return f"{n / 1e8:,.2f} 亿" if n >= 1e8 else f"{n / 1e4:,.0f} 万"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = json.load(open(src, encoding="utf-8"))
    files = data["files"]
    agg = collections.defaultdict(collections.Counter)
    dom = collections.defaultdict(collections.Counter)
    for name, s in files.items():
        key = (s["corpus_type"], s["kind"])
        rows = s.get("rows", s.get("rows_est", 0))
        for k, v in (("rows", s["manifest_rows"]), ("bytes", s["bytes"]), ("text", s["text_tokens"]),
                     ("template", s["template_tokens"]), ("content_bytes", s["content_bytes"]),
                     ("images", s.get("images", 0)), ("image_tok", s.get("image_tokens") or 0),
                     ("videos", s.get("videos", 0)), ("video_tok", s.get("video_tokens") or 0)):
            agg[key][k] += v
            if s["corpus_type"] == "book":
                dom[(s["domain"], s["kind"])][k] += v
        if s["method"] == "sampled":
            lo, hi = s["text_tokens_ci95"]
            agg[key]["ci_lo"] += lo
            agg[key]["ci_hi"] += hi
        else:
            agg[key]["ci_lo"] += s["text_tokens"]
            agg[key]["ci_hi"] += s["text_tokens"]
        del rows

    L = ["# field_full 训练数据 token 统计", "",
         f"统计时间 {data['created']}（集群时间）；分词器：Qwen3-VL-32B（`{data['tokenizer']}`）。", "",
         "## 口径", "",
         "- **文本 token**：用真实分词器对内容分词。问答取每一轮的内容，去掉 `<image>`/`<video>` 占位符；pt 取全文。"
         "**不含** id、图片路径、JSON 引号这些外壳。",
         "- **模板 token**：Qwen 对话模板每轮约 5 个（`<|im_start|>`、角色、换行、`<|im_end|>`、换行）；pt 每行加 1 个结束符。",
         "- **图片 token**：Qwen3-VL 每 32×32 像素 1 个 token。按 32B 训练配置的像素范围（32,768～262,144，"
         "即每张图 32～256 个）缩放，再加 2 个起止符；每个文件抽最多 2,000 张图读尺寸，取平均后乘以图片总数。",
         "- **视频 token**：按训练配置每秒 1 帧、最多 32 帧，每 2 帧一组，每组 ≤16,384 像素（≤16 个 token），再加 2；"
         "每个文件抽最多 600 段视频读时长和分辨率。每组的时间戳文字没算在内。",
         "- **精确与估算**：除书籍 CPT 外，所有文件逐行分词，是精确数。书籍 CPT（`books_combined`）超过 2 GB 的 6 个文件，"
         "每个按字节位置随机抽 4,000 行，按文件大小推算，给出 95% 区间。",
         ""]

    T = collections.Counter()
    L += ["## 总量", "",
          "| 类型 | 类别 | 行数 | 文件 GB | 文本 token | 模板 token | 图片 token | 视频 token | **合计 token** |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ct in ORDER:
        for kind in ("pt", "qa", "vqa"):
            a = agg.get((ct, kind))
            if not a:
                continue
            tot = a["text"] + a["template"] + a["image_tok"] + a["video_tok"]
            T.update(a)
            T["total"] += tot
            ci = "" if a["ci_lo"] == a["ci_hi"] else f"（95% 区间 {fmt(a['ci_lo'])}～{fmt(a['ci_hi'])}）"
            L.append(f"| {ct}（{NAMES[ct]}） | {kind} | {a['rows']:,} | {a['bytes'] / 1e9:.2f} | {fmt(a['text'])}{ci} | "
                     f"{fmt(a['template'])} | {fmt(a['image_tok']) if a['image_tok'] else '—'} | "
                     f"{fmt(a['video_tok']) if a['video_tok'] else '—'} | **{fmt(tot)}** |")
    L.append(f"| **合计** | | {T['rows']:,} | {T['bytes'] / 1e9:.2f} | {fmt(T['text'])} | {fmt(T['template'])} | "
             f"{fmt(T['image_tok'])} | {fmt(T['video_tok'])} | **{fmt(T['total'])}** |")
    pt_text = sum(a["text"] for (ct, k), a in agg.items() if k == "pt")
    sft = sum(a["text"] + a["template"] + a["image_tok"] + a["video_tok"] for (ct, k), a in agg.items() if k != "pt")
    L += ["", f"其中 **pt（预训练文本）约 {fmt(pt_text)} token**，**qa + vqa（SFT）约 {fmt(sft)} token**（含模板和视觉 token）。", ""]

    L += ["## 书籍按领域", "", "| 领域 | pt 文本 token | qa token | vqa token（文本+图片） | 合计 |", "|---|---|---|---|---|"]
    doms = sorted({d for d, _ in dom}, key=lambda d: (d.startswith("其他"), d))
    for d in doms:
        pt = dom.get((d, "pt"), collections.Counter())
        qa = dom.get((d, "qa"), collections.Counter())
        vq = dom.get((d, "vqa"), collections.Counter())
        a = pt["text"] + pt["template"]
        b = qa["text"] + qa["template"]
        c = vq["text"] + vq["template"] + vq["image_tok"]
        L.append(f"| {d} | {fmt(a) if a else '—'} | {fmt(b) if b else '—'} | {fmt(c) if c else '—'} | {fmt(a + b + c)} |")
    L.append("")

    L += ["## 字节与 token 的实测换算", "",
          "| 数据 | 文件字节 / 文本 token | 内容字节 / 文本 token | 1 MB 文件约多少文本 token |", "|---|---|---|---|"]
    for ct in ORDER:
        for kind in ("pt", "qa", "vqa"):
            a = agg.get((ct, kind))
            if not a or not a["text"]:
                continue
            L.append(f"| {ct} {kind} | {a['bytes'] / a['text']:.2f} | {a['content_bytes'] / a['text']:.2f} | "
                     f"{1e6 / (a['bytes'] / a['text']) / 1e4:.1f} 万 |")
    L += ["", "- 「内容字节」只算文本本身（UTF-8），「文件字节」还包括 JSON 外壳、id 和图片路径，所以带图的 vqa 文件字节/token 偏大。",
          "- 经验值「中文 3 字节 ≈ 1 token」：可以和上表「内容字节 / 文本 token」对照。中文为主的数据大约是 2～3 字节一个 token，"
          "英文和代码比例更高的数据会更大。", ""]

    L += ["## 每张图、每段视频平均多少 token", "", "| 文件 | 图片数 | 每图 token | 视频数 | 每段视频 token |", "|---|---|---|---|---|"]
    for name, s in sorted(files.items()):
        if s.get("images") or s.get("videos"):
            L.append(f"| `{name}` | {s.get('images', 0):,} | {s.get('image_tokens_per_image') or '—'} | "
                     f"{s.get('videos', 0):,} | {s.get('video_tokens_per_video') or '—'} |")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
