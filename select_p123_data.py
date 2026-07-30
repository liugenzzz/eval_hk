"""从大批 ShareGPT 格式问答数据里，随机重抽一部分，调用裁判模型判断每条题目属于
P1/P2/P3/R1/R2/R3/其他 中的哪一类能力，筛出 P1/P2/P3（当前有训练任务的三类），
产出可直接用于测试的 JSON 和 TSV（TSV 与 aero_vqa.tsv 同结构，能直接接入本项目的评估流程）。

输入数据格式（每条一个对象）：
{
  "id": "...",
  "images": ["/path/to/xxx.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\n问题..."},
    {"from": "gpt", "value": "回答..."}
  ]
}

用法：
  python select_p123_data.py --input data.json --out_dir p123_selection --sample_size 2000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_tool.cache import JsonlCache
from eval_tool.judge import JudgeClient, JudgeSettings, extract_json_object

CATEGORY_LABELS = {"P1", "P2", "P3", "R1", "R2", "R3", "其他"}
KEEP_CATEGORIES = {"P1", "P2", "P3"}


def _tqdm(iterable: Any, **kwargs: Any) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def load_records(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入文件应是一个 JSON 数组")
    return data


def extract_qa(record: dict[str, Any]) -> tuple[str, str] | None:
    conversations = record.get("conversations") or []
    question = ""
    answer = ""
    for turn in conversations:
        role = turn.get("from")
        value = str(turn.get("value", ""))
        if role == "human" and not question:
            question = value.replace("<image>", "").strip("\n ").strip()
        elif role == "gpt" and not answer:
            answer = value.strip()
        if question and answer:
            break
    if not question:
        return None
    return question, answer


def encode_image(path: str | Path, max_side: int = 1280) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        from PIL import Image

        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = max_side / float(max(w, h))
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(p.read_bytes()).decode("ascii")


def parse_category_judge(content: str) -> tuple[str, str]:
    obj = extract_json_object(content)
    category = str(obj.get("category", "其他")).strip()
    if category not in CATEGORY_LABELS:
        category = "其他"
    return category, str(obj.get("reason", ""))


def classify_one(client: JudgeClient, prompt: str, question: str, answer: str, image_b64: str | None) -> dict[str, Any]:
    user_text = f"问题：{question}\n参考答案：{answer}\n\n请看着图片判断这道题属于哪一类能力，只输出 JSON。"
    last_err = ""
    for _ in range(client.settings.max_retries):
        try:
            content = client.judge_raw(prompt, user_text, image_b64=image_b64)
            category, reason = parse_category_judge(content)
            return {"category": category, "reason": reason}
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return {"category": "其他", "reason": f"[judge_error] {last_err}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="重抽数据并用裁判模型筛选适合 P1/P2/P3 的题目")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径（ShareGPT 格式数组）")
    parser.add_argument("--out_dir", default="p123_selection", help="输出目录")
    parser.add_argument("--sample_size", type=int, default=2000, help="随机重抽条数，<=0 表示全量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--workers", type=int, default=16, help="并发数")
    parser.add_argument("--prompt_file", default="prompts/judge_category_classify.txt", help="分类裁判 prompt 文件")
    parser.add_argument("--api_base", default=JudgeSettings.api_base)
    parser.add_argument("--api_key", default=JudgeSettings.api_key)
    parser.add_argument("--model", default=JudgeSettings.model)
    parser.add_argument("--timeout", type=int, default=JudgeSettings.timeout)
    parser.add_argument("--max_retries", type=int, default=JudgeSettings.max_retries)
    parser.add_argument("--image_max_side", type=int, default=1280)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    records = load_records(args.input)
    rng = random.Random(args.seed)
    if args.sample_size > 0 and args.sample_size < len(records):
        sample = rng.sample(records, args.sample_size)
    else:
        sample = records
    print(f"总数据 {len(records)} 条，本次重抽 {len(sample)} 条")

    client = JudgeClient(
        JudgeSettings(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            temperature=0.0,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    )
    cache = JsonlCache(out_dir / "classify_cache.jsonl", ("id",))

    prepared: list[dict[str, Any]] = []
    skipped_no_qa = 0
    skipped_no_image = 0
    for record in sample:
        qa = extract_qa(record)
        if qa is None:
            skipped_no_qa += 1
            continue
        question, answer = qa
        images = record.get("images") or []
        image_b64 = None
        if images:
            image_b64 = encode_image(images[0], max_side=args.image_max_side)
        if images and image_b64 is None:
            skipped_no_image += 1
            continue
        prepared.append({"record": record, "question": question, "answer": answer, "image_b64": image_b64})
    if skipped_no_qa:
        print(f"跳过 {skipped_no_qa} 条：没有可用的问题文本")
    if skipped_no_image:
        print(f"跳过 {skipped_no_image} 条：图片文件读取失败")

    results: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for item in prepared:
        rid = str(item["record"].get("id", ""))
        cached = cache.get({"id": rid})
        if cached is not None:
            results[rid] = cached
        else:
            pending.append(item)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for item in pending:
            rid = str(item["record"].get("id", ""))
            fut = executor.submit(classify_one, client, prompt, item["question"], item["answer"], item["image_b64"])
            futures[fut] = rid
        iterator = as_completed(futures)
        if futures:
            iterator = _tqdm(iterator, total=len(futures), desc="Classifying")
        for fut in iterator:
            rid = futures[fut]
            result = fut.result()
            results[rid] = result
            cache.set({"id": rid}, result)

    audit_rows: list[dict[str, Any]] = []
    kept_json: list[dict[str, Any]] = []
    tsv_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {label: 0 for label in CATEGORY_LABELS}

    for item in prepared:
        record = item["record"]
        rid = str(record.get("id", ""))
        result = results.get(rid, {"category": "其他", "reason": "[missing_result]"})
        category = result.get("category", "其他")
        counts[category] = counts.get(category, 0) + 1
        audit_rows.append(
            {
                "id": rid,
                "category": category,
                "reason": result.get("reason", ""),
                "question": item["question"][:120],
            }
        )
        if category in KEEP_CATEGORIES:
            kept_record = dict(record)
            kept_record["category"] = category
            kept_json.append(kept_record)
            tsv_rows.append(
                {
                    "index": rid,
                    "image": item["image_b64"] or "",
                    "question": item["question"],
                    "answer": item["answer"],
                    "category": category,
                    "l2-category": "",
                    "split": "test",
                    "source_id": rid,
                }
            )

    import csv

    audit_path = out_dir / "classify_result.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "reason", "question"])
        writer.writeheader()
        writer.writerows(audit_rows)

    json_path = out_dir / "p123_selected.json"
    json_path.write_text(json.dumps(kept_json, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = out_dir / "p123_selected.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["index", "image", "question", "answer", "category", "l2-category", "split", "source_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(tsv_rows)

    print("\n各类别数量：")
    for label in ("P1", "P2", "P3", "R1", "R2", "R3", "其他"):
        print(f"  {label}: {counts.get(label, 0)}")
    print(f"\n保留 P1/P2/P3 共 {len(kept_json)} 条")
    print(f"- 明细审查: {audit_path}")
    print(f"- 筛选结果(JSON, 原格式+category): {json_path}")
    print(f"- 筛选结果(TSV, 可直接接入 eval_tool): {tsv_path}")


if __name__ == "__main__":
    main()
