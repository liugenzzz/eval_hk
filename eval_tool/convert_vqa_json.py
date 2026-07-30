"""Convert ShareGPT-style VQA JSON (single or multi-turn, single or multi-image) into aero_vqa.tsv.

Each input record looks like:
{
  "id": "military_equipment_vqa_000001",
  "image": ["path/1.jpg", "path/2.jpg", ...],
  "conversations": [
    {"from": "human", "value": "<image>\\n<image>\\n问题..."},
    {"from": "gpt", "value": "参考答案..."},
    ... (more human/gpt pairs for multi-turn records)
  ]
}

Multi-turn records are split into one TSV row per turn: the "question" column holds
only the current turn's question, while prior turns go into the "history" column as a
JSON list of {"q", "a", "n_img"} objects (a = the reference/gold answer, n_img = how
many images that turn introduced). At inference time the history is replayed as real
chat messages (user/assistant alternation, gold answers as assistant turns), and only
the current turn's answer is scored.
Multi-image records accumulate every image referenced up to and including the current
turn into the "image" cell (as a JSON list of base64 strings); history n_img values
say how the images distribute across turns.

This data source has no capability taxonomy (P1-3/R1-3), so "category" instead
records the turn type: 单轮 (no prior context) vs 多轮 (has prior context), which makes
score_summary report total + per-type scores. "l2-category" is left empty.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Any

from .imaging import encode_image_cell

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=""):
        return iterable


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_records(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if not stripped.startswith("["):
        # Not a JSON array (e.g. a .json file that's actually one-object-per-line) —
        # fall back to JSONL parsing instead of failing outright.
        return _load_jsonl_records(path, text=text)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of records")
    return data


def _load_jsonl_records(path: Path, text: str | None = None) -> list[dict[str, Any]]:
    if text is None:
        text = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON line ({exc})") from exc
    return records


def _encode_image_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _clean_question(value: str) -> str:
    return str(value or "").replace("<image>", "").strip()


def _split_turns(conversations: list[dict[str, Any]]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(conversations):
        human, gpt = conversations[i], conversations[i + 1]
        if human.get("from") != "human" or gpt.get("from") != "gpt":
            i += 1
            continue
        turns.append((str(human.get("value", "")), str(gpt.get("value", ""))))
        i += 2
    return turns


def record_to_rows(record: dict[str, Any], image_cache: dict[str, str]) -> list[dict[str, Any]]:
    record_id = str(record.get("id", ""))
    # 兼容 ShareGPT 的 "image" 与 "images" 两种字段名
    image_paths = record.get("image") or record.get("images") or []
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if not image_paths:
        print(f"[warn] 记录 {record_id} 没有图片字段（image/images 均为空），该记录将以无图方式评估")
    turns = _split_turns(record.get("conversations") or [])

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    images_used = 0
    for turn_idx, (human_value, gpt_value) in enumerate(turns):
        tag_count = human_value.count("<image>")
        images_used += tag_count
        # If no <image> tags were found (data doesn't tag images), fall back to all images.
        cumulative_paths = image_paths[:images_used] if images_used else image_paths
        question_text = _clean_question(human_value)

        images_b64 = []
        for path in cumulative_paths:
            if path not in image_cache:
                image_cache[path] = _encode_image_file(path)
            images_b64.append(image_cache[path])

        rows.append(
            {
                "index": f"{record_id}__t{turn_idx}",
                "image": encode_image_cell(images_b64) if images_b64 else "",
                "question": question_text,
                "answer": gpt_value.strip(),
                "history": json.dumps(history, ensure_ascii=False) if history else "",
                # rows with prior context score as 多轮, first/only turns as 单轮 —
                # score_summary then reports total + per-type columns automatically
                "category": "多轮" if history else "单轮",
                "l2-category": "",
                "split": "test",
                "source_id": record_id,
            }
        )
        history.append({"q": question_text, "a": gpt_value.strip(), "n_img": tag_count})
    return rows


def convert(input_json: str | Path, output_dir: str | Path) -> Path:
    records = load_records(input_json)
    out_path = Path(output_dir) / "aero_vqa.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "image", "question", "answer", "history", "category", "l2-category", "split", "source_id"]
    image_cache: dict[str, str] = {}

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        total_rows = 0
        for record in tqdm(records, total=len(records), desc="转 aero_vqa.tsv"):
            for row in record_to_rows(record, image_cache):
                writer.writerow(row)
                total_rows += 1

    print(f"写入 {total_rows} 行（来自 {len(records)} 条记录）-> {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ShareGPT-style VQA JSON/JSONL into aero_vqa.tsv")
    parser.add_argument("input_json", help="Path to the ShareGPT-format JSON array or JSONL file (one record per line)")
    parser.add_argument("output_dir", help="Directory to write aero_vqa.tsv into (this becomes tsv_dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.input_json, args.output_dir)


if __name__ == "__main__":
    main()
