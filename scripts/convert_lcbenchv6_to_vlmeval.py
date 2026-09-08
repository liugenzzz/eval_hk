#!/usr/bin/env python3
"""Convert LCBenchV6 / LiveCodeBench JSONL into a VLMEvalKit TSV file.

The input JSONL rows are expected to contain fields such as:
  question_title, question_content, starter_code, public_test_cases,
  private_test_cases, platform, question_id, contest_id, difficulty

The output TSV can be placed under $LMUData as LCBenchV6.tsv, then evaluated with:
  python run.py --data LCBenchV6 --model <your_model>
"""

import argparse
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_lmu_data_root() -> Path:
    env_root = os.environ.get("LMUData")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / "LMUData"


def build_question(item: dict) -> str:
    title = str(item.get("question_title", "")).strip()
    content = str(item.get("question_content", "")).strip()
    starter_code = str(item.get("starter_code", "")).strip()

    prompt = ""
    if title:
        prompt += f"# {title}\n\n"
    prompt += content
    if starter_code:
        prompt += "\n\nStarter code:\n```python\n" + starter_code + "\n```"
    prompt += (
        "\n\nWrite a complete Python 3 program that reads from standard input "
        "and writes the answer to standard output. Return only the code."
    )
    return prompt


def convert(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "question",
        "question_title",
        "question_id",
        "contest_id",
        "contest_date",
        "platform",
        "difficulty",
        "starter_code",
        "public_test_cases",
        "private_test_cases",
        "metadata",
        "category",
        "split",
    ]

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.DictWriter(
            fout, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()

        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question_content" not in item:
                raise KeyError(
                    f"Line {idx + 1} misses required field 'question_content'. "
                    "Expected LiveCodeBench/LCBench-style JSONL."
                )
            writer.writerow(
                {
                    "index": idx,
                    "question": build_question(item),
                    "question_title": item.get("question_title", ""),
                    "question_id": item.get("question_id", ""),
                    "contest_id": item.get("contest_id", ""),
                    "contest_date": item.get("contest_date", ""),
                    "platform": item.get("platform", ""),
                    "difficulty": item.get("difficulty", ""),
                    "starter_code": item.get("starter_code", ""),
                    "public_test_cases": item.get("public_test_cases", "[]"),
                    "private_test_cases": item.get("private_test_cases", ""),
                    "metadata": item.get("metadata", "{}"),
                    "category": item.get("difficulty", "unknown") or "unknown",
                    "split": "test",
                }
            )
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/code/VLMEvalKit-main/custom_data/data/lcbenchv6.jsonl",
        help="Path to lcbenchv6.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=str(default_lmu_data_root() / "LCBenchV6.tsv"),
        help="Output TSV path. Default: $LMUData/LCBenchV6.tsv or ~/LMUData/LCBenchV6.tsv.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    count = convert(input_path, output_path)
    print(f"Converted {count} rows to {output_path}")


if __name__ == "__main__":
    main()
