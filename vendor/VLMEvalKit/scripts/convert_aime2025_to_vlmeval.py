#!/usr/bin/env python3
"""Convert AIME 2025 JSONL into a VLMEvalKit TSV file.

Input JSONL rows are expected to contain:
  {"question": "...", "answer": "..."}

The output TSV can be placed under $LMUData as AIME2025.tsv, then evaluated with:
  python run.py --data AIME2025 --model <your_model>
"""

import argparse
import csv
import json
import os
from pathlib import Path


def default_lmu_data_root() -> Path:
    env_root = os.environ.get("LMUData")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / "LMUData"


def convert(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["index", "question", "answer", "category", "split"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            question = str(item["question"]).strip()
            answer = str(item["answer"]).strip()

            writer.writerow(
                {
                    "index": idx,
                    "question": question,
                    "answer": answer,
                    "category": "AIME2025",
                    "split": "test",
                }
            )
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/code/VLMEvalKit-main/custom_data/data/aime2025.jsonl",
        help="Path to aime2025.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=str(default_lmu_data_root() / "AIME2025.tsv"),
        help="Output TSV path. Default: $LMUData/AIME2025.tsv or ~/LMUData/AIME2025.tsv.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    count = convert(input_path, output_path)
    print(f"Converted {count} rows to {output_path}")


if __name__ == "__main__":
    main()
