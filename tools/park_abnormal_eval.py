"""Validate human annotations and optionally score predictions by dimension."""

import argparse
import json
from pathlib import Path

from eval_tool.park_abnormal import load_annotations, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--require-time", action="store_true")
    args = parser.parse_args()
    references = load_annotations(args.annotations)
    if args.predictions is None:
        print(json.dumps({"verified_clips": len(references)}, ensure_ascii=False))
        return
    with args.predictions.open(encoding="utf-8") as stream:
        predictions = [json.loads(line) for line in stream if line.strip()]
    rows = score_predictions(references, predictions, require_time=args.require_time)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
