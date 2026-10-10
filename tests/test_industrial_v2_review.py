import csv
import json

import pandas as pd
import pytest

from eval_tool.industrial_v2_review import RUNS, build_review_queue


def test_review_queue_unions_judge_conflicts_and_label_issues(tmp_path):
    source = tmp_path / "industrial.jsonl"
    examples = [("good.jpg", "good_weld"), ("unknown.jpg", "unknown"), ("disputed.jpg", "crack")]
    records = []
    for key, dtype in examples:
        for lang in ("zh", "en"):
            records.append({"id": f"{lang}_{key}", "images": [f"/images/{key}"],
                            "conversations": [{"value": "question"}, {"value": f"reference {lang} {key}"}],
                            "metadata": {"language": lang, "source_image_key": key,
                                         "source_domain": "fixture", "defect_type": dtype,
                                         "defect_status": "abnormal"}})
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    details = tmp_path / "details"
    for run in RUNS:
        folder = details / run
        folder.mkdir(parents=True)
        pd.DataFrame([{"dataset": f"industrial_{row['metadata']['language']}",
                       "index": row["id"] + "__t1",
                       "industrial_reference_status": (
                           "conflict" if run == RUNS[0] and row["id"] == "zh_disputed.jpg" else "usable"
                       )} for row in records]).to_excel(folder / "judge_detail_all.xlsx", index=False)
    out = tmp_path / "review.csv"
    summary = build_review_queue(source, details, out)
    with out.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["images"] == 3
    assert summary["triggers"] == {"judge_reference_questioned": 1,
                                    "status_corrected": 1, "type_unspecified": 1}
    by_key = {row["source_image_key"]: row for row in rows}
    assert by_key["good.jpg"]["candidate_defect_status"] == "normal"
    assert by_key["disputed.jpg"]["judge_status_disagreement"] == "True"
    assert by_key["disputed.jpg"]["human_reference_validity"] == ""
    assert len(json.loads(by_key["disputed.jpg"]["judge_statuses_by_run"])) == 8
    with pytest.raises(FileExistsError):
        build_review_queue(source, details, out)
