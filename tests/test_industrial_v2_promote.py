import csv
import json

import pandas as pd
import pytest

from eval_tool.industrial_v2 import build_v2
from eval_tool.industrial_v2_promote import promote
from eval_tool.industrial_v2_review import RUNS, build_full_review_template


def _fixture(tmp_path):
    source = tmp_path / "v1.jsonl"
    records = []
    for key, defect in (("good.jpg", "good_weld"), ("crack.jpg", "crack")):
        for lang in ("zh", "en"):
            records.append({"id": f"{lang}_{key}", "images": [f"/images/{key}"],
                            "conversations": [{"value": "question"}, {"value": "reference"}],
                            "metadata": {"language": lang, "source_image_key": key,
                                         "source_domain": "fixture", "defect_type": defect,
                                         "defect_status": "abnormal"}})
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    candidate = tmp_path / "candidate.jsonl"
    build_v2(source, candidate)
    details = tmp_path / "details"
    for run in RUNS:
        folder = details / run
        folder.mkdir(parents=True)
        pd.DataFrame([{"dataset": f"industrial_{r['metadata']['language']}",
                       "index": r["id"] + "__t1", "industrial_reference_status": "usable"}
                      for r in records]).to_excel(folder / "judge_detail_all.xlsx", index=False)
    decisions = tmp_path / "reviews.csv"
    build_full_review_template(source, details, decisions)
    return source, candidate, details, decisions


def _rewrite(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_promotion_requires_all_images_and_two_reviews(tmp_path):
    source, candidate, details, decisions = _fixture(tmp_path)
    out = tmp_path / "released.jsonl"
    with pytest.raises(ValueError, match="incomplete/invalid"):
        promote(source, candidate, details, decisions, out)
    with decisions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["reviewer_id_1"] = "a"
    rows[0]["reviewer_id_2"] = "b"
    rows[0]["reviewer_1_decision"] = rows[0]["reviewer_2_decision"] = "usable"
    rows[0]["human_reference_validity"] = "usable"
    _rewrite(decisions, rows)
    with pytest.raises(ValueError, match="incomplete/invalid"):
        promote(source, candidate, details, decisions, out)
    rows[1]["reviewer_id_1"] = "c"
    rows[1]["reviewer_id_2"] = "d"
    rows[1]["reviewer_1_decision"] = "usable"
    rows[1]["reviewer_2_decision"] = "conflict"
    rows[1]["human_reference_validity"] = "conflict"
    _rewrite(decisions, rows)
    with pytest.raises(ValueError, match="adjudication"):
        promote(source, candidate, details, decisions, out)
    rows[1]["adjudicator_id"] = "e"
    rows[1]["adjudication_reason"] = "Image and reference checked independently"
    _rewrite(decisions, rows)
    manifest = promote(source, candidate, details, decisions, out)
    assert manifest["images_reviewed"] == 2
    assert manifest["decisions"] == {"usable": 1, "conflict": 1}
    assert manifest["released_rows"] == 2
    released = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {row["metadata"]["source_image_key"] for row in released} == {rows[0]["source_image_key"]}
    assert all(row["metadata"]["reference_validity"] == "usable" for row in released)
    with pytest.raises(FileExistsError):
        promote(source, candidate, details, decisions, out)


def test_promotion_rejects_missing_or_modified_evidence(tmp_path):
    source, candidate, details, decisions = _fixture(tmp_path)
    with decisions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _rewrite(decisions, rows[:1])
    with pytest.raises(ValueError, match="missing, duplicate, or unknown"):
        promote(source, candidate, details, decisions, tmp_path / "released.jsonl")
    rows[1]["source_image_key"] = rows[0]["source_image_key"]
    _rewrite(decisions, rows)
    with pytest.raises(ValueError, match="missing, duplicate, or unknown"):
        promote(source, candidate, details, decisions, tmp_path / "released.jsonl")
