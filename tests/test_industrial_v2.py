import json

import pandas as pd
import pytest

from eval_tool.industrial_report import summarize_industrial_errors
from eval_tool.industrial_v2 import build_v2


def test_v2_corrects_only_reviewed_status_and_keeps_unspecified_separate(tmp_path):
    source = tmp_path / "v1.jsonl"
    rows = []
    for key, defect in (("good.jpg", "good_weld"), ("unclear.jpg", "unknown")):
        for language in ("zh", "en"):
            rows.append({"id": f"{language}_{key}", "conversations": [], "images": [key],
                         "metadata": {"language": language, "source_image_key": key,
                                      "defect_type": defect, "defect_status": "abnormal"}})
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "v2.jsonl"
    audit = build_v2(source, out)
    updated = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert audit["corrected_normal_images"] == ["good.jpg"]
    assert audit["unspecified_type_images"] == ["unclear.jpg"]
    assert [row["metadata"]["defect_status"] for row in updated] == ["normal", "normal", "abnormal", "abnormal"]
    assert [row["metadata"]["defect_bucket"] for row in updated] == ["normal", "normal", "unspecified", "unspecified"]
    assert [row["metadata"]["defect_status"] for row in rows] == ["abnormal"] * 4
    with pytest.raises(FileExistsError):
        build_v2(source, out)


def test_error_breakdown_uses_eligible_denominators():
    frame = pd.DataFrame([
        {"model": "base", "dataset": "industrial_zh", "quality_score": 20,
         "industrial_reference_status": "usable", "industrial_critical_errors": "missed_defect,wrong_object",
         "meta.defect_status": "abnormal", "meta.defect_bucket": "unspecified"},
        {"model": "base", "dataset": "industrial_zh", "quality_score": 50,
         "industrial_reference_status": "usable", "industrial_critical_errors": "false_alarm",
         "meta.defect_status": "normal", "meta.defect_bucket": "normal"},
        {"model": "base", "dataset": "industrial_zh", "quality_score": None,
         "industrial_reference_status": "conflict", "industrial_critical_errors": "missed_defect",
         "meta.defect_status": "abnormal", "meta.defect_bucket": "unspecified"},
    ])
    report = summarize_industrial_errors(frame)
    overall = report[report.dim.eq("overall")].set_index("flag")
    assert overall.loc["missed_defect", "n_eligible"] == 1
    assert overall.loc["missed_defect", "n_flagged"] == 1
    assert overall.loc["false_alarm", "n_eligible"] == 1
    assert overall.loc["wrong_object", "n_eligible"] == 2
