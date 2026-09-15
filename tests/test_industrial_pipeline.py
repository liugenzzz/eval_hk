"""Exercise bilingual pointwise reports with a mocked judge and real image IO."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from eval_tool.config import ModelConfig, load_pipeline_config
from eval_tool.industrial_report import summarize_industrial
from eval_tool.industrial_rubric import INDUSTRIAL_COLUMNS, INDUSTRIAL_PROMPT_MARKER
from eval_tool.judge import JudgeClient
from eval_tool.run_eval import run


def test_bilingual_pointwise_runs_without_pairwise_and_preserves_categories(tmp_path, monkeypatch):
    config = load_pipeline_config(Path(__file__).resolve().parents[1] / "industrial.example.json").to_eval_config()
    Image.new("RGB", (8, 8), "white").save(tmp_path / "fixture.png")
    predictions = {}
    for language in ("zh", "en"):
        key = f"industrial_{language}"
        records = [{
            "id": str(i), "images": ["fixture.png"],
            "messages": [{"role": "user", "content": f"<image>{language} fixture_case_{i} describe"},
                         {"role": "assistant", "content": "reference"}],
            "metadata": {"object_category": category, "defect_type": "crack"},
        } for i, category in enumerate(("bottle", "cable", "bottle"))]
        (tmp_path / f"{key}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8")
        predictions[key] = str(tmp_path / f"pred_{key}.csv")
        pd.DataFrame([{"index": f"{i}__t1", "prediction": "candidate"} for i in range(3)]).to_csv(
            predictions[key], index=False)
    calls = []

    def post(self, system_prompt, user_text, image_b64=None, **kwargs):
        assert INDUSTRIAL_PROMPT_MARKER in system_prompt
        assert image_b64
        calls.append(user_text)
        score = 80 if "zh fixture_case" in user_text else 90
        data = {"rubric": "industrial_v1", "reference_status": "usable",
                "object_score": score, "defect_score": score, "appearance_score": score,
                "location_score": score, "fabrication_score": 100, "style_score": 90,
                "critical_errors": [], "reasoning": "fixture"}
        if "fixture_case_1" in user_text:
            # A normal sample does not need an invented defect shape or location.
            data.update(appearance_score=None, location_score=None)
        if "fixture_case_2" in user_text:
            data.update(reference_status="conflict", reasoning="reference conflicts with image")
            for name in ("object_score", "defect_score", "appearance_score", "location_score",
                         "fabrication_score", "style_score"):
                data[name] = None
        return json.dumps(data)

    def forbidden(*args, **kwargs):
        raise AssertionError("industrial evaluation must never call pairwise")

    monkeypatch.setattr(JudgeClient, "_post", post)
    monkeypatch.setattr(JudgeClient, "judge_pairwise", forbidden)
    config = replace(config, tsv_dir=tmp_path, out_dir=tmp_path / "report",
                     cache_dir=tmp_path / "cache", bootstrap_n=20, min_category_n=1,
                     models=[ModelConfig(name=name, paths=predictions) for name in ("base", "sft")],
                     dataset_params={key: {**params, "image_root": str(tmp_path)}
                                     for key, params in config.dataset_params.items()},
                     report_dims=[{**dim, "min_n": 1} for dim in config.report_dims])
    written = run(config)
    assert len(calls) == 12
    assert not any("pairwise" in name for name in written)
    breakdown = pd.read_csv(written["breakdown.csv"])
    assert set(breakdown[breakdown.dim == "object_category"].value) == {"bottle", "cable"}
    combined = pd.read_excel(written["judge_detail_all.xlsx"])
    assert set(INDUSTRIAL_COLUMNS) <= set(combined.columns)
    assert set(combined.rubric) == {"industrial_v1"}
    assert len(combined) == 12
    conflicts = combined[combined.industrial_reference_status == "conflict"]
    assert len(conflicts) == 4
    assert conflicts[["quality_score", "accuracy_score", "hit", "object_score", "defect_score",
                      "appearance_score", "location_score", "fabrication_score", "style_score"]].isna().all().all()
    normal = combined[combined["index"] == "1__t1"]
    assert len(normal) == 4
    assert normal[["appearance_score", "location_score"]].isna().all().all()
    assert normal.defect_score.notna().all()
    for model in ("base", "sft"):
        for language in ("zh", "en"):
            detail = pd.read_excel(written[f"detail_{model}_industrial_{language}.xlsx"])
            assert set(INDUSTRIAL_COLUMNS) <= set(detail.columns)
            assert set(detail.industrial_reference_status) == {"usable", "conflict"}
            assert detail.object_score.notna().sum() == 2
    cache_path = tmp_path / "cache" / "judge_cache_pointwise.jsonl"
    cached = [json.loads(line)["value"] for line in cache_path.read_text(encoding="utf-8").splitlines()]
    assert len(cached) == 12
    assert all(set(INDUSTRIAL_COLUMNS) <= set(row) for row in cached)
    assert sum(row["industrial_reference_status"] == "conflict" for row in cached) == 4
    assert all(row["quality_score"] is None for row in cached if row["industrial_reference_status"] == "conflict")
    summary = summarize_industrial(combined, min_n=1, bootstrap_n=20)
    overall = summary[(summary.dim == "overall") & (summary.metric == "quality_score")]
    assert set(overall[overall.dataset == "industrial_zh"].score) == {80}
    assert set(overall[overall.dataset == "industrial_en"].score) == {90}
    assert "industrial_quality_scores.csv" in written
    assert "industrial_judge_status.csv" in written
    quality_report = pd.read_csv(written["industrial_quality_scores.csv"])
    assert {"object_score", "defect_score", "appearance_score", "location_score"} <= set(quality_report.metric)
    assert set(quality_report.dataset) == {"industrial_zh", "industrial_en"}
    assert not pd.read_csv(written["industrial_judge_status.csv"]).empty

    # Resume must use the saved parsed verdicts, including evidence exclusions.
    monkeypatch.setattr(JudgeClient, "_post", forbidden)
    second = run(config)
    assert len(calls) == 12
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 12
    pd.testing.assert_frame_equal(combined, pd.read_excel(second["judge_detail_all.xlsx"]))
