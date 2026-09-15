"""Industrial evaluation stays in its own profile and uses the existing text judge."""

import json
from pathlib import Path

import pandas as pd
import pytest

from eval_tool.aggregate import make_weighted_score_summary
from eval_tool.config import load_pipeline_config


ROOT = Path(__file__).resolve().parents[1]


def test_industrial_example_loads_two_isolated_pointwise_datasets():
    config = load_pipeline_config(ROOT / "industrial.example.json")
    expected = {"industrial_zh", "industrial_en"}
    assert set(config.datasets) == expected
    assert set(config.enabled_datasets) == expected
    assert config.dataset_kinds == {key: "industrial_describe" for key in expected}
    assert not config.do_pairwise
    assert not config.do_length_control
    assert not config.derive_plans
    assert config.judge.pointwise_prompt == (
        ROOT / "prompts/judge_industrial_pointwise_v1.txt"
    ).read_text(encoding="utf-8").strip()
    for key in expected:
        assert config.dataset_params[key]["category_field"] == "object_category"
        assert config.infer.prompt_files[key].name == "infer_vqa_raw.txt"
    assert config.dataset_weights == {}
    for path in (config.work_dir, config.out_dir, config.cache_dir):
        assert "industrial" in path.parts


def test_industrial_dimensions_are_metadata_slices_with_sample_size_gate():
    config = load_pipeline_config(ROOT / "industrial.example.json")
    dimensions = {"object_category", "source_domain", "defect_type"}
    assert config.report_dims[0]["key"] == "object_category"
    assert {item["key"] for item in config.report_dims} == dimensions
    assert all(item["from"] == f"meta.{item['key']}" for item in config.report_dims)
    assert all(item["min_n"] == 30 for item in config.report_dims)
    assert config.min_category_n == 30


def test_industrial_categories_do_not_need_legacy_weight_names():
    config = load_pipeline_config(ROOT / "industrial.example.json")
    details = pd.DataFrame(
        [{"model": "base", "category": "bottle", "hit": 1}] * 30
        + [{"model": "base", "category": "cable", "hit": 0}] * 30
        + [{"model": "base", "category": "rare", "hit": 0}]
    )
    row = make_weighted_score_summary(
        details, config.category_weights, config.min_category_n
    ).iloc[0]
    assert row["total_score"] == pytest.approx(0.5)
    assert row["total_score_by_n"] == pytest.approx(30 / 61, abs=0.0001)
    assert row["categories_excluded_small_n"] == "rare(n=1)"


def test_industrial_example_keeps_scoring_rules_in_profile():
    raw = json.loads((ROOT / "industrial.example.json").read_text(encoding="utf-8"))
    assert not {"datasets", "enabled_datasets", "report", "derive", "eval_set"} & raw.keys()
    assert "api_key" not in raw["judge"]
