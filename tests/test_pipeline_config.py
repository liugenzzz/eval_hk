import json
from pathlib import Path

import pytest

from eval_tool.artifacts import ArtifactLayout
from eval_tool.config import (
    ConfigError,
    is_pipeline_config,
    load_config,
    load_pipeline_config,
)


def _write_pipeline(tmp_path, *, models=None, **overrides):
    raw = {
        "tsv_dir": "tsv",
        "datasets": {"mcq": "aero_mcq", "vqa": "aero_vqa"},
        "work_dir": "work",
        "out_dir": "report",
        "cache_dir": "cache",
        "models": models
        if models is not None
        else [{"name": "base", "model_path": "models/base"}],
        "baseline_model": "base",
        "infer": {"batch_size": 2},
    }
    raw.update(overrides)
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_artifact_layout_derives_every_pipeline_path(tmp_path):
    layout = ArtifactLayout(work_dir=tmp_path / "work", out_dir=tmp_path / "report")

    assert layout.model_dir("base") == tmp_path / "work" / "base"
    assert layout.prediction("base", "aero_vqa") == (
        tmp_path / "work" / "base" / "base_aero_vqa.xlsx"
    )
    assert layout.partial_dir("base") == tmp_path / "work" / "base" / "_partial"
    assert layout.manifest("base", "aero_vqa") == (
        tmp_path / "work" / "base" / "base_aero_vqa.infer.json"
    )
    assert layout.rubric_out("v4") == tmp_path / "report_v4"


def test_load_pipeline_config_resolves_shared_and_model_paths(tmp_path):
    path = _write_pipeline(
        tmp_path,
        models=[
            {
                "name": "base",
                "model_path": "models/base",
                "pred": {"mcq": "external/base.csv"},
                "scored": {"vqa": "scored/base.xlsx"},
            }
        ],
        convert={"input_json": "input/sharegpt.json"},
    )

    config = load_pipeline_config(path)
    model = config.models[0]

    assert config.config_path == path.resolve()
    assert config.tsv_dir == (tmp_path / "tsv").resolve()
    assert config.work_dir == (tmp_path / "work").resolve()
    assert config.out_dir == (tmp_path / "report").resolve()
    assert config.cache_dir == (tmp_path / "cache").resolve()
    assert config.convert_input == (tmp_path / "input/sharegpt.json").resolve()
    assert model.model_path == (tmp_path / "models/base").resolve()
    assert model.pred_paths["mcq"] == (tmp_path / "external/base.csv").resolve()
    assert model.scored_paths["vqa"] == (tmp_path / "scored/base.xlsx").resolve()
    assert is_pipeline_config(path) is True


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (
            [
                {"name": "base", "model_path": "a"},
                {"name": "base", "model_path": "b"},
            ],
            "duplicate model name",
        ),
        ([{"name": "", "model_path": "a"}], r"models\[0\]\.name"),
    ],
)
def test_pipeline_rejects_invalid_model_names(tmp_path, models, message):
    with pytest.raises(ConfigError, match=message):
        load_pipeline_config(_write_pipeline(tmp_path, models=models))


def test_pipeline_rejects_missing_model_path_for_unsupplied_dataset(tmp_path):
    models = [{"name": "base", "pred": {"mcq": "base.csv"}}]

    with pytest.raises(ConfigError, match=r"models\[0\]\.model_path.*vqa"):
        load_pipeline_config(_write_pipeline(tmp_path, models=models))


@pytest.mark.parametrize("field", ["pred", "scored"])
def test_pipeline_rejects_unknown_dataset_override(tmp_path, field):
    models = [
        {
            "name": "base",
            "model_path": "model",
            field: {"unknown": "file.xlsx"},
        }
    ]

    with pytest.raises(ConfigError, match="unknown dataset.*unknown"):
        load_pipeline_config(_write_pipeline(tmp_path, models=models))


def test_pipeline_rejects_missing_baseline_model(tmp_path):
    with pytest.raises(ConfigError, match="baseline_model.*missing"):
        load_pipeline_config(
            _write_pipeline(
                tmp_path,
                models=[{"name": "base", "model_path": "model"}],
                baseline_model="missing",
            )
        )


def test_non_pipeline_legacy_config_is_not_misdetected(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "infer": {"model_name": "base", "model_path": "model"},
                "models": [{"name": "base", "vqa": "prediction.xlsx"}],
            }
        ),
        encoding="utf-8",
    )

    assert is_pipeline_config(path) is False


def test_scored_only_pipeline_is_detected(tmp_path):
    path = _write_pipeline(
        tmp_path,
        models=[
            {
                "name": "base",
                "scored": {
                    "mcq": "scored/base_mcq.xlsx",
                    "vqa": "scored/base_vqa.xlsx",
                },
            }
        ],
    )

    assert is_pipeline_config(path) is True


def _adaptation_config(tmp_path):
    return load_pipeline_config(
        _write_pipeline(
            tmp_path,
            datasets={
                "mcq": "aero_mcq",
                "judge": "aero_judge",
                "vqa": "aero_vqa",
            },
            models=[
                {
                    "name": "base",
                    "model_path": "models/base",
                    "pred": {"mcq": "external/base_mcq.csv"},
                    "scored": {"vqa": "scored/detail_base_vqa.xlsx"},
                },
                {"name": "sft", "model_path": "models/sft"},
            ],
        )
    )


def test_eval_derivation_uses_scored_then_pred_then_convention(tmp_path):
    config = _adaptation_config(tmp_path)

    derived = config.to_eval_config(["base", "sft"])
    base, sft = derived.models

    assert Path(base.scored_path_for("vqa")) == (
        tmp_path / "scored" / "detail_base_vqa.xlsx"
    )
    assert Path(base.path_for("mcq")) == tmp_path / "external" / "base_mcq.csv"
    assert base.path_for("vqa") is None
    assert Path(sft.path_for("vqa")) == (
        tmp_path / "work" / "sft" / "sft_aero_vqa.xlsx"
    )


def test_infer_derivation_skips_external_sources_and_enables_resume(tmp_path):
    config = _adaptation_config(tmp_path)

    configs = config.to_infer_configs(["base"])

    assert len(configs) == 1
    assert configs[0].datasets == {"judge": "aero_judge"}
    assert configs[0].resume is True
    assert configs[0].overwrite is False
    assert configs[0].clean_partial is False
    assert configs[0].out_dir == tmp_path / "work" / "base"


def test_infer_derivation_skips_model_when_all_datasets_are_external(tmp_path):
    config = load_pipeline_config(
        _write_pipeline(
            tmp_path,
            models=[
                {
                    "name": "base",
                    "pred": {"mcq": "mcq.xlsx", "vqa": "vqa.xlsx"},
                }
            ],
        )
    )

    assert config.to_infer_configs() == []


def test_filter_without_baseline_disables_pairwise(tmp_path):
    config = _adaptation_config(tmp_path)

    without_baseline = config.to_eval_config(["sft"])
    with_baseline = config.to_eval_config(["base", "sft"])

    assert without_baseline.do_pairwise is False
    assert without_baseline.do_pointwise is True
    assert with_baseline.do_pairwise is True


def test_model_filter_preserves_requested_order_and_rejects_bad_names(tmp_path):
    config = _adaptation_config(tmp_path)

    assert [model.name for model in config.select_models(["sft", "base"])] == [
        "sft",
        "base",
    ]
    with pytest.raises(ConfigError, match="unknown models: missing"):
        config.select_models(["missing"])
    with pytest.raises(ConfigError, match="duplicate names in --models"):
        config.select_models(["base", "base"])


def test_pipeline_example_is_complete_and_parseable():
    example_path = Path(__file__).parents[1] / "pipeline.example.json"

    config = load_pipeline_config(example_path)

    assert [model.name for model in config.models] == ["base", "sft_ep2"]
    assert config.convert_input is not None
    assert config.infer.max_new_tokens == 1024
    assert config.judge.api_key == "sk-local"
    assert config.to_infer_configs()[0].resume is True


def test_legacy_example_references_existing_prompts():
    example_path = Path(__file__).parents[1] / "config.example.json"

    config = load_config(example_path)

    assert config.judge.pointwise_prompt
    assert config.judge.pairwise_prompt
