import json

import pytest

from eval_tool.artifacts import ArtifactLayout
from eval_tool.config import ConfigError, is_pipeline_config, load_pipeline_config


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
