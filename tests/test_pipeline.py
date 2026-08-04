import json

import pytest

from eval_tool.config import load_pipeline_config
from eval_tool.pipeline import PipelineError, run_all, run_inference


def _pipeline_config(tmp_path, *, convert_input=True, models=None, datasets=None):
    raw = {
        "tsv_dir": "tsv",
        "datasets": datasets or {"vqa": "aero_vqa"},
        "work_dir": "work",
        "out_dir": "report",
        "cache_dir": "cache",
        "models": models
        or [
            {"name": "base", "model_path": "models/base"},
            {"name": "sft", "model_path": "models/sft"},
        ],
        "baseline_model": "base",
        "infer": {},
    }
    if convert_input:
        raw["convert"] = {"input_json": "input.json"}
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_pipeline_config(path)


def test_run_all_orders_convert_infer_eval(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        "eval_tool.pipeline.convert_vqa",
        lambda source, destination: calls.append(("convert", source, destination)),
    )
    monkeypatch.setattr(
        "eval_tool.pipeline.run_infer",
        lambda infer_config, generator=None: calls.append(
            ("infer", infer_config.model_name, generator)
        )
        or {},
    )
    monkeypatch.setattr(
        "eval_tool.pipeline.run_eval",
        lambda eval_config: calls.append(
            ("eval", [model.name for model in eval_config.models])
        )
        or {},
    )

    run_all(
        config,
        generator_factory=lambda model_name: f"generator:{model_name}",
    )

    assert [call[0] for call in calls] == ["convert", "infer", "infer", "eval"]
    assert calls[1][2] == "generator:base"
    assert calls[2][2] == "generator:sft"


def test_run_all_can_finish_with_selected_rubric(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        "eval_tool.pipeline.run_conversion",
        lambda pipeline: calls.append("convert"),
    )
    monkeypatch.setattr(
        "eval_tool.pipeline.run_inference",
        lambda *args, **kwargs: calls.append("infer") or {},
    )
    monkeypatch.setattr(
        "eval_tool.pipeline.run_rubric_evaluation",
        lambda pipeline, version, model_names: calls.append(
            ("rubric", version, model_names)
        )
        or {"report": tmp_path / "report_v4"},
    )

    result = run_all(config, model_names=["base"], rubric="v4")

    assert calls == ["convert", "infer", ("rubric", "v4", ["base"])]
    assert result["eval"] == {"report": tmp_path / "report_v4"}


def test_inference_stops_after_first_failed_model(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    calls = []

    def fail_first(infer_config, generator=None):
        calls.append(infer_config.model_name)
        raise RuntimeError("stop")

    monkeypatch.setattr("eval_tool.pipeline.run_infer", fail_first)

    with pytest.raises(RuntimeError, match="stop"):
        run_inference(config)
    assert calls == ["base"]


def test_all_without_convert_input_accepts_existing_tsv(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path, convert_input=False)
    config.tsv_dir.mkdir()
    (config.tsv_dir / "aero_vqa.tsv").write_text("index\tquestion\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "eval_tool.pipeline.run_inference",
        lambda *args, **kwargs: calls.append("infer") or {},
    )
    monkeypatch.setattr(
        "eval_tool.pipeline.run_evaluation",
        lambda *args, **kwargs: calls.append("eval") or {},
    )

    run_all(config)

    assert calls == ["infer", "eval"]


def test_all_without_convert_input_rejects_missing_tsv(tmp_path):
    config = _pipeline_config(tmp_path, convert_input=False)

    with pytest.raises(PipelineError, match="missing TSV.*aero_vqa"):
        run_all(config)


def test_external_pred_and_scored_datasets_are_not_inferred(tmp_path, monkeypatch):
    config = _pipeline_config(
        tmp_path,
        datasets={"mcq": "aero_mcq", "judge": "aero_judge", "vqa": "aero_vqa"},
        models=[
            {
                "name": "base",
                "model_path": "models/base",
                "pred": {"mcq": "external/mcq.xlsx"},
                "scored": {"vqa": "scored/vqa.xlsx"},
            }
        ],
    )
    seen = []
    monkeypatch.setattr(
        "eval_tool.pipeline.run_infer",
        lambda infer_config, generator=None: seen.append(infer_config.datasets) or {},
    )

    run_inference(config)

    assert seen == [{"judge": "aero_judge"}]
