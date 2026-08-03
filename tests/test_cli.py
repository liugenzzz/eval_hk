import json

import pytest

from eval_tool.cli import main


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["convert", "input.json", "--config", "pipeline.json"], "convert"),
        (["infer", "--config", "pipeline.json", "--models", "base,sft"], "infer"),
        (["eval", "--config", "pipeline.json"], "eval"),
        (["all", "--config", "pipeline.json"], "all"),
    ],
)
def test_cli_routes_subcommands(argv, handler, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "eval_tool.cli.HANDLERS",
        {handler: lambda args: calls.append((handler, args))},
    )

    main(argv)

    assert [call[0] for call in calls] == [handler]


def test_no_subcommand_routes_to_legacy_eval(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "eval_tool.cli.legacy_eval_main", lambda argv=None: calls.append(argv)
    )

    main(["--config", "old.json"])

    assert calls == [["--config", "old.json"]]


def test_infer_parser_splits_models_and_reads_safety_flags(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "eval_tool.cli.HANDLERS", {"infer": lambda args: captured.append(args)}
    )

    main(
        [
            "infer",
            "--config",
            "pipeline.json",
            "--models",
            "base,sft",
            "--overwrite",
            "--clean-partial",
        ]
    )

    assert captured[0].models == ["base", "sft"]
    assert captured[0].overwrite is True
    assert captured[0].clean_partial is True


def test_infer_subcommand_accepts_legacy_infer_schema(tmp_path, monkeypatch):
    config_path = tmp_path / "infer.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "base",
                "model_path": "model",
                "tsv_dir": "tsv",
                "out_dir": "out",
                "datasets": {"vqa": "aero_vqa"},
                "prompt_files": {},
            }
        ),
        encoding="utf-8",
    )
    seen = []
    monkeypatch.setattr(
        "eval_tool.cli.run_infer_stage",
        lambda config, generator=None: seen.append(config) or {},
    )

    main(["infer", "--config", str(config_path), "--overwrite"])

    assert seen[0].model_name == "base"
    assert seen[0].overwrite is True


def test_eval_subcommand_accepts_legacy_eval_schema(tmp_path, monkeypatch):
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        json.dumps(
            {
                "tsv_dir": "tsv",
                "out_dir": "out",
                "cache_dir": "cache",
                "datasets": {"vqa": "aero_vqa"},
                "models": [{"name": "base", "vqa": "base.xlsx"}],
                "baseline_model": "base",
            }
        ),
        encoding="utf-8",
    )
    seen = []
    monkeypatch.setattr(
        "eval_tool.cli.run_eval_stage", lambda config: seen.append(config) or {}
    )

    main(["eval", "--config", str(config_path)])

    assert [model.name for model in seen[0].models] == ["base"]


def test_unknown_pipeline_model_is_rendered_as_parser_error(tmp_path, capsys):
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "tsv_dir": "tsv",
                "datasets": {"vqa": "aero_vqa"},
                "work_dir": "work",
                "out_dir": "out",
                "cache_dir": "cache",
                "models": [{"name": "base", "model_path": "model"}],
                "baseline_model": "base",
                "infer": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        main(["infer", "--config", str(config_path), "--models", "missing"])

    assert "unknown models: missing" in capsys.readouterr().err
