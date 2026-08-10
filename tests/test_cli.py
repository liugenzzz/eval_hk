import json

import pytest

import run_rubric
import eval_tool.cli as cli
from eval_tool.cli import main


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["convert", "input.json", "--config", "pipeline.json"], "convert"),
        (["infer", "--config", "pipeline.json", "--models", "base,sft"], "infer"),
        (["eval", "--config", "pipeline.json"], "eval"),
        (["all", "--config", "pipeline.json"], "all"),
        (["build-dpo", "--config", "dpo.json"], "build-dpo"),
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


def test_build_dpo_routes_loaded_config_to_pipeline(monkeypatch, tmp_path):
    loaded_config = object()
    result = object()
    calls = []
    monkeypatch.setattr(
        cli,
        "load_dpo_config",
        lambda path, **kwargs: calls.append(("load", path, kwargs)) or loaded_config,
    )
    monkeypatch.setattr(
        cli,
        "run_build_dpo",
        lambda config, **kwargs: calls.append(("run", config, kwargs)) or result,
    )
    monkeypatch.setattr(cli.Path, "cwd", lambda: tmp_path)

    actual = main(["build-dpo", "--config", "dpo.json"])

    assert actual is result
    assert calls == [
        (
            "load",
            "dpo.json",
            {"input_overrides": None, "invocation_dir": tmp_path},
        ),
        (
            "run",
            loaded_config,
            {"dry_run": False, "overwrite": False, "clean_partial": False},
        ),
    ]


def test_build_dpo_parser_collects_repeated_inputs_as_replacement(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_dpo_config",
        lambda path, **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(cli, "run_build_dpo", lambda config, **kwargs: None)

    main(
        [
            "build-dpo",
            "--config",
            "dpo.json",
            "--input",
            "first.json",
            "--input",
            "second.jsonl",
        ]
    )

    assert calls[0]["input_overrides"] == ["first.json", "second.jsonl"]


def test_build_dpo_parser_accepts_dry_run_overwrite_and_clean_partial(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "eval_tool.cli.HANDLERS",
        {"build-dpo": lambda args: captured.append(args)},
    )

    main(
        [
            "build-dpo",
            "--config",
            "dpo.json",
            "--dry-run",
            "--overwrite",
            "--clean-partial",
        ]
    )

    assert captured[0].dry_run is True
    assert captured[0].overwrite is True
    assert captured[0].clean_partial is True


def test_build_dpo_config_error_is_rendered_as_parser_error(monkeypatch, capsys):
    from eval_tool.dpo_config import DpoConfigError

    def fail_load(*args, **kwargs):
        raise DpoConfigError("invalid DPO configuration")

    monkeypatch.setattr(cli, "load_dpo_config", fail_load)

    with pytest.raises(SystemExit, match="2"):
        main(["build-dpo", "--config", "bad.json"])

    assert "invalid DPO configuration" in capsys.readouterr().err


def test_build_dpo_pipeline_error_is_rendered_as_parser_error(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dpo_config", lambda *args, **kwargs: object())

    def fail_run(*args, **kwargs):
        raise cli.DpoPipelineError("DPO pipeline failed")

    monkeypatch.setattr(cli, "run_build_dpo", fail_run)

    with pytest.raises(SystemExit, match="2"):
        main(["build-dpo", "--config", "dpo.json"])

    assert "DPO pipeline failed" in capsys.readouterr().err


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


def test_unified_infer_makes_legacy_schema_resume_safe_by_default(
    tmp_path, monkeypatch
):
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

    main(["infer", "--config", str(config_path)])

    assert seen[0].resume is True
    assert seen[0].overwrite is False


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


def test_sweep_cli_parses_rubrics_models_and_statistics_options(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "eval_tool.cli.HANDLERS", {"sweep": lambda args: captured.append(args)}
    )

    main(
        [
            "sweep",
            "--config",
            "pipeline.json",
            "--rubrics",
            "v3,v4",
            "--models",
            "base,sft",
            "--out-root",
            "custom",
            "--no-pairwise",
            "--metric",
            "quality_score",
            "--n-bootstrap",
            "50",
        ]
    )

    assert captured[0].rubrics == ["v3", "v4"]
    assert captured[0].models == ["base", "sft"]
    assert captured[0].out_root == "custom"
    assert captured[0].no_pairwise is True
    assert captured[0].metric == "quality_score"
    assert captured[0].n_bootstrap == 50


def test_all_cli_passes_selected_rubric(monkeypatch):
    calls = []
    monkeypatch.setattr("eval_tool.cli._require_pipeline", lambda path: "pipeline")
    monkeypatch.setattr(
        "eval_tool.cli.run_all",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {},
    )

    main(["all", "--config", "pipeline.json", "--rubric", "v4"])

    assert calls == [
        (
            ("pipeline", None),
            {
                "overwrite": False,
                "clean_partial": False,
                "rubric": "v4",
            },
        )
    ]


def test_eval_rubric_dry_run_derives_without_running(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "tsv_dir": "tsv",
                "datasets": {"vqa": "aero_vqa"},
                "work_dir": "work",
                "out_dir": "reports/eval_report",
                "cache_dir": "cache",
                "models": [{"name": "base", "model_path": "model"}],
                "baseline_model": "base",
                "infer": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "eval_tool.cli.run_eval_stage",
        lambda config: pytest.fail("dry-run must not evaluate"),
    )

    derived = main(
        [
            "eval",
            "--config",
            str(config_path),
            "--rubric",
            "v4",
            "--out-root",
            str(tmp_path / "custom"),
            "--no-pairwise",
            "--dry-run",
        ]
    )

    assert derived.out_dir == tmp_path / "custom" / "eval_report_v4"
    assert derived.do_pairwise is False
    output = capsys.readouterr().out
    assert "fingerprint" in output
    assert "v4" in output


def test_run_rubric_shim_calls_new_cli(monkeypatch):
    calls = []
    monkeypatch.setattr("run_rubric.cli_main", lambda argv: calls.append(argv))

    run_rubric.main(
        ["--config", "old.json", "--rubric", "v4", "--no-pairwise"]
    )

    assert calls == [
        [
            "eval",
            "--config",
            "old.json",
            "--rubric",
            "v4",
            "--no-pairwise",
        ]
    ]


@pytest.mark.parametrize(
    "command", ["convert", "infer", "eval", "sweep", "all", "build-dpo"]
)
def test_each_subcommand_has_help(command):
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0


def test_top_level_help_lists_unified_subcommands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command in ("convert", "infer", "eval", "sweep", "all"):
        assert command in output


def test_build_dpo_help_is_available(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["build-dpo", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for flag in ("--config", "--input", "--dry-run", "--overwrite", "--clean-partial"):
        assert flag in output


def test_top_level_help_lists_build_dpo(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "build-dpo" in capsys.readouterr().out
