# Unified Configuration and CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce one pipeline configuration, deterministic artifact paths, multi-model orchestration, and `convert/infer/eval/all` subcommands while retaining every legacy entry point and config format.

**Architecture:** Parse the new schema into explicit pipeline dataclasses and adapt them into the existing `InferConfig`/`EvalConfig` execution APIs. Keep path policy in `ArtifactLayout`, workflow logic in `pipeline.py`, and argument parsing in `cli.py`. Detect legacy invocations before subcommand parsing so `python -m eval_tool --config old.json` remains unchanged.

**Tech Stack:** Python 3.12, dataclasses, argparse, pathlib/json, pandas, pytest and monkeypatch.

---

## File map

- Create `eval_tool/artifacts.py`: one source of truth for model, partial, manifest, and rubric paths.
- Create `eval_tool/pipeline.py`: model filtering and convert/infer/eval/all orchestration.
- Create `eval_tool/cli.py`: subcommand parser and legacy pre-routing.
- Create `tests/test_pipeline_config.py`: new schema and config adaptation.
- Create `tests/test_pipeline.py`: workflow and multi-model integration.
- Create `tests/test_cli.py`: command parsing and compatibility matrix.
- Create `pipeline.example.json`: complete unified configuration.
- Modify `eval_tool/config.py`: pipeline dataclasses, loader, shared judge parsing, relative model paths.
- Modify `eval_tool/__main__.py`: delegate to the new CLI.
- Modify `eval_tool/run_infer.py`: accept new resume fields parsed from pipeline configs.
- Modify `tests/test_run_eval.py`: refresh the stale one-row scored-path fixture for the 30-row aggregation gate.

### Task 1: Centralize artifact paths

**Files:**
- Create: `eval_tool/artifacts.py`
- Create: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write failing path-layout tests**

```python
def test_artifact_layout_derives_every_pipeline_path(tmp_path):
    layout = ArtifactLayout(work_dir=tmp_path / "work", out_dir=tmp_path / "report")
    assert layout.model_dir("base") == tmp_path / "work" / "base"
    assert layout.prediction("base", "aero_vqa") == tmp_path / "work" / "base" / "base_aero_vqa.xlsx"
    assert layout.partial_dir("base") == tmp_path / "work" / "base" / "_partial"
    assert layout.manifest("base", "aero_vqa") == tmp_path / "work" / "base" / "base_aero_vqa.infer.json"
    assert layout.rubric_out("v4") == tmp_path / "report_v4"
```

- [ ] **Step 2: Run the path test and confirm RED**

Run: `python -m pytest tests/test_pipeline_config.py::test_artifact_layout_derives_every_pipeline_path -q --basetemp=.pytest_cache/plan2_task1`

Expected: FAIL because `eval_tool.artifacts.ArtifactLayout` does not exist.

- [ ] **Step 3: Implement the immutable layout**

```python
@dataclass(frozen=True)
class ArtifactLayout:
    work_dir: Path
    out_dir: Path

    def model_dir(self, model_name: str) -> Path:
        return self.work_dir / model_name

    def prediction(self, model_name: str, dataset_name: str) -> Path:
        return self.model_dir(model_name) / f"{model_name}_{dataset_name}.xlsx"

    def partial_dir(self, model_name: str) -> Path:
        return self.model_dir(model_name) / "_partial"

    def manifest(self, model_name: str, dataset_name: str) -> Path:
        return self.model_dir(model_name) / f"{model_name}_{dataset_name}.infer.json"

    def rubric_out(self, rubric: str) -> Path:
        return self.out_dir.with_name(f"{self.out_dir.name}_{rubric}")
```

- [ ] **Step 4: Run the path test and confirm GREEN**

Run: `python -m pytest tests/test_pipeline_config.py::test_artifact_layout_derives_every_pipeline_path -q --basetemp=.pytest_cache/plan2_task1`

Expected: PASS.

- [ ] **Step 5: Commit the path policy**

```bash
git add eval_tool/artifacts.py tests/test_pipeline_config.py
git commit -m "feat: centralize pipeline artifact paths"
```

### Task 2: Parse and validate the pipeline schema

**Files:**
- Modify: `eval_tool/config.py`
- Modify: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write failing schema and validation tests**

```python
def test_load_pipeline_config_resolves_shared_and_model_paths(tmp_path):
    path = write_pipeline(tmp_path, models=[{
        "name": "base", "model_path": "models/base",
        "pred": {"mcq": "external/base.csv"},
        "scored": {"vqa": "scored/base.xlsx"},
    }])
    config = load_pipeline_config(path)
    model = config.models[0]
    assert config.tsv_dir == (tmp_path / "tsv").resolve()
    assert model.model_path == (tmp_path / "models/base").resolve()
    assert model.pred_paths["mcq"] == (tmp_path / "external/base.csv").resolve()
    assert model.scored_paths["vqa"] == (tmp_path / "scored/base.xlsx").resolve()


@pytest.mark.parametrize("models,message", [
    ([{"name": "base", "model_path": "a"}, {"name": "base", "model_path": "b"}], "duplicate model name"),
    ([{"name": "", "model_path": "a"}], "models\[0\].name"),
])
def test_pipeline_rejects_invalid_models(tmp_path, models, message):
    with pytest.raises(ConfigError, match=message):
        load_pipeline_config(write_pipeline(tmp_path, models=models))
```

Also cover absent `model_path` when every enabled dataset is not externally supplied, unknown dataset overrides, missing baseline, `convert.input_json`, and Python/JSON legacy loaders remaining unchanged.

- [ ] **Step 2: Run config tests and confirm RED**

Run: `python -m pytest tests/test_pipeline_config.py tests/test_infer_config.py -q --basetemp=.pytest_cache/plan2_task2`

Expected: FAIL because pipeline dataclasses and loader are absent; legacy tests still pass.

- [ ] **Step 3: Add explicit dataclasses and a dedicated loader**

```python
@dataclass(frozen=True)
class PipelineModelConfig:
    name: str
    model_path: Path | None
    pred_paths: dict[str, Path] = field(default_factory=dict)
    scored_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineInferSettings:
    prompt_files: dict[str, Path] = field(default_factory=dict)
    max_new_tokens: int = 512
    batch_size: int = 1
    limit: int | None = None
    torch_dtype: str = "auto"
    device_map: str = "auto"
    gpu_ids: list[int] = field(default_factory=list)
    workers_per_gpu: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    tsv_dir: Path
    work_dir: Path
    out_dir: Path
    cache_dir: Path
    datasets: dict[str, str]
    models: list[PipelineModelConfig]
    baseline_model: str
    infer: PipelineInferSettings
    judge: JudgeSettings
    convert_input: Path | None = None
```

Add `max_workers=8`, `do_pointwise=True`, `do_pairwise=True`, `do_length_control=True`, `mcq_llm_extract_fallback=False`, `bootstrap_n=1000`, `seed=42`, `enabled_datasets`, and `category_weights` to `PipelineConfig`, matching the existing `EvalConfig` types and defaults. Implement `load_pipeline_config()` separately from both legacy loaders, share only small parsing helpers, and expose `is_pipeline_config(path)` using top-level `infer` plus at least one `models[].model_path/pred`. Resolve all new relative paths against the config directory.

- [ ] **Step 4: Run new and legacy config tests and confirm GREEN**

Run: `python -m pytest tests/test_pipeline_config.py tests/test_infer_config.py -q --basetemp=.pytest_cache/plan2_task2`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit schema parsing**

```bash
git add eval_tool/config.py tests/test_pipeline_config.py
git commit -m "feat: parse unified pipeline configuration"
```

### Task 3: Derive legacy execution configs with correct precedence

**Files:**
- Modify: `eval_tool/config.py`
- Modify: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write failing derivation and filtering tests**

```python
def test_eval_derivation_uses_scored_then_pred_then_convention(pipeline_config):
    derived = pipeline_config.to_eval_config(["base", "sft"])
    base, sft = derived.models
    assert base.scored_path_for("vqa").endswith("detail_base_vqa.xlsx")
    assert base.path_for("mcq").endswith("external_base_mcq.csv")
    assert sft.path_for("vqa").endswith("work/sft/sft_aero_vqa.xlsx")


def test_infer_derivation_skips_external_sources(pipeline_config):
    configs = pipeline_config.to_infer_configs(["base"])
    assert configs[0].datasets == {"judge": "aero_judge"}
    assert configs[0].resume is True
    assert configs[0].out_dir.name == "base"


def test_filter_without_baseline_disables_pairwise(pipeline_config):
    derived = pipeline_config.to_eval_config(["sft"])
    assert derived.do_pairwise is False
```

Add unknown model, repeated filter name, stable requested order, all-datasets-external model, and baseline-present pairwise cases.

- [ ] **Step 2: Run focused derivation tests and confirm RED**

Run: `python -m pytest tests/test_pipeline_config.py -k 'derivation or filter' -q --basetemp=.pytest_cache/plan2_task3`

Expected: FAIL because adaptation methods do not exist.

- [ ] **Step 3: Implement selection and adaptation methods**

```python
def select_models(self, names=None):
    by_name = {model.name: model for model in self.models}
    requested = list(by_name) if names is None else list(names)
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ConfigError(f"unknown models: {','.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ConfigError("duplicate names in --models")
    return [by_name[name] for name in requested]
```

`to_infer_configs()` must omit each dataset supplied by either `scored` or `pred`, set `out_dir=ArtifactLayout.model_dir(name)`, and skip models with no remaining datasets. `to_eval_config()` must materialize `ModelConfig(paths=prediction_paths, scored_paths=scored_paths)` using `scored > pred > layout.prediction`; when a filtered set omits baseline, set `do_pairwise=False` and keep pointwise evaluation valid.

- [ ] **Step 4: Run all pipeline config tests and confirm GREEN**

Run: `python -m pytest tests/test_pipeline_config.py -q --basetemp=.pytest_cache/plan2_task3`

Expected: all tests PASS and derived final prediction paths exactly match inference outputs.

- [ ] **Step 5: Commit adapters**

```bash
git add eval_tool/config.py tests/test_pipeline_config.py
git commit -m "feat: derive inference and evaluation configs"
```

### Task 4: Orchestrate convert, multi-model infer, eval, and all

**Files:**
- Create: `eval_tool/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing workflow tests with injected stage functions**

```python
def test_run_all_orders_convert_infer_eval(pipeline_config, monkeypatch):
    calls = []
    monkeypatch.setattr("eval_tool.pipeline.convert_vqa", lambda src, dst: calls.append(("convert", src, dst)))
    monkeypatch.setattr("eval_tool.pipeline.run_infer", lambda cfg, generator=None: calls.append(("infer", cfg.model_name)))
    monkeypatch.setattr("eval_tool.pipeline.run_eval", lambda cfg: calls.append(("eval", [m.name for m in cfg.models])) or {})
    run_all(pipeline_config, generator_factory=lambda model_name: FakeGenerator(model_name))
    assert [call[0] for call in calls] == ["convert", "infer", "infer", "eval"]


def test_infer_stops_after_first_failed_model(pipeline_config, monkeypatch):
    calls = []
    def fake_run(config, generator=None):
        calls.append(config.model_name)
        raise RuntimeError("stop")
    monkeypatch.setattr("eval_tool.pipeline.run_infer", fake_run)
    with pytest.raises(RuntimeError, match="stop"):
        run_inference(pipeline_config)
    assert calls == ["base"]
```

Also test no `convert.input_json` with an existing TSV continues, with a missing TSV fails, and `pred/scored` datasets are never sent to inference.

- [ ] **Step 2: Run workflow tests and confirm RED**

Run: `python -m pytest tests/test_pipeline.py -q --basetemp=.pytest_cache/plan2_task4`

Expected: collection FAIL because `eval_tool.pipeline` does not exist.

- [ ] **Step 3: Implement small orchestration functions**

```python
def run_inference(config, model_names=None, generator_factory=None,
                  overwrite=False, clean_partial=False):
    written = {}
    for infer_config in config.to_infer_configs(model_names,
                                                overwrite=overwrite,
                                                clean_partial=clean_partial):
        generator = None if generator_factory is None else generator_factory(infer_config.model_name)
        written[infer_config.model_name] = run_infer(infer_config, generator=generator)
    return written


def run_all(config, model_names=None, generator_factory=None,
            overwrite=False, clean_partial=False):
    if config.convert_input is not None:
        convert_vqa(config.convert_input, config.tsv_dir)
    else:
        for dataset_name in config.datasets.values():
            if not (config.tsv_dir / f"{dataset_name}.tsv").exists():
                raise PipelineError(f"missing TSV and convert.input_json is unset: {dataset_name}")
    inferred = run_inference(config, model_names, generator_factory, overwrite, clean_partial)
    evaluated = run_evaluation(config, model_names)
    return {"infer": inferred, "eval": evaluated}
```

Keep functions synchronous, model order deterministic, and fail fast while preserving stage artifacts already written to disk.

- [ ] **Step 4: Run orchestration tests and confirm GREEN**

Run: `python -m pytest tests/test_pipeline.py tests/test_pipeline_config.py -q --basetemp=.pytest_cache/plan2_task4`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit orchestration**

```bash
git add eval_tool/pipeline.py tests/test_pipeline.py
git commit -m "feat: orchestrate unified evaluation pipeline"
```

### Task 5: Add subcommands and preserve the legacy compatibility matrix

**Files:**
- Create: `eval_tool/cli.py`
- Create: `tests/test_cli.py`
- Modify: `eval_tool/__main__.py`

- [ ] **Step 1: Write failing parser and routing tests**

```python
@pytest.mark.parametrize("argv,handler", [
    (["convert", "input.json", "--config", "pipeline.json"], "convert"),
    (["infer", "--config", "pipeline.json", "--models", "base,sft"], "infer"),
    (["eval", "--config", "pipeline.json"], "eval"),
    (["all", "--config", "pipeline.json"], "all"),
])
def test_cli_routes_subcommands(argv, handler, monkeypatch):
    calls = []
    monkeypatch.setattr("eval_tool.cli.HANDLERS", {handler: lambda args: calls.append(handler)})
    main(argv)
    assert calls == [handler]


def test_no_subcommand_routes_to_legacy_eval(monkeypatch):
    calls = []
    monkeypatch.setattr("eval_tool.cli.legacy_eval_main", lambda argv=None: calls.append(argv))
    main(["--config", "old.json"])
    assert calls == [["--config", "old.json"]]
```

Add tests for new `infer/eval` with their matching old schemas, comma parsing, `--overwrite`, `--clean-partial`, and unknown model errors rendered through `parser.error`.

- [ ] **Step 2: Run CLI tests and confirm RED**

Run: `python -m pytest tests/test_cli.py -q --basetemp=.pytest_cache/plan2_task5`

Expected: collection FAIL because `eval_tool.cli` does not exist.

- [ ] **Step 3: Implement `main(argv=None)` and legacy pre-routing**

```python
SUBCOMMANDS = {"convert", "infer", "eval", "sweep", "all"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS:
        return legacy_eval_main(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigError, PipelineError) as exc:
        parser.error(str(exc))
```

For `infer` and `eval`, call `is_pipeline_config()` and dispatch either through the new orchestrator or the matching old loader/stage function. `convert` defaults `--config` to `pipeline.json` so the documented short command works in a configured directory. Replace `__main__.py` with `from .cli import main` plus the standard guard.

- [ ] **Step 4: Run CLI and legacy entry tests and confirm GREEN**

Run: `python -m pytest tests/test_cli.py tests/test_run_eval.py tests/test_run_infer.py tests/test_infer_config.py -q --basetemp=.pytest_cache/plan2_task5`

Expected: all selected tests PASS, except no pre-existing failure is hidden or reclassified.

- [ ] **Step 5: Commit the CLI**

```bash
git add eval_tool/cli.py eval_tool/__main__.py tests/test_cli.py
git commit -m "feat: add unified pipeline subcommands"
```

### Task 6: Refresh the stale scored-detail fixture and add the example config

**Files:**
- Modify: `tests/test_run_eval.py`
- Create: `pipeline.example.json`

- [ ] **Step 1: Reproduce the stale-fixture failure in isolation**

Run: `python -m pytest tests/test_run_eval.py::test_run_eval_reuses_previously_scored_vqa_results_without_calling_judge -vv --basetemp=.pytest_cache/plan2_task6`

Expected: FAIL because the one-row category is correctly excluded by `MIN_CATEGORY_N = 30`, making `score_summary.csv.total_score` `NaN`.

- [ ] **Step 2: Expand both truth and scored fixtures to 30 aligned rows**

```python
rows = [
    {
        "index": str(index),
        "image": "img",
        "question": f"描述这张图 {index}",
        "answer": "参考答案",
        "category": "单轮",
        "l2-category": "",
        "source_id": f"s{index}",
    }
    for index in range(30)
]
vqa_truth = pd.DataFrame(rows)
scored = pd.DataFrame([
    {
        "index": row["index"],
        "question": row["question"],
        "answer": row["answer"],
        "prediction": "旧结果",
        "category": "单轮",
        "l2-category": "",
        "hit": 1,
        "judge_reason": "reused",
        "pred_len": 3,
    }
    for row in rows
])
```

Keep the judge monkeypatch that raises if called, so the test still proves `scored` reuse while satisfying the intentional sample-size gate.

- [ ] **Step 3: Run the focused test without modifying aggregation or evaluation code**

```python
python -m pytest tests/test_run_eval.py::test_run_eval_reuses_previously_scored_vqa_results_without_calling_judge -q --basetemp=.pytest_cache/plan2_task6
```

Expected: PASS with `total_score == 1.0`; `eval_tool/aggregate.py` and `eval_tool/run_eval.py` remain unchanged in this task.

- [ ] **Step 4: Add a complete, non-secret pipeline example and run tests**

Create `pipeline.example.json` with `convert.input_json`, two models, shared datasets, work/report/cache directories, inference settings, judge settings, and valid existing prompt filenames. Use placeholder local paths and `sk-local`, not real credentials.

Run: `python -m pytest tests/test_run_eval.py tests/test_pipeline_config.py tests/test_pipeline.py tests/test_cli.py -q --basetemp=.pytest_cache/plan2_task6`

Expected: all selected tests PASS and the example parses through `load_pipeline_config()`.

- [ ] **Step 5: Commit compatibility fixes and example**

```bash
git add tests/test_run_eval.py pipeline.example.json
git commit -m "test: refresh scored reuse fixture"
```
