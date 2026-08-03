# Rubric Sweep and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run rubric versions in memory, execute multi-rubric sweeps serially, compare every challenger model automatically, retain old scripts as shims, and document the complete unified workflow.

**Architecture:** Keep rubric metadata and immutable `EvalConfig` derivation in `rubrics.py`; let `pipeline.py` own serial execution and artifact collection. Refactor `compare_rubrics.py` into importable DataFrame-producing functions while preserving its CLI. Reject `scored` inputs during sweep because their verdicts cannot be proven to match each rubric.

**Tech Stack:** Python 3.12, dataclasses.replace, pathlib, pandas/numpy, argparse, pytest.

---

## File map

- Create `eval_tool/rubrics.py`: rubric registry, validation, and in-memory config derivation.
- Create `tests/test_rubrics.py`: rubric path, fingerprint, output suffix, and validation tests.
- Create `tests/test_rubric_sweep.py`: serial sweep, scored guard, and comparison artifact tests.
- Modify `eval_tool/pipeline.py`: `run_rubric_eval()` and `run_sweep()`.
- Modify `eval_tool/cli.py`: rubric options and sweep dispatch.
- Modify `compare_rubrics.py`: importable all-challenger comparison APIs.
- Modify `run_rubric.py`: compatibility shim without derived JSON or subprocess.
- Modify `README.md`, `开放问答评估_使用说明.md`, and `e2e_check.py`: unified workflow and offline smoke path.
- Modify `tests/test_cli.py`: rubric/sweep arguments and legacy shim coverage.

### Task 1: Derive rubric configs in memory

**Files:**
- Create: `eval_tool/rubrics.py`
- Create: `tests/test_rubrics.py`

- [ ] **Step 1: Write failing registry and derivation tests**

```python
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v3b", "v4", "v4b"])
def test_rubric_registry_points_to_existing_prompts(version):
    assert rubric_prompt_path(version).is_file()


def test_apply_rubric_changes_prompt_fingerprint_and_output(eval_config):
    derived = apply_rubric(eval_config, "v4")
    assert derived.out_dir.name == f"{eval_config.out_dir.name}_v4"
    assert derived.judge.pointwise_prompt != eval_config.judge.pointwise_prompt
    assert derived.judge.fingerprint != eval_config.judge.fingerprint
    assert derived.judge.pairwise_prompt == eval_config.judge.pairwise_prompt


def test_unknown_rubric_is_rejected(eval_config):
    with pytest.raises(RubricError, match="unknown rubric: bad"):
        apply_rubric(eval_config, "bad")
```

Also test `out_root`, `no_pairwise`, no mutation of the source config, and the exact output suffix.

- [ ] **Step 2: Run rubric tests and confirm RED**

Run: `python -m pytest tests/test_rubrics.py -q --basetemp=.pytest_cache/plan3_task1`

Expected: collection FAIL because `eval_tool.rubrics` does not exist.

- [ ] **Step 3: Implement the registry and immutable derivation**

```python
RUBRIC_PROMPTS = {
    version: Path(__file__).resolve().parent.parent / "prompts" / f"judge_equip_pointwise_{version}.txt"
    for version in ("v1", "v2", "v3", "v3b", "v4", "v4b")
}


def apply_rubric(config, version, out_root=None, no_pairwise=False):
    prompt_path = rubric_prompt_path(version)
    prompt = load_prompt_text(prompt_path)
    judge = replace(config.judge, pointwise_prompt=prompt)
    root = config.out_dir.parent if out_root is None else Path(out_root)
    out_dir = root / f"{config.out_dir.name}_{version}"
    return replace(
        config,
        out_dir=out_dir,
        judge=judge,
        do_pairwise=False if no_pairwise else config.do_pairwise,
    )
```

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `python -m pytest tests/test_rubrics.py -q --basetemp=.pytest_cache/plan3_task1`

Expected: all rubric tests PASS.

- [ ] **Step 5: Commit rubric derivation**

```bash
git add eval_tool/rubrics.py tests/test_rubrics.py
git commit -m "feat: derive rubric evaluations in memory"
```

### Task 2: Compare every challenger model through an importable API

**Files:**
- Modify: `compare_rubrics.py`
- Create: `tests/test_rubric_sweep.py`

- [ ] **Step 1: Write failing multi-challenger comparison tests**

```python
def test_compare_runs_returns_each_rubric_and_challenger():
    runs = {
        "v3": scored_frame(models=("base", "sft1", "sft2")),
        "v4": scored_frame(models=("base", "sft1", "sft2")),
    }
    result = compare_runs(runs, baseline="base", n_bootstrap=50)
    assert set(zip(result["rubric"], result["model"])) == {
        ("v3", "sft1"), ("v3", "sft2"),
        ("v4", "sft1"), ("v4", "sft2"),
    }


def test_compare_runs_keeps_error_rows_for_missing_pairs():
    result = compare_runs({"v4": scored_frame(models=("base",))}, baseline="base")
    assert result.loc[0, "error"] == "only the baseline model is present"
```

Add coverage for metric auto-selection, human agreement columns, and pairwise d-z tests grouped independently by challenger.

- [ ] **Step 2: Run comparison tests and confirm RED**

Run: `python -m pytest tests/test_rubric_sweep.py -k compare -q --basetemp=.pytest_cache/plan3_task2`

Expected: FAIL because `compare_runs()` is absent and `paired_stats()` only examines `others[0]`.

- [ ] **Step 3: Refactor statistics without changing legacy CLI arguments**

```python
def paired_stats(df, baseline, metric, target=None, n_bootstrap=5000, seed=42):
    wide = df.pivot_table(index="index", columns="model", values=metric, aggfunc="first")
    if baseline not in wide.columns:
        return {"error": f"baseline '{baseline}' not in {list(wide.columns)}"}
    candidates = [name for name in wide.columns if name != baseline]
    if target is None:
        if not candidates:
            return {"error": "only the baseline model is present"}
        target = candidates[0]
    if target not in candidates:
        return {"error": f"model '{target}' is not a challenger"}
    pair = wide[[target, baseline]].dropna()
    if len(pair) < 10:
        return {"model": target, "error": f"only {len(pair)} paired rows"}
    a = pair[target].to_numpy(dtype=float)
    b = pair[baseline].to_numpy(dtype=float)
    differences = a - b
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    bootstrap_means = differences[sample_index].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    sd = differences.std(ddof=1)
    return {
        "model": target,
        "n_paired": len(differences),
        f"{baseline}_mean": round(float(b.mean()), 4),
        f"{target}_mean": round(float(a.mean()), 4),
        "delta": round(float(differences.mean()), 4),
        "delta_ci": f"[{low:+.4f}, {high:+.4f}]",
        "significant": bool(low > 0 or high < 0),
        "effect_size_dz": round(float(differences.mean() / sd), 3) if sd > 0 else float("nan"),
        "better/equal/worse": (
            f"{int((differences > 0).sum())}/"
            f"{int((differences == 0).sum())}/"
            f"{int((differences < 0).sum())}"
        ),
        "distinct_values": int(pd.unique(np.concatenate([a, b])).size),
    }
```

Implement `compare_runs(runs, baseline, metric=None, n_bootstrap=5000, seed=42)` by iterating all candidate models for each rubric and returning a DataFrame. Update `dz_pairwise_test()` to add a `model` column and compare rubric pairs separately for each challenger. Make `main()` call these APIs, print the same tables, and retain `LABEL=PATH` parsing.

- [ ] **Step 4: Run comparison tests and the script help smoke test**

Run: `python -m pytest tests/test_rubric_sweep.py -k compare -q --basetemp=.pytest_cache/plan3_task2`

Run: `python compare_rubrics.py --help`

Expected: tests PASS and help exits 0 with existing options intact.

- [ ] **Step 5: Commit reusable comparisons**

```bash
git add compare_rubrics.py tests/test_rubric_sweep.py
git commit -m "feat: compare all rubric challengers"
```

### Task 3: Execute sweeps serially and write the comparison table

**Files:**
- Modify: `eval_tool/pipeline.py`
- Modify: `tests/test_rubric_sweep.py`

- [ ] **Step 1: Write failing sweep orchestration tests**

```python
def test_sweep_runs_serially_and_writes_comparison(tmp_path, pipeline_config, monkeypatch):
    calls = []
    def fake_eval(config):
        calls.append(config.out_dir.name)
        path = config.out_dir / "judge_detail_all.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)
        scored_frame().to_excel(path, index=False)
        return {"judge_detail_all.xlsx": path}
    monkeypatch.setattr("eval_tool.pipeline.run_eval", fake_eval)
    result = run_sweep(pipeline_config, ["v3", "v4"], n_bootstrap=50)
    assert calls == [f"{pipeline_config.out_dir.name}_v3", f"{pipeline_config.out_dir.name}_v4"]
    assert result.comparison_path.is_file()
    assert pd.read_csv(result.comparison_path)["rubric"].tolist() == ["v3", "v4"]


def test_sweep_rejects_scored_overrides(pipeline_config_with_scored):
    with pytest.raises(PipelineError, match="sweep cannot use models\[\].scored"):
        run_sweep(pipeline_config_with_scored, ["v3", "v4"])
```

Also test duplicate rubric names, an empty list, failure stopping later rubrics, missing `judge_detail_all.xlsx`, model filtering, and `no_pairwise=True` propagation.

- [ ] **Step 2: Run sweep tests and confirm RED**

Run: `python -m pytest tests/test_rubric_sweep.py -k sweep -q --basetemp=.pytest_cache/plan3_task3`

Expected: FAIL because `run_sweep()` does not exist.

- [ ] **Step 3: Implement fail-fast serial sweep**

```python
@dataclass(frozen=True)
class SweepResult:
    reports: dict[str, dict[str, Path]]
    comparison_path: Path


def run_sweep(config, rubrics, model_names=None, out_root=None,
              no_pairwise=False, metric=None, n_bootstrap=5000):
    versions = validate_rubric_list(rubrics)
    selected = config.select_models(model_names)
    if any(model.scored_paths for model in selected):
        raise PipelineError("sweep cannot use models[].scored because its rubric is unverifiable")
    reports = {}
    frames = {}
    for version in versions:
        eval_config = apply_rubric(config.to_eval_config(model_names), version, out_root, no_pairwise)
        reports[version] = run_eval(eval_config)
        detail = reports[version].get("judge_detail_all.xlsx")
        if detail is None or not detail.is_file():
            raise PipelineError(f"rubric {version} did not produce judge_detail_all.xlsx")
        frames[version] = pd.read_excel(detail)
    comparison = compare_runs(frames, baseline=config.baseline_model,
                              metric=metric, n_bootstrap=n_bootstrap)
    comparison_path = Path(out_root or config.out_dir.parent) / "rubric_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    return SweepResult(reports, comparison_path)
```

- [ ] **Step 4: Run sweep and pipeline tests and confirm GREEN**

Run: `python -m pytest tests/test_rubric_sweep.py tests/test_pipeline.py -q --basetemp=.pytest_cache/plan3_task3`

Expected: all selected tests PASS and call order proves serial execution.

- [ ] **Step 5: Commit sweep orchestration**

```bash
git add eval_tool/pipeline.py tests/test_rubric_sweep.py
git commit -m "feat: run serial rubric sweeps"
```

### Task 4: Expose rubric commands and convert `run_rubric.py` into a shim

**Files:**
- Modify: `eval_tool/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `run_rubric.py`

- [ ] **Step 1: Write failing CLI and shim tests**

```python
def test_sweep_cli_parses_rubrics_and_models(monkeypatch):
    captured = []
    monkeypatch.setattr("eval_tool.cli.handle_sweep", lambda args: captured.append(args))
    main(["sweep", "--config", "pipeline.json", "--rubrics", "v3,v4", "--models", "base,sft"])
    assert captured[0].rubrics == ["v3", "v4"]
    assert captured[0].models == ["base", "sft"]


def test_run_rubric_shim_calls_new_cli(monkeypatch):
    calls = []
    monkeypatch.setattr("run_rubric.cli_main", lambda argv: calls.append(argv))
    run_rubric.main(["--config", "old.json", "--rubric", "v4", "--no-pairwise"])
    assert calls == [["eval", "--config", "old.json", "--rubric", "v4", "--no-pairwise"]]
```

Cover `--out-root`, `--dry-run`, `--metric`, `--n-bootstrap`, and preservation of old `run_rubric.py --help`.

- [ ] **Step 2: Run CLI tests and confirm RED**

Run: `python -m pytest tests/test_cli.py -k 'rubric or sweep' -q --basetemp=.pytest_cache/plan3_task4`

Expected: FAIL because rubric handlers and the shim do not exist.

- [ ] **Step 3: Add parser options and a direct shim**

```python
sweep = subparsers.add_parser("sweep")
sweep.add_argument("--config", required=True)
sweep.add_argument("--rubrics", required=True, type=_csv_values)
sweep.add_argument("--models", type=_csv_values)
sweep.add_argument("--no-pairwise", action="store_true")
sweep.add_argument("--metric")
sweep.add_argument("--n-bootstrap", type=int, default=5000)
sweep.set_defaults(handler=handle_sweep)
```

Add `--rubric`, `--out-root`, `--no-pairwise`, and `--dry-run` to `eval`. Rewrite `run_rubric.main(argv=None)` to parse its established options, translate them to the new `eval` arguments, and call `eval_tool.cli.main()` in-process without writing a derived JSON or spawning Python.

- [ ] **Step 4: Run CLI, shim, and rubric tests**

Run: `python -m pytest tests/test_cli.py tests/test_rubrics.py tests/test_rubric_sweep.py -q --basetemp=.pytest_cache/plan3_task4`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit command integration**

```bash
git add eval_tool/cli.py tests/test_cli.py run_rubric.py
git commit -m "feat: expose rubric sweep commands"
```

### Task 5: Update documentation and the offline end-to-end check

**Files:**
- Modify: `README.md`
- Modify: `开放问答评估_使用说明.md`
- Modify: `e2e_check.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add command-help smoke tests before editing docs**

```python
@pytest.mark.parametrize("command", ["convert", "infer", "eval", "sweep", "all"])
def test_each_subcommand_has_help(command):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
```

- [ ] **Step 2: Run smoke tests and confirm current behavior**

Run: `python -m pytest tests/test_cli.py -k help -q --basetemp=.pytest_cache/plan3_task5`

Expected: PASS after Task 4; these tests pin the commands documented next.

- [ ] **Step 3: Replace stale two-config instructions with the unified workflow**

Document these verified commands verbatim:

```bash
python -m eval_tool convert data.json --config pipeline.json
python -m eval_tool infer --config pipeline.json
python -m eval_tool eval --config pipeline.json --rubric v4
python -m eval_tool sweep --config pipeline.json --rubrics v1,v3,v4,v4b
python -m eval_tool all --config pipeline.json
```

Explain default resume, `--overwrite`, `--clean-partial`, the manifest and `_partial` layout, `scored > pred > derived`, serial model/rubric execution, the sweep prohibition on `scored`, and every retained legacy command. Correct nonexistent prompt filenames and remove the false instruction that judge caches must be manually deleted after changing prompt text.

- [ ] **Step 4: Make `e2e_check.py` exercise the new orchestrator with fakes**

Construct a temporary pipeline JSON, invoke `load_pipeline_config()` and `run_all()` with a fake generator factory and monkeypatched fake judge calls, and retain assertions for single-turn/multi-turn conversion, two-model predictions, score reports, and image-free xlsx outputs. Allocate its temporary directory through `tempfile.TemporaryDirectory()` rather than deleting a fixed repository folder.

Run: `$env:PYTHONDONTWRITEBYTECODE='1'; python e2e_check.py`

Expected: exit 0 and a final success line identifying the unified `convert -> infer -> eval` path.

- [ ] **Step 5: Run documentation-adjacent tests and commit**

Run: `python -m pytest tests/test_cli.py tests/test_pipeline.py tests/test_rubric_sweep.py -q --basetemp=.pytest_cache/plan3_task5`

Expected: all selected tests PASS.

```bash
git add README.md 开放问答评估_使用说明.md e2e_check.py tests/test_cli.py
git commit -m "docs: document unified evaluation workflow"
```

### Task 6: Final verification

**Files:**
- Verify all changed source, tests, examples, and documentation.

- [ ] **Step 1: Run the complete test suite in a writable base temp**

Run: `$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q --basetemp=.pytest_cache/final_verify`

Expected: all tests PASS; no setup errors from the system temp directory.

- [ ] **Step 2: Run both offline smoke paths**

Run: `$env:PYTHONDONTWRITEBYTECODE='1'; python e2e_check.py`

Run: `python -m eval_tool --help`

Expected: the end-to-end check exits 0; top-level help lists all five subcommands and explains legacy `--config` routing.

- [ ] **Step 3: Verify example config parsing and dry-run rubric derivation**

Run: `python -c "from eval_tool.config import load_pipeline_config; print([m.name for m in load_pipeline_config('pipeline.example.json').models])"`

Run: `python run_rubric.py --config config.example.json --rubric v4 --dry-run`

Expected: the first command prints the example model names; dry-run prints rubric, prompt, output directory, and a fingerprint without creating a derived JSON.

- [ ] **Step 4: Inspect repository integrity**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors, no modified tracked bytecode, and no generated report/cache/derived-config files.

- [ ] **Step 5: Record external verification limits in the handoff**

Report real model inference and live judge API calls as not run because they require the user's actual paths and services. Include the exact offline pytest and `e2e_check.py` results as the completion evidence.
