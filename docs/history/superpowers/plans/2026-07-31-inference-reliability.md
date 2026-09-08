# Inference Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict prediction validation, crash-safe JSONL batch persistence, fingerprinted resume, and safe final artifact publication without breaking the legacy single-model inference API.

**Architecture:** Keep `InferConfig` and `run_infer.run()` as the legacy boundary. Put fingerprinting, shards, manifests, and locking in a new `infer_cache.py`; use opt-in `resume` fields so old configs retain their current overwrite behavior. Sequential and parallel inference both publish exclusively from reloaded shards, so a worker failure cannot create a partial xlsx.

**Tech Stack:** Python 3.12, dataclasses, pathlib, hashlib/json, pandas/openpyxl, pytest.

---

## File map

- Create `eval_tool/infer_cache.py`: inference snapshots, shard persistence, manifests, locks, and atomic JSON writes.
- Create `tests/test_infer_cache.py`: storage, fingerprint, conflict, manifest, and lock tests.
- Create `tests/test_run_infer_resume.py`: sequential/parallel resume integration tests.
- Modify `eval_tool/config.py`: backward-compatible `InferConfig.resume` and `clean_partial` fields.
- Modify `eval_tool/infer.py`: prompt snapshot support, batch cardinality checks, and strict positional merge.
- Modify `eval_tool/run_infer.py`: lazy model loading, resume orchestration, atomic xlsx publication, worker shard writes.
- Modify `tests/test_infer_parallel.py` and `tests/test_run_infer.py`: strict merge and compatibility coverage.

### Task 1: Make generation and positional merging strict

**Files:**
- Modify: `eval_tool/infer.py`
- Modify: `tests/test_infer_parallel.py`
- Modify: `tests/test_run_infer.py`

- [ ] **Step 1: Write failing tests for short batches and missing/duplicate positions**

```python
def test_merge_rejects_missing_positions():
    with pytest.raises(PredictionMergeError, match="missing positions: 1"):
        merge_indexed_predictions([[(0, "first")]], total=2)


def test_merge_rejects_duplicate_positions():
    with pytest.raises(PredictionMergeError, match="duplicate position: 0"):
        merge_indexed_predictions([[(0, "a")], [(0, "b")]], total=1)


class ShortBatchGenerator:
    def generate_batch(self, prompts, image_b64s=None):
        return ["only-one"]


def test_generate_predictions_rejects_short_batch():
    with pytest.raises(PredictionBatchError, match="returned 1 predictions for 2 rows"):
        generate_predictions(
            [{"index": "1", "question": "q1"}, {"index": "2", "question": "q2"}],
            "vqa", {}, ShortBatchGenerator(), batch_size=2,
        )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_infer_parallel.py tests/test_run_infer.py -q --basetemp=.pytest_cache/plan1_task1`

Expected: FAIL because `PredictionMergeError` and `PredictionBatchError` do not exist and missing positions still become `""`.

- [ ] **Step 3: Add strict exceptions, prompt snapshots, and cardinality validation**

```python
class PredictionBatchError(RuntimeError):
    pass


class PredictionMergeError(RuntimeError):
    pass


def resolve_infer_prompt_text(dataset_key, prompt_files):
    if dataset_key in prompt_files:
        return load_prompt_text(prompt_files[dataset_key])
    return DEFAULT_INFER_PROMPTS.get(dataset_key, DEFAULT_INFER_PROMPTS["vqa"])


def merge_indexed_predictions(chunks, total):
    merged: list[str | None] = [None] * total
    for chunk in chunks:
        for position, prediction in chunk:
            if not isinstance(position, int) or position < 0 or position >= total:
                raise PredictionMergeError(f"invalid position: {position}")
            if merged[position] is not None:
                raise PredictionMergeError(f"duplicate position: {position}")
            merged[position] = str(prediction)
    missing = [i for i, value in enumerate(merged) if value is None]
    if missing:
        raise PredictionMergeError(f"missing positions: {','.join(map(str, missing[:20]))}")
    return [value for value in merged if value is not None]
```

Extend `generate_predictions(rows, dataset_key, prompt_files, generator, batch_size=1, progress=False, progress_desc="Infer", prompt_texts=None, on_batch=None)` so it freezes prompt text, checks `len(batch_predictions) == len(batch)`, then calls `on_batch(batch, batch_predictions)` before extending the in-memory result. Keep both new parameters optional so all existing callers remain valid.

- [ ] **Step 4: Run focused and legacy inference tests and confirm GREEN**

Run: `python -m pytest tests/test_infer_parallel.py tests/test_run_infer.py tests/test_convert_vqa_json.py -q --basetemp=.pytest_cache/plan1_task1`

Expected: all selected tests PASS; a model-returned empty string remains a present prediction.

- [ ] **Step 5: Commit the strict contracts**

```bash
git add eval_tool/infer.py tests/test_infer_parallel.py tests/test_run_infer.py
git commit -m "fix: reject incomplete inference batches"
```

### Task 2: Build the inference shard store and fingerprint

**Files:**
- Create: `eval_tool/infer_cache.py`
- Create: `tests/test_infer_cache.py`

- [ ] **Step 1: Write failing tests for stable fingerprints, reload, conflicts, and manifests**

```python
def test_fingerprint_changes_when_inference_input_changes():
    rows = [{"index": "1", "question": "q", "history": "", "image": "img"}]
    first = build_infer_fingerprint(Path("model"), "prompt", 32, "vqa", "aero_vqa", rows)
    changed = build_infer_fingerprint(Path("model"), "prompt", 32, "vqa", "aero_vqa", [{**rows[0], "question": "new"}])
    assert first != changed


def test_store_reloads_all_worker_shards(tmp_path):
    store = InferShardStore(tmp_path, "vqa", "abc")
    store.append_batch([("1", "a")], worker_id=0)
    store.append_batch([("2", "b")], worker_id=3)
    assert InferShardStore(tmp_path, "vqa", "abc").load() == {"1": "a", "2": "b"}


def test_store_rejects_duplicate_index_across_shards(tmp_path):
    store = InferShardStore(tmp_path, "vqa", "abc")
    store.append_batch([("1", "a")], worker_id=0)
    store.append_batch([("1", "a")], worker_id=1)
    with pytest.raises(InferCacheError, match="duplicate index '1'"):
        store.load()
```

Also test: malformed trailing JSON is reported, `write_manifest()` uses the documented fields, `InferenceLock` rejects a second acquisition, and cleanup only removes files under the exact dataset partial directory.

- [ ] **Step 2: Run the new tests and confirm RED**

Run: `python -m pytest tests/test_infer_cache.py -q --basetemp=.pytest_cache/plan1_task2`

Expected: collection FAIL because `eval_tool.infer_cache` does not exist.

- [ ] **Step 3: Implement focused cache primitives**

```python
@dataclass(frozen=True)
class InferManifest:
    fingerprint: str
    dataset_name: str
    row_count: int
    index_digest: str


def build_infer_fingerprint(model_path, prompt_text, max_new_tokens,
                            dataset_key, dataset_name, rows):
    digest = hashlib.sha256()
    for row in rows:
        item = {key: _stable_value(row.get(key)) for key in
                ("index", "question", "history", "image")}
        digest.update(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    payload = {
        "model_path": str(Path(model_path).resolve()),
        "prompt": prompt_text,
        "max_new_tokens": int(max_new_tokens),
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "input_digest": digest.hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
```

Implement `InferShardStore.load()`, `append_batch()`, `foreign_fingerprints()`, `clear_current()`, and `clean_dataset()` with flat JSONL records. Implement `InferenceLock` using `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` and always release it in `__exit__`. Implement manifest JSON through a temporary sibling and `Path.replace()`.

- [ ] **Step 4: Run the store tests and confirm GREEN**

Run: `python -m pytest tests/test_infer_cache.py -q --basetemp=.pytest_cache/plan1_task2`

Expected: all cache tests PASS, including reload after changing worker IDs.

- [ ] **Step 5: Commit the cache boundary**

```bash
git add eval_tool/infer_cache.py tests/test_infer_cache.py
git commit -m "feat: add fingerprinted inference shards"
```

### Task 3: Integrate sequential resume and atomic publication

**Files:**
- Modify: `eval_tool/config.py`
- Modify: `eval_tool/run_infer.py`
- Create: `tests/test_run_infer_resume.py`

- [ ] **Step 1: Write failing sequential-resume tests**

```python
class FailingSecondBatchGenerator:
    def __init__(self):
        self.calls = 0
    def generate_batch(self, prompts, image_b64s=None):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated interruption")
        return [f"pred:{prompt}" for prompt in prompts]


def test_resume_only_generates_missing_rows(infer_fixture):
    config = infer_fixture.config(resume=True, overwrite=False, batch_size=1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(config, generator=FailingSecondBatchGenerator())
    resumed = RecordingGenerator()
    written = run(config, generator=resumed)
    assert len(resumed.prompts) == infer_fixture.row_count - 1
    assert pd.read_excel(written["vqa"])["prediction"].notna().all()
```

Add tests that: empty/duplicate dataset indices fail before model creation; a changed fingerprint fails without overwrite; overwrite reruns every row; matching manifest skips model creation; a legacy xlsx without manifest requires explicit `pred` or overwrite; and `clean_partial=True` removes shards only after successful publication.

- [ ] **Step 2: Run the resume tests and confirm RED**

Run: `python -m pytest tests/test_run_infer_resume.py -q --basetemp=.pytest_cache/plan1_task3`

Expected: FAIL because `InferConfig` has no resume fields and `run()` does not persist batches.

- [ ] **Step 3: Add opt-in resume fields and orchestration**

```python
@dataclass(frozen=True)
class InferConfig:
    # existing fields stay unchanged
    resume: bool = False
    clean_partial: bool = False
```

In `run()` validate index uniqueness, freeze the prompt, compute the fingerprint, acquire the lock, apply the manifest/foreign-fingerprint rules, load cached records, and delay `QwenVLGenerator` construction until pending rows exist. Pass an `on_batch` callback that writes `[(str(row["index"]), prediction) for row, prediction in zip(batch_rows, batch_predictions)]`. After generation, reload the shard store, order predictions by truth rows, call strict validation, write xlsx to a temporary sibling, replace the final file, then write its manifest. On any exception, append a UTF-8 warning to `config.out_dir / "warnings.log"` and leave the old final xlsx untouched.

- [ ] **Step 4: Run resume plus all existing single-process inference tests**

Run: `python -m pytest tests/test_run_infer_resume.py tests/test_run_infer.py tests/test_infer_config.py -q --basetemp=.pytest_cache/plan1_task3`

Expected: all selected tests PASS; old `InferConfig` construction and default overwrite behavior are unchanged.

- [ ] **Step 5: Commit sequential resume**

```bash
git add eval_tool/config.py eval_tool/run_infer.py tests/test_run_infer_resume.py
git commit -m "feat: resume interrupted inference batches"
```

### Task 4: Persist each parallel worker independently

**Files:**
- Modify: `eval_tool/run_infer.py`
- Modify: `tests/test_run_infer_resume.py`

- [ ] **Step 1: Write failing tests for worker persistence and incomplete results**

```python
def test_parallel_failure_keeps_completed_worker_shards(tmp_path, monkeypatch, parallel_fixture):
    executor = FakeExecutor([
        FakeFuture(result=[("1", "one")]),
        FakeFuture(error=RuntimeError("gpu worker failed")),
    ])
    monkeypatch.setattr(run_infer, "ProcessPoolExecutor", lambda **_: executor)
    with pytest.raises(RuntimeError, match="gpu worker failed"):
        run(parallel_fixture.config(resume=True), generator=None)
    assert parallel_fixture.store().load() == {"1": "one"}
    assert not parallel_fixture.final_xlsx.exists()
```

Also test that a later run with a different worker count consumes every old `w*` shard and that a worker returning fewer predictions triggers `PredictionBatchError` before publication.

- [ ] **Step 2: Run the parallel resume tests and confirm RED**

Run: `python -m pytest tests/test_run_infer_resume.py -k parallel -q --basetemp=.pytest_cache/plan1_task4`

Expected: FAIL because workers return only in-memory lists and do not receive shard paths.

- [ ] **Step 3: Add a resumable parallel branch without changing the legacy branch signature**

```python
def _run_dataset_parallel_resumable(config, dataset_key, rows, prompt_text, store):
    cached = store.load()
    pending = [row for row in rows if str(row["index"]) not in cached]
    worker_gpus = [gpu for gpu in config.gpu_ids for _ in range(config.workers_per_gpu)]
    chunks = chunk_records(pending, len(worker_gpus))
    errors = []
    with ProcessPoolExecutor(max_workers=len(worker_gpus)) as executor:
        futures = {
            executor.submit(
                _infer_worker_resumable,
                worker_id, gpu_id, chunk, dataset_key, prompt_text,
                str(config.model_path), config.max_new_tokens,
                config.batch_size, config.torch_dtype, str(store.shard_path(worker_id)),
            ): worker_id
            for worker_id, (gpu_id, chunk) in enumerate(zip(worker_gpus, chunks))
            if chunk
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future], exc))
    if errors:
        summary = "; ".join(f"worker {worker}: {error}" for worker, error in errors)
        raise InferenceWorkerError(summary)
    return store.load()
```

Keep `_run_dataset_parallel(config, dataset_key, rows)` unchanged for legacy callers and existing monkeypatch tests. Extend `_infer_worker` through a new `_infer_worker_resumable` that uses the same strict batch callback and writes only its `__w<id>.jsonl`. Catch individual future errors, continue draining futures, append a summary warning, then raise an aggregated `InferenceWorkerError`.

- [ ] **Step 4: Run parallel, resume, and legacy inference tests**

Run: `python -m pytest tests/test_infer_parallel.py tests/test_run_infer.py tests/test_run_infer_resume.py -q --basetemp=.pytest_cache/plan1_task4`

Expected: all selected tests PASS; the original three-argument parallel function remains patchable.

- [ ] **Step 5: Commit parallel persistence**

```bash
git add eval_tool/run_infer.py tests/test_run_infer_resume.py
git commit -m "feat: persist parallel inference workers"
```

### Task 5: Verify the standalone reliability increment

**Files:**
- Test: all existing and new inference tests

- [ ] **Step 1: Run the inference-focused suite with bytecode writes disabled**

Run: `$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest tests/test_infer_cache.py tests/test_infer_config.py tests/test_infer_parallel.py tests/test_run_infer.py tests/test_run_infer_resume.py tests/test_convert_vqa_json.py -q --basetemp=.pytest_cache/plan1_verify`

Expected: all selected tests PASS with no modified tracked `__pycache__` files.

- [ ] **Step 2: Run the complete baseline suite**

Run: `$env:PYTHONDONTWRITEBYTECODE='1'; python -m pytest -q --basetemp=.pytest_cache/plan1_all`

Expected: new inference tests PASS; record separately whether the pre-existing `test_run_eval_reuses_previously_scored_vqa_results_without_calling_judge` failure remains.

- [ ] **Step 3: Inspect the exact diff and generated-file status**

Run: `git diff --check` and `git status --short`

Expected: no whitespace errors; only intentional source/test changes plus the user's pre-existing untracked files.

- [ ] **Step 4: Record the verification result in the implementation handoff**

Report the exact selected-suite and full-suite pass/fail counts, identify the pre-existing scored-detail regression separately, and list real GPU/model inference as not run in this phase.
