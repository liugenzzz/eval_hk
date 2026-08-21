# LLaMAFactory Image Pixels Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLaMAFactory 0.9.5-compatible `image_min_pixels` / `image_max_pixels` preprocessing to every local Qwen image-inference path.

**Architecture:** A shared Pillow helper in `eval_tool/imaging.py` owns the exact LLaMAFactory resize/conversion sequence. Configuration remains optional for backward compatibility, is propagated through normal and DPO workers, and participates in inference identity only when enabled.

**Tech Stack:** Python 3.11+, dataclasses, Pillow, pytest, multiprocessing DTOs, JSON configuration.

---

### Task 1: Shared LLaMAFactory Qwen image regularizer

**Files:**
- Modify: `eval_tool/imaging.py`
- Create: `tests/test_imaging.py`

- [ ] **Step 1: Write failing validation and image-sequence tests**

Add tests that import `IMAGE_PREPROCESS_PROFILE`, `validate_image_pixel_bounds`, and `prepare_llamafactory_qwen_image`. Assert pair-or-neither validation, strict integer rejection, the exact max/min resize dimensions, the 28-pixel clamp, 200-to-180 aspect-ratio correction, RGB output, source immutability, and returned-image ownership. Use an instrumented Pillow image or monkeypatch to prove resize occurs before `convert("RGB")`.

```python
def test_prepare_llamafactory_qwen_image_scales_area_before_rgb_conversion():
    source = Image.new("RGBA", (1600, 800), (10, 20, 30, 128))
    prepared = prepare_llamafactory_qwen_image(
        source, image_min_pixels=65536, image_max_pixels=589824
    )
    try:
        assert prepared.size == (1086, 543)
        assert prepared.mode == "RGB"
        assert source.size == (1600, 800)
        assert source.mode == "RGBA"
    finally:
        prepared.close()
        source.close()
```

- [ ] **Step 2: Run the new tests and verify the missing API fails**

Run: `python -m pytest tests/test_imaging.py -q`

Expected: collection/import failure for the not-yet-defined preprocessing API.

- [ ] **Step 3: Implement the minimal shared helper**

Implement:

```python
IMAGE_PREPROCESS_PROFILE = "llamafactory-0.9.5-qwen-static-v1"

def validate_image_pixel_bounds(
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> tuple[int | None, int | None]:
    ...

def prepare_llamafactory_qwen_image(
    source: Image.Image,
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> Image.Image:
    ...
```

The implementation must call Pillow `resize((width, height))` for every upstream resize stage, convert after all resizing, return a distinct owned object, and close intermediate objects on success and failure.

- [ ] **Step 4: Run the focused tests and verify green**

Run: `python -m pytest tests/test_imaging.py -q`

Expected: all tests in the file pass.

### Task 2: Normal inference configuration, image entry point, workers, and cache

**Files:**
- Modify: `eval_tool/config.py`
- Modify: `eval_tool/infer.py`
- Modify: `eval_tool/run_infer.py`
- Modify: `eval_tool/infer_cache.py`
- Modify: `tests/test_infer_config.py`
- Modify: `tests/test_pipeline_config.py`
- Modify: `tests/test_run_infer.py`
- Modify: `tests/test_infer_parallel.py`
- Modify: `tests/test_run_infer_resume.py`
- Modify: `tests/test_infer_cache.py`

- [ ] **Step 1: Write failing config and cache identity tests**

Assert that standalone and pipeline configuration load `65536 / 589824`, pipeline derivation preserves them, invalid types/pairs raise `ConfigError`, and both omitted or both null yield `None`. Extend fingerprint tests with keyword-only parameters:

```python
build_infer_fingerprint(
    model_path,
    prompt,
    max_new_tokens,
    dataset_key,
    dataset_name,
    rows,
    image_min_pixels=65536,
    image_max_pixels=589824,
)
```

Assert configured bounds change the fingerprint while two `None` values preserve the legacy digest.

- [ ] **Step 2: Run focused tests and verify expected failures**

Run: `python -m pytest tests/test_infer_config.py tests/test_pipeline_config.py tests/test_infer_cache.py -q`

Expected: failures for missing dataclass fields, loader parsing, and fingerprint parameters.

- [ ] **Step 3: Add fields, parsing, validation, and fingerprint payload**

Add optional fields to `InferConfig` and `PipelineInferSettings`, copy them in `PipelineConfig.to_infer_configs`, and parse them through a shared config helper that converts `ValueError` from `validate_image_pixel_bounds` into `ConfigError`. Standalone parsing supports lowercase and uppercase aliases. Extend `build_infer_fingerprint` with keyword-only optional bounds and include this payload only when enabled:

```python
payload["image_preprocess"] = {
    "profile": IMAGE_PREPROCESS_PROFILE,
    "min_pixels": image_min_pixels,
    "max_pixels": image_max_pixels,
}
```

- [ ] **Step 4: Verify config and cache tests pass**

Run: `python -m pytest tests/test_infer_config.py tests/test_pipeline_config.py tests/test_infer_cache.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing image-entry and worker propagation tests**

Extend generator tests so `_decode_images` receives the bounds and the fake processor observes the exact prepared dimensions/mode. Capture sequential, parallel, and resumable worker `QwenVLGenerator` construction and assert both values are retained. Assert programmatic invalid generator construction fails before model loading.

- [ ] **Step 6: Run propagation tests and verify red**

Run: `python -m pytest tests/test_run_infer.py tests/test_infer_parallel.py tests/test_run_infer_resume.py -q`

Expected: failures because generator fields and worker arguments are absent.

- [ ] **Step 7: Implement normal inference propagation and preprocessing**

Add optional fields to `QwenVLGenerator`, validate them before importing/loading the model, pass them from every construction site, and call `prepare_llamafactory_qwen_image` inside `_decode_images` before the source image closes. Pass configured bounds into `build_infer_fingerprint`. Do not add Hugging Face `min_pixels` or `max_pixels` kwargs.

- [ ] **Step 8: Verify all normal-inference tests pass**

Run: `python -m pytest tests/test_run_infer.py tests/test_infer_parallel.py tests/test_run_infer_resume.py tests/test_infer_config.py tests/test_pipeline_config.py tests/test_infer_cache.py -q`

Expected: all selected tests pass.

### Task 3: DPO configuration, image entry point, workers, cache, and manifest

**Files:**
- Modify: `eval_tool/dpo_config.py`
- Modify: `eval_tool/dpo_multimodal.py`
- Modify: `eval_tool/dpo_infer.py`
- Modify: `eval_tool/dpo_cache.py`
- Modify: `eval_tool/dpo_pipeline.py`
- Modify: `tests/test_dpo_config.py`
- Modify: `tests/test_dpo_multimodal.py`
- Modify: `tests/test_dpo_infer.py`
- Modify: `tests/test_dpo_cache.py`
- Modify: `tests/test_dpo_pipeline.py`

- [ ] **Step 1: Write failing DPO config, DTO, cache, and manifest tests**

Assert the strict DPO schema accepts both fields, rejects invalid or single-sided settings, round-trips them through `_WorkerConfigDto`, includes configured bounds/profile in inference fingerprint, and writes them into the final manifest model identity.

- [ ] **Step 2: Run DPO identity tests and verify red**

Run: `python -m pytest tests/test_dpo_config.py tests/test_dpo_cache.py tests/test_dpo_pipeline.py tests/test_dpo_infer.py -q`

Expected: failures for unknown keys, missing DTO members, and unchanged identity.

- [ ] **Step 3: Implement DPO configuration and identity propagation**

Add optional fields to `DpoInferConfig`, `_INFER_KEYS`, `_load_infer`, `_WorkerConfigDto`, DTO conversion, and the default Qwen factory. Include the configured preprocessing identity in DPO fingerprint and manifest; omit it from legacy fingerprint input when disabled.

- [ ] **Step 4: Verify DPO identity tests pass**

Run: `python -m pytest tests/test_dpo_config.py tests/test_dpo_cache.py tests/test_dpo_pipeline.py tests/test_dpo_infer.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing DPO image-sequence and recursive-propagation tests**

Extend `open_model_messages` tests with RGBA/large/small images. Assert returned model images have LLaMAFactory-compatible size and RGB mode, source bytes and judge data URL remain unchanged, images remain live inside the context and close afterward. Add isolation tests proving both bounds survive recursive batch splitting and worker execution.

- [ ] **Step 6: Run DPO multimodal tests and verify red**

Run: `python -m pytest tests/test_dpo_multimodal.py tests/test_dpo_infer.py -q`

Expected: failures because bounds are not accepted or propagated.

- [ ] **Step 7: Implement DPO image preprocessing at the original-mode entry point**

Extend `open_model_messages` with keyword-only optional bounds and replace the current convert/copy sequence with `prepare_llamafactory_qwen_image`. Pass bounds through public/private isolation functions, recursive calls, sequential inference, and worker inference. Ensure the helper runs exactly once per local model image and never alters `_verified_asset_bytes` or `image_data_url`.

- [ ] **Step 8: Verify DPO tests pass**

Run: `python -m pytest tests/test_dpo_multimodal.py tests/test_dpo_infer.py tests/test_dpo_config.py tests/test_dpo_cache.py tests/test_dpo_pipeline.py -q`

Expected: all selected tests pass.

### Task 4: Examples and user documentation

**Files:**
- Modify: `infer_config.example.json`
- Modify: `pipeline.example.json`
- Modify: `dpo.example.json`
- Modify: `README.md`
- Modify: `开放问答评估_使用说明.md`
- Modify: `DPO_使用说明.md`

- [ ] **Step 1: Write failing example-schema tests**

Extend existing example configuration tests to assert all three example inference blocks contain:

```json
"image_min_pixels": 65536,
"image_max_pixels": 589824
```

and load successfully through their production loaders.

- [ ] **Step 2: Run example tests and verify red**

Run: `python -m pytest tests/test_infer_config.py tests/test_pipeline_config.py tests/test_dpo_config.py -q`

Expected: failures because the example keys are absent.

- [ ] **Step 3: Update examples and focused documentation**

Add the screenshot values to all inference examples. Document that they are total-pixel-area thresholds applied with the LLaMAFactory 0.9.5 Qwen preprocessing sequence, that both are required together, that lowering max reduces visual tokens/memory, and that changing them invalidates inference resume identity. State that offline JPEG/1280 preprocessing is a separate source-image transformation.

- [ ] **Step 4: Verify examples load**

Run: `python -m pytest tests/test_infer_config.py tests/test_pipeline_config.py tests/test_dpo_config.py -q`

Expected: all selected tests pass.

### Task 5: Full verification and review

**Files:**
- Review all files changed by Tasks 1-4

- [ ] **Step 1: Run formatting/static checks available in the repository**

Run the project-provided formatter/linter commands discovered from `pyproject.toml` or CI configuration. If none are configured, run `python -m compileall -q eval_tool tests`.

Expected: exit code 0.

- [ ] **Step 2: Run the complete test suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Inspect the final diff and requirement coverage**

Run: `git diff --check` and `git diff --stat`.

Verify each design requirement has an implementation and a test, no `AutoProcessor` min/max kwargs were introduced, and unrelated user files were not changed.

- [ ] **Step 4: Request independent specification and code-quality reviews**

Give reviewers the approved design, starting SHA, final diff, and test output. Resolve every Critical or Important issue and rerun the relevant focused tests plus the full suite.

- [ ] **Step 5: Report the exact verification evidence**

Report test counts, commands, changed configuration examples, and the remaining environment caveat: source bytes, Pillow, Transformers, and checkpoint processor config must match for pixel-level equivalence.

No git commit is created unless the user explicitly requests one.
