# Direct JSON/JSONL DPO Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an independent `build-dpo` command that reads mixed Alpaca/ShareGPT JSON or JSONL files directly, generates rejected answers with the configured local model, optionally keeps only Judge-confirmed wrong answers, and atomically publishes one strict ShareGPT DPO JSONL plus complete audit artifacts.

**Architecture:** Keep the new workflow isolated from the existing TSV/XLSX pipeline. Normalize all inputs into immutable DPO candidates, validate/deduplicate before GPU work, run structured multimodal inference through a backward-compatible extension of `QwenVLGenerator`, persist inference/Judge state in strict phase-specific JSONL shards, and publish verified artifacts through a manifest-last transaction. Reuse existing v1/v4 scoring only after DPO-specific strict schema validation.

**Tech Stack:** Python 3, stdlib dataclasses/json/hashlib/mimetypes/concurrent.futures/pathlib, Pillow, PyTorch/Transformers, existing OpenAI-compatible `JudgeClient`, pytest.

---

## Working rules

- Implement in an isolated `codex/direct-json-dpo-builder` worktree because the main worktree contains unrelated user-owned changes.
- Follow red-green-refactor for every task: write the named failing test, run it and confirm the intended failure, implement the smallest behavior, rerun the focused test, then commit.
- Do not modify `eval_tool/config.py`, `eval_tool/convert_vqa_json.py`, `eval_tool/run_infer.py`, `eval_tool/cache.py`, `eval_tool/pipeline.py`, `eval_tool/report.py`, or their existing semantics.
- Tests use fake generators and fake Judge clients. A real GPU/Judge smoke test is optional and must be reported as not executed when unavailable.
- JSONL output is UTF-8, `ensure_ascii=False`, and exactly one complete object per line.
- Use stable SHA-256 identities internally. Do not include the Judge API key, authorization headers, or image data URLs in logs/manifests.

## Task 1: Add strict DPO configuration types and loading

**Files:**

- Create: `eval_tool/dpo_config.py`
- Create: `tests/test_dpo_config.py`

**Step 1: Write failing configuration tests**

Cover these nodes:

```text
test_load_dpo_config_resolves_config_paths_and_cwd_image_root
test_explicit_image_root_resolves_from_config_directory
test_cli_inputs_replace_config_inputs_and_resolve_from_invocation_dir
test_wrong_only_requires_literal_json_boolean
test_enable_thinking_requires_literal_json_boolean
test_full_mode_ignores_optional_rubric_and_judge
test_wrong_only_requires_binary_or_v4_and_complete_judge
test_model_path_and_inputs_must_exist
test_output_name_must_end_with_jsonl
test_output_name_must_be_a_safe_nonreserved_basename
test_numeric_fields_reject_bool_and_invalid_ranges
```

Use real temporary input/model directories and assert that config-relative paths differ from invocation-relative CLI overrides.

**Step 2: Run the test and confirm the missing-module failure**

Run:

```powershell
python -m pytest tests/test_dpo_config.py -v
```

Expected: FAIL because `eval_tool.dpo_config` does not exist.

**Step 3: Implement immutable strict configuration objects**

Implement these public shapes:

```python
RubricName = Literal["binary", "v4"]

@dataclass(frozen=True)
class DpoInferConfig:
    model_name: str
    model_path: Path
    enable_thinking: bool = False
    max_new_tokens: int = 1024
    batch_size: int = 1
    torch_dtype: str = "auto"
    device_map: str = "auto"
    gpu_ids: tuple[int, ...] = ()
    workers_per_gpu: int = 1

@dataclass(frozen=True)
class DpoJudgeConfig:
    settings: JudgeSettings
    max_workers: int = 8

@dataclass(frozen=True)
class DpoBuildConfig:
    config_path: Path
    inputs: tuple[Path, ...]
    output_dir: Path
    output_name: str
    work_dir: Path
    image_root: Path
    wrong_only: bool
    rubric: RubricName | None
    infer: DpoInferConfig
    judge: DpoJudgeConfig | None

def load_dpo_config(
    path: str | Path,
    *,
    input_overrides: Sequence[str | Path] | None = None,
    invocation_dir: Path | None = None,
) -> DpoBuildConfig: ...
```

Rules to encode directly:

- `wrong_only` is required and accepted only when `type(value) is bool`.
- `enable_thinking` is optional, defaults to false, and is also a literal JSON bool.
- Config `inputs`, `model_path`, `output_dir`, `work_dir`, and explicit `image_root` resolve from the config directory.
- CLI input overrides resolve from `invocation_dir` and replace, never append to, config inputs.
- Omitted `image_root` is exactly `invocation_dir`.
- `wrong_only=false` returns `rubric=None` and `judge=None`, even if stale values are present.
- `wrong_only=true` requires one rubric and complete Judge settings.
- Validate all files/directories and numeric ranges before model loading; reject duplicate GPU IDs.
- When `gpu_ids` is nonempty, reject a `device_map` that embeds physical device numbers; spawned workers own GPU visibility and each process must see its assigned card as device zero.
- `output_name` must be one nonreserved basename ending in `.jsonl`: no absolute/drive path, slash, backslash, `..`, Windows device name, trailing dot/space, or collision with `audit_records.jsonl`, `rejected_records.jsonl`, `summary.json`, `manifest.json`, or `warnings.log`. Revalidate the resolved publish parent before replacement.
- Reject unknown keys at the DPO top level and inside `infer`/`judge`, so misspelled safety settings cannot be silently ignored. Nonempty strings are required for model name/path, dtype/device map, Judge endpoint/key/model, output/work names, and every input path. Integer validation uses `type(value) is int`: `max_new_tokens`, `batch_size`, `workers_per_gpu`, Judge timeout, and `max_workers` are at least 1; `max_retries` is at least 0; every GPU ID is unique and at least 0. Judge `temperature` is a finite non-bool JSON number in `[0, 2]`.
- Raise a DPO configuration exception derived from the existing `ConfigError` so CLI error rendering stays consistent.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest tests/test_dpo_config.py -v
```

Expected: PASS.

**Step 5: Commit**

```powershell
git add eval_tool/dpo_config.py tests/test_dpo_config.py
git commit -m "feat: add strict DPO configuration"
```

## Task 2: Load JSON/JSONL by content with complete provenance

**Files:**

- Create: `eval_tool/dpo_input.py`
- Create: `tests/test_dpo_input.py`

**Step 1: Write failing container/provenance tests**

Cover:

```text
test_loads_json_array_by_content
test_loads_single_json_object_by_content
test_loads_pretty_multiline_json_array
test_loads_nonempty_jsonl_lines_in_order
test_extension_does_not_choose_container_format
test_invalid_json_or_jsonl_file_is_fatal
test_jsonl_non_object_line_is_fatal
test_valid_container_with_schema_bad_record_is_returned_for_audit
test_json_array_scalar_null_and_list_items_are_invalid_schema_records
test_source_ref_contains_file_index_array_index_or_line_and_digest
```

**Step 2: Confirm the tests fail for missing loader APIs**

Run:

```powershell
python -m pytest tests/test_dpo_input.py -k "loads or container or source_ref or invalid_json" -v
```

Expected: FAIL for missing types/functions.

**Step 3: Implement source records and content detection**

Add immutable types and loader entry points:

```python
@dataclass(frozen=True)
class SourceRef:
    input_index: int
    source_path: Path
    container_format: Literal["json_array", "json_object", "jsonl"]
    record_index: int
    line_number: int | None
    source_id: str | None
    raw_digest: str

@dataclass(frozen=True)
class RawRecord:
    source: SourceRef
    value: Any

def load_source_records(path: Path, *, input_index: int) -> list[RawRecord]: ...
```

Detection order must be:

```python
try:
    top_level = json.loads(text)
except json.JSONDecodeError:
    return parse_nonempty_jsonl_lines(text)
else:
    return wrap_dict_or_list(top_level)
```

A syntactically broken JSONL line, or a JSONL line whose top level is not an object, is a file-level startup error. A syntactically valid top-level array may contain any JSON value: object entries proceed to schema normalization, while scalar/null/list entries remain `RawRecord` values and become per-record `invalid_schema` audit/rejected dispositions.

**Step 4: Run focused tests and commit**

```powershell
python -m pytest tests/test_dpo_input.py -k "loads or container or source_ref or invalid_json" -v
git add eval_tool/dpo_input.py tests/test_dpo_input.py
git commit -m "feat: load DPO JSON and JSONL inputs"
```

Expected: PASS before commit.

## Task 3: Normalize Alpaca/ShareGPT, images, turns, and conflicts

**Files:**

- Modify: `eval_tool/dpo_input.py`
- Modify: `tests/test_dpo_input.py`

**Step 1: Write failing normalization tests**

Add all of these cases:

```text
test_alpaca_instruction_and_nonempty_input_join_with_newline
test_alpaca_empty_or_missing_input_uses_instruction_only
test_alpaca_missing_blank_or_nonstring_instruction_is_missing_question
test_alpaca_missing_blank_or_nonstring_output_is_missing_chosen
test_alpaca_history_expands_every_gold_turn
test_alpaca_system_rejects_the_whole_record
test_sharegpt_role_aliases_normalize_to_human_and_gpt
test_sharegpt_every_turn_uses_only_prior_gold_history
test_unsupported_role_rejects_the_whole_record
test_misordered_or_unpaired_turn_rejects_the_whole_record
test_image_and_images_equal_forms_merge
test_conflicting_image_fields_reject_the_whole_record
test_text_single_image_and_multi_image_modalities
test_zero_placeholders_prepend_images_to_first_human
test_nonzero_placeholder_mismatch_rejects_the_whole_record
test_missing_image_is_rejected
test_image_root_only_changes_resolved_path_not_output_string
test_future_turn_images_do_not_leak_into_earlier_candidates
test_exact_duplicate_group_keeps_first_and_audits_every_copy
test_reference_conflict_excludes_every_source_in_group
test_sample_ids_and_candidate_order_are_stable
```

Include records whose chosen answers are one character long to prove the old 8-character/2.2-ratio rules are not present.

**Step 2: Confirm the new tests fail**

Run:

```powershell
python -m pytest tests/test_dpo_input.py -v
```

Expected: FAIL on normalization APIs.

**Step 3: Implement normalized candidates and dispositions**

Use explicit types:

```python
@dataclass(frozen=True)
class ImageRef:
    original: str
    resolved: Path

@dataclass(frozen=True)
class DpoTurn:
    from_: Literal["human", "gpt"]
    value: str

@dataclass(frozen=True)
class DpoCandidate:
    sample_id: str
    conversations: tuple[DpoTurn, ...]
    chosen: str
    images: tuple[ImageRef, ...]
    source: SourceRef
    source_format: Literal["alpaca", "sharegpt"]
    turn_index: int
    turn_count: int

ReasonCode = Literal[
    "invalid_schema", "unsupported_role", "conflicting_image_fields",
    "missing_question", "missing_chosen", "unpaired_turns", "missing_image",
    "image_placeholder_mismatch", "duplicate", "reference_conflict",
    "inference_error", "empty_rejected", "identical_pair", "judge_error",
    "judge_pass",
]

@dataclass(frozen=True)
class InputIssue:
    source: SourceRef
    sample_id: str | None
    reason_code: ReasonCode
    detail: str

@dataclass(frozen=True)
class NormalizationResult:
    candidates: tuple[DpoCandidate, ...]
    issues: tuple[InputIssue, ...]

@dataclass(frozen=True)
class InputSummary:
    source_path: Path
    sha256: str
    byte_count: int
    container_format: Literal["json_array", "json_object", "jsonl"]
    record_count: int

@dataclass(frozen=True)
class InputResult:
    candidates: tuple[DpoCandidate, ...]
    issues: tuple[InputIssue, ...]
    inputs: tuple[InputSummary, ...]

@dataclass(frozen=True)
class DedupeResult:
    candidates: tuple[DpoCandidate, ...]
    issues: tuple[InputIssue, ...]

def normalize_record(raw: RawRecord, *, image_root: Path) -> NormalizationResult: ...
def normalize_inputs(paths: Sequence[Path], *, image_root: Path) -> InputResult: ...
def deduplicate_candidates(candidates: Sequence[DpoCandidate]) -> DedupeResult: ...
```

Implementation rules:

- Alpaca history is strictly a list of two-string pairs; append the current instruction/input/output pair, then emit one candidate per pair.
- ShareGPT requires exact alternating user/assistant pairs after alias normalization. Any bad position rejects the original record; never skip and repair.
- Count `<image>` tokens across the entire source conversation. Zero placeholders plus images means prepend all placeholders to the first human turn. A nonzero mismatch rejects the record.
- While emitting turn `n`, include only image refs consumed through that turn's human message.
- Resolve absolute image paths directly and relative paths only against `image_root`; retain `ImageRef.original` unchanged.
- Compute `sample_id` from input content digest, source position, turn index, and canonical normalized content.
- Group dedupe by `(conversations, images)` first. More than one distinct chosen value makes every source `reference_conflict`; otherwise keep the first and mark every later source `duplicate`.
- Stable reason codes must use the approved list.
- `InputIssue` represents only rejected source/candidate states. Final report assembly creates one `AuditRecord(disposition="selected", reason_code=None)` or `AuditRecord(disposition="rejected", reason_code=...)` per normalized candidate. A legal multi-turn source therefore produces one audit row per candidate; a source value that cannot normalize produces exactly one source-level rejected audit row.

The primary reason-code set is fixed as:

```text
invalid_schema, unsupported_role, conflicting_image_fields, missing_question,
missing_chosen, unpaired_turns, missing_image, image_placeholder_mismatch,
duplicate, reference_conflict, inference_error, empty_rejected, identical_pair,
judge_error, judge_pass
```

**Step 4: Run focused tests and commit**

```powershell
python -m pytest tests/test_dpo_input.py -v
git add eval_tool/dpo_input.py tests/test_dpo_input.py
git commit -m "feat: normalize and deduplicate DPO conversations"
```

Expected: PASS before commit.

## Task 4: Build exact multimodal messages and extend Qwen safely

**Files:**

- Create: `eval_tool/dpo_multimodal.py`
- Modify: `eval_tool/infer.py`
- Create: `tests/test_dpo_multimodal.py`
- Modify: `tests/test_run_infer.py`

**Step 1: Write failing multimodal and compatibility tests**

Cover:

```text
test_split_image_placeholders_interleaves_text_and_images_once
test_multiturn_messages_consume_only_visible_images
test_mixed_png_and_jpeg_detect_mime_from_file_content
test_open_images_closes_source_files_and_returns_rgb_copies
test_preflight_rejects_unreadable_or_unsupported_images_before_dry_run
test_qwen_message_batch_flattens_images_in_message_order
test_qwen_message_batch_passes_false_and_true_enable_thinking
test_legacy_generate_batch_does_not_pass_enable_thinking
```

Patch the processor/model in tests; do not load Transformers.

**Step 2: Confirm focused failures**

```powershell
python -m pytest tests/test_dpo_multimodal.py tests/test_run_infer.py -k "thinking or message_batch or multimodal or generate_batch" -v
```

Expected: FAIL on missing structured-message support.

**Step 3: Add shared DPO multimodal helpers**

Implement:

```python
@dataclass(frozen=True)
class ImageAsset:
    ref: ImageRef
    mime_type: str
    sha256: str
    byte_count: int

@dataclass(frozen=True)
class ImagePreflightResult:
    assets: Mapping[Path, ImageAsset]
    issues_by_sample_id: Mapping[str, InputIssue]

def inspect_image(ref: ImageRef) -> ImageAsset: ...
def preflight_candidate_images(candidates: Sequence[DpoCandidate]) -> ImagePreflightResult: ...
def interleave_content(text: str, images: Iterator[Any]) -> list[dict[str, Any]]: ...
@contextmanager
def open_model_messages(
    candidate: DpoCandidate,
    assets: Mapping[Path, ImageAsset],
) -> Iterator[list[dict[str, Any]]]: ...
def image_data_url(asset: ImageAsset) -> str: ...
```

Split literal placeholders out of sent text and insert exactly one image part at each position. Preflight every unique image visible to a candidate before the dry-run return: hash bytes, use Pillow to verify readability/format, and derive MIME from content rather than extension. A corrupt/unreadable/unsupported image rejects only candidates that can see it using primary `missing_image` plus a detailed `unreadable_image` subreason. `open_model_messages()` opens each path with a context manager, converts/copies to RGB, yields role-preserving messages, and closes every owned RGB copy after the generator call; `dpo_infer` uses an `ExitStack` for a whole batch. OpenAI/Judge messages are built later from the same `ImageAsset` records and raw bytes, not a single-turn content shortcut.

**Step 4: Extend `QwenVLGenerator` without changing old inference**

Add:

```python
class MessageBatchGenerator(Protocol):
    def generate_message_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        enable_thinking: bool | None = None,
    ) -> list[str]: ...
```

`QwenVLGenerator.generate_message_batch()` applies the chat template to supplied messages, flattens image objects in message order, and performs the existing processor/model generation. The caller retains image ownership and keeps the `open_model_messages()` contexts alive until this method returns. Only add `enable_thinking` to `apply_chat_template` kwargs when it is not `None`. Existing `generate()`/`generate_batch()` must delegate with `None`, preserving all old tests and TSV behavior.

**Step 5: Run tests and commit**

```powershell
python -m pytest tests/test_dpo_multimodal.py tests/test_run_infer.py -k "thinking or message_batch or multimodal or generate_batch" -v
python -m pytest tests/test_run_infer.py -v
git add eval_tool/dpo_multimodal.py eval_tool/infer.py tests/test_dpo_multimodal.py tests/test_run_infer.py
git commit -m "feat: add structured multimodal generation"
```

Expected: PASS before commit.

## Task 5: Add strict phase caches and identity fingerprints

**Files:**

- Create: `eval_tool/dpo_cache.py`
- Create: `tests/test_dpo_cache.py`

**Step 1: Write failing strict-store tests**

Cover:

```text
test_store_loads_all_worker_shards_and_orders_by_expected_ids
test_store_resumes_only_missing_sample_ids
test_store_repairs_and_physically_truncates_only_final_partial_line
test_store_repairs_utf8_multibyte_truncation_only_at_the_binary_tail
test_store_rejects_middle_corruption
test_store_rejects_duplicate_or_conflicting_keys_across_shards
test_store_rejects_unknown_sample_id_and_invalid_schema
test_append_flushes_and_bounded_fsync_syncs
test_run_lock_rejects_a_live_second_process_and_releases_after_owner_death
test_nonowner_cannot_release_the_run_lock
test_overwrite_switches_to_a_new_attempt_without_in_place_delete
test_interrupted_overwrite_attempt_resumes_the_new_fingerprint
test_attempt_state_recovers_at_each_active_pointer_and_complete_marker_crash_point
```

**Step 2: Write failing fingerprint tests**

Cover:

```text
test_infer_fp_covers_order_normalizer_model_checkpoint_and_actual_messages
test_infer_fp_changes_for_image_bytes_thinking_batch_gpu_dtype_or_device
test_infer_fp_is_stable_when_only_workers_per_gpu_changes
test_checkpoint_identity_uses_config_hash_and_weight_name_size_mtime_ns
test_judge_request_fp_covers_api_base_model_temperature_timeout_tokens_rubric_prompt_messages_mime_and_image_bytes
test_judge_request_fp_excludes_api_key
test_judge_parse_fp_covers_raw_response_schema_parser_and_score_versions
```

**Step 3: Confirm failures**

```powershell
python -m pytest tests/test_dpo_cache.py -v
```

Expected: FAIL because strict stores/fingerprints do not exist.

**Step 4: Implement phase-specific stores**

Expose:

```python
class DpoCacheError(RuntimeError): ...

class StrictJsonlShardStore:
    def __init__(
        self,
        phase_dir: Path,
        fingerprint: str,
        key_fields: tuple[str, ...],
        expected_keys: Sequence[tuple[str, ...]],
        schema_validator: Callable[[Mapping[str, Any]], None],
        *,
        fsync_every: int = 32,
    ) -> None: ...
    def load(self) -> dict[tuple[str, ...], dict[str, Any]]: ...
    def append_batch(self, records: Sequence[Mapping[str, Any]], *, worker_id: int = 0) -> None: ...
    def sync(self) -> None: ...

class DpoInferenceStore(StrictJsonlShardStore): ...
class DpoJudgeRawStore(StrictJsonlShardStore): ...
class DpoJudgeParseStore(StrictJsonlShardStore): ...
```

Inference records are terminal records:

```json
{"sample_id":"...","status":"ok","prediction":"...","error_type":null,"error_message":null}
```

Judge raw and parse records are respectively:

```json
{"sample_id":"...","judge_request_fp":"...","raw_response":"..."}
{"sample_id":"...","judge_parse_fp":"...","status":"ok","result":{},"error_type":null,"error_message":null}
```

Use separate directories/manifests for inference, Judge raw request fingerprints, and Judge parse fingerprints. Inference records use key fields `(sample_id,)`; Judge raw records use `(sample_id, judge_request_fp)`; Judge parsed records use `(sample_id, judge_parse_fp)`. Each Judge phase also has a shared phase fingerprint for common configuration/prompt/parser identity, while the record-level fingerprint covers each sample's actual messages and images. `load()` validates schema/expected composite keys and inserts returned entries in expected-key order. Read JSONL as bytes: only an invalid UTF-8/JSON fragment after the last newline may be truncated; complete but invalid last records and all middle corruption are fatal. A file may have one writer or per-worker shards; it may never have uncoordinated shared appends. Append always flushes, fsync occurs at a bounded interval, and each worker syncs/closes its own handle in `finally` before the parent writes a phase-complete marker.

Add a run-wide cross-process advisory lock keyed by canonical output/work identity, held from startup verification through cleanup. On Windows use an OS-owned file lock (`msvcrt.locking`); on POSIX use `fcntl`, so process death releases ownership. The lock file stores an informational owner token/PID, but ownership is determined by the held descriptor. Do not change `InferenceLock`, `InferShardStore`, or `JsonlCache`.

Define a durable phase state machine rather than deleting state in place:

```text
PhaseAttempt = phase + schema_version + fingerprint + expected_id_digest/count + attempt_id
attempts/<attempt_id>/...worker shards...
active.json             # atomically selects the resumable attempt
complete.json           # written only after every writer syncs and strict reload succeeds
```

`--overwrite` first creates a new empty attempt and fsyncs its metadata, then atomically switches `active.json`; resume always follows the active attempt. Starting a new inference attempt invalidates downstream Judge attempts. Crash-point tests cover creation, active-pointer switch, first append, and complete-marker write.

**Step 5: Implement canonical fingerprints**

Add helpers:

```python
def canonical_sha256(value: Any) -> str: ...
def build_phase_fp(
    phase: Literal["inference", "judge_raw", "judge_parse"],
    common_identity: Mapping[str, Any],
    expected_keys: Sequence[tuple[str, ...]],
) -> str: ...
def checkpoint_identity(model_path: Path) -> dict[str, Any]: ...
def build_dpo_inference_fp(
    input_summaries: Sequence[InputSummary],
    candidates: Sequence[DpoCandidate],
    assets: Mapping[Path, ImageAsset],
    config: DpoInferConfig,
    checkpoint: Mapping[str, Any],
) -> str: ...
def build_judge_request_fp(
    inference_fp: str,
    settings: JudgeSettings,
    request_fingerprint_material: Mapping[str, Any],
    assets: Sequence[ImageAsset],
    *,
    max_tokens: int = 1024,
) -> str: ...
def build_judge_parse_fp(
    request_fp: str,
    raw_response: str,
    rubric: RubricName,
    has_images: bool,
) -> str: ...
```

Checkpoint identity recursively and stably hashes small model/config/tokenizer/processor/chat-template/generation/remote-code/index files. Weight files contribute stable relative path, size, and `mtime_ns` metadata as the approved minimum; document that this is a stat identity rather than a full multi-gigabyte weight hash. `workers_per_gpu` is a scheduling-only field and intentionally excluded so interrupted jobs can be repartitioned. `batch_size`, `gpu_ids`, dtype, device map, and all other generation-affecting settings remain included. Judge request identity includes every serialized request parameter, including `temperature`, `max_tokens`, model, endpoint, and timeout/retry policy when it can change the accepted raw response; it never includes the API key.

**Step 6: Run tests and commit**

```powershell
python -m pytest tests/test_dpo_cache.py -v
git add eval_tool/dpo_cache.py tests/test_dpo_cache.py
git commit -m "feat: add strict resumable DPO caches"
```

Expected: PASS before commit.

## Task 6: Add four-way Judge prompts, structured transport, and strict parsing

**Files:**

- Create: `prompts/judge_dpo_binary_visual.txt`
- Create: `prompts/judge_dpo_binary_text.txt`
- Create: `prompts/judge_dpo_v4_text.txt`
- Create: `eval_tool/dpo_prompts.py`
- Modify: `eval_tool/judge.py`
- Create: `eval_tool/dpo_judge.py`
- Create: `tests/test_dpo_prompts.py`
- Create: `tests/test_dpo_judge.py`
- Modify: `tests/test_judge_prompts.py`

**Step 1: Write failing prompt-routing tests**

Assert:

- binary visual and binary text use different prompts;
- visual v4 resolves to the existing `prompts/judge_equip_pointwise_v4.txt`;
- text v4 requires numeric `fact_score`, `visual_score=null`, and defines `equipment_correct` as task-subject understanding;
- text binary contains no must-see-image criterion.

Use this API:

```python
def dpo_prompt_path(rubric: str, has_images: bool) -> Path: ...
def load_dpo_prompt(rubric: str, has_images: bool) -> str: ...
```

**Step 2: Write failing Judge request/schema tests**

Cover:

```text
test_structured_transport_preserves_existing_post_override_compatibility
test_multiturn_judge_includes_gold_history_current_chosen_and_rejected
test_judge_interleaves_png_and_jpeg_at_original_placeholder_positions
test_binary_accepts_only_literal_boolean_correct
test_v4_accepts_only_literal_boolean_equipment_correct
test_v4_numeric_fields_reject_bool_string_nan_and_out_of_range
test_text_v4_requires_numeric_fact_and_null_visual
test_visual_v4_keeps_existing_nullable_dimensions
test_v4_uses_parser_hit_not_rounded_quality_score
test_transport_or_parse_failure_is_judge_error_not_wrong_answer
```

**Step 3: Confirm failures**

```powershell
python -m pytest tests/test_dpo_prompts.py tests/test_dpo_judge.py tests/test_judge_prompts.py -v
```

Expected: FAIL on missing prompt/Judge APIs.

**Step 4: Add structured Judge transport compatibly**

Keep all existing `_post()`, `judge_pointwise()`, and `judge_pairwise()` signatures and call paths. Add:

```python
def _post_messages(self, messages: list[dict[str, Any]], *, max_tokens: int = 1024) -> str: ...
def judge_messages(self, messages: list[dict[str, Any]], *, max_tokens: int = 1024) -> str: ...
```

Existing public methods must still call `_post()` so subclasses in current tests continue to work.

**Step 5: Build and strictly parse DPO Judge requests**

Implement:

```python
@dataclass(frozen=True)
class DpoJudgeRequest:
    rubric: Literal["binary", "v4"]
    has_images: bool
    system_prompt: str
    messages: tuple[dict[str, Any], ...]
    image_assets: tuple[ImageAsset, ...]
    fingerprint_material: Mapping[str, Any]

def build_dpo_judge_request(
    candidate: DpoCandidate,
    rejected: str,
    prompt: str,
    assets: Mapping[Path, ImageAsset],
) -> DpoJudgeRequest: ...
def validate_dpo_judge_object(obj: dict[str, Any], rubric: str, has_images: bool) -> None: ...
def parse_dpo_judge_response(raw: str, rubric: str, has_images: bool) -> dict[str, object]: ...
```

Use `extract_json_object()`, run strict validation first, then call `parse_v1()` for binary or `parse_v4()` for v4. Explicitly exclude Python bool from numeric fields. Do not call permissive `parse_pointwise()` first. Preserve the parser-returned `hit` for selection.

Judge messages contain system prompt; all prior human/gpt gold turns; current human; and an explicit current chosen/rejected comparison. Replace each `<image>` with one data URL carrying its own MIME. `fingerprint_material` mirrors every role/text/image position but replaces each data URL with original/resolved path, MIME, byte count, and SHA-256; it is therefore complete without storing base64 in fingerprints or audit output.

**Step 6: Run tests and commit**

```powershell
python -m pytest tests/test_dpo_prompts.py tests/test_dpo_judge.py tests/test_judge_prompts.py -v
git add prompts/judge_dpo_binary_visual.txt prompts/judge_dpo_binary_text.txt prompts/judge_dpo_v4_text.txt eval_tool/dpo_prompts.py eval_tool/judge.py eval_tool/dpo_judge.py tests/test_dpo_prompts.py tests/test_dpo_judge.py tests/test_judge_prompts.py
git commit -m "feat: add strict multimodal DPO judging"
```

Expected: PASS before commit.

## Task 7: Run resilient local inference and concurrent judging

**Files:**

- Create: `eval_tool/dpo_infer.py`
- Modify: `eval_tool/dpo_judge.py`
- Create: `tests/test_dpo_infer.py`
- Modify: `tests/test_dpo_judge.py`

**Step 1: Write failing inference isolation tests**

Cover:

```text
test_model_receives_gold_history_and_current_human_only
test_successful_batch_appends_each_terminal_result_immediately
test_data_batch_failure_bisects_in_original_order_to_one_bad_sample
test_empty_and_identical_outputs_remain_cached_successes_for_pipeline_filtering
test_oom_cuda_context_worker_exit_or_model_load_failure_aborts_the_run
test_short_batch_response_is_a_fatal_generator_protocol_error
test_completed_worker_shards_survive_a_parallel_worker_failure
test_resume_with_changed_workers_per_gpu_runs_only_pending_samples
test_parallel_completion_order_does_not_change_result_order
test_windows_spawn_worker_receives_only_picklable_dto_values
test_one_fatal_worker_terminates_a_blocked_peer_without_more_appends
```

**Step 2: Implement resilient inference APIs**

```python
@dataclass(frozen=True)
class InferenceOutcome:
    sample_id: str
    prediction: str | None
    error_type: str | None = None
    error_message: str | None = None

def infer_batch_with_isolation(
    batch: Sequence[DpoCandidate],
    generator: MessageBatchGenerator,
    *,
    enable_thinking: bool,
    append: Callable[[Sequence[InferenceOutcome]], None],
) -> list[InferenceOutcome]: ...

def run_dpo_inference(
    candidates: Sequence[DpoCandidate],
    assets: Mapping[Path, ImageAsset],
    config: DpoInferConfig,
    store: DpoInferenceStore,
    *,
    generator_factory: Callable[[DpoInferConfig], MessageBatchGenerator] | None = None,
) -> dict[str, InferenceOutcome]: ...
```

Load the model outside recursive isolation. Non-infrastructure exceptions bisect batches; a single failing sample becomes `inference_error`. `MemoryError`, CUDA OOM/context/NCCL failures, broken worker process, nonzero worker exit, and model unavailability abort the run. Prediction count mismatch is fatal because attribution is unreliable.

For explicit GPUs, use module-level worker entry points with `multiprocessing.get_context("spawn")`. Cross-process arguments are limited to serializable candidate DTOs, config scalars, paths, fingerprint/attempt/worker IDs, and queues/events. Never pass a generator, store, callback, PIL object, open handle, lambda, or local factory. Workers reconstruct their store/writer and default generator after setting `CUDA_VISIBLE_DEVICES`; do not import torch/transformers at module import time. Custom generator factories are supported only in the sequential fake-test path unless expressed as an importable module-level factory.

In explicit-GPU mode normalize `device_map` to `auto`/worker-local `cuda:0`; never pass a host GPU ordinal into the masked worker. Use a readiness barrier so every worker reports successful GPU binding and model load before inference begins. A shared fatal event stops submission; on the first fatal outcome, terminate/join live workers, preserve already synced shards, and raise immediately. Prefer explicit `multiprocessing.Process`/Queue/Event ownership over `ProcessPoolExecutor` where forced peer termination is otherwise unavailable. The parent strictly reloads and orders results by original candidate IDs.

**Step 3: Write failing Judge runner tests**

Cover max-retry backoff, `max_workers`, raw-before-parse cache ordering, parse-only resume without API calls, single-writer cache coordination, and writer failure before parse append. Network workers return values only; the coordinator owns all cache writes. Construct one client per worker/thread unless the supplied factory explicitly documents thread safety.

Run:

```powershell
python -m pytest tests/test_dpo_judge.py -k "retry or worker or cache or resume" -v
```

Expected: FAIL because the concurrent runner is not implemented.

**Step 4: Implement the Judge runner**

Expose:

```python
@dataclass(frozen=True)
class JudgeJob:
    candidate: DpoCandidate
    rejected: str
    inference_fp: str
    request: DpoJudgeRequest

@dataclass(frozen=True)
class JudgeOutcome:
    sample_id: str
    request_fp: str
    parse_fp: str | None
    raw_response: str | None
    parsed: Mapping[str, object] | None
    error_type: str | None
    error_message: str | None

def judge_candidates(
    jobs: Sequence[JudgeJob],
    config: DpoJudgeConfig,
    raw_store: DpoJudgeRawStore,
    parse_store: DpoJudgeParseStore,
    *,
    client_factory: Callable[[JudgeSettings], JudgeClient] | None = None,
) -> dict[str, JudgeOutcome]: ...
```

API success writes raw first. Parse failure writes a terminal `judge_error` parse outcome while preserving raw for later parser-only replay. Transport failures retry with bounded backoff and do not get interpreted as failed answers.

**Step 5: Run focused tests and commit**

```powershell
python -m pytest tests/test_dpo_infer.py tests/test_dpo_judge.py -v
git add eval_tool/dpo_infer.py eval_tool/dpo_judge.py tests/test_dpo_infer.py tests/test_dpo_judge.py
git commit -m "feat: run resumable DPO inference and judging"
```

Expected: PASS before commit.

## Task 8: Build verified audit artifacts and manifest-last publication

**Files:**

- Create: `eval_tool/dpo_report.py`
- Create: `tests/test_dpo_report.py`

**Step 1: Write failing artifact tests**

Cover:

```text
test_training_rows_have_exactly_conversations_chosen_rejected_images
test_invalid_valid_json_record_appears_in_audit_and_rejected
test_every_duplicate_and_conflict_source_has_its_own_disposition
test_summary_counts_match_audit_and_rejected_by_stable_reason
test_manifest_records_sha256_byte_count_and_line_count
test_manifest_and_warnings_recursively_redact_api_keys
test_manifest_records_input_model_checkpoint_thinking_and_phase_fingerprints
test_audit_fields_include_source_turn_digests_prediction_judge_and_disposition
test_summary_groups_by_input_format_modality_turn_shape_and_reason
test_configured_output_name_is_used_without_changing_other_artifact_names
test_warnings_log_is_always_published
test_manifest_is_replaced_last
test_publish_failure_restores_previous_committed_artifacts
test_forced_process_exit_after_each_replace_recovers_previous_committed_set
test_output_volume_temp_copy_is_rehashed_before_replace
test_committed_manifest_mismatch_is_rejected_on_next_start
test_existing_training_output_without_manifest_is_not_trusted_or_overwritten
test_zero_selected_does_not_touch_output_and_writes_failed_run_diagnostics
```

**Step 2: Confirm failures**

```powershell
python -m pytest tests/test_dpo_report.py -v
```

Expected: FAIL because report APIs do not exist.

**Step 3: Implement report/publication APIs**

```python
@dataclass(frozen=True)
class ArtifactStat:
    sha256: str
    byte_count: int
    line_count: int | None

@dataclass(frozen=True)
class AuditRecord:
    source: Mapping[str, Any]
    sample_id: str | None
    source_format: Literal["alpaca", "sharegpt"] | None
    turn_index: int | None
    turn_count: int | None
    modality: Literal["text", "single_image", "multi_image"] | None
    digests: Mapping[str, str]
    prediction: str | None
    inference_error: str | None
    judge: Mapping[str, Any] | None
    disposition: Literal["selected", "rejected"]
    reason_code: ReasonCode | None
    detail: str

def redact_config(value: Any) -> Any: ...
def compute_artifact_stat(path: Path, *, jsonl: bool) -> ArtifactStat: ...
def build_summary(audit_records: Sequence[AuditRecord]) -> dict[str, Any]: ...
def stage_dpo_artifacts(
    training_rows: Sequence[Mapping[str, Any]],
    audit_records: Sequence[AuditRecord],
    summary: Mapping[str, Any],
    warnings: Sequence[str],
    manifest_data: Mapping[str, Any],
    *,
    output_name: str,
    staging_dir: Path,
) -> dict[str, Path]: ...
def validate_staged_artifacts(
    staged: Mapping[str, Path], *, output_name: str
) -> dict[str, ArtifactStat]: ...
def recover_incomplete_publication(output_dir: Path, *, output_name: str) -> None: ...
def publish_staged_artifacts(
    staged: Mapping[str, Path],
    stats: Mapping[str, ArtifactStat],
    *,
    output_dir: Path,
    output_name: str,
    run_id: str,
) -> dict[str, Path]: ...
def verify_committed_artifacts(
    output_dir: Path, *, output_name: str
) -> dict[str, Any] | None: ...
def write_failed_run(
    work_dir: Path,
    run_id: str,
    audit_records: Sequence[AuditRecord],
    summary: Mapping[str, Any],
    warnings: Sequence[str],
    error: str,
) -> Path: ...
```

Each audit record has a field-level contract: `sample_id` when normalization reached candidate scope; source path/input index/container kind/array index or JSONL line/source ID/raw digest; source format; turn index/count; conversation/chosen/image digests; modality; prediction or inference error; Judge request/parse fingerprints and parsed verdict when applicable; final `disposition`; primary `reason_code`; and a readable detail. Invalid but syntactically legal source values use the same source fields with candidate-only fields set to null. `rejected_records.jsonl` contains the corresponding nonselected audit records, not a lossy count-only projection.

`summary.json` aggregates totals by input file, original Alpaca/ShareGPT/invalid format, pure-text/single-image/multi-image modality, single-turn/multi-turn shape, selected/nonselected disposition, and stable primary reason code. Tests must reconcile each aggregate against audit records.

`manifest.json` has a fixed schema version and contains: redacted effective config; ordered input path/content summaries; normalizer version; model name/resolved path/checkpoint identity; `enable_thinking` and all generation settings; inference fingerprint; Judge request/parse fingerprints when used; run/result counts; and every non-manifest delivered artifact's digest/statistics. It never attempts to hash itself. API keys, authorization values, and data URLs are forbidden. Verification receives the expected `output_name` and rejects an absent manifest or any managed orphan/mismatched artifact rather than trusting or overwriting it.

Stage under `work_dir/staging/<run_id>`. Because work/output may be on different volumes, first copy every artifact into an output-volume transaction directory, flush/fsync it, then recompute SHA-256/byte/line statistics and compare them with staging before any replacement.

Use a durable output-side journal under `.dpo_publish/<run_id>/` with the old manifest, per-file backups/existence flags, the expected new manifest, and replacement progress. Fsync journal updates. Publish the configured `output_name`, `audit_records.jsonl`, `rejected_records.jsonl`, `summary.json`, and `warnings.log` first and `manifest.json` last. Mark the journal committed only after the new manifest and every fixed-name artifact verify. On startup, while holding the run lock, recover an uncommitted journal by restoring the complete previous set (and removing files that previously did not exist); if the new manifest already verifies fully, finalize that committed set instead. Then verify the committed manifest. This preserves fixed delivery filenames while making process-kill interruption recoverable rather than merely detectable.

The training JSONL manifest entry must include SHA-256, byte count, and line count. Transaction/temp/backup names contain a validated run ID and stay below the resolved output directory. After success, remove only this run's staging/journal backups; after failure, remove unpublished output temps but keep failed-run diagnostics.

With zero selected samples, never create/replace any successful output artifact. Write `audit_records.jsonl`, `rejected_records.jsonl`, `summary.json`, and an error summary beneath `work_dir/failed_runs/<run_id>/`, then raise an error carrying that path.

**Step 4: Run tests and commit**

```powershell
python -m pytest tests/test_dpo_report.py -v
git add eval_tool/dpo_report.py tests/test_dpo_report.py
git commit -m "feat: publish verified DPO artifacts"
```

Expected: PASS before commit.

## Task 9: Orchestrate dry-run, selection, resume, and cleanup

**Files:**

- Create: `eval_tool/dpo_pipeline.py`
- Create: `tests/test_dpo_pipeline.py`
- Create: `tests/test_dpo_pipeline_resume.py`

**Step 1: Write failing fake end-to-end tests**

Cover:

```text
test_full_mode_never_constructs_or_calls_judge
test_fake_mixed_input_pipeline_publishes_reconciled_strict_artifacts
test_wrong_only_binary_keeps_all_and_only_correct_false
test_wrong_only_v4_keeps_all_and_only_hit_zero
test_judge_error_and_judge_pass_have_distinct_dispositions
test_empty_rejected_identical_pair_and_inference_error_are_filtered
test_short_valid_answers_are_not_length_filtered
test_dry_run_constructs_neither_generator_nor_judge
test_dry_run_reports_normalization_dedupe_conflicts_and_images
test_resume_runs_only_missing_inference_and_judge_jobs
test_judge_setting_or_rubric_change_reuses_inference
test_input_model_generation_image_or_checkpoint_change_requires_overwrite
test_parser_only_change_reuses_judge_raw
test_overwrite_resets_inference_judge_raw_and_judge_parse_then_rebuilds
test_output_follows_input_turn_order_not_completion_order
test_clean_partial_runs_only_after_success_and_keeps_delivery_artifacts
test_unreadable_input_or_uncreatable_output_work_dir_fails_before_generator
test_existing_output_without_manifest_fails_before_gpu_work
test_zero_valid_raises_with_failed_run_path
```

**Step 2: Confirm failures**

```powershell
python -m pytest tests/test_dpo_pipeline.py tests/test_dpo_pipeline_resume.py -v
```

Expected: FAIL because pipeline entry points do not exist.

**Step 3: Implement the orchestration boundary**

```python
class DpoPipelineError(RuntimeError): ...

@dataclass(frozen=True)
class DpoRunResult:
    output_path: Path | None
    artifacts: Mapping[str, Path]
    selected_count: int
    dry_run: bool
    failed_run_dir: Path | None
    summary: Mapping[str, Any]

def run_build_dpo(
    config: DpoBuildConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    clean_partial: bool = False,
    generator_factory=None,
    judge_client_factory=None,
) -> DpoRunResult: ...
```

Pipeline order is fixed and runs entirely inside one run-wide lock:

1. Validate/create output/work directories, acquire the canonical run lock, recover any incomplete publish journal, then verify existing committed artifacts.
2. Load, normalize, validate, deduplicate, preflight/hash every candidate-visible image, and build complete audit dispositions.
3. For dry-run, return stats only after image readability/MIME/hash validation, but before constructing generator/Judge.
4. Build inference identity, validate/prepare phase state, and run only missing candidates.
5. Filter `inference_error`, `empty_rejected`, and whitespace-trimmed `identical_pair`.
6. In full mode, print that Judge is skipped and select every remaining candidate without constructing a client.
7. In wrong-only mode, build request/parse fingerprints, reuse raw/parsed caches independently, and keep only parser `hit == 0` results.
8. Build strict training rows in original order; stage, validate, and atomically publish artifacts.
9. Only after successful final publication, honor `clean_partial` for reproducible intermediate caches and this run's staging/transaction leftovers; release the run lock in `finally`.

`overwrite=True` coordinates all three phase stores without an in-place deletion window: while holding the run lock, durably create a new inference attempt, atomically switch its active pointer, then create/switch fresh downstream Judge raw and Judge parse attempts derived from it. Rebuild through the new attempts and garbage-collect old attempts only after successful publication. A startup read/permission/directory/manifest failure occurs before either factory is invoked.

Every syntactically valid source record and every normalized candidate must end in exactly one audit disposition. Every nonselected item must also appear in `rejected_records.jsonl` with one stable primary reason.

**Step 4: Run focused tests and commit**

```powershell
python -m pytest tests/test_dpo_pipeline.py tests/test_dpo_pipeline_resume.py -v
git add eval_tool/dpo_pipeline.py tests/test_dpo_pipeline.py tests/test_dpo_pipeline_resume.py
git commit -m "feat: orchestrate direct DPO dataset builds"
```

Expected: PASS before commit.

## Task 10: Expose `build-dpo` CLI and document the workflow

**Files:**

- Modify: `eval_tool/cli.py`
- Modify: `tests/test_cli.py`
- Create: `dpo.example.json`
- Modify: `README.md`

**Step 1: Write failing CLI tests**

Cover:

```text
test_build_dpo_routes_loaded_config_to_pipeline
test_build_dpo_parser_collects_repeated_inputs_as_replacement
test_build_dpo_parser_accepts_dry_run_overwrite_and_clean_partial
test_build_dpo_config_error_is_rendered_as_parser_error
test_build_dpo_pipeline_error_is_rendered_as_parser_error
test_build_dpo_help_is_available
test_top_level_help_lists_build_dpo
```

Also extend the existing subcommand parametrization without changing legacy no-subcommand behavior.

**Step 2: Confirm failures**

```powershell
python -m pytest tests/test_cli.py -v
```

Expected: FAIL because `build-dpo` is not registered.

**Step 3: Wire the isolated handler**

Add `build-dpo` to `SUBCOMMANDS`, parser, and `HANDLERS`:

```python
def _handle_build_dpo(args: argparse.Namespace) -> DpoRunResult:
    config = load_dpo_config(
        args.config,
        input_overrides=args.inputs,
        invocation_dir=Path.cwd(),
    )
    return run_build_dpo(
        config,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        clean_partial=args.clean_partial,
    )
```

Flags:

```text
--config PATH       required
--input PATH        repeatable; replaces config inputs when present
--dry-run
--overwrite
--clean-partial
```

Catch DPO config/pipeline errors through the same parser error presentation as existing commands. Ensure adding the name to `SUBCOMMANDS` prevents fallback into legacy eval.

**Step 4: Add example and README documentation**

Document:

- mixed multi-file inputs;
- Alpaca and ShareGPT examples including multi-turn/multi-image;
- `wrong_only` false versus binary/v4 true;
- model/Judge configuration and `enable_thinking` default false;
- all CLI flags;
- output, work cache, failed run, and unchanged image-path semantics;
- no TSV/XLSX conversion;
- optional real-service smoke testing status.

**Step 5: Run tests and commit**

```powershell
python -m pytest tests/test_cli.py -v
git add eval_tool/cli.py tests/test_cli.py dpo.example.json README.md
git commit -m "feat: expose direct JSON DPO builder CLI"
```

Expected: PASS before commit.

## Task 11: Run regression, offline end-to-end verification, and review

**Files:**

- Modify only files required by failures attributable to this feature.
- Optionally create small temporary fixtures outside tracked source; do not commit generated outputs.

**Step 1: Run targeted legacy regressions**

```powershell
python -m pytest tests/test_run_infer.py tests/test_run_infer_resume.py tests/test_infer_cache.py tests/test_judge_prompts.py tests/test_run_eval.py tests/test_cache.py tests/test_rubrics.py tests/test_convert_vqa_json.py tests/test_pipeline.py tests/test_pipeline_config.py -v
```

Expected: PASS; old TSV infer/eval/all behavior is unchanged.

**Step 2: Run the full test suite**

```powershell
python -m pytest -q
```

Expected: all tests PASS.

**Step 3: Run the repository offline end-to-end check**

```powershell
python e2e_check.py
```

Expected: exit code 0.

**Step 4: Run process-safety tests explicitly**

```powershell
python -m pytest tests/test_dpo_cache.py -k "process or utf8 or attempt" -v
python -m pytest tests/test_dpo_infer.py -k "spawn or fatal_worker" -v
python -m pytest tests/test_dpo_report.py -k "forced_process_exit or output_volume" -v
```

Expected on Windows: all spawn, owner-death lock, UTF-8 truncation, worker termination, and publish-recovery tests PASS. Platform-specific cases may be conditionally skipped elsewhere, and the final report must name any skips.

**Step 5: Exercise CLI dry-run with mixed temporary fixtures**

Create temporary Alpaca JSON and ShareGPT JSONL inputs plus tiny PNG/JPEG fixtures, then run:

```powershell
python -m eval_tool build-dpo --config <temporary-config> --dry-run
```

Expected: no model/Judge construction, no TSV/XLSX, and stable normalization/dedupe/image statistics.

**Step 6: Exercise the fixed fake full-pipeline integration node**

Run:

```powershell
python -m pytest tests/test_dpo_pipeline.py::test_fake_mixed_input_pipeline_publishes_reconciled_strict_artifacts -v
```

The node itself must assert that:

- every training line has only `conversations`, `chosen`, `rejected`, `images`;
- manifest SHA-256/byte/line counts match;
- audit/rejected/summary counts reconcile;
- image path strings equal the inputs.

**Step 7: Request code review and fix verified issues**

Use `superpowers:requesting-code-review`. Ask one reviewer to check requirements coverage and another to inspect cache/publish failure safety. Apply feedback through `superpowers:receiving-code-review`, rerun the affected focused tests, then rerun the full suite.

**Step 8: Final verification and completion commit**

Use `superpowers:verification-before-completion`, capture fresh command output, and commit only if any review fixes or documentation changes remain:

```powershell
git status --short
git diff --check
python -m pytest -q
python e2e_check.py
```

Expected: only intended feature files differ from the base; whitespace check and both verification commands pass.

If no real local model/Judge call was run, final handoff must state: “真实本地模型/Judge API 冒烟测试未执行；离线 fake 集成与完整回归已执行。”
