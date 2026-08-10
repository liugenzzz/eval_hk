from __future__ import annotations

import hashlib
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from .dpo_cache import (
    DPO_NORMALIZER_VERSION,
    DpoInferenceStore,
    DpoJudgeParseStore,
    DpoJudgeRawStore,
    DpoRunLock,
    build_dpo_inference_fp,
    build_judge_request_fp,
    build_phase_fp,
    canonical_sha256,
    checkpoint_identity,
)
from .dpo_config import DpoBuildConfig
from .dpo_infer import run_dpo_inference
from .dpo_input import (
    DpoCandidate,
    ImageRef,
    InputIssue,
    InputResult,
    ReasonCode,
    SourceRef,
    deduplicate_candidates,
    normalize_inputs,
)
from .dpo_judge import JudgeJob, build_dpo_judge_request, judge_candidates
from .dpo_multimodal import ImagePreflightResult, preflight_candidate_images
from .dpo_prompts import load_dpo_prompt
from .dpo_report import (
    AuditRecord,
    build_summary,
    publish_staged_artifacts,
    recover_incomplete_publication,
    stage_dpo_artifacts,
    validate_staged_artifacts,
    verify_committed_artifacts,
    write_failed_run,
)


DPO_PIPELINE_VERSION = "dpo-pipeline-v1"
_PHASE_DIRS = ("inference", "judge_raw", "judge_parse")


class DpoPipelineError(RuntimeError):
    """A direct JSON/JSONL DPO build cannot proceed or publish safely."""


@dataclass(frozen=True)
class DpoRunResult:
    output_path: Path | None
    artifacts: Mapping[str, Path]
    selected_count: int
    dry_run: bool
    failed_run_dir: Path | None
    summary: Mapping[str, Any]


@dataclass
class _Unit:
    """One audit row: a normalized candidate or a rejected source value."""

    source: SourceRef
    sample_id: str | None
    candidate: DpoCandidate | None
    reason_code: ReasonCode | None = None
    detail: str = ""
    prediction: str | None = None
    inference_error: str | None = None
    judge: dict[str, Any] | None = None


def run_build_dpo(
    config: DpoBuildConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    clean_partial: bool = False,
    generator_factory=None,
    judge_client_factory=None,
) -> DpoRunResult:
    """Build one DPO dataset from JSON/JSONL inputs under a single run lock."""

    if not isinstance(config, DpoBuildConfig):
        raise DpoPipelineError("run_build_dpo requires a DpoBuildConfig")
    for name, flag in (
        ("dry_run", dry_run),
        ("overwrite", overwrite),
        ("clean_partial", clean_partial),
    ):
        if type(flag) is not bool:
            raise DpoPipelineError(f"{name} must be a literal boolean")

    run_id = _new_run_id()
    output_dir = _prepare_directory(config.output_dir, "output")
    work_dir = _prepare_directory(config.work_dir, "work")

    lock = DpoRunLock.for_identity(output_dir, work_dir)
    with _translate("DPO run lock"):
        lock.acquire()
    try:
        return _run_locked(
            config,
            run_id=run_id,
            output_dir=output_dir,
            work_dir=work_dir,
            dry_run=dry_run,
            overwrite=overwrite,
            clean_partial=clean_partial,
            generator_factory=generator_factory,
            judge_client_factory=judge_client_factory,
        )
    finally:
        lock.release()


def _run_locked(
    config: DpoBuildConfig,
    *,
    run_id: str,
    output_dir: Path,
    work_dir: Path,
    dry_run: bool,
    overwrite: bool,
    clean_partial: bool,
    generator_factory,
    judge_client_factory,
) -> DpoRunResult:
    warnings: list[str] = []

    # 1. Recover an interrupted publication and refuse to touch an output set
    #    that no longer matches its own manifest, before any model work.
    with _translate("existing DPO output"):
        recover_incomplete_publication(output_dir, output_name=config.output_name)
        verify_committed_artifacts(output_dir, output_name=config.output_name)

    # 2. Normalize every input into candidates plus complete rejections.
    with _translate("DPO input"):
        inputs = normalize_inputs(config.inputs, image_root=config.image_root)
        dedupe = deduplicate_candidates(inputs.candidates)
        preflight = preflight_candidate_images(dedupe.candidates)

    by_sample = {candidate.sample_id: candidate for candidate in inputs.candidates}
    runnable = tuple(
        candidate
        for candidate in dedupe.candidates
        if candidate.sample_id not in preflight.issues_by_sample_id
    )
    units = _build_units(
        issues=(
            *inputs.issues,
            *dedupe.issues,
            *preflight.issues_by_sample_id.values(),
        ),
        runnable=runnable,
        by_sample=by_sample,
    )

    # 3. A dry run stops after image readability/MIME/hash validation.
    if dry_run:
        summary = _dry_run_summary(config, inputs, units, runnable, preflight)
        _print_dry_run(summary)
        return DpoRunResult(
            output_path=None,
            artifacts=MappingProxyType({}),
            selected_count=0,
            dry_run=True,
            failed_run_dir=None,
            summary=summary,
        )

    if not runnable:
        raise _failed_run(
            config,
            work_dir,
            run_id,
            units,
            warnings,
            "no DPO candidate survived normalization, deduplication, and image preflight",
        )

    # 4. Run only the candidates that are still missing from the inference cache.
    with _translate("DPO model identity"):
        checkpoint = checkpoint_identity(config.infer.model_path)
        inference_fp = build_dpo_inference_fp(
            inputs.inputs, runnable, preflight.assets, config.infer, checkpoint
        )
    with _translate("DPO inference cache"):
        store = DpoInferenceStore(
            work_dir / "inference",
            inference_fp,
            [candidate.sample_id for candidate in runnable],
            overwrite=overwrite,
        )
        if overwrite:
            store.start_new_attempt()
    with _translate("DPO inference"):
        outcomes = run_dpo_inference(
            runnable,
            preflight.assets,
            config.infer,
            store,
            generator_factory=generator_factory,
        )
    warnings.extend(store.warnings)

    # 5. Filter unusable rejected answers before any Judge call.
    remaining = _filter_inference_outcomes(units, outcomes)

    judge_fingerprints: dict[str, str] = {}
    if not config.wrong_only:
        # 6. Full mode never constructs a Judge client.
        print(
            "wrong_only=false: the Judge is skipped and every generated answer is kept."
        )
        selected = remaining
    else:
        # 7. Wrong-only mode keeps parser hit == 0 verdicts only.
        selected = _judge_remaining(
            config,
            remaining,
            work_dir=work_dir,
            inference_fp=inference_fp,
            assets=preflight.assets,
            overwrite=overwrite,
            judge_client_factory=judge_client_factory,
            fingerprints=judge_fingerprints,
        )

    audit_records = [_audit_record(unit) for unit in units]
    summary = build_summary(audit_records)
    if not selected:
        raise _failed_run(
            config,
            work_dir,
            run_id,
            units,
            warnings,
            "no DPO sample was selected; nothing was published",
        )

    # 8. Publish in original source and turn order.
    training_rows = [
        _training_row(unit.candidate, prediction) for unit, prediction in selected
    ]
    manifest_data = _manifest_data(
        config,
        run_id=run_id,
        inputs=inputs,
        checkpoint=checkpoint,
        inference_fp=inference_fp,
        judge_fingerprints=judge_fingerprints,
        candidate_count=len(runnable),
    )
    staging_dir = work_dir / "staging" / run_id
    with _translate("DPO artifact publication"):
        staged = stage_dpo_artifacts(
            training_rows,
            audit_records,
            summary,
            warnings,
            manifest_data,
            output_name=config.output_name,
            staging_dir=staging_dir,
        )
        stats = validate_staged_artifacts(staged, output_name=config.output_name)
        published = publish_staged_artifacts(
            staged,
            stats,
            output_dir=output_dir,
            output_name=config.output_name,
            run_id=run_id,
        )

    # 9. Only a fully published run may drop reproducible intermediate state.
    if clean_partial:
        _clean_partial(work_dir, staging_dir)

    return DpoRunResult(
        output_path=published[config.output_name],
        artifacts=MappingProxyType(dict(published)),
        selected_count=len(training_rows),
        dry_run=False,
        failed_run_dir=None,
        summary=MappingProxyType(dict(summary)),
    )


# ---------------------------------------------------------------------------
# normalization bookkeeping
# ---------------------------------------------------------------------------


def _build_units(
    *,
    issues: Sequence[InputIssue],
    runnable: Sequence[DpoCandidate],
    by_sample: Mapping[str, DpoCandidate],
) -> list[_Unit]:
    units = [
        _Unit(
            source=issue.source,
            sample_id=issue.sample_id,
            candidate=by_sample.get(issue.sample_id) if issue.sample_id else None,
            reason_code=issue.reason_code,
            detail=issue.detail,
        )
        for issue in issues
    ]
    units.extend(
        _Unit(
            source=candidate.source,
            sample_id=candidate.sample_id,
            candidate=candidate,
        )
        for candidate in runnable
    )
    units.sort(key=_order_key)
    return units


def _order_key(unit: _Unit) -> tuple[int, int, int]:
    turn_index = unit.candidate.turn_index if unit.candidate is not None else -1
    return (unit.source.input_index, unit.source.record_index, turn_index)


def _filter_inference_outcomes(
    units: Sequence[_Unit], outcomes: Mapping[str, Any]
) -> list[tuple[_Unit, str]]:
    remaining: list[tuple[_Unit, str]] = []
    for unit in units:
        candidate = unit.candidate
        if candidate is None or unit.reason_code is not None:
            continue
        outcome = outcomes.get(candidate.sample_id)
        if outcome is None:
            raise DpoPipelineError(
                f"inference result is missing for candidate {candidate.sample_id}"
            )
        if outcome.error_type is not None:
            unit.reason_code = "inference_error"
            unit.inference_error = f"{outcome.error_type}: {outcome.error_message}"
            unit.detail = "the model did not produce an answer for this candidate"
            continue
        prediction = outcome.prediction or ""
        unit.prediction = prediction
        if not prediction.strip():
            unit.reason_code = "empty_rejected"
            unit.detail = "the generated rejected answer is empty"
            continue
        if prediction.strip() == candidate.chosen.strip():
            unit.reason_code = "identical_pair"
            unit.detail = "the generated answer equals the reference answer"
            continue
        remaining.append((unit, prediction))
    return remaining


# ---------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------


def _judge_remaining(
    config: DpoBuildConfig,
    remaining: Sequence[tuple[_Unit, str]],
    *,
    work_dir: Path,
    inference_fp: str,
    assets: Mapping[Path, Any],
    overwrite: bool,
    judge_client_factory,
    fingerprints: dict[str, str],
) -> list[tuple[_Unit, str]]:
    if config.rubric is None or config.judge is None:
        raise DpoPipelineError(
            "wrong_only requires one rubric and complete Judge settings"
        )
    if not remaining:
        return []

    settings = config.judge.settings
    with _translate("DPO Judge request"):
        jobs = [
            JudgeJob(
                candidate=unit.candidate,
                rejected=prediction,
                inference_fp=inference_fp,
                request=build_dpo_judge_request(
                    unit.candidate,
                    prediction,
                    load_dpo_prompt(config.rubric, bool(unit.candidate.images)),
                    assets,
                ),
            )
            for unit, prediction in remaining
        ]
        expected_raw_keys = tuple(
            (
                job.candidate.sample_id,
                build_judge_request_fp(
                    job.inference_fp,
                    settings,
                    job.request.fingerprint_material,
                    job.request.image_assets,
                ),
            )
            for job in jobs
        )

    common_identity = {
        "pipeline_version": DPO_PIPELINE_VERSION,
        "inference_fp": inference_fp,
        "rubric": config.rubric,
        "judge": {
            "api_base": settings.api_base,
            "model": settings.model,
            "temperature": settings.temperature,
            "timeout": settings.timeout,
            "max_retries": settings.max_retries,
        },
    }

    with _translate("DPO Judge cache"):
        raw_store = DpoJudgeRawStore(
            work_dir / "judge_raw",
            build_phase_fp("judge_raw", common_identity, expected_raw_keys),
            expected_raw_keys,
            overwrite=True,
        )
        if overwrite:
            raw_store.start_new_attempt()
    fingerprints["judge_raw"] = raw_store.fingerprint

    def parse_store_factory(keys: tuple[tuple[str, str], ...]) -> DpoJudgeParseStore:
        parse_store = DpoJudgeParseStore(
            work_dir / "judge_parse",
            build_phase_fp("judge_parse", common_identity, keys),
            keys,
            # A derived phase always follows its raw responses instead of
            # demanding an explicit overwrite of its own.
            overwrite=True,
        )
        if overwrite:
            parse_store.start_new_attempt()
        fingerprints["judge_parse"] = parse_store.fingerprint
        return parse_store

    with _translate("DPO judging"):
        results = judge_candidates(
            jobs,
            config.judge,
            raw_store,
            parse_store_factory,
            client_factory=judge_client_factory,
        )

    selected: list[tuple[_Unit, str]] = []
    for unit, prediction in remaining:
        outcome = results[unit.candidate.sample_id]
        record: dict[str, Any] = {
            "rubric": config.rubric,
            "request_fp": outcome.request_fp,
            "parse_fp": outcome.parse_fp,
        }
        if outcome.error_type is not None:
            record["status"] = "error"
            record["error_type"] = outcome.error_type
            record["error_message"] = outcome.error_message
            unit.judge = record
            unit.reason_code = "judge_error"
            unit.detail = "the Judge response could not be obtained or parsed"
            continue
        parsed = dict(outcome.parsed or {})
        record["status"] = "ok"
        record["hit"] = parsed.get("hit")
        record["result"] = parsed
        unit.judge = record
        if parsed.get("hit") != 0:
            unit.reason_code = "judge_pass"
            unit.detail = "the Judge accepted the generated answer"
            continue
        selected.append((unit, prediction))
    return selected


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def _training_row(candidate: DpoCandidate, prediction: str) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": turn.from_, "value": turn.value}
            for turn in candidate.conversations
        ],
        "chosen": candidate.chosen,
        "rejected": prediction,
        "images": [ref.original for ref in candidate.images],
    }


def _audit_record(unit: _Unit) -> AuditRecord:
    candidate = unit.candidate
    modality = _modality(candidate.images) if candidate is not None else None
    return AuditRecord(
        source=_source_json(unit.source),
        sample_id=unit.sample_id,
        source_format=candidate.source_format if candidate is not None else None,
        turn_index=candidate.turn_index if candidate is not None else None,
        turn_count=candidate.turn_count if candidate is not None else None,
        modality=modality,
        digests=_digests(candidate),
        prediction=unit.prediction,
        inference_error=unit.inference_error,
        judge=unit.judge,
        disposition="rejected" if unit.reason_code is not None else "selected",
        reason_code=unit.reason_code,
        detail=unit.detail,
    )


def _source_json(source: SourceRef) -> dict[str, Any]:
    return {
        "input_index": source.input_index,
        "source_path": str(source.source_path),
        "container_format": source.container_format,
        "record_index": source.record_index,
        "line_number": source.line_number,
        "source_id": source.source_id,
        "raw_digest": source.raw_digest,
    }


def _digests(candidate: DpoCandidate | None) -> dict[str, str]:
    if candidate is None:
        return {}
    return {
        "conversations": canonical_sha256(
            [[turn.from_, turn.value] for turn in candidate.conversations]
        ),
        "chosen": hashlib.sha256(candidate.chosen.encode("utf-8")).hexdigest(),
        "images": canonical_sha256([ref.original for ref in candidate.images]),
    }


def _modality(images: Sequence[ImageRef]) -> str:
    if not images:
        return "text"
    return "single_image" if len(images) == 1 else "multi_image"


def _manifest_data(
    config: DpoBuildConfig,
    *,
    run_id: str,
    inputs: InputResult,
    checkpoint: Mapping[str, Any],
    inference_fp: str,
    judge_fingerprints: Mapping[str, str],
    candidate_count: int,
) -> dict[str, Any]:
    judge = config.judge
    return {
        "run_id": run_id,
        "pipeline_version": DPO_PIPELINE_VERSION,
        "normalizer_version": DPO_NORMALIZER_VERSION,
        "config": {
            "config_path": str(config.config_path),
            "inputs": [str(path) for path in config.inputs],
            "output_dir": str(config.output_dir),
            "output_name": config.output_name,
            "work_dir": str(config.work_dir),
            "image_root": str(config.image_root),
            "wrong_only": config.wrong_only,
            "rubric": config.rubric,
            "judge": None
            if judge is None
            else {
                "api_base": judge.settings.api_base,
                "api_key": judge.settings.api_key,
                "model": judge.settings.model,
                "temperature": judge.settings.temperature,
                "timeout": judge.settings.timeout,
                "max_retries": judge.settings.max_retries,
                "max_workers": judge.max_workers,
            },
        },
        "inputs": [
            {
                "source_path": str(summary.source_path),
                "sha256": summary.sha256,
                "byte_count": summary.byte_count,
                "container_format": summary.container_format,
                "record_count": summary.record_count,
            }
            for summary in inputs.inputs
        ],
        "model": {
            "model_name": config.infer.model_name,
            "model_path": str(config.infer.model_path),
            "enable_thinking": config.infer.enable_thinking,
            "max_new_tokens": config.infer.max_new_tokens,
            "batch_size": config.infer.batch_size,
            "torch_dtype": config.infer.torch_dtype,
            "device_map": config.infer.device_map,
            "gpu_ids": list(config.infer.gpu_ids),
            "workers_per_gpu": config.infer.workers_per_gpu,
            "checkpoint": dict(checkpoint),
        },
        "fingerprints": {
            "inference": inference_fp,
            "judge_raw": judge_fingerprints.get("judge_raw"),
            "judge_parse": judge_fingerprints.get("judge_parse"),
        },
        "counts": {
            "inputs": len(inputs.inputs),
            "source_records": sum(summary.record_count for summary in inputs.inputs),
            "candidates": candidate_count,
        },
    }


def _print_dry_run(summary: Mapping[str, Any]) -> None:
    """Show the stats a dry run exists to produce; nothing else reports them."""
    print("dry-run: no model was loaded, no Judge was called, nothing was published")
    for item in summary["inputs"]:
        print(
            f"input       {item['source_path']} "
            f"({item['container_format']}, {item['record_count']} records)"
        )
    print(f"candidates  {summary['normalized_candidates']} normalized")
    print(f"pending     {summary['pending_candidates']} would run inference")
    print(f"rejected    {summary['rejected_before_inference']} before inference")
    for reason, count in summary["by_reason_code"].items():
        print(f"  {reason:<26} {count}")
    print("modality    of the pending candidates")
    for modality, count in summary["by_modality"].items():
        print(f"  {modality:<26} {count}")
    images = summary["images"]
    print(
        f"images      {images['unique_readable']} unique readable, "
        f"{images['visible_references']} visible references"
    )


def _dry_run_summary(
    config: DpoBuildConfig,
    inputs: InputResult,
    units: Sequence[_Unit],
    runnable: Sequence[DpoCandidate],
    preflight: ImagePreflightResult,
) -> Mapping[str, Any]:
    reasons: Counter[str] = Counter(
        unit.reason_code for unit in units if unit.reason_code is not None
    )
    modality: Counter[str] = Counter(
        _modality(candidate.images) for candidate in runnable
    )
    return MappingProxyType(
        {
            "dry_run": True,
            "inputs": [
                {
                    "source_path": str(summary.source_path),
                    "sha256": summary.sha256,
                    "byte_count": summary.byte_count,
                    "container_format": summary.container_format,
                    "record_count": summary.record_count,
                }
                for summary in inputs.inputs
            ],
            "source_records": sum(summary.record_count for summary in inputs.inputs),
            "normalized_candidates": len(inputs.candidates),
            "pending_candidates": len(runnable),
            "rejected_before_inference": sum(
                1 for unit in units if unit.reason_code is not None
            ),
            "by_reason_code": dict(sorted(reasons.items())),
            "by_modality": dict(sorted(modality.items())),
            "images": {
                "unique_readable": len(preflight.assets),
                "visible_references": sum(
                    len(candidate.images) for candidate in runnable
                ),
            },
            "wrong_only": config.wrong_only,
            "rubric": config.rubric,
        }
    )


# ---------------------------------------------------------------------------
# failure and cleanup
# ---------------------------------------------------------------------------


def _failed_run(
    config: DpoBuildConfig,
    work_dir: Path,
    run_id: str,
    units: Sequence[_Unit],
    warnings: Sequence[str],
    error: str,
) -> DpoPipelineError:
    audit_records = [_audit_record(unit) for unit in units]
    summary = build_summary(audit_records)
    with _translate("DPO failed-run diagnostics"):
        failed_dir = write_failed_run(
            work_dir, run_id, audit_records, summary, warnings, error
        )
    return DpoPipelineError(f"{error}; diagnostics: {failed_dir}")


def _clean_partial(work_dir: Path, staging_dir: Path) -> None:
    for name in _PHASE_DIRS:
        shutil.rmtree(work_dir / name, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


@contextmanager
def _translate(context: str) -> Iterator[None]:
    """Present every known subsystem failure as one CLI-renderable error."""
    try:
        yield
    except (DpoPipelineError, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001 - deliberate boundary translation
        detail = str(exc) or type(exc).__name__
        raise DpoPipelineError(f"{context}: {detail}") from exc


def _prepare_directory(path: Path, label: str) -> Path:
    try:
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise DpoPipelineError(
            f"DPO {label} directory cannot be created: {path} ({exc})"
        ) from exc


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:12]}"
