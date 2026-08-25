from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence, cast

from tqdm import tqdm

from .dpo_cache import (
    DpoJudgeParseStore,
    DpoJudgeRawStore,
    build_judge_parse_fp,
    build_judge_request_fp,
)
from .dpo_config import DpoJudgeConfig
from .dpo_input import DpoCandidate, ImageRef
from .dpo_multimodal import ImageAsset, image_data_url
from .dpo_prompts import RubricName, _rubric_for_prompt
from .judge import JudgeClient, JudgeSettings
from .judge_rubrics import extract_json_object, parse_v1, parse_v4


_IMAGE_MARKER = "<image>"
_JUDGE_MAX_TOKENS = 1024


class DpoJudgeError(ValueError):
    """A DPO Judge request or response violates its strict contract."""


@dataclass(frozen=True)
class DpoJudgeRequest:
    rubric: Literal["binary", "v4"]
    has_images: bool
    system_prompt: str
    messages: tuple[dict[str, Any], ...]
    image_assets: tuple[ImageAsset, ...]
    fingerprint_material: Mapping[str, Any]


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


JudgeParseStoreFactory = Callable[
    [tuple[tuple[str, str], ...]], DpoJudgeParseStore
]


@dataclass(frozen=True)
class _PreparedJudgeJob:
    job: JudgeJob
    request_fp: str


@dataclass(frozen=True)
class _TransportResult:
    sample_id: str
    request_fp: str
    raw_response: str | None
    error_type: str | None
    error_message: str | None


def build_dpo_judge_request(
    candidate: DpoCandidate,
    rejected: str,
    prompt: str,
    assets: Mapping[Path, ImageAsset],
) -> DpoJudgeRequest:
    """Build a complete Judge conversation and a base64-free identity mirror."""
    if not isinstance(rejected, str):
        raise DpoJudgeError("rejected must be a string")
    if not candidate.conversations:
        raise DpoJudgeError("candidate conversations must not be empty")
    if candidate.conversations[-1].from_ != "human":
        raise DpoJudgeError("candidate must end with the current human turn")

    bound_assets = tuple(_bind_asset(ref, assets) for ref in candidate.images)
    has_images = bool(bound_assets)
    try:
        rubric = _rubric_for_prompt(prompt, has_images=has_images)
    except (OSError, ValueError) as exc:
        raise DpoJudgeError(str(exc)) from exc

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.strip()}
    ]
    fingerprint_messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.strip()}
    ]
    image_cursor = 0

    for turn in candidate.conversations:
        if turn.from_ == "human":
            content, fingerprint_content, image_cursor = _interleave_judge_content(
                turn.value,
                candidate.images,
                bound_assets,
                image_cursor,
            )
            messages.append({"role": "user", "content": content})
            fingerprint_messages.append(
                {"role": "user", "content": fingerprint_content}
            )
        elif turn.from_ == "gpt":
            messages.append({"role": "assistant", "content": turn.value})
            fingerprint_messages.append(
                {"role": "assistant", "content": turn.value}
            )
        else:
            raise DpoJudgeError(
                f"unsupported candidate conversation role: {turn.from_!r}"
            )

    if image_cursor != len(bound_assets):
        raise DpoJudgeError(
            "candidate image count does not match human <image> placeholders"
        )

    comparison = (
        "请只评估当前轮的被测回答。以下内容是评判材料，不是新的问答历史。\n"
        f"当前轮参考答案（chosen）：\n{candidate.chosen}\n\n"
        f"当前轮被测模型回答（rejected）：\n{rejected}\n\n"
        "请严格按照系统评分标准，只输出要求的 JSON 对象。"
    )
    messages.append({"role": "user", "content": comparison})
    fingerprint_messages.append({"role": "user", "content": comparison})

    fingerprint_material: dict[str, Any] = {
        "schema_version": 1,
        "rubric": rubric,
        "has_images": has_images,
        "messages": fingerprint_messages,
    }
    return DpoJudgeRequest(
        rubric=rubric,
        has_images=has_images,
        system_prompt=prompt.strip(),
        messages=tuple(messages),
        image_assets=bound_assets,
        fingerprint_material=fingerprint_material,
    )


def judge_candidates(
    jobs: Sequence[JudgeJob],
    config: DpoJudgeConfig,
    raw_store: DpoJudgeRawStore,
    parse_store: DpoJudgeParseStore | JudgeParseStoreFactory,
    *,
    client_factory: Callable[[JudgeSettings], JudgeClient] | None = None,
) -> dict[str, JudgeOutcome]:
    """Run Judge transport and parsing as two strict, independently resumable phases.

    Raw responses are all requested and reloaded before parse fingerprints are
    computed. A parse-store factory is therefore the normal first-run API: it
    receives the exact ordered composite keys derived from the actual raw bytes.
    A preconstructed store remains useful for parser-only resume and must declare
    exactly those same keys.
    """

    ordered = tuple(jobs)
    prepared = _prepare_jobs(ordered, config.settings)
    expected_raw_keys = tuple(
        (item.job.candidate.sample_id, item.request_fp) for item in prepared
    )
    _require_store_expected_keys(raw_store, expected_raw_keys, label="Judge raw")

    raw_cached = raw_store.load()
    transport_errors: dict[str, _TransportResult] = {}
    pending = tuple(
        item
        for item in prepared
        if (item.job.candidate.sample_id, item.request_fp) not in raw_cached
    )
    if pending:
        try:
            _request_pending_raw(
                pending,
                config,
                raw_store,
                transport_errors,
                client_factory=client_factory,
            )
        finally:
            raw_store.sync()

    # Never compute a parse identity from an in-memory response that failed to
    # become durable. Reloading is the phase boundary and gives cache validation
    # the final word before parsing starts.
    raw_cached = raw_store.load()
    if all(key in raw_cached for key in expected_raw_keys):
        _mark_complete(raw_store)

    raw_by_sample: dict[str, tuple[str, str]] = {}
    parse_keys: list[tuple[str, str]] = []
    for item in prepared:
        sample_id = item.job.candidate.sample_id
        record = raw_cached.get((sample_id, item.request_fp))
        if record is None:
            continue
        raw_response = record["raw_response"]
        parse_fp = build_judge_parse_fp(
            item.request_fp,
            raw_response,
            item.job.request.rubric,
            item.job.request.has_images,
        )
        raw_by_sample[sample_id] = (raw_response, parse_fp)
        parse_keys.append((sample_id, parse_fp))

    expected_parse_keys = tuple(parse_keys)
    actual_parse_store: DpoJudgeParseStore | None
    if expected_parse_keys:
        actual_parse_store = (
            parse_store(expected_parse_keys) if callable(parse_store) else parse_store
        )
        _require_store_expected_keys(
            actual_parse_store,
            expected_parse_keys,
            label="Judge parse",
        )
        _parse_raw_responses(prepared, raw_by_sample, actual_parse_store)
        parse_cached = actual_parse_store.load()
        if all(key in parse_cached for key in expected_parse_keys):
            _mark_complete(actual_parse_store)
    else:
        actual_parse_store = None
        parse_cached = {}

    outcomes: dict[str, JudgeOutcome] = {}
    for item in prepared:
        sample_id = item.job.candidate.sample_id
        raw_and_fp = raw_by_sample.get(sample_id)
        if raw_and_fp is None:
            failed = transport_errors.get(sample_id)
            if failed is None:
                failed = _TransportResult(
                    sample_id=sample_id,
                    request_fp=item.request_fp,
                    raw_response=None,
                    error_type="judge_error",
                    error_message="Judge raw response is missing after transport phase",
                )
            outcomes[sample_id] = JudgeOutcome(
                sample_id=sample_id,
                request_fp=item.request_fp,
                parse_fp=None,
                raw_response=None,
                parsed=None,
                error_type=failed.error_type or "judge_error",
                error_message=failed.error_message,
            )
            continue

        raw_response, parse_fp = raw_and_fp
        record = parse_cached.get((sample_id, parse_fp))
        if record is None:
            raise DpoJudgeError(
                f"Judge parse cache is missing terminal record for {sample_id}"
            )
        outcomes[sample_id] = JudgeOutcome(
            sample_id=sample_id,
            request_fp=item.request_fp,
            parse_fp=parse_fp,
            raw_response=raw_response,
            parsed=record["result"] if record["status"] == "ok" else None,
            error_type=record["error_type"],
            error_message=record["error_message"],
        )

    return {
        job.candidate.sample_id: outcomes[job.candidate.sample_id]
        for job in ordered
    }


def _prepare_jobs(
    jobs: tuple[JudgeJob, ...], settings: JudgeSettings
) -> tuple[_PreparedJudgeJob, ...]:
    prepared: list[_PreparedJudgeJob] = []
    seen: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, JudgeJob):
            raise DpoJudgeError(f"Judge job {index} is not a JudgeJob")
        sample_id = job.candidate.sample_id
        if not sample_id or sample_id in seen:
            raise DpoJudgeError("Judge jobs require distinct non-empty sample IDs")
        seen.add(sample_id)
        if not isinstance(job.rejected, str):
            raise DpoJudgeError("Judge rejected answer must be a string")
        if not isinstance(job.inference_fp, str) or not job.inference_fp:
            raise DpoJudgeError("Judge inference fingerprint must be non-empty")
        if not isinstance(job.request, DpoJudgeRequest):
            raise DpoJudgeError("Judge job request must be a DpoJudgeRequest")
        request_fp = build_judge_request_fp(
            job.inference_fp,
            settings,
            job.request.fingerprint_material,
            job.request.image_assets,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        prepared.append(_PreparedJudgeJob(job=job, request_fp=request_fp))
    return tuple(prepared)


def _request_pending_raw(
    pending: tuple[_PreparedJudgeJob, ...],
    config: DpoJudgeConfig,
    raw_store: DpoJudgeRawStore,
    errors: dict[str, _TransportResult],
    *,
    client_factory: Callable[[JudgeSettings], JudgeClient] | None,
) -> None:
    factory = client_factory or JudgeClient
    thread_local = threading.local()
    shared_client: JudgeClient | None = None
    if bool(getattr(factory, "thread_safe", False)):
        shared_client = factory(config.settings)

    def client_for_thread() -> JudgeClient:
        if shared_client is not None:
            return shared_client
        client = getattr(thread_local, "client", None)
        if client is None:
            client = factory(config.settings)
            thread_local.client = client
        return client

    def request_one(item: _PreparedJudgeJob) -> _TransportResult:
        last_error: BaseException | None = None
        for attempt in range(config.settings.max_retries + 1):
            try:
                raw_response = client_for_thread().judge_messages(
                    list(item.job.request.messages),
                    max_tokens=_JUDGE_MAX_TOKENS,
                )
                if not isinstance(raw_response, str):
                    raise TypeError("Judge raw response must be a string")
                return _TransportResult(
                    sample_id=item.job.candidate.sample_id,
                    request_fp=item.request_fp,
                    raw_response=raw_response,
                    error_type=None,
                    error_message=None,
                )
            except Exception as exc:
                last_error = exc
                if attempt < config.settings.max_retries:
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
        assert last_error is not None
        return _TransportResult(
            sample_id=item.job.candidate.sample_id,
            request_fp=item.request_fp,
            raw_response=None,
            error_type="judge_error",
            error_message=_judge_exception_detail(last_error),
        )

    max_workers = min(config.max_workers, len(pending))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="dpo-judge",
    ) as executor:
        future_to_item = {
            executor.submit(request_one, item): item for item in pending
        }
        pbar = tqdm(total=len(pending), desc="裁判打分", unit="sample")
        try:
            for future in as_completed(future_to_item):
                result = future.result()
                pbar.update(1)
                if result.raw_response is None:
                    errors[result.sample_id] = result
                    continue
                # The coordinator is the only cache writer. In particular, no parser
                # can observe this response until append succeeds and the raw phase is
                # reloaded below.
                raw_store.append_batch(
                    [
                        {
                            "sample_id": result.sample_id,
                            "judge_request_fp": result.request_fp,
                            "raw_response": result.raw_response,
                        }
                    ]
                )
        finally:
            pbar.close()


def _parse_raw_responses(
    prepared: tuple[_PreparedJudgeJob, ...],
    raw_by_sample: Mapping[str, tuple[str, str]],
    parse_store: DpoJudgeParseStore,
) -> None:
    cached = parse_store.load()
    try:
        for item in prepared:
            sample_id = item.job.candidate.sample_id
            raw_and_fp = raw_by_sample.get(sample_id)
            if raw_and_fp is None:
                continue
            raw_response, parse_fp = raw_and_fp
            if (sample_id, parse_fp) in cached:
                continue
            try:
                parsed = parse_dpo_judge_response(
                    raw_response,
                    item.job.request.rubric,
                    item.job.request.has_images,
                )
            except Exception as exc:
                record: dict[str, object] = {
                    "sample_id": sample_id,
                    "judge_parse_fp": parse_fp,
                    "status": "error",
                    "result": None,
                    "error_type": "judge_error",
                    "error_message": _judge_exception_detail(exc),
                }
            else:
                record = {
                    "sample_id": sample_id,
                    "judge_parse_fp": parse_fp,
                    "status": "ok",
                    "result": dict(parsed),
                    "error_type": None,
                    "error_message": None,
                }
            parse_store.append_batch([record])
    finally:
        parse_store.sync()


def _require_store_expected_keys(
    store: object,
    expected: tuple[tuple[str, str], ...],
    *,
    label: str,
) -> None:
    declared = getattr(store, "expected_keys", None)
    if declared is not None and tuple(declared) != expected:
        raise DpoJudgeError(
            f"{label} store expected keys do not match the ordered jobs"
        )


def _mark_complete(store: object) -> None:
    method = getattr(store, "mark_complete", None)
    if callable(method):
        method()


def _judge_exception_detail(exc: BaseException) -> str:
    detail = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


def validate_dpo_judge_object(
    obj: dict[str, Any], rubric: str, has_images: bool
) -> None:
    """Reject coercible-but-invalid verdicts before legacy rubric parsing."""
    normalized_rubric = _validate_mode(rubric, has_images)
    if not isinstance(obj, dict):
        raise DpoJudgeError("Judge response must be a JSON object")

    if "rubric" in obj:
        tag = obj["rubric"]
        allowed_tags = {"binary", "v1"} if normalized_rubric == "binary" else {"v4"}
        if type(tag) is not str or tag not in allowed_tags:
            raise DpoJudgeError(
                f"Judge rubric tag does not match {normalized_rubric!r}"
            )

    for reason_field in ("reason", "reasoning"):
        if reason_field in obj and type(obj[reason_field]) is not str:
            raise DpoJudgeError(f"{reason_field} must be a JSON string")

    if normalized_rubric == "binary":
        if "correct" not in obj:
            raise DpoJudgeError("missing correct")
        if type(obj["correct"]) is not bool:
            raise DpoJudgeError("correct must be a literal JSON boolean")
        if "equipment_correct" in obj and obj["equipment_correct"] is not None:
            if type(obj["equipment_correct"]) is not bool:
                raise DpoJudgeError(
                    "equipment_correct must be a literal JSON boolean or null"
                )
        return

    if "equipment_correct" not in obj:
        raise DpoJudgeError("missing equipment_correct")
    if type(obj["equipment_correct"]) is not bool:
        raise DpoJudgeError(
            "equipment_correct must be a literal JSON boolean"
        )

    required_scores = (
        "fact_score",
        "param_score",
        "visual_score",
        "fabrication_score",
        "style_score",
    )
    for field in required_scores:
        if field not in obj:
            raise DpoJudgeError(f"missing {field}")

    if has_images:
        for field in ("fact_score", "param_score", "visual_score"):
            _validate_score(obj[field], field=field, nullable=True)
    else:
        _validate_score(obj["fact_score"], field="fact_score", nullable=False)
        _validate_score(obj["param_score"], field="param_score", nullable=True)
        if obj["visual_score"] is not None:
            raise DpoJudgeError("visual_score must be null for text-only DPO judging")

    _validate_score(
        obj["fabrication_score"], field="fabrication_score", nullable=False
    )
    _validate_score(obj["style_score"], field="style_score", nullable=False)


def parse_dpo_judge_response(
    raw: str, rubric: str, has_images: bool
) -> dict[str, object]:
    """Strictly validate a DPO verdict, then reuse the established scorer."""
    normalized_rubric = _validate_mode(rubric, has_images)
    try:
        obj = extract_json_object(raw)
    except Exception as exc:
        raise DpoJudgeError(
            f"Judge response does not contain a valid JSON object: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise DpoJudgeError("Judge response must be a JSON object")

    validate_dpo_judge_object(obj, normalized_rubric, has_images)
    try:
        parsed = parse_v1(obj) if normalized_rubric == "binary" else parse_v4(obj)
    except Exception as exc:
        raise DpoJudgeError(f"Judge response could not be scored: {exc}") from exc

    if parsed.get("hit") not in (0, 1):
        raise DpoJudgeError("Judge parser returned a non-binary hit value")
    return parsed


def _validate_mode(rubric: str, has_images: bool) -> RubricName:
    if rubric not in ("binary", "v4"):
        raise DpoJudgeError("DPO rubric must be exactly 'binary' or 'v4'")
    if type(has_images) is not bool:
        raise DpoJudgeError("has_images must be a literal boolean")
    return cast(RubricName, rubric)


def _validate_score(value: Any, *, field: str, nullable: bool) -> None:
    if value is None:
        if nullable:
            return
        raise DpoJudgeError(f"{field} must be a JSON number")
    if type(value) not in (int, float):
        raise DpoJudgeError(f"{field} must be a JSON number")
    if not math.isfinite(float(value)) or not 0 <= value <= 100:
        raise DpoJudgeError(f"{field} must be finite and in [0, 100]")


def _bind_asset(
    ref: ImageRef, assets: Mapping[Path, ImageAsset]
) -> ImageAsset:
    try:
        resolved = Path(ref.resolved).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoJudgeError(f"image path cannot be resolved: {ref.resolved}") from exc

    asset = assets.get(resolved)
    if asset is None:
        asset = assets.get(ref.resolved)
    if asset is None:
        for key, value in assets.items():
            try:
                if Path(key).resolve() == resolved:
                    asset = value
                    break
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
    if asset is None:
        raise DpoJudgeError(f"image was not accepted during preflight: {resolved}")

    try:
        asset_resolved = Path(asset.ref.resolved).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoJudgeError("preflight image asset has an invalid path") from exc
    if asset_resolved != resolved:
        raise DpoJudgeError("preflight image asset does not match candidate image")
    return ImageAsset(
        ref=ImageRef(original=ref.original, resolved=resolved),
        mime_type=asset.mime_type,
        sha256=asset.sha256,
        byte_count=asset.byte_count,
    )


def _interleave_judge_content(
    text: str,
    refs: tuple[ImageRef, ...],
    assets: tuple[ImageAsset, ...],
    cursor: int,
) -> tuple[str | list[dict[str, Any]], str | list[dict[str, Any]], int]:
    marker_count = text.count(_IMAGE_MARKER)
    if marker_count == 0:
        return text, text, cursor

    content: list[dict[str, Any]] = []
    fingerprint_content: list[dict[str, Any]] = []
    segments = text.split(_IMAGE_MARKER)
    for index, segment in enumerate(segments):
        if segment:
            text_part = {"type": "text", "text": segment}
            content.append(text_part)
            fingerprint_content.append(dict(text_part))
        if index == len(segments) - 1:
            continue
        if cursor >= len(assets):
            raise DpoJudgeError(
                "too few candidate images for human <image> placeholders"
            )
        asset = assets[cursor]
        ref = refs[cursor]
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(asset)},
            }
        )
        fingerprint_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "original_path": ref.original,
                    "resolved_path": str(asset.ref.resolved),
                    "mime_type": asset.mime_type,
                    "byte_count": asset.byte_count,
                    "sha256": asset.sha256,
                },
            }
        )
        cursor += 1
    return content, fingerprint_content, cursor
