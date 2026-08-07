from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, Sequence


ContainerFormat = Literal["json_array", "json_object", "jsonl"]


@dataclass(frozen=True)
class SourceRef:
    input_index: int
    source_path: Path
    container_format: ContainerFormat
    record_index: int
    line_number: int | None
    source_id: str | None
    raw_digest: str


@dataclass(frozen=True)
class RawRecord:
    source: SourceRef
    value: Any


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
    "invalid_schema",
    "unsupported_role",
    "conflicting_image_fields",
    "missing_question",
    "missing_chosen",
    "unpaired_turns",
    "missing_image",
    "image_placeholder_mismatch",
    "duplicate",
    "reference_conflict",
    "inference_error",
    "empty_rejected",
    "identical_pair",
    "judge_error",
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
    container_format: ContainerFormat
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


class DpoInputError(RuntimeError):
    """A DPO input file cannot be read as a supported JSON container."""


def load_source_records(path: Path, *, input_index: int) -> list[RawRecord]:
    """Load JSON records based on file content while retaining source identity."""
    source_path = _resolve_source_path(path)
    text = _read_utf8(source_path)
    if not text.strip():
        raise DpoInputError(f"DPO input file is empty: {source_path}")

    try:
        value = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError):
        return _load_jsonl(text, source_path=source_path, input_index=input_index)

    if isinstance(value, dict):
        return [
            _raw_record(
                value,
                source_path=source_path,
                input_index=input_index,
                container_format="json_object",
                record_index=0,
                line_number=None,
            )
        ]
    if isinstance(value, list):
        return [
            _raw_record(
                item,
                source_path=source_path,
                input_index=input_index,
                container_format="json_array",
                record_index=record_index,
                line_number=None,
            )
            for record_index, item in enumerate(value)
        ]
    raise DpoInputError(
        f"DPO input must contain a top-level JSON object or array: {source_path}"
    )


def normalize_record(raw: RawRecord, *, image_root: Path) -> NormalizationResult:
    """Normalize one Alpaca or ShareGPT source value into turn candidates."""
    value = raw.value
    if not isinstance(value, dict):
        return _normalization_issue(
            raw, "invalid_schema", "record must be a JSON object"
        )

    has_conversations = "conversations" in value
    has_alpaca_core = any(
        field in value for field in ("instruction", "output", "history")
    )
    if has_conversations and has_alpaca_core:
        return _normalization_issue(
            raw,
            "invalid_schema",
            "record mixes ShareGPT and Alpaca core fields",
        )

    if has_conversations:
        pairs, problem = _sharegpt_pairs(value)
        source_format: Literal["alpaca", "sharegpt"] = "sharegpt"
    elif has_alpaca_core:
        pairs, problem = _alpaca_pairs(value)
        source_format = "alpaca"
    else:
        return _normalization_issue(
            raw, "invalid_schema", "record format is not recognized"
        )
    if problem is not None:
        reason_code, detail = problem
        return _normalization_issue(raw, reason_code, detail)

    image_strings, image_problem = _image_strings(value)
    if image_problem is not None:
        reason_code, detail = image_problem
        return _normalization_issue(raw, reason_code, detail)

    placeholder_count = sum(question.count("<image>") for question, _ in pairs)
    if placeholder_count == 0 and image_strings:
        prefix = "<image>\n" * len(image_strings)
        first_question, first_chosen = pairs[0]
        pairs[0] = (prefix + first_question, first_chosen)
    elif placeholder_count != len(image_strings):
        return _normalization_issue(
            raw,
            "image_placeholder_mismatch",
            "human image placeholder count does not match image count",
        )

    image_refs, resolution_errors = _resolve_image_refs(
        image_strings, image_root=image_root
    )
    turn_count = len(pairs)
    candidates: list[DpoCandidate] = []
    issues: list[InputIssue] = []
    file_status: dict[int, bool] = {}
    visible_image_count = 0
    gold_history: list[DpoTurn] = []

    for turn_index, (question, chosen) in enumerate(pairs):
        visible_image_count += question.count("<image>")
        conversations = tuple(
            [*gold_history, DpoTurn(from_="human", value=question)]
        )
        visible_images = image_refs[:visible_image_count]
        sample_id = _candidate_sample_id(
            raw.source,
            turn_index=turn_index,
            conversations=conversations,
            chosen=chosen,
            images=visible_images,
        )
        candidate = DpoCandidate(
            sample_id=sample_id,
            conversations=conversations,
            chosen=chosen,
            images=visible_images,
            source=raw.source,
            source_format=source_format,
            turn_index=turn_index,
            turn_count=turn_count,
        )

        missing_index = _first_missing_visible_image(
            image_refs,
            visible_image_count=visible_image_count,
            resolution_errors=resolution_errors,
            file_status=file_status,
        )
        if missing_index is None:
            candidates.append(candidate)
        else:
            issues.append(
                InputIssue(
                    source=raw.source,
                    sample_id=sample_id,
                    reason_code="missing_image",
                    detail=(
                        f"visible image {missing_index + 1} is missing or not a file"
                    ),
                )
            )

        gold_history.extend(
            (
                DpoTurn(from_="human", value=question),
                DpoTurn(from_="gpt", value=chosen),
            )
        )

    return NormalizationResult(candidates=tuple(candidates), issues=tuple(issues))


def normalize_inputs(
    paths: Sequence[Path], *, image_root: Path
) -> InputResult:
    """Normalize input files in deterministic source and turn order."""
    resolved_image_root = _resolve_image_root(image_root)
    candidates: list[DpoCandidate] = []
    issues: list[InputIssue] = []
    inputs: list[InputSummary] = []

    for input_index, path in enumerate(paths):
        source_path = _resolve_source_path(Path(path))
        records = load_source_records(source_path, input_index=input_index)
        source_bytes = _read_bytes(source_path)
        container_format: ContainerFormat = (
            records[0].source.container_format if records else "json_array"
        )
        inputs.append(
            InputSummary(
                source_path=source_path,
                sha256=hashlib.sha256(source_bytes).hexdigest(),
                byte_count=len(source_bytes),
                container_format=container_format,
                record_count=len(records),
            )
        )
        for raw in records:
            result = normalize_record(raw, image_root=resolved_image_root)
            candidates.extend(result.candidates)
            issues.extend(result.issues)

    return InputResult(
        candidates=tuple(candidates), issues=tuple(issues), inputs=tuple(inputs)
    )


def deduplicate_candidates(
    candidates: Sequence[DpoCandidate],
) -> DedupeResult:
    """Remove exact duplicates and exclude groups with conflicting references."""
    ordered = tuple(candidates)
    groups: dict[
        tuple[tuple[tuple[str, str], ...], tuple[str, ...]], list[int]
    ] = {}
    for index, candidate in enumerate(ordered):
        key = (
            tuple(
                (turn.from_, turn.value) for turn in candidate.conversations
            ),
            tuple(image.original for image in candidate.images),
        )
        groups.setdefault(key, []).append(index)

    conflict_indices: set[int] = set()
    duplicate_indices: set[int] = set()
    for indices in groups.values():
        chosen_values = {ordered[index].chosen for index in indices}
        if len(chosen_values) > 1:
            conflict_indices.update(indices)
        else:
            duplicate_indices.update(indices[1:])

    kept: list[DpoCandidate] = []
    issues: list[InputIssue] = []
    for index, candidate in enumerate(ordered):
        if index in conflict_indices:
            issues.append(
                InputIssue(
                    source=candidate.source,
                    sample_id=candidate.sample_id,
                    reason_code="reference_conflict",
                    detail="same prompt and images have conflicting chosen answers",
                )
            )
        elif index in duplicate_indices:
            issues.append(
                InputIssue(
                    source=candidate.source,
                    sample_id=candidate.sample_id,
                    reason_code="duplicate",
                    detail="exact normalized candidate duplicates an earlier source",
                )
            )
        else:
            kept.append(candidate)

    return DedupeResult(candidates=tuple(kept), issues=tuple(issues))


def _normalization_issue(
    raw: RawRecord, reason_code: ReasonCode, detail: str
) -> NormalizationResult:
    return NormalizationResult(
        candidates=(),
        issues=(
            InputIssue(
                source=raw.source,
                sample_id=None,
                reason_code=reason_code,
                detail=detail,
            ),
        ),
    )


def _alpaca_pairs(
    value: dict[str, Any],
) -> tuple[
    list[tuple[str, str]], tuple[ReasonCode, str] | None
]:
    if "system" in value:
        return [], (
            "unsupported_role",
            "Alpaca system fields are not supported",
        )

    instruction = value.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return [], ("missing_question", "instruction must be a nonempty string")

    chosen = value.get("output")
    if not isinstance(chosen, str) or not chosen.strip():
        return [], ("missing_chosen", "output must be a nonempty string")

    input_value = value.get("input")
    if input_value is None or (
        isinstance(input_value, str) and not input_value.strip()
    ):
        current_question = instruction
    elif isinstance(input_value, str):
        current_question = instruction + "\n" + input_value
    else:
        return [], ("invalid_schema", "input must be a string or null")

    history_present = "history" in value
    history = value.get("history")
    if not history_present or history == []:
        pairs: list[tuple[str, str]] = []
    elif not isinstance(history, list):
        return [], ("unpaired_turns", "history must be a list of pairs")
    else:
        pairs = []
        for pair in history:
            if not isinstance(pair, list) or len(pair) != 2:
                return [], (
                    "unpaired_turns",
                    "each history item must be a two-element list",
                )
            question, answer = pair
            if not isinstance(question, str) or not question.strip():
                return [], (
                    "missing_question",
                    "history question must be a nonempty string",
                )
            if not isinstance(answer, str) or not answer.strip():
                return [], (
                    "missing_chosen",
                    "history answer must be a nonempty string",
                )
            pairs.append((question, answer))

    pairs.append((current_question, chosen))
    return pairs, None


def _sharegpt_pairs(
    value: dict[str, Any],
) -> tuple[
    list[tuple[str, str]], tuple[ReasonCode, str] | None
]:
    conversations = value.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return [], (
            "invalid_schema",
            "conversations must be a nonempty list",
        )

    normalized_messages: list[tuple[Literal["human", "gpt"], str]] = []
    role_aliases: dict[str, Literal["human", "gpt"]] = {
        "human": "human",
        "user": "human",
        "gpt": "gpt",
        "assistant": "gpt",
    }
    for message in conversations:
        if not isinstance(message, dict):
            return [], ("invalid_schema", "conversation message must be an object")
        role = message.get("from")
        message_value = message.get("value")
        if not isinstance(role, str) or not isinstance(message_value, str):
            return [], (
                "invalid_schema",
                "conversation from and value must be strings",
            )
        normalized_role = role_aliases.get(role)
        if normalized_role is None:
            return [], ("unsupported_role", "conversation role is not supported")
        normalized_messages.append((normalized_role, message_value))

    if len(normalized_messages) % 2:
        return [], (
            "unpaired_turns",
            "conversations must contain complete human and gpt pairs",
        )
    for index, (role, _) in enumerate(normalized_messages):
        expected = "human" if index % 2 == 0 else "gpt"
        if role != expected:
            return [], (
                "unpaired_turns",
                "conversations must strictly alternate human then gpt",
            )

    pairs: list[tuple[str, str]] = []
    for index in range(0, len(normalized_messages), 2):
        question = normalized_messages[index][1]
        answer = normalized_messages[index + 1][1]
        if not question.strip():
            return [], (
                "missing_question",
                "human message must be a nonempty string",
            )
        if not answer.strip():
            return [], (
                "missing_chosen",
                "gpt message must be a nonempty string",
            )
        pairs.append((question, answer))
    return pairs, None


def _image_strings(
    value: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[ReasonCode, str] | None]:
    image_present = "image" in value
    images_present = "images" in value
    image_items: tuple[str, ...] = ()
    images_items: tuple[str, ...] = ()

    if image_present:
        image = value["image"]
        if not isinstance(image, str) or not image.strip():
            return (), ("invalid_schema", "image must be a nonempty string")
        image_items = (image,)

    if images_present:
        images = value["images"]
        if not isinstance(images, list):
            return (), ("invalid_schema", "images must be a string array")
        if any(not isinstance(item, str) or not item.strip() for item in images):
            return (), (
                "invalid_schema",
                "every images entry must be a nonempty string",
            )
        images_items = tuple(images)

    if image_present and images_present and image_items != images_items:
        return (), (
            "conflicting_image_fields",
            "image and images fields do not describe the same ordered paths",
        )
    if images_present:
        return images_items, None
    return image_items, None


def _resolve_image_refs(
    originals: tuple[str, ...], *, image_root: Path
) -> tuple[tuple[ImageRef, ...], frozenset[int]]:
    refs: list[ImageRef] = []
    errors: set[int] = set()
    root_resolved: Path | None = None
    root_attempted = False

    for index, original in enumerate(originals):
        try:
            image_path = Path(original)
            is_absolute = image_path.is_absolute()
        except (OSError, RuntimeError, TypeError, ValueError):
            refs.append(ImageRef(original=original, resolved=Path(".")))
            errors.add(index)
            continue

        if is_absolute:
            unresolved = image_path
        elif image_path.drive or image_path.root:
            refs.append(ImageRef(original=original, resolved=image_path))
            errors.add(index)
            continue
        else:
            if not root_attempted:
                root_attempted = True
                try:
                    root_resolved = Path(image_root).resolve()
                except (OSError, RuntimeError, TypeError, ValueError):
                    root_resolved = None
            if root_resolved is None:
                refs.append(ImageRef(original=original, resolved=image_path))
                errors.add(index)
                continue
            unresolved = root_resolved / image_path

        try:
            resolved = unresolved.resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            resolved = unresolved
            errors.add(index)
        refs.append(ImageRef(original=original, resolved=resolved))

    return tuple(refs), frozenset(errors)


def _first_missing_visible_image(
    refs: tuple[ImageRef, ...],
    *,
    visible_image_count: int,
    resolution_errors: frozenset[int],
    file_status: dict[int, bool],
) -> int | None:
    for index in range(visible_image_count):
        if index in resolution_errors:
            return index
        if index not in file_status:
            try:
                file_status[index] = refs[index].resolved.is_file()
            except (OSError, RuntimeError, TypeError, ValueError):
                file_status[index] = False
        if not file_status[index]:
            return index
    return None


def _candidate_sample_id(
    source: SourceRef,
    *,
    turn_index: int,
    conversations: tuple[DpoTurn, ...],
    chosen: str,
    images: tuple[ImageRef, ...],
) -> str:
    payload = {
        "raw_digest": source.raw_digest,
        "input_index": source.input_index,
        "record_index": source.record_index,
        "line_number": source.line_number,
        "turn_index": turn_index,
        "conversations": [
            {"from": turn.from_, "value": turn.value} for turn in conversations
        ],
        "chosen": chosen,
        "images": [image.original for image in images],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_image_root(image_root: Path) -> Path:
    try:
        return Path(image_root).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoInputError("cannot resolve DPO image root") from exc


def _resolve_source_path(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise DpoInputError(f"cannot resolve DPO input path: {path}") from exc


def _read_utf8(source_path: Path) -> str:
    try:
        return source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        raise DpoInputError(
            f"cannot read DPO input {source_path}: {type(exc).__name__}"
        ) from exc


def _read_bytes(source_path: Path) -> bytes:
    try:
        return source_path.read_bytes()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoInputError(
            f"cannot read DPO input bytes {source_path}: {type(exc).__name__}"
        ) from exc


def _reject_nonstandard_constant(constant: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {constant}")


def _parse_finite_float(number: str) -> float:
    value = float(number)
    if not math.isfinite(value):
        raise ValueError("JSON number is outside the finite float range")
    return value


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
    )


def _load_jsonl(
    text: str,
    *,
    source_path: Path,
    input_index: int,
) -> list[RawRecord]:
    records: list[RawRecord] = []
    lines = io.StringIO(text, newline="\n")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise DpoInputError(
                f"invalid JSONL in DPO input {source_path} at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise DpoInputError(
                f"JSONL record in DPO input {source_path} at line {line_number} "
                "must be an object"
            )
        records.append(
            _raw_record(
                value,
                source_path=source_path,
                input_index=input_index,
                container_format="jsonl",
                record_index=len(records),
                line_number=line_number,
            )
        )
    if not records:
        raise DpoInputError(f"DPO input file is empty: {source_path}")
    return records


def _raw_record(
    value: Any,
    *,
    source_path: Path,
    input_index: int,
    container_format: ContainerFormat,
    record_index: int,
    line_number: int | None,
) -> RawRecord:
    source_id_value = value.get("id") if isinstance(value, dict) else None
    source_id = (
        source_id_value
        if isinstance(source_id_value, str) and source_id_value != ""
        else None
    )
    try:
        raw_digest = _raw_digest(value)
    except (TypeError, UnicodeError, ValueError) as exc:
        line_location = (
            f" at line {line_number}" if line_number is not None else ""
        )
        raise DpoInputError(
            f"cannot canonicalize DPO input {source_path}, "
            f"record index {record_index}{line_location}"
        ) from exc
    source = SourceRef(
        input_index=input_index,
        source_path=source_path,
        container_format=container_format,
        record_index=record_index,
        line_number=line_number,
        source_id=source_id,
        raw_digest=raw_digest,
    )
    return RawRecord(source=source, value=value)


def _raw_digest(value: Any) -> str:
    normalized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()
