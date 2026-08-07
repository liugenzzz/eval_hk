from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn


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
