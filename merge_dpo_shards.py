from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from eval_tool.dpo_config import load_dpo_config
from eval_tool.dpo_input import DpoCandidate, deduplicate_candidates, normalize_inputs


_ATTEMPT_ID = re.compile(r"^[0-9a-f]{32}$")
_SHARD_NAME = re.compile(r"^worker-(\d{6,})\.jsonl$")
_INFERENCE_FIELDS = {
    "sample_id",
    "status",
    "prediction",
    "error_type",
    "error_message",
}


class RecoveryError(RuntimeError):
    """Existing DPO inference shards cannot be recovered safely."""


@dataclass(frozen=True)
class RecoveryResult:
    output_path: Path
    shard_record_count: int
    training_row_count: int
    inference_error_count: int
    empty_rejected_count: int
    identical_pair_count: int
    sha256: str


def recover_dpo_shards(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> RecoveryResult:
    """Build a strict training JSONL directly from completed inference shards."""

    if type(force) is not bool:
        raise RecoveryError("force must be a literal boolean")
    config = load_dpo_config(config_path)
    if config.wrong_only:
        print(
            "临时全量恢复：忽略 wrong_only=true 和 Judge，不进行正确性筛选。",
            flush=True,
        )

    print("[1/3] 正在读取并归一化原始输入...", flush=True)
    normalized = normalize_inputs(config.inputs, image_root=config.image_root)
    deduplicated = deduplicate_candidates(normalized.candidates)
    candidates = deduplicated.candidates
    candidate_by_id = _candidate_index(candidates)

    attempt_dir, expected_count = _active_attempt(config.work_dir / "inference")
    shards = _shard_paths(attempt_dir)
    print(
        f"[2/3] 正在读取 {len(shards)} 个推理分片，预期 {expected_count} 条...",
        flush=True,
    )
    records = _load_shards(shards, expected_count=expected_count)

    unknown = sorted(set(records) - set(candidate_by_id))
    if unknown:
        preview = ", ".join(unknown[:3])
        raise RecoveryError(
            f"{len(unknown)} shard sample_id values do not match current inputs: {preview}"
        )

    destination = _destination(config.output_dir, config.output_name, output_path)
    if destination.exists() and not force:
        raise RecoveryError(
            f"output already exists: {destination}; pass --force to replace it"
        )

    print("[3/3] 正在按原始顺序汇合训练数据...", flush=True)
    counters = {
        "inference_error": 0,
        "empty_rejected": 0,
        "identical_pair": 0,
    }
    rows: list[dict[str, Any]] = []
    for candidate in tqdm(candidates, desc="汇合训练数据", unit="sample"):
        record = records.get(candidate.sample_id)
        if record is None:
            continue
        if record["status"] == "error":
            counters["inference_error"] += 1
            continue
        prediction = record["prediction"]
        if not prediction.strip():
            counters["empty_rejected"] += 1
            continue
        if prediction.strip() == candidate.chosen.strip():
            counters["identical_pair"] += 1
            continue
        rows.append(_training_row(candidate, prediction))

    if not rows:
        raise RecoveryError("no trainable DPO rows remain after validation")
    digest = _write_jsonl_atomic(destination, rows, force=force)
    return RecoveryResult(
        output_path=destination,
        shard_record_count=len(records),
        training_row_count=len(rows),
        inference_error_count=counters["inference_error"],
        empty_rejected_count=counters["empty_rejected"],
        identical_pair_count=counters["identical_pair"],
        sha256=digest,
    )


def _active_attempt(phase_dir: Path) -> tuple[Path, int]:
    active_path = phase_dir / "active.json"
    active = _read_json_object(active_path, label="active inference pointer")
    attempt_id = active.get("attempt_id")
    expected_count = active.get("expected_id_count")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise RecoveryError(f"invalid attempt_id in {active_path}")
    if type(expected_count) is not int or expected_count < 1:
        raise RecoveryError(f"invalid expected_id_count in {active_path}")

    attempts_dir = (phase_dir / "attempts").resolve()
    attempt_dir = (attempts_dir / attempt_id).resolve()
    if attempt_dir.parent != attempts_dir or not attempt_dir.is_dir():
        raise RecoveryError(f"active inference attempt directory is missing: {attempt_dir}")
    return attempt_dir, expected_count


def _shard_paths(attempt_dir: Path) -> tuple[Path, ...]:
    indexed: list[tuple[int, Path]] = []
    for path in attempt_dir.glob("worker-*.jsonl"):
        match = _SHARD_NAME.fullmatch(path.name)
        if match is not None and path.is_file():
            indexed.append((int(match.group(1)), path))
    shards = tuple(path for _, path in sorted(indexed))
    if not shards:
        raise RecoveryError(f"no worker JSONL shards found in {attempt_dir}")
    return shards


def _load_shards(
    shards: Sequence[Path], *, expected_count: int
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    locations: dict[str, str] = {}
    with tqdm(total=expected_count, desc="读取推理分片", unit="record") as progress:
        for shard in shards:
            try:
                stream = shard.open("r", encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise RecoveryError(f"cannot open shard {shard}: {exc}") from exc
            with stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        raise RecoveryError(f"{shard}:{line_number}: blank JSONL line")
                    record = _parse_inference_record(line, shard, line_number)
                    sample_id = record["sample_id"]
                    previous = locations.get(sample_id)
                    if previous is not None:
                        raise RecoveryError(
                            f"duplicate sample_id {sample_id!r} at {previous} "
                            f"and {shard}:{line_number}"
                        )
                    records[sample_id] = record
                    locations[sample_id] = f"{shard}:{line_number}"
                    progress.update(1)
    if len(records) != expected_count:
        raise RecoveryError(
            f"inference shards contain {len(records)} records, expected {expected_count}"
        )
    return records


def _parse_inference_record(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise RecoveryError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _INFERENCE_FIELDS:
        raise RecoveryError(
            f"{path}:{line_number}: inference record fields are invalid"
        )
    sample_id = value["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise RecoveryError(f"{path}:{line_number}: sample_id must be non-empty")
    status = value["status"]
    if status == "ok":
        if not isinstance(value["prediction"], str):
            raise RecoveryError(f"{path}:{line_number}: ok prediction must be a string")
        if value["error_type"] is not None or value["error_message"] is not None:
            raise RecoveryError(f"{path}:{line_number}: ok errors must be null")
    elif status == "error":
        if value["prediction"] is not None:
            raise RecoveryError(f"{path}:{line_number}: failed prediction must be null")
        if not isinstance(value["error_type"], str) or not value["error_type"]:
            raise RecoveryError(f"{path}:{line_number}: error_type must be non-empty")
        if not isinstance(value["error_message"], str) or not value["error_message"]:
            raise RecoveryError(f"{path}:{line_number}: error_message must be non-empty")
    else:
        raise RecoveryError(f"{path}:{line_number}: status must be ok or error")
    return value


def _object_without_duplicate_keys(pairs):
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _candidate_index(candidates: Sequence[DpoCandidate]) -> dict[str, DpoCandidate]:
    indexed: dict[str, DpoCandidate] = {}
    for candidate in candidates:
        if candidate.sample_id in indexed:
            raise RecoveryError(f"duplicate normalized sample_id: {candidate.sample_id}")
        indexed[candidate.sample_id] = candidate
    return indexed


def _training_row(candidate: DpoCandidate, prediction: str) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": turn.from_, "value": turn.value}
            for turn in candidate.conversations
        ],
        "chosen": {"from": "gpt", "value": candidate.chosen},
        "rejected": {"from": "gpt", "value": prediction},
        "images": [image.original for image in candidate.images],
    }


def _destination(
    output_dir: Path, output_name: str, output_path: str | Path | None
) -> Path:
    if output_path is not None:
        destination = Path(output_path).resolve()
    else:
        configured = Path(output_name)
        destination = (
            output_dir / f"{configured.stem}.recovered{configured.suffix}"
        ).resolve()
    if destination.suffix.casefold() != ".jsonl":
        raise RecoveryError("output path must end with .jsonl")
    return destination


def _write_jsonl_atomic(
    destination: Path, rows: Sequence[Mapping[str, Any]], *, force: bool
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            for row in tqdm(rows, desc="写入训练文件", unit="row"):
                line = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                stream.write(line)
                digest.update(line)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() and not force:
            raise RecoveryError(f"output already exists: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecoveryError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object: {path}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly recover a trainable DPO JSONL from inference shards."
    )
    parser.add_argument("--config", default="dpo_full.json")
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = recover_dpo_shards(
            args.config,
            output_path=args.output,
            force=args.force,
        )
    except (RecoveryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print("恢复完成", flush=True)
    print(f"分片记录: {result.shard_record_count}", flush=True)
    print(f"训练样本: {result.training_row_count}", flush=True)
    print(f"推理失败过滤: {result.inference_error_count}", flush=True)
    print(f"空回答过滤: {result.empty_rejected_count}", flush=True)
    print(f"同答案过滤: {result.identical_pair_count}", flush=True)
    print(f"输出文件: {result.output_path}", flush=True)
    print(f"SHA256: {result.sha256}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
