from __future__ import annotations

import importlib
import os
import queue
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from tqdm import tqdm

from .dpo_cache import DpoInferenceStore
from .dpo_config import DpoInferConfig
from .dpo_input import DpoCandidate, DpoTurn, ImageRef, SourceRef
from .dpo_multimodal import ImageAsset, open_model_messages


class MessageBatchGenerator(Protocol):
    def generate_message_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        enable_thinking: bool | None = None,
    ) -> list[str]: ...


class DpoInferenceFatalError(RuntimeError):
    """Inference cannot continue without risking wrong sample attribution."""


class DpoInferenceProtocolError(DpoInferenceFatalError):
    """A generator violated the message-batch response contract."""


@dataclass(frozen=True)
class InferenceOutcome:
    sample_id: str
    prediction: str | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _CandidateDto:
    sample_id: str
    conversations: tuple[tuple[str, str], ...]
    chosen: str
    images: tuple[tuple[str, str], ...]
    source_input_index: int
    source_path: str
    source_container_format: str
    source_record_index: int
    source_line_number: int | None
    source_id: str | None
    source_raw_digest: str
    source_format: str
    turn_index: int
    turn_count: int


@dataclass(frozen=True)
class _ImageAssetDto:
    original: str
    resolved: str
    mime_type: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class _WorkerConfigDto:
    model_name: str
    model_path: str
    enable_thinking: bool
    max_new_tokens: int
    batch_size: int
    torch_dtype: str
    device_map: str


@dataclass(frozen=True)
class _WorkerLaunch:
    worker_id: int
    gpu_id: int
    config: _WorkerConfigDto
    batches: tuple[tuple[_CandidateDto, ...], ...]
    assets: tuple[_ImageAssetDto, ...]
    phase_dir: str
    fingerprint: str
    expected_sample_ids: tuple[str, ...]
    attempt_id: str
    fsync_every: int
    factory_spec: tuple[str, str] | None


class _PeerFatalStop(RuntimeError):
    pass


def infer_batch_with_isolation(
    batch: Sequence[DpoCandidate],
    generator: MessageBatchGenerator,
    *,
    enable_thinking: bool,
    append: Callable[[Sequence[InferenceOutcome]], None],
) -> list[InferenceOutcome]:
    """Infer a batch, recursively isolating ordinary sample/data failures."""

    return _infer_batch_with_isolation(
        tuple(batch),
        generator,
        enable_thinking=enable_thinking,
        append=append,
        assets={},
    )


def _infer_batch_with_isolation(
    batch: tuple[DpoCandidate, ...],
    generator: MessageBatchGenerator,
    *,
    enable_thinking: bool,
    append: Callable[[Sequence[InferenceOutcome]], None],
    assets: Mapping[Path, ImageAsset],
) -> list[InferenceOutcome]:
    if not batch:
        return []

    try:
        method = getattr(generator, "generate_message_batch")
        if not callable(method):
            raise DpoInferenceProtocolError(
                "generator does not provide generate_message_batch()"
            )
        with ExitStack() as stack:
            messages_batch = [
                stack.enter_context(open_model_messages(candidate, assets))
                for candidate in batch
            ]
            predictions = method(
                messages_batch,
                enable_thinking=enable_thinking,
            )
        if (
            isinstance(predictions, (str, bytes))
            or not isinstance(predictions, Sequence)
            or len(predictions) != len(batch)
        ):
            count = (
                len(predictions)
                if isinstance(predictions, Sequence)
                and not isinstance(predictions, (str, bytes))
                else "non-sequence"
            )
            raise DpoInferenceProtocolError(
                "generator returned an invalid prediction count "
                f"(expected {len(batch)}, got {count})"
            )
        if any(not isinstance(prediction, str) for prediction in predictions):
            raise DpoInferenceProtocolError(
                "generator predictions must all be strings"
            )
    except BaseException as exc:
        if _is_infrastructure_failure(exc):
            raise
        if len(batch) > 1:
            midpoint = len(batch) // 2
            left = _infer_batch_with_isolation(
                batch[:midpoint],
                generator,
                enable_thinking=enable_thinking,
                append=append,
                assets=assets,
            )
            right = _infer_batch_with_isolation(
                batch[midpoint:],
                generator,
                enable_thinking=enable_thinking,
                append=append,
                assets=assets,
            )
            return [*left, *right]

        outcome = InferenceOutcome(
            sample_id=batch[0].sample_id,
            prediction=None,
            error_type="inference_error",
            error_message=_exception_detail(exc),
        )
        append((outcome,))
        return [outcome]

    outcomes = [
        InferenceOutcome(sample_id=candidate.sample_id, prediction=prediction)
        for candidate, prediction in zip(batch, predictions)
    ]
    # A completed result becomes durable before another result is exposed. This
    # keeps the recovery boundary precise even when the successful model batch
    # contains many samples.
    for outcome in outcomes:
        append((outcome,))
    return outcomes


def run_dpo_inference(
    candidates: Sequence[DpoCandidate],
    assets: Mapping[Path, ImageAsset],
    config: DpoInferConfig,
    store: DpoInferenceStore,
    *,
    generator_factory: Callable[[DpoInferConfig], MessageBatchGenerator] | None = None,
) -> dict[str, InferenceOutcome]:
    """Run only pending candidates and return cache results in source order."""

    ordered = tuple(candidates)
    _validate_candidates(ordered)
    loaded = store.load()
    pending_ids = {
        candidate.sample_id
        for candidate in ordered
        if (candidate.sample_id,) not in loaded
    }
    pending = tuple(
        candidate for candidate in ordered if candidate.sample_id in pending_ids
    )

    factory_spec = _importable_factory_spec(generator_factory)
    use_parallel = bool(config.gpu_ids) and (
        generator_factory is None or factory_spec is not None
    )

    if pending:
        if use_parallel:
            _run_parallel_inference(
                pending,
                assets,
                config,
                store,
                factory_spec=factory_spec,
            )
        else:
            _run_sequential_inference(
                pending,
                assets,
                config,
                store,
                generator_factory=generator_factory,
            )
        loaded = store.load()

    store.mark_complete(snapshot=loaded)
    outcomes = _inference_records(loaded)
    missing = [candidate.sample_id for candidate in ordered if candidate.sample_id not in outcomes]
    if missing:
        raise DpoInferenceFatalError(
            f"inference cache is missing {len(missing)} requested candidates"
        )
    return {candidate.sample_id: outcomes[candidate.sample_id] for candidate in ordered}


def _run_sequential_inference(
    candidates: tuple[DpoCandidate, ...],
    assets: Mapping[Path, ImageAsset],
    config: DpoInferConfig,
    store: DpoInferenceStore,
    *,
    generator_factory: Callable[[DpoInferConfig], MessageBatchGenerator] | None,
) -> None:
    factory = generator_factory or _default_generator_factory
    try:
        generator = factory(config)
    except BaseException as exc:
        raise DpoInferenceFatalError(
            f"DPO model could not be loaded: {_exception_detail(exc)}"
        ) from exc
    if not callable(getattr(generator, "generate_message_batch", None)):
        raise DpoInferenceFatalError(
            "DPO model does not implement generate_message_batch()"
        )

    def append(outcomes: Sequence[InferenceOutcome]) -> None:
        store.append_batch([_outcome_record(outcome) for outcome in outcomes])

    try:
        for start in range(0, len(candidates), config.batch_size):
            _infer_batch_with_isolation(
                candidates[start : start + config.batch_size],
                generator,
                enable_thinking=config.enable_thinking,
                append=append,
                assets=assets,
            )
    finally:
        store.sync()


def _run_parallel_inference(
    candidates: tuple[DpoCandidate, ...],
    assets: Mapping[Path, ImageAsset],
    config: DpoInferConfig,
    store: DpoInferenceStore,
    *,
    factory_spec: tuple[str, str] | None,
) -> None:
    import multiprocessing

    batches = tuple(
        candidates[start : start + config.batch_size]
        for start in range(0, len(candidates), config.batch_size)
    )
    slots = tuple(
        gpu_id
        for gpu_id in config.gpu_ids
        for _ in range(config.workers_per_gpu)
    )
    worker_count = min(len(slots), len(batches))
    assignments: list[list[tuple[DpoCandidate, ...]]] = [
        [] for _ in range(worker_count)
    ]
    for index, batch in enumerate(batches):
        assignments[index % worker_count].append(batch)

    expected_sample_ids = tuple(key[0] for key in store.expected_keys)
    asset_dtos = tuple(_asset_to_dto(asset) for asset in assets.values())
    launches = [
        _WorkerLaunch(
            worker_id=worker_id,
            gpu_id=slots[worker_id],
            config=_config_to_dto(config),
            batches=tuple(
                tuple(_candidate_to_dto(candidate) for candidate in batch)
                for batch in assignments[worker_id]
            ),
            assets=asset_dtos,
            phase_dir=str(store.phase_dir),
            fingerprint=store.fingerprint,
            expected_sample_ids=expected_sample_ids,
            attempt_id=store.attempt_id,
            fsync_every=store.fsync_every,
            factory_spec=factory_spec,
        )
        for worker_id in range(worker_count)
    ]
    total_batches = sum(len(launch.batches) for launch in launches)

    context = multiprocessing.get_context("spawn")
    status_queue = context.Queue()
    start_event = context.Event()
    fatal_event = context.Event()
    processes = [
        context.Process(
            target=_dpo_inference_worker_entry,
            args=(launch, status_queue, start_event, fatal_event),
            name=f"dpo-infer-gpu{launch.gpu_id}-worker{launch.worker_id}",
        )
        for launch in launches
    ]

    try:
        for process in processes:
            process.start()
        _wait_for_worker_readiness(processes, status_queue, fatal_event)
        start_event.set()
        _wait_for_worker_completion(
            processes, status_queue, fatal_event, total_batches=total_batches
        )
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                raise DpoInferenceFatalError(
                    f"inference worker {process.name} did not exit after completion"
                )
            if process.exitcode != 0:
                raise DpoInferenceFatalError(
                    f"inference worker {process.name} exited with code {process.exitcode}"
                )
    except BaseException:
        fatal_event.set()
        _terminate_processes(processes)
        raise
    finally:
        try:
            status_queue.close()
            status_queue.join_thread()
        except (OSError, ValueError):
            pass


def _wait_for_worker_readiness(processes, status_queue, fatal_event) -> None:
    ready: set[int] = set()
    while len(ready) < len(processes):
        message = _next_worker_message(processes, status_queue, fatal_event)
        kind, worker_id, error_type, detail = message
        if kind == "ready":
            if worker_id in ready:
                raise DpoInferenceFatalError(
                    f"inference worker {worker_id} reported readiness twice"
                )
            ready.add(worker_id)
            continue
        if kind == "fatal":
            raise DpoInferenceFatalError(
                f"inference worker {worker_id} failed before readiness: "
                f"{error_type}: {detail}"
            )
        raise DpoInferenceFatalError(
            f"inference worker {worker_id} sent {kind!r} before readiness"
        )


def _wait_for_worker_completion(
    processes,
    status_queue,
    fatal_event,
    *,
    total_batches: int,
) -> None:
    if type(total_batches) is not int or total_batches < 0:
        raise DpoInferenceFatalError(
            "total inference batches must be a non-negative integer"
        )
    pbar = tqdm(total=total_batches, desc="VLM\u63a8\u7406", unit="batch")
    try:
        _wait_for_worker_completion_messages(
            processes,
            status_queue,
            fatal_event,
            total_batches=total_batches,
            pbar=pbar,
        )
    finally:
        pbar.close()


def _wait_for_worker_completion_messages(
    processes,
    status_queue,
    fatal_event,
    *,
    total_batches: int,
    pbar,
) -> None:
    done: set[int] = set()
    peer_stopped: set[int] = set()
    completed_batches = 0
    while len(done) + len(peer_stopped) < len(processes):
        message = _next_worker_message(
            processes,
            status_queue,
            fatal_event,
            known_exits=done | peer_stopped,
        )
        kind, worker_id, error_type, detail = message
        if kind == "progress":
            if worker_id in done or worker_id in peer_stopped:
                raise DpoInferenceFatalError(
                    f"inference worker {worker_id} reported progress after exit"
                )
            if type(error_type) is not int or error_type < 1 or detail is not None:
                raise DpoInferenceFatalError(
                    f"inference worker {worker_id} sent invalid progress"
                )
            completed_batches += error_type
            if completed_batches > total_batches:
                raise DpoInferenceFatalError(
                    "inference workers reported more batches than scheduled"
                )
            pbar.update(error_type)
            continue
        if kind == "done":
            if worker_id in done or worker_id in peer_stopped:
                raise DpoInferenceFatalError(
                    f"inference worker {worker_id} reported completion twice"
                )
            done.add(worker_id)
        elif kind == "peer_stopped":
            peer_stopped.add(worker_id)
        elif kind == "fatal":
            raise DpoInferenceFatalError(
                f"inference worker {worker_id} failed: {error_type}: {detail}"
            )
        else:
            raise DpoInferenceFatalError(
                f"inference worker {worker_id} sent unexpected status {kind!r}"
            )
    if peer_stopped:
        raise DpoInferenceFatalError(
            "inference workers stopped because a fatal peer did not report its failure"
        )
    if completed_batches != total_batches:
        raise DpoInferenceFatalError(
            "inference workers exited before reporting every completed batch"
        )


def _next_worker_message(
    processes,
    status_queue,
    fatal_event,
    *,
    known_exits: set[int] | None = None,
) -> tuple[str, int, int | str | None, str | None]:
    known = known_exits or set()
    while True:
        try:
            message = status_queue.get(timeout=0.1)
        except queue.Empty:
            if fatal_event.is_set():
                # The failing worker sets the event before syncing its shard and
                # publishing the fatal message. Give that durable handoff priority
                # over exit-code polling.
                continue
            for worker_id, process in enumerate(processes):
                if worker_id in known:
                    continue
                if process.exitcode is not None:
                    raise DpoInferenceFatalError(
                        f"inference worker {worker_id} exited without a terminal status "
                        f"(code {process.exitcode})"
                    )
            continue
        if (
            not isinstance(message, tuple)
            or len(message) != 4
            or message[0]
            not in {"ready", "done", "fatal", "peer_stopped", "progress"}
            or type(message[1]) is not int
        ):
            raise DpoInferenceFatalError("inference worker sent an invalid status")
        return message


def _terminate_processes(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout=5)


def _dpo_inference_worker_entry(
    launch: _WorkerLaunch,
    status_queue,
    start_event,
    fatal_event,
) -> None:
    store: DpoInferenceStore | None = None
    terminal: tuple[str, int, str | None, str | None] | None = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(launch.gpu_id)
        config = _config_from_dto(launch.config)
        store = DpoInferenceStore(
            Path(launch.phase_dir),
            launch.fingerprint,
            launch.expected_sample_ids,
            fsync_every=launch.fsync_every,
        )
        if store.attempt_id != launch.attempt_id:
            raise DpoInferenceFatalError(
                "active inference attempt changed before worker startup"
            )
        # Populate the process-local known-key set before any worker begins
        # writing; concurrent load-and-tail-repair would be unsafe.
        store.load()
        factory = (
            _resolve_factory(launch.factory_spec)
            if launch.factory_spec is not None
            else _default_generator_factory
        )
        generator = factory(config)
        if not callable(getattr(generator, "generate_message_batch", None)):
            raise DpoInferenceFatalError(
                "DPO model does not implement generate_message_batch()"
            )
        assets = _assets_from_dtos(launch.assets)
        status_queue.put(("ready", launch.worker_id, None, None))
        start_event.wait()

        def append(outcomes: Sequence[InferenceOutcome]) -> None:
            if fatal_event.is_set():
                raise _PeerFatalStop("a peer reported a fatal inference failure")
            assert store is not None
            store.append_batch(
                [_outcome_record(outcome) for outcome in outcomes],
                worker_id=launch.worker_id,
            )

        for batch_dtos in launch.batches:
            if fatal_event.is_set():
                raise _PeerFatalStop("a peer reported a fatal inference failure")
            batch = tuple(_candidate_from_dto(dto) for dto in batch_dtos)
            _infer_batch_with_isolation(
                batch,
                generator,
                enable_thinking=config.enable_thinking,
                append=append,
                assets=assets,
            )
            status_queue.put(("progress", launch.worker_id, 1, None))
        terminal = ("done", launch.worker_id, None, None)
    except _PeerFatalStop:
        terminal = ("peer_stopped", launch.worker_id, None, None)
    except BaseException as exc:
        fatal_event.set()
        terminal = (
            "fatal",
            launch.worker_id,
            type(exc).__name__,
            str(exc) or type(exc).__name__,
        )
    finally:
        if store is not None:
            try:
                store.sync()
            except BaseException as exc:
                fatal_event.set()
                terminal = (
                    "fatal",
                    launch.worker_id,
                    type(exc).__name__,
                    str(exc) or type(exc).__name__,
                )
        if terminal is not None:
            try:
                status_queue.put(terminal)
            except BaseException:
                pass


def _default_generator_factory(config: DpoInferConfig) -> MessageBatchGenerator:
    # Keep torch/transformers imports behind worker CUDA binding.
    from .infer import QwenVLGenerator

    return QwenVLGenerator(
        model_path=config.model_path,
        max_new_tokens=config.max_new_tokens,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
    )


def _importable_factory_spec(
    factory: Callable[[DpoInferConfig], MessageBatchGenerator] | None,
) -> tuple[str, str] | None:
    if factory is None:
        return None
    module_name = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", None)
    if (
        not isinstance(module_name, str)
        or not isinstance(qualname, str)
        or "<locals>" in qualname
        or "<lambda>" in qualname
    ):
        return None
    try:
        resolved = _resolve_factory((module_name, qualname))
    except Exception:
        return None
    return (module_name, qualname) if resolved is factory else None


def _resolve_factory(spec: tuple[str, str]):
    module_name, qualname = spec
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise DpoInferenceFatalError("configured generator factory is not callable")
    return value


def _is_infrastructure_failure(exc: BaseException) -> bool:
    if isinstance(exc, (DpoInferenceFatalError, MemoryError)):
        return True
    if not isinstance(exc, Exception):
        return True
    if isinstance(
        exc,
        (BrokenPipeError, ChildProcessError, ConnectionError, EOFError),
    ):
        return True
    name = type(exc).__name__.casefold()
    detail = f"{name}: {exc}".casefold()
    tokens = (
        "outofmemory",
        "out of memory",
        "cuda error",
        "cuda context",
        "cuda initialization",
        "cuda driver",
        "cuda unavailable",
        "no cuda gpu",
        "device-side assert",
        "nccl",
        "cudnn",
        "cublas",
        "hip error",
        "brokenprocesspool",
        "broken worker",
        "worker exited",
        "model unavailable",
        "model is unavailable",
    )
    return any(token in detail for token in tokens)


def _validate_candidates(candidates: tuple[DpoCandidate, ...]) -> None:
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, DpoCandidate):
            raise DpoInferenceFatalError(
                f"inference candidate {index} is not a DpoCandidate"
            )
        if not candidate.sample_id or candidate.sample_id in seen:
            raise DpoInferenceFatalError(
                "inference candidates require distinct non-empty sample IDs"
            )
        seen.add(candidate.sample_id)


def _outcome_record(outcome: InferenceOutcome) -> dict[str, object]:
    if outcome.error_type is None:
        return {
            "sample_id": outcome.sample_id,
            "status": "ok",
            "prediction": outcome.prediction,
            "error_type": None,
            "error_message": None,
        }
    return {
        "sample_id": outcome.sample_id,
        "status": "error",
        "prediction": None,
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
    }


def _inference_records(
    records: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> dict[str, InferenceOutcome]:
    outcomes: dict[str, InferenceOutcome] = {}
    for record in records.values():
        sample_id = record["sample_id"]
        outcomes[sample_id] = InferenceOutcome(
            sample_id=sample_id,
            prediction=record["prediction"],
            error_type=record["error_type"],
            error_message=record["error_message"],
        )
    return outcomes


def _exception_detail(exc: BaseException) -> str:
    detail = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


def _config_to_dto(config: DpoInferConfig) -> _WorkerConfigDto:
    return _WorkerConfigDto(
        model_name=config.model_name,
        model_path=str(config.model_path),
        enable_thinking=config.enable_thinking,
        max_new_tokens=config.max_new_tokens,
        batch_size=config.batch_size,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
    )


def _config_from_dto(dto: _WorkerConfigDto) -> DpoInferConfig:
    return DpoInferConfig(
        model_name=dto.model_name,
        model_path=Path(dto.model_path),
        enable_thinking=dto.enable_thinking,
        max_new_tokens=dto.max_new_tokens,
        batch_size=dto.batch_size,
        torch_dtype=dto.torch_dtype,
        device_map=_worker_device_map(dto.device_map),
        gpu_ids=(),
        workers_per_gpu=1,
    )


def _worker_device_map(device_map: str) -> str:
    if re.search(r"cuda\s*:\s*\d+", device_map, flags=re.IGNORECASE):
        return "cuda:0"
    return device_map


def _candidate_to_dto(candidate: DpoCandidate) -> _CandidateDto:
    source = candidate.source
    return _CandidateDto(
        sample_id=candidate.sample_id,
        conversations=tuple(
            (turn.from_, turn.value) for turn in candidate.conversations
        ),
        chosen=candidate.chosen,
        images=tuple(
            (image.original, str(image.resolved)) for image in candidate.images
        ),
        source_input_index=source.input_index,
        source_path=str(source.source_path),
        source_container_format=source.container_format,
        source_record_index=source.record_index,
        source_line_number=source.line_number,
        source_id=source.source_id,
        source_raw_digest=source.raw_digest,
        source_format=candidate.source_format,
        turn_index=candidate.turn_index,
        turn_count=candidate.turn_count,
    )


def _candidate_from_dto(dto: _CandidateDto) -> DpoCandidate:
    source = SourceRef(
        input_index=dto.source_input_index,
        source_path=Path(dto.source_path),
        container_format=dto.source_container_format,  # type: ignore[arg-type]
        record_index=dto.source_record_index,
        line_number=dto.source_line_number,
        source_id=dto.source_id,
        raw_digest=dto.source_raw_digest,
    )
    return DpoCandidate(
        sample_id=dto.sample_id,
        conversations=tuple(
            DpoTurn(from_=role, value=value)  # type: ignore[arg-type]
            for role, value in dto.conversations
        ),
        chosen=dto.chosen,
        images=tuple(
            ImageRef(original=original, resolved=Path(resolved))
            for original, resolved in dto.images
        ),
        source=source,
        source_format=dto.source_format,  # type: ignore[arg-type]
        turn_index=dto.turn_index,
        turn_count=dto.turn_count,
    )


def _asset_to_dto(asset: ImageAsset) -> _ImageAssetDto:
    return _ImageAssetDto(
        original=asset.ref.original,
        resolved=str(asset.ref.resolved),
        mime_type=asset.mime_type,
        sha256=asset.sha256,
        byte_count=asset.byte_count,
    )


def _assets_from_dtos(
    dtos: tuple[_ImageAssetDto, ...],
) -> dict[Path, ImageAsset]:
    assets: dict[Path, ImageAsset] = {}
    for dto in dtos:
        resolved = Path(dto.resolved)
        asset = ImageAsset(
            ref=ImageRef(original=dto.original, resolved=resolved),
            mime_type=dto.mime_type,
            sha256=dto.sha256,
            byte_count=dto.byte_count,
        )
        assets[resolved] = asset
    return assets
