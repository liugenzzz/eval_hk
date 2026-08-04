from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import InferConfig, load_infer_config
from .infer import (
    QwenVLGenerator,
    VLGenerator,
    build_prediction_frame,
    chunk_records,
    count_missing_images,
    generate_predictions,
    merge_indexed_predictions,
    prediction_output_path,
    resolve_infer_prompt_text,
)
from .infer_cache import (
    InferCacheError,
    InferManifest,
    InferShardStore,
    InferenceLock,
    build_index_digest,
    build_infer_fingerprint,
    read_manifest,
    write_manifest,
)
from .io import load_truth_dataset


class InferenceWorkerError(RuntimeError):
    pass


def run(config: InferConfig, generator: VLGenerator | None = None) -> dict[str, Path]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    use_parallel = generator is None and bool(config.gpu_ids)
    overwrite = _overwrite_enabled(config)
    local_generator = generator

    def get_local_generator() -> VLGenerator:
        nonlocal local_generator
        if local_generator is None:
            local_generator = QwenVLGenerator(
                model_path=config.model_path,
                max_new_tokens=config.max_new_tokens,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
            )
        return local_generator

    written: dict[str, Path] = {}
    for dataset_key, dataset_name in config.datasets.items():
        out_path = prediction_output_path(config.out_dir, config.model_name, dataset_name)
        if out_path.exists() and not overwrite and not config.resume:
            written[dataset_key] = out_path
            continue
        print(f"[infer] 读取数据集 {dataset_name} ...", flush=True)
        try:
            data = load_truth_dataset(config.tsv_dir, dataset_name)
            if config.limit is not None:
                data = data.head(config.limit).copy()
            rows = data.to_dict("records")
            missing_count, missing_sample = count_missing_images(rows)
        except Exception as exc:
            if config.resume:
                _append_infer_warning(config.out_dir, dataset_key, dataset_name, exc)
            raise
        if missing_count:
            print(
                f"[infer] 警告：{dataset_name} 共 {len(rows)} 条中有 {missing_count} 条无图（image 列为空/NaN），"
                f"这些行将以纯文本方式推理。示例 index: {missing_sample}",
                flush=True,
            )
        print(f"[infer] {dataset_name}: 共 {len(rows)} 条，开始推理", flush=True)
        if config.resume:
            _run_dataset_resumable(
                config,
                dataset_key,
                dataset_name,
                data,
                rows,
                out_path,
                get_local_generator,
                use_parallel=use_parallel,
            )
        elif use_parallel:
            predictions = _run_dataset_parallel(config, dataset_key, rows)
        else:
            predictions = generate_predictions(
                rows,
                dataset_key,
                config.prompt_files,
                get_local_generator(),
                batch_size=config.batch_size,
                progress=True,
                progress_desc=f"Infer {dataset_name}",
            )
        if not config.resume:
            output = build_prediction_frame(data, predictions)
            output.to_excel(out_path, index=False)
        written[dataset_key] = out_path
    return written


def _run_dataset_resumable(
    config: InferConfig,
    dataset_key: str,
    dataset_name: str,
    data,
    rows: list[Mapping[str, Any]],
    out_path: Path,
    generator_factory: Callable[[], VLGenerator | None],
    *,
    use_parallel: bool,
) -> None:
    lock_path = config.out_dir / "_partial" / f"{dataset_key}.lock"
    try:
        with InferenceLock(lock_path):
            _run_dataset_resumable_unlocked(
                config,
                dataset_key,
                dataset_name,
                data,
                rows,
                out_path,
                generator_factory,
                use_parallel=use_parallel,
            )
    except Exception as exc:
        _append_infer_warning(config.out_dir, dataset_key, dataset_name, exc)
        raise


def _run_dataset_resumable_unlocked(
    config: InferConfig,
    dataset_key: str,
    dataset_name: str,
    data,
    rows: list[Mapping[str, Any]],
    out_path: Path,
    generator_factory: Callable[[], VLGenerator | None],
    *,
    use_parallel: bool,
) -> None:
    indexes = _validated_indexes(rows)
    prompt_text = resolve_infer_prompt_text(dataset_key, config.prompt_files)
    fingerprint = build_infer_fingerprint(
        config.model_path,
        prompt_text,
        config.max_new_tokens,
        dataset_key,
        dataset_name,
        rows,
    )
    store = InferShardStore(config.out_dir / "_partial", dataset_key, fingerprint)
    manifest_path = out_path.with_suffix(".infer.json")
    attempt_path = store.partial_dir / f"{dataset_key}.attempt.json"
    expected_manifest = InferManifest(
        fingerprint=fingerprint,
        dataset_name=dataset_name,
        row_count=len(rows),
        index_digest=build_index_digest(indexes),
    )
    overwrite = _overwrite_enabled(config)
    manifest = read_manifest(manifest_path) if not overwrite else None
    attempt = read_manifest(attempt_path) if not overwrite else None
    authorized_attempt = attempt == expected_manifest
    if (
        out_path.exists()
        and manifest == expected_manifest
        and not authorized_attempt
    ):
        if config.clean_partial:
            store.clean_dataset()
        return
    cached = store.load()
    if overwrite:
        store.clear_current()
        cached = {}
        write_manifest(attempt_path, expected_manifest)
    else:
        if (
            manifest is not None
            and manifest != expected_manifest
            and not authorized_attempt
        ):
            raise InferCacheError(
                f"inference manifest does not match current inputs for {dataset_key}; "
                "rerun with --overwrite"
            )
        if (
            store.foreign_fingerprints()
            and manifest != expected_manifest
            and not authorized_attempt
        ):
            raise InferCacheError(
                f"foreign inference shards found for {dataset_key}; rerun with --overwrite"
            )
        if not cached:
            if out_path.exists() and manifest is None:
                raise InferCacheError(
                    f"unverified legacy prediction file for {dataset_key}; "
                    "declare it as a pred input or rerun with --overwrite"
                )
    _validate_cached_indexes(cached, indexes, require_complete=False)
    if manifest == expected_manifest and not out_path.exists():
        missing = [index for index in indexes if index not in cached]
        if missing:
            raise InferCacheError(
                f"completed inference manifest exists but the final prediction file "
                f"and complete shards do not for {dataset_key}; rerun with --overwrite"
            )
    pending_rows = [row for row in rows if str(row["index"]) not in cached]
    if pending_rows:
        if use_parallel:
            _run_dataset_parallel_resumable(
                config,
                dataset_key,
                pending_rows,
                prompt_text,
                store,
            )
        else:
            generator = generator_factory()
            if generator is None:
                raise RuntimeError("sequential inference requires a generator")
            generate_predictions(
                pending_rows,
                dataset_key,
                config.prompt_files,
                generator,
                batch_size=config.batch_size,
                progress=True,
                progress_desc=f"Infer {dataset_name}",
                prompt_texts={dataset_key: prompt_text},
                on_batch=lambda batch_rows, batch_predictions: store.append_batch(
                    (str(row["index"]), prediction)
                    for row, prediction in zip(batch_rows, batch_predictions)
                ),
            )
    completed = store.load()
    _validate_cached_indexes(completed, indexes, require_complete=True)
    predictions = [completed[index] for index in indexes]
    _publish_prediction_artifacts(
        build_prediction_frame(data, predictions),
        out_path,
        manifest_path,
        expected_manifest,
    )
    attempt_path.unlink(missing_ok=True)
    if config.clean_partial:
        store.clean_dataset()


def _validated_indexes(rows: list[Mapping[str, Any]]) -> list[str]:
    indexes: list[str] = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        index = str(row.get("index", "")).strip()
        if not index:
            raise InferCacheError(f"empty index at row {position}")
        if index in seen:
            raise InferCacheError(f"duplicate index '{index}'")
        seen.add(index)
        indexes.append(index)
    return indexes


def _overwrite_enabled(config: InferConfig) -> bool:
    if config.overwrite is None:
        return not config.resume
    return config.overwrite


def _validate_cached_indexes(
    cached: Mapping[str, str],
    expected_indexes: list[str],
    *,
    require_complete: bool,
) -> None:
    expected = set(expected_indexes)
    actual = set(cached)
    unexpected = sorted(actual - expected)
    if unexpected:
        raise InferCacheError(
            f"unexpected cached indexes: {','.join(unexpected[:20])}"
        )
    if require_complete:
        missing = [index for index in expected_indexes if index not in actual]
        if missing:
            raise InferCacheError(f"missing cached indexes: {','.join(missing[:20])}")


def _write_prediction_xlsx_atomic(output, out_path: Path) -> None:
    temporary = out_path.with_name(f".{out_path.stem}.tmp{out_path.suffix}")
    try:
        output.to_excel(temporary, index=False)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_prediction_artifacts(
    output,
    out_path: Path,
    manifest_path: Path,
    manifest: InferManifest,
) -> None:
    backup = out_path.with_name(f".{out_path.stem}.backup{out_path.suffix}")
    had_previous = out_path.is_file()
    if had_previous:
        shutil.copy2(out_path, backup)
    try:
        _write_prediction_xlsx_atomic(output, out_path)
        write_manifest(manifest_path, manifest)
    except BaseException:
        if had_previous:
            backup.replace(out_path)
        else:
            out_path.unlink(missing_ok=True)
        raise
    finally:
        backup.unlink(missing_ok=True)


def _append_infer_warning(
    out_dir: Path,
    dataset_key: str,
    dataset_name: str,
    warning: object,
) -> None:
    with (out_dir / "warnings.log").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f"[infer] {dataset_key}/{dataset_name}: "
            f"{type(warning).__name__}: {warning}\n"
        )


def _run_dataset_parallel_resumable(
    config: InferConfig,
    dataset_key: str,
    rows: list[Mapping[str, Any]],
    prompt_text: str,
    store: InferShardStore,
) -> None:
    worker_gpus = [
        gpu for gpu in config.gpu_ids for _ in range(config.workers_per_gpu)
    ]
    chunks = chunk_records(rows, len(worker_gpus))
    errors: list[tuple[int, Exception]] = []
    with ProcessPoolExecutor(max_workers=len(worker_gpus)) as executor:
        futures = {
            executor.submit(
                _infer_worker_resumable,
                worker_id,
                gpu_id,
                chunk,
                dataset_key,
                prompt_text,
                str(config.model_path),
                config.max_new_tokens,
                config.batch_size,
                config.torch_dtype,
                str(store.partial_dir),
                store.fingerprint,
            ): worker_id
            for worker_id, (gpu_id, chunk) in enumerate(zip(worker_gpus, chunks))
            if chunk
        }
        iterator = as_completed(futures)
        iterator = _progress(
            iterator,
            total=len(futures),
            desc=f"Infer {dataset_key} workers",
        )
        for future in iterator:
            worker_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append((worker_id, exc))
    if errors:
        summary = "; ".join(
            f"worker {worker_id}: {error}" for worker_id, error in errors
        )
        raise InferenceWorkerError(summary)


def _infer_worker_resumable(
    worker_id: int,
    gpu_id: int,
    indexed_rows: list[tuple[int, Mapping[str, Any]]],
    dataset_key: str,
    prompt_text: str,
    model_path: str,
    max_new_tokens: int,
    batch_size: int,
    torch_dtype: str,
    partial_dir: str,
    fingerprint: str,
) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    generator = QwenVLGenerator(
        model_path=Path(model_path),
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    rows = [row for _, row in indexed_rows]
    store = InferShardStore(partial_dir, dataset_key, fingerprint)
    generate_predictions(
        rows,
        dataset_key,
        {},
        generator,
        batch_size=batch_size,
        progress=True,
        progress_desc=f"Infer {dataset_key} gpu{gpu_id}-w{worker_id}",
        prompt_texts={dataset_key: prompt_text},
        on_batch=lambda batch_rows, batch_predictions: store.append_batch(
            (
                (str(row["index"]), prediction)
                for row, prediction in zip(batch_rows, batch_predictions)
            ),
            worker_id=worker_id,
        ),
    )
    return len(rows)


def _run_dataset_parallel(config: InferConfig, dataset_key: str, rows: list[Mapping[str, Any]]) -> list[str]:
    worker_gpus = [gpu for gpu in config.gpu_ids for _ in range(config.workers_per_gpu)]
    chunks = chunk_records(rows, len(worker_gpus))
    indexed_results: list[list[tuple[int, str]]] = []
    with ProcessPoolExecutor(max_workers=len(worker_gpus)) as executor:
        futures = []
        for worker_id, (gpu_id, chunk) in enumerate(zip(worker_gpus, chunks)):
            if not chunk:
                continue
            futures.append(
                executor.submit(
                    _infer_worker,
                    worker_id,
                    gpu_id,
                    chunk,
                    dataset_key,
                    {k: str(v) for k, v in config.prompt_files.items()},
                    str(config.model_path),
                    config.max_new_tokens,
                    config.batch_size,
                    config.torch_dtype,
                )
            )
        iterator = as_completed(futures)
        iterator = _progress(iterator, total=len(futures), desc=f"Infer {dataset_key} workers")
        for fut in iterator:
            indexed_results.append(fut.result())
    return merge_indexed_predictions(indexed_results, total=len(rows))


def _infer_worker(
    worker_id: int,
    gpu_id: int,
    indexed_rows: list[tuple[int, Mapping[str, Any]]],
    dataset_key: str,
    prompt_files: dict[str, str],
    model_path: str,
    max_new_tokens: int,
    batch_size: int,
    torch_dtype: str,
) -> list[tuple[int, str]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    generator = QwenVLGenerator(
        model_path=Path(model_path),
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    indices = [idx for idx, _ in indexed_rows]
    rows = [row for _, row in indexed_rows]
    predictions = generate_predictions(
        rows,
        dataset_key,
        {k: Path(v) for k, v in prompt_files.items()},
        generator,
        batch_size=batch_size,
        progress=True,
        progress_desc=f"Infer {dataset_key} gpu{gpu_id}-w{worker_id}",
    )
    return list(zip(indices, predictions))


def _progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reusable Qwen-VL prediction xlsx files from Aero TSV datasets")
    parser.add_argument("--config", required=True, help="Path to infer config.json or config.py")
    args = parser.parse_args()
    written = run(load_infer_config(args.config))
    print("Written prediction files:")
    for dataset_key, path in written.items():
        print(f"- {dataset_key}: {path}")


if __name__ == "__main__":
    main()
