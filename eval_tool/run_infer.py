from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

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
)
from .io import load_truth_dataset


def run(config: InferConfig, generator: VLGenerator | None = None) -> dict[str, Path]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    use_parallel = generator is None and bool(config.gpu_ids)
    local_generator = generator
    if not use_parallel and local_generator is None:
        local_generator = QwenVLGenerator(
            model_path=config.model_path,
            max_new_tokens=config.max_new_tokens,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
        )
    written: dict[str, Path] = {}
    for dataset_key, dataset_name in config.datasets.items():
        out_path = prediction_output_path(config.out_dir, config.model_name, dataset_name)
        if out_path.exists() and not config.overwrite:
            written[dataset_key] = out_path
            continue
        print(f"[infer] 读取数据集 {dataset_name} ...", flush=True)
        data = load_truth_dataset(config.tsv_dir, dataset_name)
        if config.limit is not None:
            data = data.head(config.limit).copy()
        rows = data.to_dict("records")
        missing_count, missing_sample = count_missing_images(rows)
        if missing_count:
            print(
                f"[infer] 警告：{dataset_name} 共 {len(rows)} 条中有 {missing_count} 条无图（image 列为空/NaN），"
                f"这些行将以纯文本方式推理。示例 index: {missing_sample}",
                flush=True,
            )
        print(f"[infer] {dataset_name}: 共 {len(rows)} 条，开始推理", flush=True)
        if use_parallel:
            predictions = _run_dataset_parallel(config, dataset_key, rows)
        else:
            predictions = generate_predictions(
                rows,
                dataset_key,
                config.prompt_files,
                local_generator,
                batch_size=config.batch_size,
                progress=True,
                progress_desc=f"Infer {dataset_name}",
            )
        output = build_prediction_frame(data, predictions)
        output.to_excel(out_path, index=False)
        written[dataset_key] = out_path
    return written


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
