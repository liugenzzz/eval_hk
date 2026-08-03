from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import pandas as pd

from .history import parse_history_cell
from .imaging import decode_image_cell
from .prompting import PromptTemplate, load_prompt_text


DEFAULT_INFER_PROMPTS = {
    "mcq": (
        "请根据图片和题目选择唯一正确选项，只输出选项字母。\n"
        "题目：{question}\n"
        "A. {A}\nB. {B}\nC. {C}\nD. {D}"
    ),
    "judge": (
        "请根据图片和题目判断说法是否正确。A 表示正确，B 表示错误。只输出 A 或 B。\n"
        "题目：{question}\n"
        "A. {A}\nB. {B}"
    ),
    "vqa": (
        "请根据图片完成描述性回答。回答必须围绕图中可见内容，说明关键对象、结构/位置关系、"
        "标注信息、数值或显著差异；不要只给通用功能解释。\n"
        "题目：{question}"
    ),
}


class VLGenerator(Protocol):
    def generate(self, prompt: str, image_b64: str | None = None) -> str:
        ...


class PredictionMergeError(RuntimeError):
    """Prediction worker results cannot be merged into a complete ordered output."""


class PredictionBatchError(RuntimeError):
    """A generator returned a different number of predictions than requested."""


@dataclass
class QwenVLGenerator:
    model_path: Path
    max_new_tokens: int = 512
    torch_dtype: str = "auto"
    device_map: str = "auto"

    def __post_init__(self) -> None:
        try:
            import torch
            import transformers
        except Exception as exc:
            raise RuntimeError(
                "Qwen-VL inference requires transformers and torch. Install the matching Qwen-VL environment first."
            ) from exc
        model_cls = (
            getattr(transformers, "AutoModelForImageTextToText", None)
            or getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            or getattr(transformers, "AutoModelForVision2Seq", None)
        )
        if model_cls is None:
            raise RuntimeError(
                "Your transformers version lacks a compatible Qwen-VL auto model class. "
                "Upgrade transformers to a Qwen2.5-VL/Qwen3-VL compatible version."
            )
        self._torch = torch
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            str(self.model_path),
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )

    def generate(self, prompt: str | list[dict[str, Any]], image_b64: str | list[str] | None = None) -> str:
        return self.generate_batch([prompt], [image_b64])[0]

    def generate_batch(
        self,
        prompts: list[str | list[dict[str, Any]]],
        image_b64s: list[str | list[str] | None] | None = None,
    ) -> list[str]:
        image_b64s = image_b64s or [None] * len(prompts)
        images_per_sample = [_decode_images(cell) for cell in image_b64s]
        texts = []
        input_images = []
        for prompt, images in zip(prompts, images_per_sample):
            messages = build_chat_messages(prompt, images)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            # Qwen's processor wants a single flat list of every image in the batch,
            # matched to the <|image_pad|> placeholders in order — not a list per sample.
            input_images.extend(images)
        inputs_kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
        if input_images:
            inputs_kwargs["images"] = input_images
        inputs = self.processor(**inputs_kwargs)
        inputs = inputs.to(self.model.device)
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated[:, input_len:]
        decoded = self.processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [text.strip() for text in decoded]


def _row_prompt(
    dataset_key: str,
    row: Mapping[str, Any],
    prompt_files: Mapping[str, Path],
    prompt_texts: Mapping[str, str] | None = None,
) -> str | list[dict[str, Any]]:
    """Single-turn rows render to a prompt string; rows with history become chat turns
    (gold answers replayed as assistant messages), with the prompt template applied to
    the current question only."""
    rendered = render_infer_prompt(dataset_key, row, prompt_files, prompt_texts=prompt_texts)
    history = parse_history_cell(row.get("history"))
    if not history:
        return rendered
    turns: list[dict[str, Any]] = []
    for turn in history:
        turns.append({"role": "user", "text": turn["q"], "n_img": turn.get("n_img", 0)})
        turns.append({"role": "assistant", "text": turn["a"]})
    turns.append({"role": "user", "text": rendered})
    return turns


def resolve_infer_prompt_text(dataset_key: str, prompt_files: Mapping[str, Path]) -> str:
    if dataset_key in prompt_files:
        return load_prompt_text(prompt_files[dataset_key])
    return DEFAULT_INFER_PROMPTS.get(dataset_key, DEFAULT_INFER_PROMPTS["vqa"])


def render_infer_prompt(
    dataset_key: str,
    row: Mapping[str, Any],
    prompt_files: Mapping[str, Path],
    prompt_texts: Mapping[str, str] | None = None,
) -> str:
    template_text = (
        prompt_texts[dataset_key]
        if prompt_texts is not None and dataset_key in prompt_texts
        else resolve_infer_prompt_text(dataset_key, prompt_files)
    )
    return PromptTemplate(template_text).render(row)


def prediction_output_path(out_dir: str | Path, model_name: str, dataset_name: str) -> Path:
    return Path(out_dir) / f"{model_name}_{dataset_name}.xlsx"


def build_prediction_frame(data: pd.DataFrame, predictions: list[str]) -> pd.DataFrame:
    keep = [
        "index",
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "category",
        "l2-category",
        "source_id",
    ]
    cols = [col for col in keep if col in data.columns]
    out = data[cols].copy()
    out["prediction"] = predictions
    return out


def generate_predictions(
    rows: list[Mapping[str, Any]],
    dataset_key: str,
    prompt_files: Mapping[str, Path],
    generator: VLGenerator,
    batch_size: int = 1,
    progress: bool = False,
    progress_desc: str = "Infer",
    prompt_texts: Mapping[str, str] | None = None,
    on_batch: Callable[[list[Mapping[str, Any]], list[str]], None] | None = None,
) -> list[str]:
    predictions: list[str] = []
    prompt_snapshot = dict(prompt_texts or {})
    prompt_snapshot.setdefault(dataset_key, resolve_infer_prompt_text(dataset_key, prompt_files))
    batch_size = max(1, int(batch_size))
    starts = list(range(0, len(rows), batch_size))
    # Plain periodic prints instead of tqdm: tqdm floods redirected/nohup logs
    # with one line per refresh, ballooning the log file.
    log_every = max(1, len(starts) // 100)
    for batch_no, start in enumerate(starts):
        batch = rows[start : start + batch_size]
        prompts = [_row_prompt(dataset_key, row, prompt_files, prompt_texts=prompt_snapshot) for row in batch]
        images = [_image_cell_or_none(row.get("image")) for row in batch]
        if hasattr(generator, "generate_batch"):
            batch_predictions = list(generator.generate_batch(prompts, images))  # type: ignore[attr-defined]
        else:
            batch_predictions = [generator.generate(prompt, image) for prompt, image in zip(prompts, images)]
        if len(batch_predictions) != len(batch):
            raise PredictionBatchError(
                f"returned {len(batch_predictions)} predictions for {len(batch)} rows"
            )
        if on_batch is not None:
            on_batch(batch, batch_predictions)
        predictions.extend(batch_predictions)
        if progress and (batch_no % log_every == 0 or batch_no == len(starts) - 1):
            print(f"[{progress_desc}] {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def chunk_records(rows: list[Mapping[str, Any]], worker_count: int) -> list[list[tuple[int, Mapping[str, Any]]]]:
    worker_count = max(1, int(worker_count))
    chunks: list[list[tuple[int, Mapping[str, Any]]]] = [[] for _ in range(worker_count)]
    for idx, row in enumerate(rows):
        chunks[idx % worker_count].append((idx, row))
    return chunks


def merge_indexed_predictions(chunks: Iterable[Iterable[tuple[int, str]]], total: int) -> list[str]:
    merged: list[str | None] = [None] * total
    for chunk in chunks:
        for idx, prediction in chunk:
            if type(idx) is not int or idx < 0 or idx >= total:
                raise PredictionMergeError(f"invalid position: {idx}")
            if merged[idx] is not None:
                raise PredictionMergeError(f"duplicate position: {idx}")
            merged[idx] = prediction
    missing = [idx for idx, prediction in enumerate(merged) if prediction is None]
    if missing:
        sample = ",".join(str(idx) for idx in missing[:20])
        raise PredictionMergeError(f"missing positions: {sample}")
    return [prediction for prediction in merged if prediction is not None]


def count_missing_images(rows: list[Mapping[str, Any]]) -> tuple[int, list[str]]:
    """Return (missing_count, sample_indices) for rows whose image cell is empty/NaN."""
    missing_indices = []
    for row in rows:
        if _image_cell_or_none(row.get("image")) is None:
            missing_indices.append(str(row.get("index", "")))
    return len(missing_indices), missing_indices[:10]


def _image_cell_or_none(value: Any) -> str | None:
    """Normalize a TSV image cell: None / pandas NaN / blank all mean "no image"."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _decode_images(image_cell: str | list[str] | None) -> list[Any]:
    if not image_cell:
        return []
    from PIL import Image

    cells = image_cell if isinstance(image_cell, list) else decode_image_cell(image_cell)
    images = []
    for raw in cells:
        raw = str(raw)
        if "," in raw and raw.strip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)
        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return images


def _message_content(prompt: str, images: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return content


def build_chat_messages(prompt: str | list[dict[str, Any]], images: list[Any]) -> list[dict[str, Any]]:
    """Build Qwen-style chat messages.

    `prompt` is either a plain string (single-turn: all images + text in one user message)
    or a list of turns [{"role": "user", "text": ..., "n_img": k}, {"role": "assistant", "text": ...}, ...]
    for multi-turn rows: each user turn consumes its n_img images (in order), and any
    images left over are attached to the final user turn.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": _message_content(prompt, images)}]
    messages: list[dict[str, Any]] = []
    cursor = 0
    user_turns = [i for i, turn in enumerate(prompt) if turn.get("role") == "user"]
    last_user = user_turns[-1] if user_turns else -1
    for i, turn in enumerate(prompt):
        role = str(turn.get("role", "user"))
        text = str(turn.get("text", ""))
        if role == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            continue
        if i == last_user:
            turn_images = images[cursor:]
            cursor = len(images)
        else:
            n_img = int(turn.get("n_img", 0) or 0)
            turn_images = images[cursor : cursor + n_img]
            cursor += n_img
        messages.append({"role": "user", "content": _message_content(text, turn_images)})
    return messages
