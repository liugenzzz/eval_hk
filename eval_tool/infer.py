from __future__ import annotations

import base64
import copy
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import pandas as pd

from .history import parse_history_cell
from .imaging import (
    IMAGE_PREPROCESS_PROFILE,
    decode_image_cell,
    prepare_llamafactory_qwen_image,
    validate_image_pixel_bounds,
)
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

_PROCESSOR_KWARGS_WARNING = (
    "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` "
    "dict, not in `**kwargs`"
)
_PROCESSOR_KWARGS_FILTER_MARKER = "_eval_tool_processor_kwargs_filter"


class _ProcessorKwargsWarningFilter(logging.Filter):
    _eval_tool_processor_kwargs_filter = True

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _PROCESSOR_KWARGS_WARNING


def _force_left_padding(processor: Any) -> None:
    """把 processor 和它内部的 tokenizer 都设成左补齐。

    ``AutoProcessor.from_pretrained(padding_side="left")`` 在多数版本上会透传给
    tokenizer，但不是所有版本都会 —— 而这件事错了不会报错，只会让批里较短的那几条
    预测变成废话。多设一次不花钱，漏设一次查不出来。
    """
    for target in (processor, getattr(processor, "tokenizer", None)):
        if target is None:
            continue
        try:
            target.padding_side = "left"
        except Exception:  # noqa: BLE001 - 有的实现是只读属性，设不上就算了
            pass


def _install_processor_kwargs_warning_filter() -> None:
    """Hide one noisy Transformers compatibility message, and nothing else."""

    logger = logging.getLogger("transformers.processing_utils")
    if any(
        getattr(item, _PROCESSOR_KWARGS_FILTER_MARKER, False)
        for item in logger.filters
    ):
        return
    logger.addFilter(_ProcessorKwargsWarningFilter())


class VLGenerator(Protocol):
    def generate(self, prompt: str, image_b64: str | None = None) -> str:
        ...


class MessageBatchGenerator(Protocol):
    def generate_message_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        enable_thinking: bool | None = None,
    ) -> list[str]:
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
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None
    # 覆盖 checkpoint 自带 generation_config 的解码参数；None = 照 checkpoint 的来。
    # 评估通路总是传（默认贪心），DPO 采样通路不传，保持原样。
    generation: Mapping[str, Any] | None = None
    # 视频抽帧参数（fps / max_frames / max_pixels …），原样写进 qwen_vl_utils 的视频片段。
    video: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.image_min_pixels, self.image_max_pixels = validate_image_pixel_bounds(
            self.image_min_pixels,
            self.image_max_pixels,
        )
        try:
            import torch
            import transformers
        except Exception as exc:
            raise RuntimeError(
                "Qwen-VL inference requires transformers and torch. Install the matching Qwen-VL environment first."
            ) from exc
        _install_processor_kwargs_warning_filter()
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
        self.processor = transformers.AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            # batch_size > 1 时一批里长短不一的提示词要补齐，**必须从左边补**。
            # decoder-only 模型是接着序列最后一个 token 往下生成的：右补齐会让短提示词
            # 后面跟着一串 pad，模型从 pad 后面开始续写，短的那几条答出来的是废话。
            # transformers 只会打一行警告（"right-padding was detected"），不会报错，
            # 所以这件事在报表上表现为「这个 checkpoint 好像差一点」，查不出来。
            padding_side="left",
        )
        # 有的 processor 不把 padding_side 透传给内部 tokenizer，补一刀。
        _force_left_padding(self.processor)
        if self.image_min_pixels is not None:
            try:
                image_processor = getattr(self.processor, "image_processor", None)
                checkpoint_size = str(getattr(image_processor, "size", None))
            except Exception:
                checkpoint_size = "<unavailable>"
            try:
                print(
                    "[infer] Qwen image preprocessing active: "
                    f"profile={IMAGE_PREPROCESS_PROFILE} "
                    f"min_pixels={self.image_min_pixels} "
                    f"max_pixels={self.image_max_pixels} "
                    f"checkpoint_size={checkpoint_size}",
                    flush=True,
                )
            except Exception:
                pass
        self.model = model_cls.from_pretrained(
            str(self.model_path),
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self._generation_config = None
        if self.generation is not None:
            config = copy.deepcopy(self.model.generation_config)
            if not self.generation.get("do_sample", False):
                # 贪心时把 checkpoint 带的采样参数复位成中性值：留着 0.7 的话它们不起作用，
                # 还会让人以为这次是按 temperature=0.7 解的码。不能设 None —— transformers 5
                # 会拿 checkpoint 的默认值把 None 补回来，然后每一批都警告一次。
                for name, neutral in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50)):
                    setattr(config, name, neutral)
            for name, value in self.generation.items():
                setattr(config, name, value)
            # max_new_tokens 也放进同一个 config：和 generation_config 分开传已被弃用
            config.max_new_tokens = self.max_new_tokens
            self._generation_config = config

    def generate(self, prompt: str | list[dict[str, Any]], image_b64: str | list[str] | None = None) -> str:
        return self.generate_batch([prompt], [image_b64])[0]

    def generate_batch(
        self,
        prompts: list[str | list[dict[str, Any]]],
        image_b64s: list[str | list[str] | None] | None = None,
    ) -> list[str]:
        image_b64s = image_b64s or [None] * len(prompts)
        images_per_sample: list[list[Any]] = []
        owned_images: list[Any] = []
        try:
            for cell in image_b64s:
                images = _decode_images(
                    cell,
                    image_min_pixels=self.image_min_pixels,
                    image_max_pixels=self.image_max_pixels,
                )
                images_per_sample.append(images)
                owned_images.extend(images)
            messages_batch = [
                build_chat_messages(prompt, images)
                for prompt, images in zip(prompts, images_per_sample)
            ]
            return self.generate_message_batch(
                messages_batch, enable_thinking=None
            )
        finally:
            for image in reversed(owned_images):
                image.close()

    def generate_video_batch(self, prompts: list[str], video_lists: list[list[str]]) -> list[str]:
        """每条：若干个视频 + 一段文字（单轮）。抽帧交给 qwen_vl_utils，参数取 self.video。"""
        options = dict(self.video or {})
        messages_batch = []
        for prompt, videos in zip(prompts, video_lists):
            if not isinstance(prompt, str):
                raise ValueError("视频评估只支持单轮提示词")
            content: list[dict[str, Any]] = [{"type": "video", "video": str(v), **options} for v in videos]
            content.append({"type": "text", "text": prompt})
            messages_batch.append([{"role": "user", "content": content}])
        return self.generate_message_batch(messages_batch)

    def _video_inputs(self, messages_batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        image_processor = getattr(self.processor, "image_processor", None)
        patch = int(getattr(image_processor, "patch_size", 14) or 14)
        # Qwen3-VL 的处理器要帧的时间戳（video_metadata），Qwen2/2.5-VL 的不认这个参数。
        with_metadata = "qwen3" in type(self.processor).__name__.lower()
        _, videos, video_kwargs = process_vision_info(
            messages_batch, return_video_kwargs=True, return_video_metadata=with_metadata,
            image_patch_size=patch,
        )
        if not videos:
            return {}
        extra: dict[str, Any] = dict(video_kwargs or {})
        if with_metadata:
            frames, metadata = zip(*videos)
            extra.update(videos=list(frames), video_metadata=list(metadata), do_resize=False)
        else:
            extra["videos"] = list(videos)
        return extra

    def generate_message_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        texts: list[str] = []
        input_images: list[Any] = []
        has_video = any(
            isinstance(part, Mapping) and part.get("type") == "video"
            for messages in messages_batch for message in messages
            for part in (message.get("content") if isinstance(message.get("content"), (list, tuple)) else ())
        )
        for messages in messages_batch:
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking
            texts.append(
                self.processor.apply_chat_template(messages, **template_kwargs)
            )
            # Qwen expects one flat image list matching all message placeholders.
            for message in messages:
                content = message.get("content")
                if not isinstance(content, (list, tuple)):
                    continue
                for part in content:
                    if isinstance(part, Mapping) and part.get("type") == "image":
                        input_images.append(part["image"])
        inputs_kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
        if input_images:
            inputs_kwargs["images"] = input_images
        if has_video:
            inputs_kwargs.update(self._video_inputs(messages_batch))
        inputs = self.processor(**inputs_kwargs)
        inputs = inputs.to(self.model.device)
        generation_config = getattr(self, "_generation_config", None)
        extra: dict[str, Any] = (
            {"generation_config": generation_config} if generation_config is not None
            else {"max_new_tokens": self.max_new_tokens}
        )
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **extra)
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
        videos = [_video_list(row.get("video_files")) for row in batch]
        if any(videos):
            # 视频行不能退回纯文本推理：那样模型没看视频也会答，分数照样出，错得看不出来。
            if not hasattr(generator, "generate_video_batch"):
                raise PredictionBatchError(f"{type(generator).__name__} 不支持视频输入")
            if not all(videos):
                raise PredictionBatchError("同一批里混了有视频和没视频的行")
            batch_predictions = list(generator.generate_video_batch(prompts, videos))  # type: ignore[attr-defined]
        elif hasattr(generator, "generate_batch"):
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
    """Return (missing_count, sample_indices) for rows with neither an image nor a video."""
    missing_indices = []
    for row in rows:
        if _image_cell_or_none(row.get("image")) is None and not _video_list(row.get("video_files")):
            missing_indices.append(str(row.get("index", "")))
    return len(missing_indices), missing_indices[:10]


def _video_list(value: Any) -> list[str]:
    """``video_files`` 列（JSON 数组）→ 路径列表；空 / NaN → []。"""
    if value is None or (isinstance(value, float) and value != value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    items = json.loads(text) if text.startswith("[") else [text]
    return [str(item) for item in items if str(item).strip()]


def _image_cell_or_none(value: Any) -> str | None:
    """Normalize a TSV image cell: None / pandas NaN / blank all mean "no image"."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _decode_images(
    image_cell: str | list[str] | None,
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> list[Any]:
    if not image_cell:
        return []
    from PIL import Image

    cells = image_cell if isinstance(image_cell, list) else decode_image_cell(image_cell)
    images = []
    try:
        for raw in cells:
            raw = str(raw)
            if "," in raw and raw.strip().startswith("data:"):
                raw = raw.split(",", 1)[1]
            data = base64.b64decode(raw)
            with Image.open(io.BytesIO(data)) as source:
                images.append(
                    prepare_llamafactory_qwen_image(
                        source,
                        image_min_pixels=image_min_pixels,
                        image_max_pixels=image_max_pixels,
                    )
                )
        return images
    except Exception:
        for image in reversed(images):
            image.close()
        raise


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
