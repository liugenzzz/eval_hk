import base64
import builtins
import io
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eval_tool.config import InferConfig
from eval_tool.infer import QwenVLGenerator, generate_predictions
from eval_tool.imaging import IMAGE_PREPROCESS_PROFILE
from eval_tool.run_infer import run


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, image_b64=None):
        self.calls.append((prompt, image_b64))
        return "答案是 A" if "选项" in prompt else "开放回答"


class FakeBatchGenerator:
    def __init__(self):
        self.batch_calls = []

    def generate_batch(self, prompts, image_b64s=None):
        image_b64s = image_b64s or [None] * len(prompts)
        self.batch_calls.append((list(prompts), list(image_b64s)))
        return [f"pred-{i}" for i, _ in enumerate(prompts)]


class ShortBatchGenerator:
    def generate_batch(self, prompts, image_b64s=None):
        return ["only-one"]


class MutatingPromptGenerator:
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path
        self.prompts = []

    def generate(self, prompt, image_b64=None):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            self.prompt_path.write_text("changed {question}", encoding="utf-8")
        return "answer"


class FakeModelInputs(dict):
    def __init__(self, batch_size):
        super().__init__(input_ids=np.zeros((batch_size, 2), dtype=np.int64))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeQwenProcessor:
    def __init__(self, decoded=None, error=None):
        self.decoded = decoded or []
        self.error = error
        self.template_calls = []
        self.processor_calls = []
        self.image_pixels = []
        self.image_sizes = []
        self.image_modes = []
        self.decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, dict(kwargs)))
        return f"rendered-{len(self.template_calls)}"

    def __call__(self, **kwargs):
        self.processor_calls.append(dict(kwargs))
        images = kwargs.get("images", [])
        self.image_pixels.append(
            [image.getpixel((0, 0)) for image in images]
        )
        self.image_sizes.append([image.size for image in images])
        self.image_modes.append([image.mode for image in images])
        if self.error is not None:
            raise self.error
        return FakeModelInputs(len(kwargs["text"]))

    def batch_decode(self, generated, **kwargs):
        self.decode_calls.append((generated, dict(kwargs)))
        return list(self.decoded)


class FakeQwenModel:
    device = "fake-device"

    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        batch_size = kwargs["input_ids"].shape[0]
        return np.arange(batch_size * 5).reshape(batch_size, 5)


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class ImmediateFuture:
    def __init__(self, function, args):
        try:
            self.value = function(*args)
            self.error = None
        except Exception as exc:
            self.value = None
            self.error = exc

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class ImmediateExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def submit(self, function, *args):
        return ImmediateFuture(function, args)


def _qwen_generator(*, decoded=None, processor_error=None):
    generator = object.__new__(QwenVLGenerator)
    generator.model_path = Path("unused")
    generator.max_new_tokens = 17
    generator.torch_dtype = "auto"
    generator.device_map = "auto"
    generator.image_min_pixels = None
    generator.image_max_pixels = None
    generator.processor = FakeQwenProcessor(decoded=decoded, error=processor_error)
    generator.model = FakeQwenModel()
    generator._torch = FakeTorch()
    return generator


def _png_base64(color, *, size=(1, 1), mode="RGB"):
    buffer = io.BytesIO()
    with Image.new(mode, size, color) as image:
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _mock_qwen_dependencies(
    monkeypatch, checkpoint_size=None, *, image_processor=None
):
    processor_calls = []
    model_calls = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            processor_calls.append((args, dict(kwargs)))
            return SimpleNamespace(
                image_processor=(
                    image_processor
                    if image_processor is not None
                    else SimpleNamespace(size=checkpoint_size)
                )
            )

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model_calls.append((args, dict(kwargs)))
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeAutoProcessor,
            AutoModelForImageTextToText=FakeModelLoader,
        ),
    )
    return processor_calls, model_calls


def test_qwen_message_batch_flattens_images_in_sample_message_content_order():
    generator = _qwen_generator(decoded=[" first ", "second\n"])
    images = [
        Image.new("RGB", (1, 1), (255, 0, 0)),
        Image.new("RGB", (1, 1), (0, 255, 0)),
        Image.new("RGB", (1, 1), (0, 0, 255)),
    ]
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image", "image": images[0]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "history"}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": images[1]}],
            },
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
    ]

    try:
        result = generator.generate_message_batch(messages_batch)

        assert result == ["first", "second"]
        assert [call[0] for call in generator.processor.template_calls] == messages_batch
        assert all(
            call[1] == {"tokenize": False, "add_generation_prompt": True}
            for call in generator.processor.template_calls
        )
        processor_call = generator.processor.processor_calls[0]
        assert processor_call["text"] == ["rendered-1", "rendered-2"]
        assert processor_call["images"] == images
        assert processor_call["padding"] is True
        assert processor_call["return_tensors"] == "pt"
        assert generator.processor.image_pixels == [
            [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        ]
        assert generator.model.generate_calls[0]["max_new_tokens"] == 17
        assert generator.processor.decode_calls[0][0].shape == (2, 3)
        assert generator.processor.decode_calls[0][1] == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
    finally:
        for image in images:
            image.close()


def test_qwen_message_batch_omits_images_when_batch_has_none():
    generator = _qwen_generator(decoded=[" answer "])
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "text only"}],
            }
        ]
    ]

    assert generator.generate_message_batch(messages) == ["answer"]
    assert "images" not in generator.processor.processor_calls[0]


@pytest.mark.parametrize("enable_thinking", [False, True, None])
def test_qwen_message_batch_passes_enable_thinking_only_when_explicit(
    enable_thinking,
):
    generator = _qwen_generator(decoded=["answer"])
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "question"}],
            }
        ]
    ]

    generator.generate_message_batch(messages, enable_thinking=enable_thinking)

    template_kwargs = generator.processor.template_calls[0][1]
    if enable_thinking is None:
        assert "enable_thinking" not in template_kwargs
    else:
        assert template_kwargs["enable_thinking"] is enable_thinking


def test_legacy_generate_batch_preserves_image_order_omits_thinking_and_closes_images():
    generator = _qwen_generator(decoded=[" one ", " two "])
    red = _png_base64((255, 0, 0))
    green = _png_base64((0, 255, 0))
    blue = _png_base64((0, 0, 255))

    result = generator.generate_batch(["one", "two"], [[red, green], blue])

    assert result == ["one", "two"]
    assert generator.processor.image_pixels == [
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    ]
    assert generator.processor.image_sizes == [[(1, 1), (1, 1), (1, 1)]]
    assert generator.processor.image_modes == [["RGB", "RGB", "RGB"]]
    assert all(
        "enable_thinking" not in kwargs
        for _, kwargs in generator.processor.template_calls
    )
    owned_images = generator.processor.processor_calls[0]["images"]
    for image in owned_images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_legacy_generate_batch_applies_configured_llamafactory_image_bounds():
    generator = _qwen_generator(decoded=[" answer "])
    generator.image_min_pixels = 65_536
    generator.image_max_pixels = 589_824
    payload = _png_base64(
        (11, 22, 33, 128), size=(1000, 1000), mode="RGBA"
    )

    assert generator.generate_batch(["question"], [payload]) == ["answer"]

    assert generator.processor.image_sizes == [[(768, 768)]]
    assert generator.processor.image_modes == [["RGB"]]
    owned_image = generator.processor.processor_calls[0]["images"][0]
    with pytest.raises(ValueError):
        owned_image.getpixel((0, 0))


def test_legacy_generate_batch_closes_owned_images_when_processor_raises():
    generator = _qwen_generator(
        decoded=["unused"], processor_error=RuntimeError("processor failed")
    )
    generator.image_min_pixels = 65_536
    generator.image_max_pixels = 589_824
    payload = _png_base64(
        (11, 22, 33, 128), size=(1000, 1000), mode="RGBA"
    )

    with pytest.raises(RuntimeError, match="processor failed"):
        generator.generate_batch(["question"], [payload])

    owned_images = generator.processor.processor_calls[0]["images"]
    assert len(owned_images) == 1
    assert generator.processor.image_sizes == [[(768, 768)]]
    assert generator.processor.image_modes == [["RGB"]]
    with pytest.raises(ValueError):
        owned_images[0].getpixel((0, 0))


def test_qwen_generator_validates_pixel_pair_before_model_import(
    tmp_path, monkeypatch
):
    real_import = builtins.__import__
    attempted_model_imports = []

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            attempted_model_imports.append(name)
            raise AssertionError(f"unexpected model import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="provided together"):
        QwenVLGenerator(
            model_path=tmp_path / "model",
            image_min_pixels=65_536,
        )

    assert attempted_model_imports == []


def test_qwen_generator_keeps_processor_load_kwargs_and_logs_active_profile(
    tmp_path, monkeypatch, capsys
):
    checkpoint_size = {"shortest_edge": 56, "longest_edge": 1_003_520}
    processor_calls, model_calls = _mock_qwen_dependencies(
        monkeypatch, checkpoint_size
    )

    QwenVLGenerator(
        model_path=tmp_path / "model",
        image_min_pixels=65_536,
        image_max_pixels=589_824,
    )

    assert processor_calls == [
        ((str(tmp_path / "model"),), {"trust_remote_code": True})
    ]
    assert len(model_calls) == 1
    output = capsys.readouterr().out
    assert IMAGE_PREPROCESS_PROFILE in output
    assert "min_pixels=65536" in output
    assert "max_pixels=589824" in output
    assert f"checkpoint_size={checkpoint_size}" in output


def test_qwen_generator_checkpoint_size_logging_is_best_effort(
    tmp_path, monkeypatch, capsys
):
    class ThrowingImageProcessor:
        @property
        def size(self):
            raise RuntimeError("size unavailable")

    _, model_calls = _mock_qwen_dependencies(
        monkeypatch,
        image_processor=ThrowingImageProcessor(),
    )

    QwenVLGenerator(
        model_path=tmp_path / "model",
        image_min_pixels=65_536,
        image_max_pixels=589_824,
    )

    assert len(model_calls) == 1
    assert "checkpoint_size=<unavailable>" in capsys.readouterr().out


def test_qwen_generator_profile_logging_failure_does_not_block_model_load(
    tmp_path, monkeypatch
):
    _, model_calls = _mock_qwen_dependencies(
        monkeypatch,
        {"shortest_edge": 56, "longest_edge": 1_003_520},
    )

    def failing_print(*args, **kwargs):
        raise BrokenPipeError("stdout is closed")

    monkeypatch.setattr(builtins, "print", failing_print)

    QwenVLGenerator(
        model_path=tmp_path / "model",
        image_min_pixels=65_536,
        image_max_pixels=589_824,
    )

    assert len(model_calls) == 1


def test_qwen_generator_without_pixel_profile_does_not_log_image_profile(
    tmp_path, monkeypatch, capsys
):
    processor_calls, _ = _mock_qwen_dependencies(monkeypatch, {"edge": 28})

    QwenVLGenerator(model_path=tmp_path / "model")

    assert processor_calls == [
        ((str(tmp_path / "model"),), {"trust_remote_code": True})
    ]
    assert capsys.readouterr().out == ""


def test_generate_predictions_rejects_short_batch():
    rows = [
        {"index": "1", "question": "q1"},
        {"index": "2", "question": "q2"},
    ]

    with pytest.raises(RuntimeError, match="returned 1 predictions for 2 rows"):
        generate_predictions(rows, "vqa", {}, ShortBatchGenerator(), batch_size=2)


def test_generate_predictions_snapshots_prompt_before_first_batch(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("original {question}", encoding="utf-8")
    generator = MutatingPromptGenerator(prompt_path)

    generate_predictions(
        [{"index": "1", "question": "q1"}, {"index": "2", "question": "q2"}],
        "vqa",
        {"vqa": prompt_path},
        generator,
        batch_size=1,
    )

    assert generator.prompts == ["original q1", "original q2"]


def test_generate_predictions_reports_each_completed_batch():
    completed = []
    rows = [
        {"index": "1", "question": "q1"},
        {"index": "2", "question": "q2"},
        {"index": "3", "question": "q3"},
    ]

    predictions = generate_predictions(
        rows,
        "vqa",
        {},
        FakeBatchGenerator(),
        batch_size=2,
        on_batch=lambda batch_rows, batch_predictions: completed.append(
            ([row["index"] for row in batch_rows], list(batch_predictions))
        ),
    )

    assert predictions == ["pred-0", "pred-1", "pred-0"]
    assert completed == [
        (["1", "2"], ["pred-0", "pred-1"]),
        (["3"], ["pred-0"]),
    ]


def test_run_infer_generates_reusable_xlsx_without_image_column(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    prompt_path = tmp_path / "infer_mcq.txt"
    prompt_path.write_text("题目：{question}\n选项A：{A}\n选项B：{B}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "index": "1",
                "image": "base64-image",
                "question": "选哪个？",
                "A": "燃油泵",
                "B": "滑油泵",
                "answer": "A",
                "category": "P1",
                "l2-category": "零件图",
                "source_id": "s1",
            }
        ]
    ).to_csv(tsv_dir / "aero_mcq.tsv", sep="\t", index=False)

    generator = FakeGenerator()
    written = run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"mcq": "aero_mcq"},
            prompt_files={"mcq": prompt_path},
        ),
        generator=generator,
    )

    path = written["mcq"]
    assert path == out_dir / "base_aero_mcq.xlsx"
    output = pd.read_excel(path, dtype={"index": str})
    assert "image" not in output.columns
    assert output.loc[0, "prediction"] == "答案是 A"
    assert generator.calls == [("题目：选哪个？\n选项A：燃油泵\n选项B：滑油泵", "base64-image")]


def test_run_infer_uses_batch_generation_when_available(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    prompt_path = tmp_path / "infer_vqa.txt"
    prompt_path.write_text("题目：{question}", encoding="utf-8")
    pd.DataFrame(
        [
            {"index": "1", "image": "img1", "question": "q1", "answer": "a1"},
            {"index": "2", "image": "img2", "question": "q2", "answer": "a2"},
            {"index": "3", "image": "img3", "question": "q3", "answer": "a3"},
        ]
    ).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)

    generator = FakeBatchGenerator()
    run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"vqa": "aero_vqa"},
            prompt_files={"vqa": prompt_path},
            batch_size=2,
        ),
        generator=generator,
    )

    assert generator.batch_calls == [
        (["题目：q1", "题目：q2"], ["img1", "img2"]),
        (["题目：q3"], ["img3"]),
    ]
    output = pd.read_excel(out_dir / "base_aero_vqa.xlsx", dtype={"index": str})
    assert output["prediction"].tolist() == ["pred-0", "pred-1", "pred-0"]


def test_run_infer_passes_image_bounds_to_sequential_qwen_generator(
    tmp_path, monkeypatch
):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    pd.DataFrame(
        [{"index": "1", "image": "", "question": "q1", "answer": "a1"}]
    ).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)
    constructed = []

    def construct_generator(**kwargs):
        constructed.append(kwargs)
        return FakeBatchGenerator()

    monkeypatch.setattr("eval_tool.run_infer.QwenVLGenerator", construct_generator)

    run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"vqa": "aero_vqa"},
            prompt_files={},
            image_min_pixels=65_536,
            image_max_pixels=589_824,
        )
    )

    assert constructed == [
        {
            "model_path": tmp_path / "model",
            "max_new_tokens": 512,
            "torch_dtype": "auto",
            "device_map": "auto",
            "image_min_pixels": 65_536,
            "image_max_pixels": 589_824,
        }
    ]


def test_run_infer_passes_image_bounds_through_parallel_worker(
    tmp_path, monkeypatch
):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    pd.DataFrame(
        [{"index": "1", "image": "", "question": "q1", "answer": "a1"}]
    ).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)
    constructed = []

    def construct_generator(**kwargs):
        constructed.append(kwargs)
        return FakeBatchGenerator()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before-test")
    monkeypatch.setattr("eval_tool.run_infer.QwenVLGenerator", construct_generator)
    monkeypatch.setattr("eval_tool.run_infer.ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("eval_tool.run_infer.as_completed", lambda futures: futures)
    monkeypatch.setattr(
        "eval_tool.run_infer._progress", lambda iterable, **kwargs: iterable
    )

    run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"vqa": "aero_vqa"},
            prompt_files={},
            gpu_ids=[3],
            image_min_pixels=65_536,
            image_max_pixels=589_824,
        )
    )

    assert constructed == [
        {
            "model_path": tmp_path / "model",
            "max_new_tokens": 512,
            "torch_dtype": "auto",
            "device_map": "auto",
            "image_min_pixels": 65_536,
            "image_max_pixels": 589_824,
        }
    ]


def test_run_infer_uses_parallel_branch_without_creating_main_generator(tmp_path, monkeypatch):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    pd.DataFrame([{"index": "1", "image": "img1", "question": "q1"}]).to_csv(
        tsv_dir / "aero_vqa.tsv", sep="\t", index=False
    )

    def fake_parallel(config, dataset_key, rows):
        return ["parallel-pred"]

    monkeypatch.setattr("eval_tool.run_infer._run_dataset_parallel", fake_parallel)

    run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"vqa": "aero_vqa"},
            prompt_files={},
            gpu_ids=[0],
        )
    )

    output = pd.read_excel(out_dir / "base_aero_vqa.xlsx", dtype={"index": str})
    assert output.loc[0, "prediction"] == "parallel-pred"
