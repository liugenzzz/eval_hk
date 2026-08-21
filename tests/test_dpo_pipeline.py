import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from eval_tool.dpo_config import DpoBuildConfig, DpoInferConfig, DpoJudgeConfig
from eval_tool.dpo_pipeline import DpoPipelineError, run_build_dpo
from eval_tool.judge import JudgeSettings


# ----------------------------------------------------------------------------
# fixtures and fakes
# ----------------------------------------------------------------------------


class FakeGenerator:
    """Return one answer per structured message list without touching torch."""

    instances: list["FakeGenerator"] = []

    def __init__(self, respond, config=None):
        self.respond = respond
        self.config = config
        self.batches: list[list[list[dict]]] = []
        self.enable_thinking_values: list[object] = []

    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        self.batches.append(messages_batch)
        self.enable_thinking_values.append(enable_thinking)
        return [self.respond(last_human_text(messages)) for messages in messages_batch]


class FakeJudgeClient:
    def __init__(self, respond, settings=None):
        self.respond = respond
        self.settings = settings
        self.requests: list[list[dict]] = []

    def judge_messages(self, messages, *, max_tokens=1024):
        self.requests.append(messages)
        return self.respond(messages)


def last_human_text(messages) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return "".join(
                part["text"] for part in message["content"] if part["type"] == "text"
            )
    raise AssertionError("structured messages contain no user turn")


def judge_question_text(messages) -> str:
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            return content
        return "".join(
            part["text"] for part in content if part.get("type") == "text"
        )
    raise AssertionError("judge messages contain no user turn")


def generator_factory(respond, created=None):
    def factory(config):
        generator = FakeGenerator(respond, config)
        if created is not None:
            created.append(generator)
        return generator

    return factory


def judge_factory(respond, created=None):
    def factory(settings):
        client = FakeJudgeClient(respond, settings)
        if created is not None:
            created.append(client)
        return client

    return factory


def exploding_generator_factory(config):  # pragma: no cover - must never run
    raise AssertionError("the generator must not be constructed")


def exploding_judge_factory(settings):  # pragma: no cover - must never run
    raise AssertionError("the Judge client must not be constructed")


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def write_jsonl(path: Path, values) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(value, ensure_ascii=False)}\n" for value in values),
        encoding="utf-8",
    )
    return path


def write_png(path: Path, color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path, format="PNG")
    return path


def write_jpeg(path: Path, color=(20, 160, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path, format="JPEG")
    return path


def make_model_dir(tmp_path: Path, name: str = "model", marker: str = "fake") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": marker}), encoding="utf-8"
    )
    return model_dir


def judge_config(**overrides) -> DpoJudgeConfig:
    settings = JudgeSettings(
        api_base="http://judge.invalid/v1/chat/completions",
        api_key="sk-secret-value",
        model=overrides.pop("model", "judge-model"),
        temperature=overrides.pop("temperature", 0.0),
        timeout=overrides.pop("timeout", 5),
        max_retries=overrides.pop("max_retries", 0),
    )
    return DpoJudgeConfig(settings=settings, max_workers=overrides.pop("max_workers", 2))


def make_config(
    tmp_path: Path,
    inputs,
    *,
    wrong_only: bool = False,
    rubric=None,
    judge=None,
    output_name: str = "train_dpo.jsonl",
    image_root: Path | None = None,
    model_dir: Path | None = None,
    infer_overrides=None,
) -> DpoBuildConfig:
    infer_kwargs = {
        "model_name": "fake-vl",
        "model_path": model_dir or make_model_dir(tmp_path),
        "batch_size": 1,
    }
    infer_kwargs.update(infer_overrides or {})
    return DpoBuildConfig(
        config_path=tmp_path / "dpo.json",
        inputs=tuple(Path(item) for item in inputs),
        output_dir=tmp_path / "out",
        output_name=output_name,
        work_dir=tmp_path / "work",
        image_root=image_root or tmp_path,
        wrong_only=wrong_only,
        rubric=rubric,
        infer=DpoInferConfig(**infer_kwargs),
        judge=judge,
    )


def read_jsonl(path: Path) -> list:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def artifacts_of(result) -> dict[str, Path]:
    return {name: Path(path) for name, path in result.artifacts.items()}


def binary_wrong(_messages) -> str:
    return json.dumps({"correct": False, "reason": "wrong"})


def binary_right(_messages) -> str:
    return json.dumps({"correct": True, "reason": "right"})


V4_WRONG = {
    "equipment_correct": False,
    "fact_score": 0,
    "param_score": 0,
    "visual_score": None,
    "fabrication_score": 0,
    "style_score": 0,
    "reasoning": "wrong",
}
V4_RIGHT = {
    "equipment_correct": True,
    "fact_score": 100,
    "param_score": 100,
    "visual_score": None,
    "fabrication_score": 100,
    "style_score": 100,
    "reasoning": "right",
}


# ----------------------------------------------------------------------------
# full mode
# ----------------------------------------------------------------------------


def test_full_mode_never_constructs_or_calls_judge(tmp_path, capsys):
    source = write_json(
        tmp_path / "alpaca.json",
        [{"instruction": "问题一", "output": "参考一"}],
    )
    config = make_config(tmp_path, [source])

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"模型回答:{text}"),
        judge_client_factory=exploding_judge_factory,
    )

    assert result.selected_count == 1
    assert result.dry_run is False
    assert result.failed_run_dir is None
    printed = capsys.readouterr().out
    assert "judge" in printed.casefold()
    assert "读取并校验输入" in printed
    assert "加载推理缓存" in printed
    assert "写入训练文件" in printed
    assert "校验暂存产物" in printed
    assert "发布最终产物" in printed
    rows = read_jsonl(Path(result.output_path))
    assert rows[0]["rejected"] == {
        "from": "gpt",
        "value": "模型回答:问题一",
    }
    assert rows[0]["chosen"] == {"from": "gpt", "value": "参考一"}


def test_fake_mixed_input_pipeline_publishes_reconciled_strict_artifacts(tmp_path):
    picture = write_png(tmp_path / "images" / "a.png")
    photo = write_jpeg(tmp_path / "images" / "b.jpg")
    alpaca = write_json(
        tmp_path / "alpaca.json",
        [
            {
                "instruction": "描述装备",
                "input": "第一段背景",
                "output": "参考答案一",
                "history": [["历史问题", "历史答案"]],
            },
            {"instruction": "纯文本问题", "output": "参考答案二"},
        ],
    )
    sharegpt = write_jsonl(
        tmp_path / "sharegpt.jsonl",
        [
            {
                "conversations": [
                    {"from": "human", "value": "<image>这是什么?"},
                    {"from": "gpt", "value": "参考答案三"},
                ],
                "images": ["images/a.png"],
            },
            {
                "conversations": [
                    {"from": "user", "value": "<image><image>比较这两张图"},
                    {"from": "assistant", "value": "参考答案四"},
                ],
                "images": ["images/a.png", "images/b.jpg"],
            },
        ],
    )
    assert picture.is_file() and photo.is_file()
    config = make_config(tmp_path, [alpaca, sharegpt])

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text[:6]}"),
    )

    artifacts = artifacts_of(result)
    assert set(artifacts) == {
        "train_dpo.jsonl",
        "audit_records.jsonl",
        "rejected_records.jsonl",
        "summary.json",
        "warnings.log",
        "manifest.json",
    }

    rows = read_jsonl(artifacts["train_dpo.jsonl"])
    assert len(rows) == result.selected_count == 5
    for row in rows:
        assert set(row) == {"conversations", "chosen", "rejected", "images"}

    image_rows = [row for row in rows if row["images"]]
    assert [row["images"] for row in image_rows] == [
        ["images/a.png"],
        ["images/a.png", "images/b.jpg"],
    ]

    manifest = json.loads(artifacts["manifest.json"].read_text(encoding="utf-8"))
    for name, stat in manifest["artifacts"].items():
        raw = artifacts[name].read_bytes()
        assert stat["byte_count"] == len(raw)
        if name.endswith(".jsonl"):
            assert stat["line_count"] == len(read_jsonl(artifacts[name]))

    audits = read_jsonl(artifacts["audit_records.jsonl"])
    rejected = read_jsonl(artifacts["rejected_records.jsonl"])
    summary = json.loads(artifacts["summary.json"].read_text(encoding="utf-8"))
    assert summary["selected"] == len(rows)
    assert summary["rejected"] == len(rejected)
    assert summary["total"] == len(audits)
    assert rejected == [item for item in audits if item["disposition"] == "rejected"]
    assert summary["by_modality"]["multi_image"] == 1
    assert summary["by_turn_shape"]["multi_turn"] == 2


def test_short_valid_answers_are_not_length_filtered(tmp_path):
    source = write_json(tmp_path / "alpaca.json", [{"instruction": "一?", "output": "是"}])
    config = make_config(tmp_path, [source])

    result = run_build_dpo(
        config, generator_factory=generator_factory(lambda text: "否")
    )

    rows = read_jsonl(Path(result.output_path))
    assert rows == [
        {
            "conversations": [{"from": "human", "value": "一?"}],
            "chosen": {"from": "gpt", "value": "是"},
            "rejected": {"from": "gpt", "value": "否"},
            "images": [],
        }
    ]


def test_output_follows_input_turn_order_not_completion_order(tmp_path):
    source = write_json(
        tmp_path / "alpaca.json",
        [
            {
                "instruction": "第三问",
                "output": "第三答",
                "history": [["第一问", "第一答"], ["第二问", "第二答"]],
            }
        ],
    )
    config = make_config(
        tmp_path, [source], infer_overrides={"batch_size": 3}
    )

    def respond(text: str) -> str:
        return {"第一问": "回答A", "第二问": "回答B", "第三问": "回答C"}[
            text.splitlines()[-1]
        ]

    result = run_build_dpo(config, generator_factory=generator_factory(respond))

    rows = read_jsonl(Path(result.output_path))
    assert [row["chosen"]["value"] for row in rows] == [
        "第一答",
        "第二答",
        "第三答",
    ]
    assert [row["rejected"]["value"] for row in rows] == [
        "回答A",
        "回答B",
        "回答C",
    ]
    assert all(row["chosen"]["from"] == "gpt" for row in rows)
    assert all(row["rejected"]["from"] == "gpt" for row in rows)


# ----------------------------------------------------------------------------
# pre-judge filtering
# ----------------------------------------------------------------------------


def test_empty_rejected_identical_pair_and_inference_error_are_filtered(tmp_path):
    source = write_json(
        tmp_path / "alpaca.json",
        [
            {"instruction": "空回答", "output": "参考"},
            {"instruction": "同答", "output": "一样的答案"},
            {"instruction": "爆炸", "output": "参考"},
            {"instruction": "正常", "output": "参考"},
        ],
    )
    config = make_config(tmp_path, [source])

    def respond(text: str) -> str:
        if text == "空回答":
            return "   "
        if text == "同答":
            return "  一样的答案 "
        if text == "爆炸":
            raise ValueError("bad sample")
        return "不同的答案"

    result = run_build_dpo(config, generator_factory=generator_factory(respond))

    audits = read_jsonl(artifacts_of(result)["audit_records.jsonl"])
    by_reason = {
        item["reason_code"] for item in audits if item["disposition"] == "rejected"
    }
    assert by_reason == {"empty_rejected", "identical_pair", "inference_error"}
    assert result.selected_count == 1
    errored = [item for item in audits if item["reason_code"] == "inference_error"]
    assert "bad sample" in errored[0]["inference_error"]


# ----------------------------------------------------------------------------
# wrong-only judging
# ----------------------------------------------------------------------------


def test_wrong_only_binary_keeps_all_and_only_correct_false(tmp_path):
    source = write_json(
        tmp_path / "alpaca.json",
        [
            {"instruction": "错的", "output": "参考一"},
            {"instruction": "对的", "output": "参考二"},
        ],
    )
    config = make_config(
        tmp_path, [source], wrong_only=True, rubric="binary", judge=judge_config()
    )

    def respond(messages) -> str:
        text = judge_question_text(messages)
        return binary_wrong(messages) if "错的" in text else binary_right(messages)

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
        judge_client_factory=judge_factory(respond),
    )

    rows = read_jsonl(Path(result.output_path))
    assert [row["chosen"]["value"] for row in rows] == ["参考一"]
    audits = read_jsonl(artifacts_of(result)["audit_records.jsonl"])
    passed = [item for item in audits if item["reason_code"] == "judge_pass"]
    assert len(passed) == 1
    assert passed[0]["judge"]["hit"] == 1


def test_wrong_only_v4_keeps_all_and_only_hit_zero(tmp_path):
    source = write_json(
        tmp_path / "alpaca.json",
        [
            {"instruction": "错的", "output": "参考一"},
            {"instruction": "对的", "output": "参考二"},
        ],
    )
    config = make_config(
        tmp_path, [source], wrong_only=True, rubric="v4", judge=judge_config()
    )

    def respond(messages) -> str:
        text = judge_question_text(messages)
        payload = V4_WRONG if "错的" in text else V4_RIGHT
        return json.dumps(payload, ensure_ascii=False)

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
        judge_client_factory=judge_factory(respond),
    )

    rows = read_jsonl(Path(result.output_path))
    assert [row["chosen"]["value"] for row in rows] == ["参考一"]
    audits = read_jsonl(artifacts_of(result)["audit_records.jsonl"])
    selected = [item for item in audits if item["disposition"] == "selected"]
    assert selected[0]["judge"]["hit"] == 0


def test_judge_error_and_judge_pass_have_distinct_dispositions(tmp_path):
    source = write_json(
        tmp_path / "alpaca.json",
        [
            {"instruction": "坏响应", "output": "参考一"},
            {"instruction": "对的", "output": "参考二"},
            {"instruction": "错的", "output": "参考三"},
        ],
    )
    config = make_config(
        tmp_path, [source], wrong_only=True, rubric="binary", judge=judge_config()
    )

    def respond(messages) -> str:
        text = judge_question_text(messages)
        if "坏响应" in text:
            return "not json at all"
        return binary_right(messages) if "对的" in text else binary_wrong(messages)

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
        judge_client_factory=judge_factory(respond),
    )

    audits = read_jsonl(artifacts_of(result)["audit_records.jsonl"])
    reasons = {
        item["source"]["record_index"]: item["reason_code"] for item in audits
    }
    assert reasons == {0: "judge_error", 1: "judge_pass", 2: None}
    assert result.selected_count == 1


def test_manifest_and_warnings_never_expose_the_judge_api_key(tmp_path):
    source = write_json(tmp_path / "alpaca.json", [{"instruction": "问", "output": "参考"}])
    config = make_config(
        tmp_path, [source], wrong_only=True, rubric="binary", judge=judge_config()
    )

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: "生成"),
        judge_client_factory=judge_factory(binary_wrong),
    )

    artifacts = artifacts_of(result)
    manifest_text = artifacts["manifest.json"].read_text(encoding="utf-8")
    assert "sk-secret-value" not in manifest_text
    assert "<redacted>" in manifest_text
    assert "sk-secret-value" not in artifacts["warnings.log"].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("infer_overrides", "expected"),
    [
        pytest.param(
            {},
            {
                "image_min_pixels": None,
                "image_max_pixels": None,
                "image_preprocess_profile": None,
            },
            id="disabled",
        ),
        pytest.param(
            {"image_min_pixels": 65_536, "image_max_pixels": 589_824},
            {
                "image_min_pixels": 65_536,
                "image_max_pixels": 589_824,
                "image_preprocess_profile": "llamafactory-0.9.5-qwen-static-v1",
            },
            id="enabled",
        ),
    ],
)
def test_manifest_records_image_preprocess_identity(
    tmp_path, infer_overrides, expected
):
    source = write_json(
        tmp_path / "alpaca.json",
        [{"instruction": "question", "output": "reference"}],
    )
    config = make_config(tmp_path, [source], infer_overrides=infer_overrides)

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: "generated"),
    )

    manifest = json.loads(
        artifacts_of(result)["manifest.json"].read_text(encoding="utf-8")
    )
    assert {
        key: manifest["model"][key]
        for key in (
            "image_min_pixels",
            "image_max_pixels",
            "image_preprocess_profile",
        )
    } == expected


# ----------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------


def test_dry_run_constructs_neither_generator_nor_judge(tmp_path):
    source = write_json(tmp_path / "alpaca.json", [{"instruction": "问", "output": "参考"}])
    config = make_config(
        tmp_path, [source], wrong_only=True, rubric="binary", judge=judge_config()
    )

    result = run_build_dpo(
        config,
        dry_run=True,
        generator_factory=exploding_generator_factory,
        judge_client_factory=exploding_judge_factory,
    )

    assert result.dry_run is True
    assert result.output_path is None
    assert result.artifacts == {}
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.jsonl"))


def test_dry_run_reports_normalization_dedupe_conflicts_and_images(tmp_path, capsys):
    write_png(tmp_path / "images" / "a.png")
    source = write_json(
        tmp_path / "mixed.json",
        [
            {"instruction": "重复", "output": "参考"},
            {"instruction": "重复", "output": "参考"},
            {"instruction": "冲突", "output": "答案甲"},
            {"instruction": "冲突", "output": "答案乙"},
            {"instruction": "缺图<image>", "output": "参考", "images": ["images/missing.png"]},
            {
                "conversations": [
                    {"from": "human", "value": "<image>看图"},
                    {"from": "gpt", "value": "参考"},
                ],
                "images": ["images/a.png"],
            },
            "不是对象",
        ],
    )
    config = make_config(tmp_path, [source])

    result = run_build_dpo(
        config,
        dry_run=True,
        generator_factory=exploding_generator_factory,
    )

    summary = dict(result.summary)
    assert summary["dry_run"] is True
    assert summary["inputs"][0]["record_count"] == 7
    assert summary["inputs"][0]["container_format"] == "json_array"
    reasons = summary["by_reason_code"]
    assert reasons["duplicate"] == 1
    assert reasons["reference_conflict"] == 2
    assert reasons["invalid_schema"] == 1
    assert reasons["missing_image"] == 1
    assert summary["pending_candidates"] == 2
    assert summary["images"]["unique_readable"] == 1

    printed = capsys.readouterr().out
    assert "dry-run" in printed
    assert "reference_conflict" in printed
    assert str(source) in printed


# ----------------------------------------------------------------------------
# startup failures
# ----------------------------------------------------------------------------


def test_unreadable_input_or_uncreatable_output_work_dir_fails_before_generator(
    tmp_path,
):
    config = make_config(tmp_path, [tmp_path / "missing.json"])

    with pytest.raises(DpoPipelineError):
        run_build_dpo(config, generator_factory=exploding_generator_factory)

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    source = write_json(tmp_path / "alpaca.json", [{"instruction": "问", "output": "参考"}])
    blocked = replace(make_config(tmp_path, [source]), output_dir=blocker / "out")

    with pytest.raises(DpoPipelineError):
        run_build_dpo(blocked, generator_factory=exploding_generator_factory)


def test_existing_output_without_manifest_fails_before_gpu_work(tmp_path):
    source = write_json(tmp_path / "alpaca.json", [{"instruction": "问", "output": "参考"}])
    config = make_config(tmp_path, [source])
    (tmp_path / "out").mkdir(parents=True)
    (tmp_path / "out" / "train_dpo.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DpoPipelineError, match="manifest"):
        run_build_dpo(config, generator_factory=exploding_generator_factory)


def test_zero_valid_raises_with_failed_run_path(tmp_path):
    source = write_json(tmp_path / "alpaca.json", ["不是对象", 42])
    config = make_config(tmp_path, [source])

    with pytest.raises(DpoPipelineError) as exc_info:
        run_build_dpo(config, generator_factory=exploding_generator_factory)

    message = str(exc_info.value)
    failed_root = tmp_path / "work" / "failed_runs"
    assert failed_root.is_dir()
    failed_dir = next(failed_root.iterdir())
    assert str(failed_dir) in message
    assert (failed_dir / "audit_records.jsonl").is_file()
    assert (failed_dir / "summary.json").is_file()
    assert (failed_dir / "error.json").is_file()
    assert not (tmp_path / "out" / "train_dpo.jsonl").exists()
