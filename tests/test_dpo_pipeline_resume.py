import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval_tool.dpo_config import DpoBuildConfig, DpoInferConfig, DpoJudgeConfig
from eval_tool.dpo_pipeline import DpoPipelineError, run_build_dpo
from eval_tool.judge import JudgeSettings


# ----------------------------------------------------------------------------
# self-contained fakes and fixtures
# ----------------------------------------------------------------------------


class FakeGenerator:
    def __init__(self, respond):
        self.respond = respond

    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        return [self.respond(last_human_text(messages)) for messages in messages_batch]


class FatalGenerator:
    def __init__(self, fatal_question: str, respond):
        self.fatal_question = fatal_question
        self.respond = respond

    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        for messages in messages_batch:
            if last_human_text(messages) == self.fatal_question:
                raise MemoryError("simulated fatal interruption")
        return [self.respond(last_human_text(messages)) for messages in messages_batch]


class FakeJudgeClient:
    def __init__(self, respond):
        self.respond = respond

    def judge_messages(self, messages, *, max_tokens=1024):
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
        return "".join(part["text"] for part in content if part.get("type") == "text")
    raise AssertionError("judge messages contain no user turn")


def generator_factory(respond):
    return lambda config: FakeGenerator(respond)


def judge_factory(respond):
    return lambda settings: FakeJudgeClient(respond)


def exploding_generator_factory(config):  # pragma: no cover - must never run
    raise AssertionError("the generator must not be constructed")


def exploding_judge_factory(settings):  # pragma: no cover - must never run
    raise AssertionError("the Judge client must not be constructed")


def binary_wrong(_messages) -> str:
    return json.dumps({"correct": False, "reason": "wrong"})


def binary_right(_messages) -> str:
    return json.dumps({"correct": True, "reason": "right"})


def v4_wrong(_messages) -> str:
    return json.dumps(
        {
            "equipment_correct": False,
            "fact_score": 0,
            "param_score": 0,
            "visual_score": None,
            "fabrication_score": 0,
            "style_score": 0,
        }
    )


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def make_model_dir(tmp_path: Path, name: str = "model", marker: str = "fake") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": marker}), encoding="utf-8"
    )
    return model_dir


def judge_config() -> DpoJudgeConfig:
    return DpoJudgeConfig(
        settings=JudgeSettings(
            api_base="http://judge.invalid/v1/chat/completions",
            api_key="sk-secret-value",
            model="judge-model",
            temperature=0.0,
            timeout=5,
            max_retries=0,
        ),
        max_workers=2,
    )


def make_config(tmp_path: Path, source: Path, *, wrong_only=False, rubric=None):
    return DpoBuildConfig(
        config_path=tmp_path / "dpo.json",
        inputs=(source,),
        output_dir=tmp_path / "out",
        output_name="train_dpo.jsonl",
        work_dir=tmp_path / "work",
        image_root=tmp_path,
        wrong_only=wrong_only,
        rubric=rubric,
        infer=DpoInferConfig(
            model_name="fake-vl",
            model_path=make_model_dir(tmp_path),
            batch_size=1,
        ),
        judge=judge_config() if wrong_only else None,
    )


def alpaca_source(tmp_path: Path, count: int = 3) -> Path:
    return write_json(
        tmp_path / "alpaca.json",
        [
            {"instruction": f"问题{index}", "output": f"参考{index}"}
            for index in range(count)
        ],
    )


def count_records(work_dir: Path, phase: str) -> int:
    attempt = work_dir / phase / "attempts" / attempt_id(work_dir, phase)
    return sum(
        len(read_jsonl(shard)) for shard in sorted(attempt.glob("worker-*.jsonl"))
    )


def attempt_id(work_dir: Path, phase: str) -> str:
    active = json.loads((work_dir / phase / "active.json").read_text(encoding="utf-8"))
    return active["attempt_id"]


# ----------------------------------------------------------------------------
# resume
# ----------------------------------------------------------------------------


def test_resume_runs_only_missing_inference_and_judge_jobs(tmp_path):
    source = alpaca_source(tmp_path)
    config = make_config(tmp_path, source, wrong_only=True, rubric="binary")

    with pytest.raises(DpoPipelineError):
        run_build_dpo(
            config,
            generator_factory=lambda infer: FatalGenerator(
                "问题2", lambda text: f"生成:{text}"
            ),
            judge_client_factory=exploding_judge_factory,
        )

    work = tmp_path / "work"
    assert count_records(work, "inference") == 2

    resumed: list[str] = []
    judged: list[str] = []

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(
            lambda text: resumed.append(text) or f"生成:{text}"
        ),
        judge_client_factory=judge_factory(
            lambda messages: judged.append(judge_question_text(messages))
            or binary_wrong(messages)
        ),
    )

    assert resumed == ["问题2"]
    assert sorted(judged) == ["问题0", "问题1", "问题2"]
    assert result.selected_count == 3

    replayed: list[str] = []
    again = run_build_dpo(
        config,
        generator_factory=generator_factory(
            lambda text: replayed.append(text) or "unused"
        ),
        judge_client_factory=judge_factory(
            lambda messages: replayed.append("judge") or binary_wrong(messages)
        ),
    )
    assert replayed == []
    assert again.selected_count == 3


def test_judge_setting_or_rubric_change_reuses_inference(tmp_path):
    source = alpaca_source(tmp_path, count=2)
    config = make_config(tmp_path, source, wrong_only=True, rubric="binary")
    generated: list[str] = []

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(
            lambda text: generated.append(text) or f"生成:{text}"
        ),
        judge_client_factory=judge_factory(binary_wrong),
    )
    assert len(generated) == 2
    assert result.selected_count == 2
    inference_attempt = attempt_id(tmp_path / "work", "inference")

    generated.clear()
    v4_result = run_build_dpo(
        replace(config, rubric="v4"),
        generator_factory=generator_factory(
            lambda text: generated.append(text) or f"生成:{text}"
        ),
        judge_client_factory=judge_factory(v4_wrong),
    )

    assert generated == []
    assert v4_result.selected_count == 2
    assert attempt_id(tmp_path / "work", "inference") == inference_attempt


def test_parser_only_change_reuses_judge_raw(tmp_path, monkeypatch):
    source = alpaca_source(tmp_path, count=2)
    config = make_config(tmp_path, source, wrong_only=True, rubric="binary")
    calls: list[str] = []

    run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
        judge_client_factory=judge_factory(
            lambda messages: calls.append("api") or binary_wrong(messages)
        ),
    )
    assert len(calls) == 2
    raw_attempt = attempt_id(tmp_path / "work", "judge_raw")
    parse_attempt = attempt_id(tmp_path / "work", "judge_parse")

    import eval_tool.dpo_cache as dpo_cache

    monkeypatch.setattr(dpo_cache, "DPO_JUDGE_PARSER_VERSION", "dpo-judge-parser-v2")
    calls.clear()

    result = run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
        judge_client_factory=judge_factory(
            lambda messages: calls.append("api") or binary_wrong(messages)
        ),
    )

    assert calls == []
    assert result.selected_count == 2
    assert attempt_id(tmp_path / "work", "judge_raw") == raw_attempt
    assert attempt_id(tmp_path / "work", "judge_parse") != parse_attempt


def test_input_model_generation_image_or_checkpoint_change_requires_overwrite(tmp_path):
    source = alpaca_source(tmp_path, count=2)
    config = make_config(tmp_path, source)

    run_build_dpo(config, generator_factory=generator_factory(lambda text: f"生成:{text}"))

    other_model = make_model_dir(tmp_path, name="model-b", marker="other")
    changed = replace(config, infer=replace(config.infer, model_path=other_model))
    with pytest.raises(DpoPipelineError, match="overwrite"):
        run_build_dpo(changed, generator_factory=exploding_generator_factory)

    thinking = replace(config, infer=replace(config.infer, enable_thinking=True))
    with pytest.raises(DpoPipelineError, match="overwrite"):
        run_build_dpo(thinking, generator_factory=exploding_generator_factory)

    alpaca_source(tmp_path, count=3)
    with pytest.raises(DpoPipelineError, match="overwrite"):
        run_build_dpo(config, generator_factory=exploding_generator_factory)

    result = run_build_dpo(
        config,
        overwrite=True,
        generator_factory=generator_factory(lambda text: f"重建:{text}"),
    )
    rows = read_jsonl(Path(result.output_path))
    assert [row["rejected"] for row in rows] == ["重建:问题0", "重建:问题1", "重建:问题2"]


def test_overwrite_resets_inference_judge_raw_and_judge_parse_then_rebuilds(tmp_path):
    source = alpaca_source(tmp_path, count=2)
    config = make_config(tmp_path, source, wrong_only=True, rubric="binary")
    run_build_dpo(
        config,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
        judge_client_factory=judge_factory(binary_wrong),
    )
    work = tmp_path / "work"
    phases = ("inference", "judge_raw", "judge_parse")
    before = {phase: attempt_id(work, phase) for phase in phases}

    generated: list[str] = []
    judged: list[str] = []
    result = run_build_dpo(
        config,
        overwrite=True,
        generator_factory=generator_factory(
            lambda text: generated.append(text) or f"重建:{text}"
        ),
        judge_client_factory=judge_factory(
            lambda messages: judged.append("api") or binary_wrong(messages)
        ),
    )

    after = {phase: attempt_id(work, phase) for phase in phases}
    assert all(after[phase] != before[phase] for phase in phases)
    assert len(generated) == 2
    assert len(judged) == 2
    rows = read_jsonl(Path(result.output_path))
    assert [row["rejected"] for row in rows] == ["重建:问题0", "重建:问题1"]


def test_clean_partial_runs_only_after_success_and_keeps_delivery_artifacts(tmp_path):
    source = alpaca_source(tmp_path, count=2)
    config = make_config(tmp_path, source)

    result = run_build_dpo(
        config,
        clean_partial=True,
        generator_factory=generator_factory(lambda text: f"生成:{text}"),
    )

    work = tmp_path / "work"
    assert not (work / "inference").exists()
    staging = work / "staging"
    assert not staging.exists() or not list(staging.iterdir())
    output_dir = tmp_path / "out"
    for name in (
        "train_dpo.jsonl",
        "audit_records.jsonl",
        "rejected_records.jsonl",
        "summary.json",
        "warnings.log",
        "manifest.json",
    ):
        assert (output_dir / name).is_file()
    assert Path(result.output_path).is_file()


def test_failed_run_keeps_partial_caches_for_a_later_resume(tmp_path):
    source = alpaca_source(tmp_path, count=2)
    config = make_config(tmp_path, source)

    with pytest.raises(DpoPipelineError):
        run_build_dpo(
            config,
            clean_partial=True,
            generator_factory=lambda infer: FatalGenerator(
                "问题0", lambda text: f"生成:{text}"
            ),
        )

    assert (tmp_path / "work" / "inference").is_dir()
    assert not (tmp_path / "out" / "manifest.json").exists()
