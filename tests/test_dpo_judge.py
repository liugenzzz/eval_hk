from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from eval_tool.dpo_cache import (
    DpoJudgeParseStore,
    DpoJudgeRawStore,
    build_judge_parse_fp,
    build_judge_request_fp,
)
from eval_tool.dpo_config import DpoJudgeConfig
from eval_tool.dpo_input import (
    DpoCandidate,
    DpoTurn,
    ImageRef,
    SourceRef,
)
from eval_tool.dpo_judge import (
    JudgeJob,
    build_dpo_judge_request,
    judge_candidates,
    parse_dpo_judge_response,
    validate_dpo_judge_object,
)
from eval_tool.dpo_multimodal import inspect_image
from eval_tool.dpo_prompts import load_dpo_prompt
from eval_tool.judge import JudgeClient, JudgeSettings


def _candidate(
    conversations: tuple[DpoTurn, ...],
    *,
    sample_id: str = "sample-1",
    chosen: str = "标准答案",
    images: tuple[ImageRef, ...] = (),
) -> DpoCandidate:
    return DpoCandidate(
        sample_id=sample_id,
        conversations=conversations,
        chosen=chosen,
        images=images,
        source=SourceRef(
            input_index=0,
            source_path=Path("input.json"),
            container_format="json_array",
            record_index=0,
            line_number=None,
            source_id=None,
            raw_digest="0" * 64,
        ),
        source_format="sharegpt",
        turn_index=1,
        turn_count=2,
    )


def _v4_object(**updates):
    obj = {
        "rubric": "v4",
        "equipment_correct": True,
        "fact_score": 80,
        "param_score": None,
        "visual_score": None,
        "fabrication_score": 100,
        "style_score": 80,
        "reasoning": "ok",
    }
    obj.update(updates)
    return obj


def test_structured_transport_preserves_existing_post_override_compatibility():
    class CompatClient(JudgeClient):
        def __init__(self):
            super().__init__(JudgeSettings())
            self.legacy_calls = []
            self.message_calls = []

        def _post(self, system_prompt, user_text, image_b64=None, mime="image/jpeg"):
            self.legacy_calls.append((system_prompt, user_text, image_b64, mime))
            return '{"correct": true, "reason": "ok"}'

        def _post_messages(self, messages, *, max_tokens=1024):
            self.message_calls.append((messages, max_tokens))
            return "structured"

    client = CompatClient()

    assert client.judge_pointwise("q", "ref", "pred")["hit"] == 1
    assert client.legacy_calls
    assert client.judge_messages([{"role": "user", "content": "q"}], max_tokens=17) == "structured"
    assert client.message_calls == [([{"role": "user", "content": "q"}], 17)]


def test_multiturn_judge_includes_gold_history_current_chosen_and_rejected():
    candidate = _candidate(
        (
            DpoTurn(from_="human", value="历史问题"),
            DpoTurn(from_="gpt", value="历史标准答案"),
            DpoTurn(from_="human", value="当前问题"),
        ),
        chosen="当前标准答案",
    )
    prompt = load_dpo_prompt("binary", False)

    request = build_dpo_judge_request(candidate, "当前模型答案", prompt, {})

    assert request.rubric == "binary"
    assert request.has_images is False
    assert [message["role"] for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert request.messages[1]["content"] == "历史问题"
    assert request.messages[2]["content"] == "历史标准答案"
    assert request.messages[3]["content"] == "当前问题"
    comparison = request.messages[4]["content"]
    assert "当前标准答案" in comparison
    assert "当前模型答案" in comparison
    assert "chosen" in comparison
    assert "rejected" in comparison


def test_judge_interleaves_png_and_jpeg_at_original_placeholder_positions(tmp_path):
    png = tmp_path / "first.bin"
    jpeg = tmp_path / "second.bin"
    Image.new("RGB", (2, 2), "red").save(png, format="PNG")
    Image.new("RGB", (2, 2), "blue").save(jpeg, format="JPEG")
    refs = (
        ImageRef(original="relative/first.png", resolved=png.resolve()),
        ImageRef(original="relative/second.jpg", resolved=jpeg.resolve()),
    )
    assets = {ref.resolved: inspect_image(ref) for ref in refs}
    candidate = _candidate(
        (DpoTurn(from_="human", value="前<image>中<image>后"),),
        images=refs,
    )

    request = build_dpo_judge_request(
        candidate,
        "模型答案",
        load_dpo_prompt("binary", True),
        assets,
    )

    content = request.messages[1]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert content[0]["text"] == "前"
    assert content[2]["text"] == "中"
    assert content[4]["text"] == "后"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    fingerprint = json.dumps(request.fingerprint_material, ensure_ascii=False)
    assert "data:" not in fingerprint
    assert "base64" not in fingerprint
    assert "relative/first.png" in fingerprint
    assert assets[refs[0].resolved].sha256 in fingerprint


@pytest.mark.parametrize("correct", [True, False])
def test_binary_accepts_only_literal_boolean_correct(correct):
    parsed = parse_dpo_judge_response(
        json.dumps({"correct": correct, "reason": "verdict"}),
        "binary",
        False,
    )

    assert parsed["hit"] == (1 if correct else 0)


@pytest.mark.parametrize("correct", [None, 0, 1, "true", "false", [], {}])
def test_binary_rejects_non_boolean_correct(correct):
    with pytest.raises(ValueError):
        parse_dpo_judge_response(
            json.dumps({"correct": correct, "reason": "bad"}),
            "binary",
            False,
        )


@pytest.mark.parametrize("equipment_correct", [None, 0, 1, "true", "false", [], {}])
def test_v4_accepts_only_literal_boolean_equipment_correct(equipment_correct):
    with pytest.raises(ValueError):
        validate_dpo_judge_object(
            _v4_object(equipment_correct=equipment_correct), "v4", False
        )


@pytest.mark.parametrize("value", [True, False, "50", float("nan"), float("inf"), -1, 101])
@pytest.mark.parametrize(
    "field", [
        "fact_score",
        "param_score",
        "visual_score",
        "fabrication_score",
        "style_score",
    ]
)
def test_v4_numeric_fields_reject_bool_string_nan_and_out_of_range(field, value):
    obj = _v4_object()
    if field == "visual_score":
        has_images = True
        obj["fact_score"] = None
        obj["visual_score"] = value
    else:
        has_images = False
        obj[field] = value

    with pytest.raises(ValueError):
        validate_dpo_judge_object(obj, "v4", has_images)


def test_text_v4_requires_numeric_fact_and_null_visual():
    validate_dpo_judge_object(_v4_object(), "v4", False)

    with pytest.raises(ValueError):
        validate_dpo_judge_object(_v4_object(fact_score=None), "v4", False)
    with pytest.raises(ValueError):
        validate_dpo_judge_object(_v4_object(visual_score=50), "v4", False)


def test_visual_v4_keeps_existing_nullable_dimensions():
    raw = json.dumps(
        _v4_object(fact_score=None, param_score=None, visual_score=75)
    )

    parsed = parse_dpo_judge_response(raw, "v4", True)

    assert parsed["visual_score"] == 75


def test_v4_uses_parser_hit_not_rounded_quality_score():
    raw = json.dumps(
        _v4_object(
            fact_score=59.999,
            param_score=None,
            visual_score=None,
            fabrication_score=100,
        )
    )

    parsed = parse_dpo_judge_response(raw, "v4", False)

    assert parsed["quality_score"] == 60.0
    assert parsed["hit"] == 0


def test_transport_or_parse_failure_is_judge_error_not_wrong_answer():
    class FailingClient(JudgeClient):
        def _post_messages(self, messages, *, max_tokens=1024):
            raise OSError("transport failed")

    client = FailingClient(JudgeSettings())
    with pytest.raises(OSError):
        client.judge_messages([{"role": "user", "content": "q"}])

    with pytest.raises(ValueError):
        parse_dpo_judge_response('{"correct": null}', "binary", False)


def _judge_job(sample_id: str, rejected: str = "model answer") -> JudgeJob:
    candidate = _candidate(
        (DpoTurn(from_="human", value=f"question {sample_id}"),),
        sample_id=sample_id,
    )
    request = build_dpo_judge_request(
        candidate,
        rejected,
        load_dpo_prompt("binary", False),
        {},
    )
    return JudgeJob(
        candidate=candidate,
        rejected=rejected,
        inference_fp="inference-fingerprint",
        request=request,
    )


def _judge_config(*, max_retries=0, max_workers=4) -> DpoJudgeConfig:
    return DpoJudgeConfig(
        settings=JudgeSettings(
            api_base="http://judge.invalid/v1/chat/completions",
            api_key="secret",
            model="judge-model",
            temperature=0.0,
            timeout=5,
            max_retries=max_retries,
        ),
        max_workers=max_workers,
    )


def _request_fp(job: JudgeJob, config: DpoJudgeConfig) -> str:
    return build_judge_request_fp(
        job.inference_fp,
        config.settings,
        job.request.fingerprint_material,
        job.request.image_assets,
    )


class _RecordingStore:
    def __init__(self, expected_keys, events, phase, *, fail_append=False):
        self.expected_keys = tuple(expected_keys)
        self.events = events
        self.phase = phase
        self.fail_append = fail_append
        self.records = {}
        self.writer_threads = []

    def load(self):
        self.events.append((self.phase, "load"))
        return {
            key: self.records[key]
            for key in self.expected_keys
            if key in self.records
        }

    def append_batch(self, records, *, worker_id=0):
        self.events.append((self.phase, "append"))
        self.writer_threads.append(threading.get_ident())
        if self.fail_append:
            raise OSError(f"{self.phase} writer failed")
        for record in records:
            fingerprint_field = (
                "judge_request_fp" if self.phase == "raw" else "judge_parse_fp"
            )
            key = (record["sample_id"], record[fingerprint_field])
            assert key in self.expected_keys
            assert key not in self.records
            self.records[key] = dict(record)

    def sync(self):
        self.events.append((self.phase, "sync"))

    def mark_complete(self):
        self.events.append((self.phase, "complete"))


class _CallbackClient:
    def __init__(self, callback):
        self.callback = callback

    def judge_messages(self, messages, *, max_tokens=1024):
        return self.callback(messages, max_tokens)


def test_judge_runner_retries_with_bounded_exponential_backoff(monkeypatch):
    job = _judge_job("sample-1")
    config = _judge_config(max_retries=2, max_workers=1)
    events = []
    raw = _RecordingStore(
        [(job.candidate.sample_id, _request_fp(job, config))], events, "raw"
    )
    parse_holder = {}
    attempts = 0
    sleeps = []

    def callback(messages, max_tokens):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(f"temporary-{attempts}")
        return '{"correct": false, "reason": "wrong"}'

    def parse_factory(keys):
        parse_holder["store"] = _RecordingStore(keys, events, "parse")
        return parse_holder["store"]

    monkeypatch.setattr("eval_tool.dpo_judge.time.sleep", sleeps.append)
    outcomes = judge_candidates(
        [job],
        config,
        raw,
        parse_factory,
        client_factory=lambda settings: _CallbackClient(callback),
    )

    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert outcomes["sample-1"].parsed["hit"] == 0


def test_judge_runner_honors_max_workers_and_constructs_one_client_per_thread():
    jobs = [_judge_job(f"sample-{index}") for index in range(6)]
    config = _judge_config(max_workers=2)
    events = []
    raw = _RecordingStore(
        [(job.candidate.sample_id, _request_fp(job, config)) for job in jobs],
        events,
        "raw",
    )
    state_lock = threading.Lock()
    active = 0
    max_active = 0
    clients = 0

    def factory(settings):
        nonlocal clients
        with state_lock:
            clients += 1

        def callback(messages, max_tokens):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with state_lock:
                active -= 1
            return '{"correct": false, "reason": "wrong"}'

        return _CallbackClient(callback)

    outcomes = judge_candidates(
        jobs,
        config,
        raw,
        lambda keys: _RecordingStore(keys, events, "parse"),
        client_factory=factory,
    )

    assert max_active == 2
    assert clients == 2
    assert list(outcomes) == [job.candidate.sample_id for job in jobs]


def test_judge_runner_coordinator_writes_all_raw_before_parse_and_is_single_writer():
    jobs = [_judge_job(f"sample-{index}") for index in range(4)]
    config = _judge_config(max_workers=4)
    events = []
    raw = _RecordingStore(
        [(job.candidate.sample_id, _request_fp(job, config)) for job in jobs],
        events,
        "raw",
    )
    parse_holder = {}
    main_thread = threading.get_ident()

    def parse_factory(keys):
        events.append(("parse", "factory"))
        parse_holder["store"] = _RecordingStore(keys, events, "parse")
        return parse_holder["store"]

    judge_candidates(
        jobs,
        config,
        raw,
        parse_factory,
        client_factory=lambda settings: _CallbackClient(
            lambda messages, max_tokens: '{"correct": false}'
        ),
    )

    parse_factory_index = events.index(("parse", "factory"))
    raw_append_indices = [
        index for index, event in enumerate(events) if event == ("raw", "append")
    ]
    assert len(raw_append_indices) == len(jobs)
    assert max(raw_append_indices) < parse_factory_index
    assert set(raw.writer_threads) == {main_thread}
    assert set(parse_holder["store"].writer_threads) == {main_thread}


def test_parse_only_resume_uses_cached_raw_without_constructing_client(tmp_path):
    job = _judge_job("sample-1")
    config = _judge_config()
    request_fp = _request_fp(job, config)
    raw_response = '{"correct": false, "reason": "wrong"}'
    raw = DpoJudgeRawStore(
        tmp_path / "raw",
        "raw-phase",
        [(job.candidate.sample_id, request_fp)],
        fsync_every=1,
    )
    raw.append_batch(
        [
            {
                "sample_id": job.candidate.sample_id,
                "judge_request_fp": request_fp,
                "raw_response": raw_response,
            }
        ]
    )
    raw.sync()
    parse_fp = build_judge_parse_fp(request_fp, raw_response, "binary", False)
    parse_store = DpoJudgeParseStore(
        tmp_path / "parse",
        "parse-phase",
        [(job.candidate.sample_id, parse_fp)],
        fsync_every=1,
    )

    def forbidden_factory(settings):
        raise AssertionError("client factory must not run during parse-only resume")

    outcomes = judge_candidates(
        [job],
        config,
        raw,
        parse_store,
        client_factory=forbidden_factory,
    )

    assert outcomes["sample-1"].parsed["hit"] == 0
    assert outcomes["sample-1"].raw_response == raw_response


def test_first_run_real_stores_build_parse_store_from_actual_raw_key(tmp_path):
    job = _judge_job("sample-1")
    config = _judge_config()
    request_fp = _request_fp(job, config)
    raw = DpoJudgeRawStore(
        tmp_path / "raw",
        "raw-phase",
        [(job.candidate.sample_id, request_fp)],
        fsync_every=1,
    )
    created = {}

    def parse_factory(keys):
        created["keys"] = keys
        created["store"] = DpoJudgeParseStore(
            tmp_path / "parse",
            "parse-phase",
            keys,
            fsync_every=1,
        )
        return created["store"]

    outcomes = judge_candidates(
        [job],
        config,
        raw,
        parse_factory,
        client_factory=lambda settings: _CallbackClient(
            lambda messages, max_tokens: '{"correct": false}'
        ),
    )

    outcome = outcomes["sample-1"]
    assert created["keys"] == (("sample-1", outcome.parse_fp),)
    assert created["store"].is_complete()
    assert raw.is_complete()


def test_parse_failure_is_cached_terminal_judge_error_while_raw_survives():
    job = _judge_job("sample-1")
    config = _judge_config()
    events = []
    raw = _RecordingStore(
        [(job.candidate.sample_id, _request_fp(job, config))], events, "raw"
    )
    parse_holder = {}

    def parse_factory(keys):
        parse_holder["store"] = _RecordingStore(keys, events, "parse")
        return parse_holder["store"]

    outcomes = judge_candidates(
        [job],
        config,
        raw,
        parse_factory,
        client_factory=lambda settings: _CallbackClient(
            lambda messages, max_tokens: "not JSON"
        ),
    )

    outcome = outcomes["sample-1"]
    assert outcome.raw_response == "not JSON"
    assert outcome.error_type == "judge_error"
    assert outcome.parsed is None
    assert next(iter(raw.records.values()))["raw_response"] == "not JSON"
    assert next(iter(parse_holder["store"].records.values()))["status"] == "error"


def test_transport_failure_is_not_parsed_as_a_wrong_answer(monkeypatch):
    job = _judge_job("sample-1")
    config = _judge_config(max_retries=1)
    events = []
    raw = _RecordingStore(
        [(job.candidate.sample_id, _request_fp(job, config))], events, "raw"
    )
    parse_factory_calls = []
    monkeypatch.setattr("eval_tool.dpo_judge.time.sleep", lambda delay: None)

    outcomes = judge_candidates(
        [job],
        config,
        raw,
        lambda keys: parse_factory_calls.append(keys),
        client_factory=lambda settings: _CallbackClient(
            lambda messages, max_tokens: (_ for _ in ()).throw(OSError("offline"))
        ),
    )

    outcome = outcomes["sample-1"]
    assert outcome.error_type == "judge_error"
    assert outcome.parsed is None
    assert outcome.parse_fp is None
    assert raw.records == {}
    assert parse_factory_calls == []


def test_raw_writer_failure_happens_before_parse_store_construction():
    job = _judge_job("sample-1")
    config = _judge_config()
    events = []
    raw = _RecordingStore(
        [(job.candidate.sample_id, _request_fp(job, config))],
        events,
        "raw",
        fail_append=True,
    )
    parse_factory_calls = []

    with pytest.raises(OSError, match="raw writer failed"):
        judge_candidates(
            [job],
            config,
            raw,
            lambda keys: parse_factory_calls.append(keys),
            client_factory=lambda settings: _CallbackClient(
                lambda messages, max_tokens: '{"correct": false}'
            ),
        )

    assert parse_factory_calls == []
    assert not any(event[0] == "parse" for event in events)
