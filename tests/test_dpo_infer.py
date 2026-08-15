from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

import eval_tool.dpo_infer as dpo_infer
from eval_tool.dpo_cache import DpoInferenceStore
from eval_tool.dpo_config import DpoInferConfig
from eval_tool.dpo_infer import (
    DpoInferenceFatalError,
    DpoInferenceProtocolError,
    infer_batch_with_isolation,
    run_dpo_inference,
)
from eval_tool.dpo_input import DpoCandidate, DpoTurn, SourceRef


def _candidate(
    sample_id: str,
    question: str,
    *,
    chosen: str = "gold",
    history: tuple[DpoTurn, ...] = (),
) -> DpoCandidate:
    return DpoCandidate(
        sample_id=sample_id,
        conversations=(*history, DpoTurn(from_="human", value=question)),
        chosen=chosen,
        images=(),
        source=SourceRef(
            input_index=0,
            source_path=Path("input.json"),
            container_format="json_array",
            record_index=int(sample_id.rsplit("-", 1)[-1]) if "-" in sample_id else 0,
            line_number=None,
            source_id=None,
            raw_digest="0" * 64,
        ),
        source_format="sharegpt",
        turn_index=0,
        turn_count=1,
    )


def _config(tmp_path: Path, **updates) -> DpoInferConfig:
    config = DpoInferConfig(
        model_name="fake",
        model_path=tmp_path,
        enable_thinking=False,
        max_new_tokens=32,
        batch_size=3,
        torch_dtype="auto",
        device_map="auto",
        gpu_ids=(),
        workers_per_gpu=1,
    )
    return replace(config, **updates)


def _store(tmp_path: Path, candidates) -> DpoInferenceStore:
    return DpoInferenceStore(
        tmp_path / "inference",
        "infer-fingerprint",
        [candidate.sample_id for candidate in candidates],
        fsync_every=1,
    )


def _message_text(messages) -> str:
    pieces: list[str] = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            pieces.append(content)
        else:
            pieces.extend(
                part["text"]
                for part in content
                if part.get("type") == "text"
            )
    return "\n".join(pieces)


class _EchoGenerator:
    def __init__(self):
        self.calls = []

    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        self.calls.append((messages_batch, enable_thinking))
        return [_message_text(messages) for messages in messages_batch]


class _SelectiveGenerator:
    def __init__(self):
        self.batch_questions = []

    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        questions = [_message_text(messages) for messages in messages_batch]
        self.batch_questions.append(questions)
        if any(question == "bad" for question in questions):
            raise ValueError("bad data")
        return [f"prediction:{question}" for question in questions]


class _ShortGenerator:
    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        return ["only-one"]


class _InfrastructureGenerator:
    def __init__(self, error):
        self.error = error

    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        raise self.error


class _ValueGenerator:
    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        outputs = []
        for messages in messages_batch:
            question = _message_text(messages)
            outputs.append("" if question == "empty" else "same")
        return outputs


class _SpawnGenerator:
    def generate_message_batch(self, messages_batch, *, enable_thinking=None):
        outputs = []
        for messages in messages_batch:
            question = _message_text(messages)
            if question.startswith("slow:"):
                _, delay, value = question.split(":", 2)
                time.sleep(float(delay))
                question = value
            elif question == "fatal":
                raise MemoryError("simulated CUDA allocation failure")
            elif question == "blocked":
                time.sleep(30)
                question = "late"
            outputs.append(f"{os.environ.get('CUDA_VISIBLE_DEVICES')}:{question}")
        return outputs


def _spawn_generator_factory(config):
    return _SpawnGenerator()


def test_worker_completion_updates_and_closes_parent_progress(monkeypatch):
    events: list[tuple[str, object]] = []

    class FakeProgress:
        def __init__(self, **kwargs):
            events.append(("open", kwargs))

        def update(self, amount):
            events.append(("update", amount))

        def close(self):
            events.append(("close", None))

    class FakeQueue:
        def __init__(self):
            self.messages = [
                ("progress", 0, 1, None),
                ("done", 0, None, None),
            ]

        def get(self, timeout):
            return self.messages.pop(0)

    class FakeEvent:
        def is_set(self):
            return False

    class FakeProcess:
        exitcode = None

    monkeypatch.setattr(dpo_infer, "tqdm", FakeProgress, raising=False)

    dpo_infer._wait_for_worker_completion(
        [FakeProcess()], FakeQueue(), FakeEvent(), total_batches=1
    )

    assert events == [
        (
            "open",
            {"total": 1, "desc": "VLM\u63a8\u7406", "unit": "batch"},
        ),
        ("update", 1),
        ("close", None),
    ]


def test_model_receives_gold_history_and_current_human_only():
    candidate = _candidate(
        "sample-1",
        "current question",
        chosen="CURRENT GOLD MUST NOT LEAK",
        history=(
            DpoTurn(from_="human", value="history question"),
            DpoTurn(from_="gpt", value="history gold"),
        ),
    )
    generator = _EchoGenerator()

    outcomes = infer_batch_with_isolation(
        [candidate],
        generator,
        enable_thinking=True,
        append=lambda values: None,
    )

    messages = generator.calls[0][0][0]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert _message_text(messages) == "history question\ncurrent question"
    assert messages[1]["content"][0]["text"] == "history gold"
    assert "CURRENT GOLD MUST NOT LEAK" not in repr(messages)
    assert generator.calls[0][1] is True
    assert outcomes[0].prediction == "history question\ncurrent question"


def test_successful_batch_appends_each_terminal_result_immediately():
    candidates = [_candidate(f"sample-{index}", f"q{index}") for index in range(3)]
    appended = []

    outcomes = infer_batch_with_isolation(
        candidates,
        _EchoGenerator(),
        enable_thinking=False,
        append=lambda values: appended.append(tuple(values)),
    )

    assert [len(call) for call in appended] == [1, 1, 1]
    assert [call[0].sample_id for call in appended] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]
    assert [outcome.sample_id for outcome in outcomes] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]


def test_data_batch_failure_bisects_in_original_order_to_one_bad_sample():
    candidates = [
        _candidate("sample-0", "first"),
        _candidate("sample-1", "bad"),
        _candidate("sample-2", "last"),
    ]
    generator = _SelectiveGenerator()
    appended = []

    outcomes = infer_batch_with_isolation(
        candidates,
        generator,
        enable_thinking=False,
        append=lambda values: appended.extend(values),
    )

    assert [outcome.sample_id for outcome in outcomes] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]
    assert outcomes[1].error_type == "inference_error"
    assert outcomes[0].prediction == "prediction:first"
    assert outcomes[2].prediction == "prediction:last"
    assert [outcome.sample_id for outcome in appended] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]
    assert generator.batch_questions == [
        ["first", "bad", "last"],
        ["first"],
        ["bad", "last"],
        ["bad"],
        ["last"],
    ]


def test_empty_and_identical_outputs_remain_cached_successes_for_pipeline_filtering(tmp_path):
    candidates = [
        _candidate("sample-0", "empty", chosen="gold"),
        _candidate("sample-1", "identical", chosen="same"),
    ]
    store = _store(tmp_path, candidates)

    outcomes = run_dpo_inference(
        candidates,
        {},
        _config(tmp_path, batch_size=2),
        store,
        generator_factory=lambda config: _ValueGenerator(),
    )

    assert outcomes["sample-0"].prediction == ""
    assert outcomes["sample-1"].prediction == "same"
    assert all(outcome.error_type is None for outcome in outcomes.values())
    assert [record["status"] for record in store.load().values()] == ["ok", "ok"]


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("oom"),
        RuntimeError("CUDA out of memory"),
        RuntimeError("CUDA context was destroyed"),
        RuntimeError("NCCL communicator failed"),
    ],
)
def test_oom_cuda_context_or_nccl_failure_aborts_the_run(tmp_path, error):
    candidates = [_candidate("sample-0", "q")]
    store = _store(tmp_path, candidates)

    with pytest.raises((MemoryError, RuntimeError), match=str(error)):
        run_dpo_inference(
            candidates,
            {},
            _config(tmp_path),
            store,
            generator_factory=lambda config: _InfrastructureGenerator(error),
        )

    assert store.load() == {}


def test_model_load_failure_aborts_before_any_append(tmp_path):
    candidates = [_candidate("sample-0", "q")]
    store = _store(tmp_path, candidates)

    def fail_load(config):
        raise OSError("checkpoint unavailable")

    with pytest.raises(DpoInferenceFatalError, match="could not be loaded"):
        run_dpo_inference(
            candidates,
            {},
            _config(tmp_path),
            store,
            generator_factory=fail_load,
        )

    assert store.load() == {}


def test_short_batch_response_is_a_fatal_generator_protocol_error():
    candidates = [_candidate("sample-0", "q0"), _candidate("sample-1", "q1")]

    with pytest.raises(DpoInferenceProtocolError, match="prediction count"):
        infer_batch_with_isolation(
            candidates,
            _ShortGenerator(),
            enable_thinking=False,
            append=lambda values: None,
        )


def test_resume_with_changed_workers_per_gpu_runs_only_pending_samples(tmp_path):
    candidates = [_candidate(f"sample-{index}", f"q{index}") for index in range(3)]
    store = _store(tmp_path, candidates)
    store.append_batch(
        [
            {
                "sample_id": "sample-0",
                "status": "ok",
                "prediction": "already cached",
                "error_type": None,
                "error_message": None,
            }
        ]
    )
    store.sync()
    generator = _EchoGenerator()

    outcomes = run_dpo_inference(
        candidates,
        {},
        _config(tmp_path, workers_per_gpu=7),
        store,
        generator_factory=lambda config: generator,
    )

    generated = [
        _message_text(messages)
        for messages_batch, _ in generator.calls
        for messages in messages_batch
    ]
    assert generated == ["q1", "q2"]
    assert outcomes["sample-0"].prediction == "already cached"
    assert list(outcomes) == ["sample-0", "sample-1", "sample-2"]


def test_completed_inference_cache_is_scanned_only_once(tmp_path, monkeypatch):
    candidates = [_candidate(f"sample-{index}", f"q{index}") for index in range(3)]
    store = _store(tmp_path, candidates)
    store.append_batch(
        [
            {
                "sample_id": candidate.sample_id,
                "status": "ok",
                "prediction": f"cached:{candidate.sample_id}",
                "error_type": None,
                "error_message": None,
            }
            for candidate in candidates
        ]
    )
    store.sync()
    real_load = store.load
    load_calls = 0

    def counting_load():
        nonlocal load_calls
        load_calls += 1
        return real_load()

    monkeypatch.setattr(store, "load", counting_load)

    outcomes = run_dpo_inference(
        candidates,
        {},
        _config(tmp_path),
        store,
        generator_factory=lambda config: pytest.fail("completed cache reran inference"),
    )

    assert load_calls == 1
    assert list(outcomes) == ["sample-0", "sample-1", "sample-2"]


@pytest.mark.skipif(os.name != "nt", reason="Windows spawn contract")
def test_windows_spawn_worker_receives_picklable_dtos_and_results_stay_ordered(tmp_path):
    candidates = [
        _candidate("sample-0", "slow:0.25:first"),
        _candidate("sample-1", "second"),
        _candidate("sample-2", "third"),
    ]
    store = _store(tmp_path, candidates)

    outcomes = run_dpo_inference(
        candidates,
        {},
        _config(
            tmp_path,
            batch_size=1,
            gpu_ids=(7,),
            workers_per_gpu=2,
            device_map="cuda:0",
        ),
        store,
        generator_factory=_spawn_generator_factory,
    )

    assert list(outcomes) == ["sample-0", "sample-1", "sample-2"]
    assert outcomes["sample-0"].prediction == "7:first"
    assert outcomes["sample-1"].prediction == "7:second"
    assert outcomes["sample-2"].prediction == "7:third"


@pytest.mark.skipif(os.name != "nt", reason="Windows peer termination contract")
def test_one_fatal_worker_terminates_blocked_peer_and_preserves_completed_shard(tmp_path):
    candidates = [
        _candidate("sample-0", "finished-before-fatal"),
        _candidate("sample-1", "blocked"),
        _candidate("sample-2", "fatal"),
    ]
    store = _store(tmp_path, candidates)
    started = time.monotonic()

    with pytest.raises(DpoInferenceFatalError, match="MemoryError"):
        run_dpo_inference(
            candidates,
            {},
            _config(
                tmp_path,
                batch_size=1,
                gpu_ids=(0,),
                workers_per_gpu=2,
            ),
            store,
            generator_factory=_spawn_generator_factory,
        )

    assert time.monotonic() - started < 15
    cached = store.load()
    assert ("sample-0",) in cached
    assert cached[("sample-0",)]["prediction"] == "0:finished-before-fatal"
    assert ("sample-1",) not in cached
    assert ("sample-2",) not in cached
