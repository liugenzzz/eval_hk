from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType
from typing import get_args

import pytest


def _dpo_input() -> ModuleType:
    return importlib.import_module("eval_tool.dpo_input")


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_record(
    module: ModuleType,
    value: object,
    *,
    input_index: int = 0,
    record_index: int = 0,
    line_number: int | None = None,
    source_path: Path | None = None,
    source_id: str | None = None,
    raw_digest: str | None = None,
):
    return module.RawRecord(
        source=module.SourceRef(
            input_index=input_index,
            source_path=(source_path or (Path.cwd() / "source.json")).resolve(),
            container_format="jsonl" if line_number is not None else "json_array",
            record_index=record_index,
            line_number=line_number,
            source_id=source_id,
            raw_digest=raw_digest or _canonical_digest(value),
        ),
        value=value,
    )


def _turn_values(candidate):
    return [(turn.from_, turn.value) for turn in candidate.conversations]


def test_public_contract_uses_literal_and_frozen_dataclasses(tmp_path):
    module = _dpo_input()
    path = (tmp_path / "record.data").resolve()
    source = module.SourceRef(
        input_index=3,
        source_path=path,
        container_format="json_object",
        record_index=0,
        line_number=None,
        source_id="record-1",
        raw_digest="a" * 64,
    )
    record = module.RawRecord(source=source, value={"id": "record-1"})

    assert get_args(module.ContainerFormat) == (
        "json_array",
        "json_object",
        "jsonl",
    )
    assert issubclass(module.DpoInputError, RuntimeError)
    assert source == module.SourceRef(
        input_index=3,
        source_path=path,
        container_format="json_object",
        record_index=0,
        line_number=None,
        source_id="record-1",
        raw_digest="a" * 64,
    )
    assert record.source is source
    assert record.value == {"id": "record-1"}
    with pytest.raises(FrozenInstanceError):
        source.record_index = 1
    with pytest.raises(FrozenInstanceError):
        record.value = None


def test_loads_array_and_tracks_source_fields_and_order(tmp_path):
    module = _dpo_input()
    path = tmp_path / "records.unknown"
    values = [
        {"id": " source-1 ", "answer": "甲"},
        {"id": "", "answer": "乙"},
        {"id": 17, "answer": "丙"},
    ]
    path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")

    records = module.load_source_records(path, input_index=7)

    assert [record.value for record in records] == values
    assert [record.source.input_index for record in records] == [7, 7, 7]
    assert [record.source.source_path for record in records] == [
        path.resolve(),
        path.resolve(),
        path.resolve(),
    ]
    assert [record.source.container_format for record in records] == [
        "json_array",
        "json_array",
        "json_array",
    ]
    assert [record.source.record_index for record in records] == [0, 1, 2]
    assert [record.source.line_number for record in records] == [None, None, None]
    assert [record.source.source_id for record in records] == [
        " source-1 ",
        None,
        None,
    ]
    assert [record.source.raw_digest for record in records] == [
        _canonical_digest(value) for value in values
    ]


def test_loads_single_object_from_jsonl_extension(tmp_path):
    module = _dpo_input()
    path = tmp_path / "single.jsonl"
    value = {"id": "one", "instruction": "问题"}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    records = module.load_source_records(path, input_index=0)

    assert len(records) == 1
    assert records[0].value == value
    assert records[0].source.container_format == "json_object"
    assert records[0].source.record_index == 0
    assert records[0].source.line_number is None
    assert records[0].source.source_id == "one"


def test_loads_pretty_array_from_jsonl_extension(tmp_path):
    module = _dpo_input()
    path = tmp_path / "pretty.jsonl"
    values = [{"id": "a"}, {"id": "b"}]
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")

    records = module.load_source_records(path, input_index=2)

    assert [record.value for record in records] == values
    assert [record.source.container_format for record in records] == [
        "json_array",
        "json_array",
    ]


def test_loads_jsonl_from_json_extension_with_physical_line_numbers(tmp_path):
    module = _dpo_input()
    path = tmp_path / "records.json"
    path.write_text(
        '\n{"id":"first","value":1}\n   \n{"id":"second","value":2}\n',
        encoding="utf-8",
    )

    records = module.load_source_records(path, input_index=5)

    assert [record.value["id"] for record in records] == ["first", "second"]
    assert [record.source.container_format for record in records] == [
        "jsonl",
        "jsonl",
    ]
    assert [record.source.record_index for record in records] == [0, 1]
    assert [record.source.line_number for record in records] == [2, 4]
    assert [record.source.source_id for record in records] == ["first", "second"]


def test_array_preserves_scalar_null_and_list_items_for_later_audit(tmp_path):
    module = _dpo_input()
    path = tmp_path / "mixed.json"
    values = [{"valid": True}, "text", 42, None, ["nested"]]
    path.write_text(json.dumps(values), encoding="utf-8")

    records = module.load_source_records(path, input_index=0)

    assert [record.value for record in records] == values
    assert [record.source.record_index for record in records] == list(range(5))
    assert [record.source.source_id for record in records] == [None] * 5


def test_digest_is_canonical_across_containers_and_changes_with_value(tmp_path):
    module = _dpo_input()
    value = {"b": "舰", "a": [1, True, None]}
    array_path = tmp_path / "array.jsonl"
    array_path.write_text(
        json.dumps([value], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    jsonl_path = tmp_path / "lines.json"
    jsonl_path.write_text(
        '{"a":[1,true,null],"b":"舰"}\n{"different":true}\n',
        encoding="utf-8",
    )

    array_records = module.load_source_records(array_path, input_index=0)
    jsonl_records = module.load_source_records(jsonl_path, input_index=1)

    expected = _canonical_digest(value)
    assert array_records[0].source.raw_digest == expected
    assert jsonl_records[0].source.raw_digest == expected
    assert jsonl_records[1].source.raw_digest != expected
    assert len(expected) == 64
    assert all(character in "0123456789abcdef" for character in expected)


def test_digest_matches_known_canonical_sha256_vector(tmp_path):
    module = _dpo_input()
    path = tmp_path / "known-digest.json"
    path.write_text(
        '{"z":"\\u8230","a":[1,true,null]}', encoding="utf-8"
    )

    records = module.load_source_records(path, input_index=0)

    assert records[0].source.raw_digest == (
        "d1fafc7c1433a58a795d4e37286ed90644a06470ea4a267834772b0d8db17b8e"
    )


def test_duplicate_keys_use_stdlib_last_value_across_containers(tmp_path):
    module = _dpo_input()
    duplicate_object = (
        '{"id":"first","id":"last","payload":"old","payload":"new"}'
    )
    expected = {"id": "last", "payload": "new"}
    object_path = tmp_path / "duplicate-object.json"
    object_path.write_text(duplicate_object, encoding="utf-8")
    jsonl_path = tmp_path / "duplicate-lines.jsonl"
    jsonl_path.write_text(
        duplicate_object + '\n{"id":"tail"}\n', encoding="utf-8"
    )

    object_records = module.load_source_records(object_path, input_index=0)
    jsonl_records = module.load_source_records(jsonl_path, input_index=1)

    assert object_records[0].value == expected
    assert object_records[0].source.container_format == "json_object"
    assert object_records[0].source.source_id == "last"
    assert object_records[0].source.raw_digest == _canonical_digest(expected)
    assert jsonl_records[0].value == expected
    assert jsonl_records[0].source.container_format == "jsonl"
    assert jsonl_records[0].source.source_id == "last"
    assert jsonl_records[0].source.raw_digest == _canonical_digest(expected)


@pytest.mark.parametrize("text", ["null", '"text"', "1", "true"])
def test_rejects_complete_top_level_json_scalar(tmp_path, text):
    module = _dpo_input()
    path = tmp_path / "scalar.data"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    assert str(path.resolve()) in str(caught.value)
    assert "object" in str(caught.value).lower()
    assert "array" in str(caught.value).lower()


@pytest.mark.parametrize("text", ["", " \n\t\n"])
def test_rejects_empty_or_whitespace_only_files(tmp_path, text):
    module = _dpo_input()
    path = tmp_path / "empty.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    assert str(path.resolve()) in str(caught.value)
    assert "empty" in str(caught.value).lower()


@pytest.mark.parametrize(
    ("text", "line_number"),
    [
        ("{not-json}", 1),
        ('{"ok":1}\n{"broken":}\n', 2),
    ],
)
def test_invalid_jsonl_reports_source_path_and_physical_line(
    tmp_path, text, line_number
):
    module = _dpo_input()
    path = tmp_path / "invalid.input"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value)
    assert str(path.resolve()) in message
    assert f"line {line_number}" in message.lower()


def test_rejects_multiline_jsonl_record_at_its_first_physical_line(tmp_path):
    module = _dpo_input()
    path = tmp_path / "multiline.jsonl"
    path.write_text(
        '{\n  "id": "multiline"\n}\n{"id":"second"}\n', encoding="utf-8"
    )

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value)
    assert str(path.resolve()) in message
    assert "line 1" in message.lower()


def test_jsonl_keeps_raw_u2028_inside_string_on_same_physical_line(tmp_path):
    module = _dpo_input()
    path = tmp_path / "raw-line-separator.jsonl"
    separator = "\u2028"
    path.write_text(
        '{"id":"first","text":"before'
        + separator
        + 'after"}\n{"id":"second"}\n',
        encoding="utf-8",
    )

    records = module.load_source_records(path, input_index=0)

    assert [record.value for record in records] == [
        {"id": "first", "text": "before" + separator + "after"},
        {"id": "second"},
    ]
    assert [record.source.line_number for record in records] == [1, 2]


@pytest.mark.parametrize(
    "separator",
    [
        pytest.param("\u2028", id="line-separator"),
        pytest.param("\u2029", id="paragraph-separator"),
        pytest.param("\u0085", id="next-line"),
    ],
)
def test_unicode_separators_between_objects_do_not_create_jsonl_lines(
    tmp_path, separator
):
    module = _dpo_input()
    path = tmp_path / "not-physical-lines.jsonl"
    path.write_text(
        '{"id":"first"}' + separator + '{"id":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value)
    assert str(path.resolve()) in message
    assert "line 1" in message.lower()
    assert "line 2" not in message.lower()


@pytest.mark.parametrize("value", [123, [], None])
def test_rejects_non_object_jsonl_records(tmp_path, value):
    module = _dpo_input()
    path = tmp_path / "non-object.jsonl"
    path.write_text(
        '{"ok":true}\n' + json.dumps(value) + '\n', encoding="utf-8"
    )

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value).lower()
    assert str(path.resolve()).lower() in message
    assert "line 2" in message
    assert "object" in message


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_rejects_nonstandard_numeric_constants_in_arrays(tmp_path, constant):
    module = _dpo_input()
    path = tmp_path / "nonfinite.json"
    path.write_text(f"[{constant}]", encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    assert str(path.resolve()) in str(caught.value)


def test_rejects_nonstandard_numeric_constants_in_jsonl(tmp_path):
    module = _dpo_input()
    path = tmp_path / "nonfinite.jsonl"
    path.write_text('{"value":NaN}\n{"ok":true}\n', encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value)
    assert str(path.resolve()) in message
    assert "line 1" in message.lower()


def test_rejects_json_number_that_overflows_to_infinity(tmp_path):
    module = _dpo_input()
    path = tmp_path / "overflow.json"
    path.write_text("[1e400]", encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    assert str(path.resolve()) in str(caught.value)


@pytest.mark.parametrize(
    ("text", "record_index", "line_number"),
    [
        pytest.param(
            r'{"id":"single","text":"\ud800","marker":"SURROGATE_SECRET"}',
            0,
            None,
            id="single-object",
        ),
        pytest.param(
            r'[{"id":"ok"},{"id":"array","text":"\udfff","marker":"SURROGATE_SECRET"}]',
            1,
            None,
            id="array-item",
        ),
        pytest.param(
            '{"id":"ok"}\n\n'
            + r'{"id":"jsonl","text":"\ud800","marker":"SURROGATE_SECRET"}'
            + "\n",
            1,
            3,
            id="jsonl-record",
        ),
    ],
)
def test_rejects_lone_surrogate_with_record_location(
    tmp_path, text, record_index, line_number
):
    module = _dpo_input()
    path = tmp_path / "lone-surrogate.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value)
    assert str(path.resolve()) in message
    assert f"record index {record_index}" in message.lower()
    if line_number is not None:
        assert f"line {line_number}" in message.lower()
    assert "SURROGATE_SECRET" not in message


def test_normalizes_read_errors_with_resolved_path(tmp_path, monkeypatch):
    module = _dpo_input()
    path = tmp_path / "unreadable.json"
    path.write_text("{}", encoding="utf-8")

    def deny_read_text(self, *args, **kwargs):
        raise PermissionError("fixture denied")

    monkeypatch.setattr(Path, "read_text", deny_read_text)

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    assert str(path.resolve()) in str(caught.value)
    assert isinstance(caught.value.__cause__, PermissionError)


def test_normalizes_invalid_utf8_with_resolved_path(tmp_path):
    module = _dpo_input()
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b'{"text":"\xff"}')

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    assert str(path.resolve()) in str(caught.value)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_jsonl_error_does_not_echo_large_record(tmp_path):
    module = _dpo_input()
    path = tmp_path / "large-invalid.jsonl"
    marker = "SENSITIVE_RECORD_CONTENT_" * 500
    path.write_text('{"payload":"' + marker, encoding="utf-8")

    with pytest.raises(module.DpoInputError) as caught:
        module.load_source_records(path, input_index=0)

    message = str(caught.value)
    assert str(path.resolve()) in message
    assert "line 1" in message.lower()
    assert marker not in message


def test_normalization_public_contract_is_literal_and_immutable(tmp_path):
    module = _dpo_input()
    expected_reasons = (
        "invalid_schema",
        "unsupported_role",
        "conflicting_image_fields",
        "missing_question",
        "missing_chosen",
        "unpaired_turns",
        "missing_image",
        "image_placeholder_mismatch",
        "duplicate",
        "reference_conflict",
        "inference_error",
        "empty_rejected",
        "identical_pair",
        "judge_error",
        "judge_pass",
    )

    assert get_args(module.ReasonCode) == expected_reasons
    image = module.ImageRef(original="image.bin", resolved=tmp_path / "image.bin")
    turn = module.DpoTurn(from_="human", value="question")
    with pytest.raises(FrozenInstanceError):
        image.original = "changed.bin"
    with pytest.raises(FrozenInstanceError):
        turn.value = "changed"


def test_alpaca_instruction_and_nonempty_input_join_with_newline(tmp_path):
    module = _dpo_input()
    value = {
        "instruction": "  keep question whitespace  ",
        "input": "  keep input whitespace  ",
        "output": "x",
        "business_metadata": {"accepted": True},
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert _turn_values(candidate) == [
        ("human", "  keep question whitespace  \n  keep input whitespace  ")
    ]
    assert candidate.chosen == "x"
    assert candidate.source_format == "alpaca"
    assert candidate.turn_index == 0
    assert candidate.turn_count == 1


@pytest.mark.parametrize("input_value", [pytest.param(None, id="null"), "", "  \t"])
def test_alpaca_empty_or_null_input_uses_instruction_only(tmp_path, input_value):
    module = _dpo_input()
    value = {"instruction": "question", "input": input_value, "output": "a"}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert _turn_values(result.candidates[0]) == [("human", "question")]


def test_alpaca_missing_input_uses_instruction_only(tmp_path):
    module = _dpo_input()
    value = {"instruction": "question", "output": "a"}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert _turn_values(result.candidates[0]) == [("human", "question")]


@pytest.mark.parametrize("instruction", [None, "", " \t", 7, [], {}])
def test_alpaca_bad_instruction_is_missing_question(tmp_path, instruction):
    module = _dpo_input()
    value = {"instruction": instruction, "output": "answer"}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].sample_id is None
    assert result.issues[0].reason_code == "missing_question"


@pytest.mark.parametrize("output", [None, "", " \t", 7, [], {}])
def test_alpaca_bad_output_is_missing_chosen(tmp_path, output):
    module = _dpo_input()
    value = {"instruction": "question", "output": output}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].sample_id is None
    assert result.issues[0].reason_code == "missing_chosen"


@pytest.mark.parametrize("input_value", [7, [], {}])
def test_alpaca_nonstring_input_is_invalid_schema(tmp_path, input_value):
    module = _dpo_input()
    value = {"instruction": "question", "input": input_value, "output": "a"}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert [issue.reason_code for issue in result.issues] == ["invalid_schema"]


def test_alpaca_history_expands_every_gold_turn(tmp_path):
    module = _dpo_input()
    value = {
        "instruction": "q3",
        "output": "c",
        "history": [["q1", "a"], ["q2", "b"]],
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert [candidate.chosen for candidate in result.candidates] == ["a", "b", "c"]
    assert [_turn_values(candidate) for candidate in result.candidates] == [
        [("human", "q1")],
        [("human", "q1"), ("gpt", "a"), ("human", "q2")],
        [
            ("human", "q1"),
            ("gpt", "a"),
            ("human", "q2"),
            ("gpt", "b"),
            ("human", "q3"),
        ],
    ]
    assert [candidate.turn_index for candidate in result.candidates] == [0, 1, 2]
    assert [candidate.turn_count for candidate in result.candidates] == [3, 3, 3]


@pytest.mark.parametrize(
    ("history", "reason"),
    [
        (None, "unpaired_turns"),
        ("not-a-list", "unpaired_turns"),
        (["not-a-pair"], "unpaired_turns"),
        ([[]], "unpaired_turns"),
        ([["q", "a", "extra"]], "unpaired_turns"),
        ([{"q": "a"}], "unpaired_turns"),
        ([[7, "a"]], "missing_question"),
        ([["", "a"]], "missing_question"),
        ([["q", 7]], "missing_chosen"),
        ([["q", "  "]], "missing_chosen"),
    ],
)
def test_alpaca_bad_history_rejects_whole_record(tmp_path, history, reason):
    module = _dpo_input()
    value = {"instruction": "current", "output": "z", "history": history}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == reason


@pytest.mark.parametrize("system_value", [None, "", "system prompt", 7])
def test_alpaca_system_key_rejects_the_whole_record(tmp_path, system_value):
    module = _dpo_input()
    value = {"instruction": "q", "output": "a", "system": system_value}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert [issue.reason_code for issue in result.issues] == ["unsupported_role"]


@pytest.mark.parametrize("core_field", ["instruction", "output", "history"])
def test_ambiguous_sharegpt_and_alpaca_record_is_invalid_schema(
    tmp_path, core_field
):
    module = _dpo_input()
    value = {"conversations": [], core_field: [] if core_field == "history" else "x"}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert [issue.reason_code for issue in result.issues] == ["invalid_schema"]


@pytest.mark.parametrize("value", [None, 7, "text", [], {}, {"input": "only"}])
def test_unrecognized_raw_value_has_one_source_level_issue(tmp_path, value):
    module = _dpo_input()

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].sample_id is None
    assert result.issues[0].reason_code == "invalid_schema"


def test_sharegpt_role_aliases_normalize_to_human_and_gpt(tmp_path):
    module = _dpo_input()
    value = {
        "conversations": [
            {"from": "user", "value": "q1", "extra": "allowed"},
            {"from": "assistant", "value": "a"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "b"},
        ],
        "business_metadata": 17,
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert [candidate.source_format for candidate in result.candidates] == [
        "sharegpt",
        "sharegpt",
    ]
    assert [candidate.chosen for candidate in result.candidates] == ["a", "b"]
    assert [_turn_values(candidate) for candidate in result.candidates] == [
        [("human", "q1")],
        [("human", "q1"), ("gpt", "a"), ("human", "q2")],
    ]


def test_sharegpt_every_turn_uses_only_prior_gold_history(tmp_path):
    module = _dpo_input()
    value = {
        "conversations": [
            {"from": "human", "value": "question one"},
            {"from": "gpt", "value": "gold one"},
            {"from": "human", "value": "question two"},
            {"from": "gpt", "value": "gold two"},
            {"from": "human", "value": "question three"},
            {"from": "gpt", "value": "gold three"},
        ]
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert "gold two" not in [
        turn.value for turn in result.candidates[1].conversations
    ]
    assert _turn_values(result.candidates[2]) == [
        ("human", "question one"),
        ("gpt", "gold one"),
        ("human", "question two"),
        ("gpt", "gold two"),
        ("human", "question three"),
    ]
    assert result.candidates[2].chosen == "gold three"


@pytest.mark.parametrize(
    "conversations",
    [
        None,
        "not-a-list",
        [],
        ["not-an-object", {"from": "gpt", "value": "a"}],
        [{"from": 7, "value": "q"}, {"from": "gpt", "value": "a"}],
        [{"from": "human", "value": 7}, {"from": "gpt", "value": "a"}],
        [{"value": "q"}, {"from": "gpt", "value": "a"}],
    ],
)
def test_sharegpt_bad_message_schema_is_invalid_schema(tmp_path, conversations):
    module = _dpo_input()
    value = {"conversations": conversations}

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "invalid_schema"


@pytest.mark.parametrize("role", ["system", "tool", "developer", "USER"])
def test_sharegpt_unsupported_role_rejects_the_whole_record(tmp_path, role):
    module = _dpo_input()
    value = {
        "conversations": [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
            {"from": role, "value": "unsupported"},
        ]
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "unsupported_role"


@pytest.mark.parametrize(
    "conversations",
    [
        [{"from": "human", "value": "q"}],
        [
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "q"},
        ],
        [
            {"from": "human", "value": "q1"},
            {"from": "human", "value": "q2"},
        ],
        [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "gpt", "value": "a2"},
            {"from": "human", "value": "q2"},
        ],
    ],
)
def test_sharegpt_misordered_or_unpaired_turn_rejects_whole_record(
    tmp_path, conversations
):
    module = _dpo_input()

    result = module.normalize_record(
        _raw_record(module, {"conversations": conversations}), image_root=tmp_path
    )

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "unpaired_turns"


@pytest.mark.parametrize(
    ("role", "value", "reason"),
    [
        ("human", "", "missing_question"),
        ("user", "  \t", "missing_question"),
        ("gpt", "", "missing_chosen"),
        ("assistant", "  \t", "missing_chosen"),
    ],
)
def test_sharegpt_blank_turn_has_role_specific_reason(tmp_path, role, value, reason):
    module = _dpo_input()
    if role in {"human", "user"}:
        conversations = [
            {"from": role, "value": value},
            {"from": "gpt", "value": "a"},
        ]
    else:
        conversations = [
            {"from": "human", "value": "q"},
            {"from": role, "value": value},
        ]

    result = module.normalize_record(
        _raw_record(module, {"conversations": conversations}), image_root=tmp_path
    )

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == reason


def test_image_and_images_equal_forms_merge(tmp_path):
    module = _dpo_input()
    (tmp_path / "one.bin").write_bytes(b"one")
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": "one.bin",
        "images": ["one.bin"],
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert [ref.original for ref in result.candidates[0].images] == ["one.bin"]
    assert result.candidates[0].images[0].resolved == (tmp_path / "one.bin").resolve()


@pytest.mark.parametrize(
    ("image_value", "images_value", "reason"),
    [
        ("one.bin", ["two.bin"], "conflicting_image_fields"),
        ("one.bin", [], "conflicting_image_fields"),
        (7, ["one.bin"], "invalid_schema"),
        ("", [], "invalid_schema"),
        ("  ", [], "invalid_schema"),
        ("one.bin", "one.bin", "invalid_schema"),
        ("one.bin", [7], "invalid_schema"),
        ("one.bin", [""], "invalid_schema"),
    ],
)
def test_bad_or_conflicting_image_fields_reject_whole_record(
    tmp_path, image_value, images_value, reason
):
    module = _dpo_input()
    value = {
        "instruction": "question",
        "output": "a",
        "image": image_value,
        "images": images_value,
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].sample_id is None
    assert result.issues[0].reason_code == reason


def test_text_single_image_and_multi_image_modalities(tmp_path):
    module = _dpo_input()
    for name in ("one.bin", "two.bin"):
        (tmp_path / name).write_bytes(name.encode("ascii"))
    values = [
        {"instruction": "text only", "output": "a"},
        {
            "instruction": "<image> single",
            "output": "b",
            "image": "one.bin",
        },
        {
            "instruction": "left <image> middle <image> right",
            "output": "c",
            "images": ["one.bin", "two.bin"],
        },
    ]

    results = [
        module.normalize_record(_raw_record(module, value), image_root=tmp_path)
        for value in values
    ]

    assert all(result.issues == () for result in results)
    assert [[ref.original for ref in result.candidates[0].images] for result in results] == [
        [],
        ["one.bin"],
        ["one.bin", "two.bin"],
    ]


def test_zero_placeholders_prepend_every_image_to_first_human(tmp_path):
    module = _dpo_input()
    for name in ("one.bin", "two.bin"):
        (tmp_path / name).write_bytes(b"content")
    value = {
        "conversations": [
            {"from": "human", "value": "first question"},
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "second question"},
            {"from": "gpt", "value": "b"},
        ],
        "images": ["one.bin", "two.bin"],
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.issues == ()
    assert _turn_values(result.candidates[0]) == [
        ("human", "<image>\n<image>\nfirst question")
    ]
    assert _turn_values(result.candidates[1]) == [
        ("human", "<image>\n<image>\nfirst question"),
        ("gpt", "a"),
        ("human", "second question"),
    ]
    assert [ref.original for ref in result.candidates[0].images] == [
        "one.bin",
        "two.bin",
    ]
    assert [ref.original for ref in result.candidates[1].images] == [
        "one.bin",
        "two.bin",
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"instruction": "<image> question", "output": "a"},
        {
            "instruction": "<image> question",
            "output": "a",
            "images": ["one.bin", "two.bin"],
        },
        {
            "instruction": "<image><image> question",
            "output": "a",
            "image": "one.bin",
        },
    ],
)
def test_nonzero_placeholder_mismatch_rejects_whole_record(tmp_path, value):
    module = _dpo_input()

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].sample_id is None
    assert result.issues[0].reason_code == "image_placeholder_mismatch"


def test_assistant_placeholders_do_not_consume_images(tmp_path):
    module = _dpo_input()
    (tmp_path / "one.bin").write_bytes(b"one")
    without_images = {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "literal <image> in answer"},
        ]
    }
    with_images = {**without_images, "images": ["one.bin"]}

    text_result = module.normalize_record(
        _raw_record(module, without_images), image_root=tmp_path
    )
    image_result = module.normalize_record(
        _raw_record(module, with_images), image_root=tmp_path
    )

    assert text_result.issues == ()
    assert _turn_values(text_result.candidates[0]) == [("human", "question")]
    assert image_result.issues == ()
    assert _turn_values(image_result.candidates[0]) == [
        ("human", "<image>\nquestion")
    ]
    assert image_result.candidates[0].chosen == "literal <image> in answer"


def test_original_image_string_is_preserved_and_root_only_changes_resolution(tmp_path):
    module = _dpo_input()
    root_one = tmp_path / "root-one"
    root_two = tmp_path / "root-two"
    root_one.mkdir()
    root_two.mkdir()
    (root_one / "image.bin").write_bytes(b"one")
    (root_two / "image.bin").write_bytes(b"two")
    original = "nested/../image.bin"
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": original,
    }
    raw = _raw_record(module, value)

    first = module.normalize_record(raw, image_root=root_one)
    second = module.normalize_record(raw, image_root=root_two)

    assert first.issues == second.issues == ()
    assert first.candidates[0].images[0].original == original
    assert second.candidates[0].images[0].original == original
    assert first.candidates[0].images[0].resolved == (root_one / "image.bin").resolve()
    assert second.candidates[0].images[0].resolved == (root_two / "image.bin").resolve()
    assert first.candidates[0].sample_id == second.candidates[0].sample_id
    assert (root_one / "image.bin").read_bytes() == b"one"
    assert (root_two / "image.bin").read_bytes() == b"two"


def test_absolute_image_path_is_not_rebased_under_image_root(tmp_path):
    module = _dpo_input()
    image_path = tmp_path / "absolute.bin"
    image_path.write_bytes(b"absolute")
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": str(image_path.resolve()),
    }

    result = module.normalize_record(
        _raw_record(module, value), image_root=tmp_path / "unused-root"
    )

    assert result.issues == ()
    assert result.candidates[0].images[0].resolved == image_path.resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows path classification")
@pytest.mark.parametrize(
    "original",
    [r"\outside.bin", "/outside.bin", r"C:outside.bin"],
)
def test_windows_rooted_or_drive_relative_images_are_rejected_without_stat(
    tmp_path, monkeypatch, original
):
    module = _dpo_input()
    stat_calls = []

    def accept_any_file(path):
        stat_calls.append(path)
        return True

    monkeypatch.setattr(Path, "is_file", accept_any_file)
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": original,
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "missing_image"
    assert result.issues[0].sample_id is not None
    assert stat_calls == []


def test_future_turn_images_do_not_leak_and_future_missing_image_keeps_early_turn(
    tmp_path,
):
    module = _dpo_input()
    (tmp_path / "first.bin").write_bytes(b"first")
    value = {
        "conversations": [
            {"from": "human", "value": "<image> first"},
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "<image> second"},
            {"from": "gpt", "value": "b"},
        ],
        "images": ["first.bin", "future-missing.bin"],
    }
    raw = _raw_record(module, value)

    missing_result = module.normalize_record(raw, image_root=tmp_path)

    assert len(missing_result.candidates) == 1
    assert missing_result.candidates[0].turn_index == 0
    assert [ref.original for ref in missing_result.candidates[0].images] == [
        "first.bin"
    ]
    assert len(missing_result.issues) == 1
    assert missing_result.issues[0].reason_code == "missing_image"
    assert missing_result.issues[0].sample_id is not None
    assert missing_result.issues[0].sample_id != missing_result.candidates[0].sample_id

    (tmp_path / "future-missing.bin").write_bytes(b"future")
    complete_result = module.normalize_record(raw, image_root=tmp_path)

    assert complete_result.issues == ()
    assert [ref.original for ref in complete_result.candidates[0].images] == [
        "first.bin"
    ]
    assert [ref.original for ref in complete_result.candidates[1].images] == [
        "first.bin",
        "future-missing.bin",
    ]
    assert (
        missing_result.issues[0].sample_id
        == complete_result.candidates[1].sample_id
    )


def test_missing_visible_image_rejects_each_affected_candidate(tmp_path):
    module = _dpo_input()
    (tmp_path / "second.bin").write_bytes(b"second")
    value = {
        "conversations": [
            {"from": "human", "value": "<image> first"},
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "<image> second"},
            {"from": "gpt", "value": "b"},
        ],
        "images": ["first-missing.bin", "second.bin"],
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert [issue.reason_code for issue in result.issues] == [
        "missing_image",
        "missing_image",
    ]
    assert all(issue.sample_id is not None for issue in result.issues)
    assert result.issues[0].sample_id != result.issues[1].sample_id


def test_zero_placeholder_global_missing_image_affects_every_turn(tmp_path):
    module = _dpo_input()
    value = {
        "conversations": [
            {"from": "human", "value": "first"},
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "second"},
            {"from": "gpt", "value": "b"},
        ],
        "images": ["global-missing.bin"],
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert [issue.reason_code for issue in result.issues] == [
        "missing_image",
        "missing_image",
    ]
    assert all(issue.sample_id is not None for issue in result.issues)


def test_image_resolve_value_error_becomes_candidate_issue(
    tmp_path, monkeypatch
):
    module = _dpo_input()
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": "explode.bin",
        "large_business_field": "DO_NOT_ECHO_" * 1000,
    }
    raw = _raw_record(module, value)
    original_resolve = Path.resolve

    def fail_one_resolve(self, *args, **kwargs):
        if self.name == "explode.bin":
            raise ValueError("synthetic resolve failure")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_one_resolve)

    result = module.normalize_record(raw, image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.reason_code == "missing_image"
    assert issue.sample_id is not None
    assert "DO_NOT_ECHO_" not in issue.detail


def test_image_is_file_oserror_becomes_candidate_issue(tmp_path, monkeypatch):
    module = _dpo_input()
    image_path = tmp_path / "check.bin"
    image_path.write_bytes(b"content")
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": "check.bin",
    }
    raw = _raw_record(module, value)
    original_is_file = Path.is_file

    def fail_one_is_file(self):
        if self.name == "check.bin":
            raise OSError("synthetic stat failure")
        return original_is_file(self)

    monkeypatch.setattr(Path, "is_file", fail_one_is_file)

    result = module.normalize_record(raw, image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "missing_image"
    assert result.issues[0].sample_id is not None


def test_invalid_image_path_construction_is_normalized(tmp_path):
    module = _dpo_input()
    value = {
        "instruction": "<image> question",
        "output": "a",
        "image": "bad\x00path.bin",
    }

    result = module.normalize_record(_raw_record(module, value), image_root=tmp_path)

    assert result.candidates == ()
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "missing_image"
    assert result.issues[0].sample_id is not None


def test_sample_ids_are_full_stable_sha256_and_ignore_source_id(tmp_path):
    module = _dpo_input()
    value = {
        "instruction": "question",
        "output": "x",
        "history": [["earlier", "a"]],
    }
    raw = _raw_record(module, value, source_id="first-id")
    same_position_other_id = _raw_record(
        module,
        value,
        source_id="other-id",
        raw_digest=raw.source.raw_digest,
    )

    first = module.normalize_record(raw, image_root=tmp_path)
    repeated = module.normalize_record(raw, image_root=tmp_path)
    renamed = module.normalize_record(same_position_other_id, image_root=tmp_path)

    first_ids = [candidate.sample_id for candidate in first.candidates]
    assert first_ids == [candidate.sample_id for candidate in repeated.candidates]
    assert first_ids == [candidate.sample_id for candidate in renamed.candidates]
    assert len(set(first_ids)) == 2
    assert all(len(sample_id) == 64 for sample_id in first_ids)
    assert all(
        all(character in "0123456789abcdef" for character in sample_id)
        for sample_id in first_ids
    )


def test_same_record_content_from_different_source_position_gets_new_sample_id(
    tmp_path,
):
    module = _dpo_input()
    value = {"instruction": "question", "output": "x"}
    first = module.normalize_record(
        _raw_record(module, value, input_index=0, record_index=0),
        image_root=tmp_path,
    )
    copied = module.normalize_record(
        _raw_record(module, value, input_index=1, record_index=0),
        image_root=tmp_path,
    )
    another_record = module.normalize_record(
        _raw_record(module, value, input_index=0, record_index=1),
        image_root=tmp_path,
    )

    assert first.candidates[0].sample_id != copied.candidates[0].sample_id
    assert first.candidates[0].sample_id != another_record.candidates[0].sample_id


def test_jsonl_line_number_participates_in_sample_id(tmp_path):
    module = _dpo_input()
    value = {"instruction": "question", "output": "x"}
    first = module.normalize_record(
        _raw_record(module, value, line_number=1), image_root=tmp_path
    )
    later_line = module.normalize_record(
        _raw_record(module, value, line_number=7), image_root=tmp_path
    )

    assert first.candidates[0].sample_id != later_line.candidates[0].sample_id


def test_normalize_inputs_preserves_order_and_reports_exact_file_summaries(tmp_path):
    module = _dpo_input()
    first_path = tmp_path / "first.data"
    empty_array_path = tmp_path / "empty.data"
    third_path = tmp_path / "third.data"
    first_values = [
        {
            "instruction": "current",
            "output": "b",
            "history": [["prior", "a"]],
        },
        17,
    ]
    first_bytes = json.dumps(first_values, separators=(",", ":")).encode("utf-8")
    empty_bytes = b"[\r\n]"
    third_bytes = (
        b'{"instruction":"last","output":"c"}\r\n'
        b"\r\n"
    )
    first_path.write_bytes(first_bytes)
    empty_array_path.write_bytes(empty_bytes)
    third_path.write_bytes(third_bytes)

    result = module.normalize_inputs(
        [first_path, empty_array_path, third_path], image_root=tmp_path
    )

    assert [candidate.chosen for candidate in result.candidates] == ["a", "b", "c"]
    assert [candidate.source.input_index for candidate in result.candidates] == [
        0,
        0,
        2,
    ]
    assert [candidate.turn_index for candidate in result.candidates] == [0, 1, 0]
    assert len(result.issues) == 1
    assert result.issues[0].reason_code == "invalid_schema"
    assert result.issues[0].source.input_index == 0
    assert result.issues[0].source.record_index == 1
    assert [summary.source_path for summary in result.inputs] == [
        first_path.resolve(),
        empty_array_path.resolve(),
        third_path.resolve(),
    ]
    assert [summary.sha256 for summary in result.inputs] == [
        hashlib.sha256(first_bytes).hexdigest(),
        hashlib.sha256(empty_bytes).hexdigest(),
        hashlib.sha256(third_bytes).hexdigest(),
    ]
    assert [summary.byte_count for summary in result.inputs] == [
        len(first_bytes),
        len(empty_bytes),
        len(third_bytes),
    ]
    assert [summary.container_format for summary in result.inputs] == [
        "json_array",
        "json_array",
        "json_object",
    ]
    assert [summary.record_count for summary in result.inputs] == [2, 0, 1]


def test_normalize_inputs_keeps_jsonl_container_and_physical_line_summary(tmp_path):
    module = _dpo_input()
    path = tmp_path / "records.data"
    payload = (
        b'\n{"instruction":"one","output":"a"}\n'
        b'  \n{"instruction":"two","output":"b"}\n'
    )
    path.write_bytes(payload)

    result = module.normalize_inputs([path], image_root=tmp_path)

    assert result.inputs[0].container_format == "jsonl"
    assert result.inputs[0].record_count == 2
    assert [candidate.source.line_number for candidate in result.candidates] == [2, 4]


def test_normalize_inputs_wraps_image_root_resolution_error(
    tmp_path, monkeypatch
):
    module = _dpo_input()
    input_path = tmp_path / "record.json"
    input_path.write_text(
        json.dumps(
            {
                "instruction": "<image> question",
                "output": "a",
                "image": "one.bin",
            }
        ),
        encoding="utf-8",
    )
    image_root = tmp_path / "bad-image-root"
    original_resolve = Path.resolve

    def fail_root_resolve(self, *args, **kwargs):
        if self == image_root:
            raise ValueError("synthetic image-root failure")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_root_resolve)

    with pytest.raises(module.DpoInputError) as caught:
        module.normalize_inputs([input_path], image_root=image_root)

    assert "image root" in str(caught.value).lower()
    assert isinstance(caught.value.__cause__, ValueError)


def test_exact_duplicate_group_keeps_first_and_audits_every_later_copy(tmp_path):
    module = _dpo_input()
    value = {"instruction": "question", "output": "x"}
    candidates = tuple(
        module.normalize_record(
            _raw_record(module, value, input_index=index), image_root=tmp_path
        ).candidates[0]
        for index in range(3)
    )

    result = module.deduplicate_candidates(candidates)

    assert result.candidates == (candidates[0],)
    assert [issue.reason_code for issue in result.issues] == [
        "duplicate",
        "duplicate",
    ]
    assert [issue.sample_id for issue in result.issues] == [
        candidates[1].sample_id,
        candidates[2].sample_id,
    ]
    assert [issue.source for issue in result.issues] == [
        candidates[1].source,
        candidates[2].source,
    ]


def test_reference_conflict_excludes_every_source_in_group(tmp_path):
    module = _dpo_input()
    values = [
        {"instruction": "question", "output": "x"},
        {"instruction": "question", "output": "y"},
        {"instruction": "question", "output": "x"},
    ]
    candidates = tuple(
        module.normalize_record(
            _raw_record(module, value, input_index=index), image_root=tmp_path
        ).candidates[0]
        for index, value in enumerate(values)
    )

    result = module.deduplicate_candidates(candidates)

    assert result.candidates == ()
    assert [issue.reason_code for issue in result.issues] == [
        "reference_conflict",
        "reference_conflict",
        "reference_conflict",
    ]
    assert [issue.sample_id for issue in result.issues] == [
        candidate.sample_id for candidate in candidates
    ]


def test_deduplication_uses_original_image_strings_not_resolved_paths(tmp_path):
    module = _dpo_input()
    source = _raw_record(module, {"instruction": "q", "output": "a"}).source
    conversation = (module.DpoTurn(from_="human", value="q"),)
    first = module.DpoCandidate(
        sample_id="1" * 64,
        conversations=conversation,
        chosen="a",
        images=(module.ImageRef("same.bin", tmp_path / "root-one" / "same.bin"),),
        source=source,
        source_format="alpaca",
        turn_index=0,
        turn_count=1,
    )
    second = module.DpoCandidate(
        sample_id="2" * 64,
        conversations=conversation,
        chosen="a",
        images=(module.ImageRef("same.bin", tmp_path / "root-two" / "same.bin"),),
        source=source,
        source_format="alpaca",
        turn_index=0,
        turn_count=1,
    )
    different_original = module.DpoCandidate(
        sample_id="3" * 64,
        conversations=conversation,
        chosen="a",
        images=(module.ImageRef("other.bin", tmp_path / "root-one" / "same.bin"),),
        source=source,
        source_format="alpaca",
        turn_index=0,
        turn_count=1,
    )

    result = module.deduplicate_candidates([first, second, different_original])

    assert result.candidates == (first, different_original)
    assert [issue.sample_id for issue in result.issues] == [second.sample_id]
    assert [issue.reason_code for issue in result.issues] == ["duplicate"]


def test_deduplication_is_exact_without_trimming_and_keeps_stable_order(tmp_path):
    module = _dpo_input()
    values = [
        {"instruction": "q1", "output": "a"},
        {"instruction": "q2", "output": "b"},
        {"instruction": "q1", "output": "a "},
        {"instruction": "q3", "output": "c"},
    ]
    candidates = tuple(
        module.normalize_record(
            _raw_record(module, value, input_index=index), image_root=tmp_path
        ).candidates[0]
        for index, value in enumerate(values)
    )

    result = module.deduplicate_candidates(candidates)

    assert result.candidates == (candidates[1], candidates[3])
    assert [issue.reason_code for issue in result.issues] == [
        "reference_conflict",
        "reference_conflict",
    ]
    assert [issue.sample_id for issue in result.issues] == [
        candidates[0].sample_id,
        candidates[2].sample_id,
    ]
