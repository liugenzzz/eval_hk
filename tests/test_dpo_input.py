from __future__ import annotations

import hashlib
import importlib
import json
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
