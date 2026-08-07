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
