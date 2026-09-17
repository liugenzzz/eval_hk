"""Optional roots must still deliver actual image bytes to inference and judging."""
import base64
import hashlib
import json
from pathlib import Path

import pytest

from eval_tool.io import load_truth_dataset
from eval_tool.industrial_frozen_check import verify_frozen_dataset


def make_truth(tmp_path, image):
    record = {"id": "p1", "images": [str(image)], "conversations": [
        {"from": "human", "value": "<image>describe"},
        {"from": "gpt", "value": "reference"}]}
    target = tmp_path / "sample.jsonl"
    target.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return target


@pytest.mark.parametrize("params", [{}, {"image_root": None}, {"image_root": ""}])
@pytest.mark.parametrize("absolute", [True, False])
def test_images_load_with_omitted_null_or_empty_root(tmp_path, monkeypatch, params, absolute):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"actual image payload")
    make_truth(tmp_path, image if absolute else image.name)
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    frame = load_truth_dataset(tmp_path, "sample", params)
    assert base64.b64decode(frame.iloc[0]["image"]) == b"actual image payload"


def test_explicit_root_overrides_relative_base(tmp_path):
    root = tmp_path / "external"; root.mkdir()
    (root / "pic.jpg").write_bytes(b"external")
    (tmp_path / "pic.jpg").write_bytes(b"local")
    make_truth(tmp_path, "pic.jpg")
    frame = load_truth_dataset(tmp_path, "sample", {"image_root": str(root)})
    assert base64.b64decode(frame.iloc[0]["image"]) == b"external"


def test_explicit_root_does_not_rewrite_absolute_images(tmp_path):
    image = tmp_path / "pic.jpg"; image.write_bytes(b"absolute")
    make_truth(tmp_path, image)
    frame = load_truth_dataset(tmp_path, "sample", {"image_root": str(tmp_path / "wrong")})
    assert base64.b64decode(frame.iloc[0]["image"]) == b"absolute"


@pytest.mark.parametrize("image_field", ["images", "image"])
def test_two_absolute_directories_keep_same_filename_distinct_without_root(tmp_path, image_field):
    records = []
    expected = [b"original equipment image", b"revised equipment image"]
    for index, (directory, payload) in enumerate(zip(["zb_images", "zb_images_v1"], expected)):
        root = tmp_path / directory
        root.mkdir()
        image = root / "001.jpg"
        image.write_bytes(payload)
        records.append({
            "id": f"p{index}", image_field: [str(image.resolve())],
            "conversations": [
                {"from": "human", "value": "<image>describe"},
                {"from": "gpt", "value": "reference"}],
        })
    (tmp_path / "sample.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    frame = load_truth_dataset(tmp_path, "sample")

    assert len(frame) == 2
    assert [base64.b64decode(value) for value in frame["image"]] == expected


def test_code_scoring_still_skips_image_access_and_missing_image_fails_for_inference(tmp_path):
    make_truth(tmp_path, "missing.jpg")
    assert load_truth_dataset(tmp_path, "sample", need_images=False).iloc[0]["image"] == ""
    with pytest.raises(FileNotFoundError):
        load_truth_dataset(tmp_path, "sample")


def test_frozen_check_resolves_relative_images_like_loader(tmp_path, monkeypatch):
    image = tmp_path / "pic.jpg"; image.write_bytes(b"payload")
    source = make_truth(tmp_path, image.name)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_ids": ["p1"], "image_hashes": {"p1": hashlib.sha256(image.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(); monkeypatch.chdir(elsewhere)
    assert verify_frozen_dataset(source)["records"] == 1
