"""Video rows: carried as paths, routed to the video generator, never answered blind."""

import json

import pytest

from eval_tool.eval_set import record_to_rows
from eval_tool.infer import PredictionBatchError, _video_list, generate_predictions

RECORD = {
    "id": "clip1",
    "videos": ["/data/clip1.mp4"],
    "conversations": [{"from": "human", "value": "<video>\n视频里的人在做什么？"},
                      {"from": "gpt", "value": "在跑步。"}],
    "metadata": {"task_type": "park_open", "behavior_category": "移动运动类"},
}


def test_video_path_is_carried_and_tag_removed():
    row = record_to_rows(RECORD)[0]
    assert json.loads(row["video_files"]) == ["/data/clip1.mp4"]
    assert row["question"] == "视频里的人在做什么？"


def test_image_records_get_no_video_column():
    record = {"id": "i", "images": [], "conversations": RECORD["conversations"], "metadata": {}}
    assert "video_files" not in record_to_rows(record)[0]


class VideoGenerator:
    def __init__(self):
        self.calls = []

    def generate_batch(self, prompts, images):
        raise AssertionError("video rows must not take the image path")

    def generate_video_batch(self, prompts, videos):
        self.calls.append((prompts, videos))
        return ["跑步"] * len(prompts)


class ImageOnlyGenerator:
    def generate_batch(self, prompts, images):
        return ["瞎猜"] * len(prompts)


def _rows(*videos):
    rows = []
    for i, video in enumerate(videos):
        row = {"index": f"r{i}", "question": "在做什么？", "image": ""}
        if video is not None:
            row["video_files"] = json.dumps([video])
        rows.append(row)
    return rows


def test_video_rows_go_to_the_video_generator():
    gen = VideoGenerator()
    out = generate_predictions(_rows("/a.mp4", "/b.mp4"), "k", {}, gen, batch_size=2,
                               prompt_texts={"k": "{question}"})
    assert out == ["跑步", "跑步"]
    assert gen.calls[0][1] == [["/a.mp4"], ["/b.mp4"]]


def test_a_generator_without_video_support_is_an_error_not_a_blind_answer():
    with pytest.raises(PredictionBatchError):
        generate_predictions(_rows("/a.mp4"), "k", {}, ImageOnlyGenerator(), prompt_texts={"k": "{question}"})


def test_mixed_batch_is_rejected():
    with pytest.raises(PredictionBatchError):
        generate_predictions(_rows("/a.mp4", None), "k", {}, VideoGenerator(), batch_size=2,
                             prompt_texts={"k": "{question}"})


@pytest.mark.parametrize("cell, expected", [
    ('["/a.mp4", "/b.mp4"]', ["/a.mp4", "/b.mp4"]),
    ("/a.mp4", ["/a.mp4"]),
    ("", []),
    (None, []),
    (float("nan"), []),
])
def test_video_list(cell, expected):
    assert _video_list(cell) == expected
