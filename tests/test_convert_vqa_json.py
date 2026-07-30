import base64
import json

from eval_tool.convert_vqa_json import convert, record_to_rows
from eval_tool.history import format_history_for_judge, parse_history_cell
from eval_tool.imaging import decode_image_cell
from eval_tool.infer import build_chat_messages, _row_prompt


def test_single_turn_single_image(tmp_path):
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    record = {
        "id": "rec_1",
        "image": [str(image_path)],
        "conversations": [
            {"from": "human", "value": "<image>\n这是什么？"},
            {"from": "gpt", "value": "一艘船。"},
        ],
    }

    rows = record_to_rows(record, {})

    assert len(rows) == 1
    row = rows[0]
    assert row["index"] == "rec_1__t0"
    assert row["question"] == "这是什么？"
    assert row["answer"] == "一艘船。"
    assert row["history"] == ""
    assert row["category"] == "单轮"
    assert decode_image_cell(row["image"]) == [base64.b64encode(b"fake-image-bytes").decode("ascii")]


def test_multi_turn_stores_structured_history_and_cumulative_images(tmp_path):
    img1 = tmp_path / "1.jpg"
    img2 = tmp_path / "2.jpg"
    img1.write_bytes(b"one")
    img2.write_bytes(b"two")
    record = {
        "id": "rec_2",
        "image": [str(img1), str(img2)],
        "conversations": [
            {"from": "human", "value": "<image>\n第一艘船是什么？"},
            {"from": "gpt", "value": "驱逐舰。"},
            {"from": "human", "value": "<image>\n第二艘呢？"},
            {"from": "gpt", "value": "护卫舰。"},
        ],
    }

    rows = record_to_rows(record, {})

    assert len(rows) == 2
    assert rows[0]["question"] == "第一艘船是什么？"
    assert rows[0]["history"] == ""
    assert rows[0]["category"] == "单轮"
    assert rows[1]["category"] == "多轮"
    assert decode_image_cell(rows[0]["image"]) == [base64.b64encode(b"one").decode("ascii")]

    # second turn: clean question, structured history, cumulative images
    assert rows[1]["question"] == "第二艘呢？"
    history = parse_history_cell(rows[1]["history"])
    assert history == [{"q": "第一艘船是什么？", "a": "驱逐舰。", "n_img": 1}]
    assert decode_image_cell(rows[1]["image"]) == [
        base64.b64encode(b"one").decode("ascii"),
        base64.b64encode(b"two").decode("ascii"),
    ]
    assert rows[1]["answer"] == "护卫舰。"


def test_row_prompt_builds_chat_turns_with_gold_history():
    row = {
        "question": "第二艘呢？",
        "history": json.dumps([{"q": "第一艘船是什么？", "a": "驱逐舰。", "n_img": 1}], ensure_ascii=False),
    }

    prompt = _row_prompt("vqa", row, {})

    assert isinstance(prompt, list)
    assert [t["role"] for t in prompt] == ["user", "assistant", "user"]
    assert prompt[0]["text"] == "第一艘船是什么？"
    assert prompt[0]["n_img"] == 1
    assert prompt[1]["text"] == "驱逐舰。"
    assert "第二艘呢？" in prompt[2]["text"]


def test_build_chat_messages_distributes_images_across_turns():
    turns = [
        {"role": "user", "text": "第一问", "n_img": 1},
        {"role": "assistant", "text": "第一答"},
        {"role": "user", "text": "第二问"},
    ]
    images = ["IMG1", "IMG2", "IMG3"]

    messages = build_chat_messages(turns, images)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    first_user_images = [c["image"] for c in messages[0]["content"] if c["type"] == "image"]
    assert first_user_images == ["IMG1"]
    # remaining images all attach to the final user turn
    last_user_images = [c["image"] for c in messages[2]["content"] if c["type"] == "image"]
    assert last_user_images == ["IMG2", "IMG3"]
    assert messages[1]["content"] == [{"type": "text", "text": "第一答"}]


def test_build_chat_messages_single_turn_string():
    messages = build_chat_messages("这是什么？", ["IMG1"])

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert [c["image"] for c in messages[0]["content"] if c["type"] == "image"] == ["IMG1"]


def test_format_history_for_judge():
    history = [{"q": "第一问", "a": "第一答", "n_img": 1}]

    text = format_history_for_judge(history, "第二问")

    assert "第1轮问：第一问" in text
    assert "第1轮答：第一答" in text
    assert text.endswith("当前问题：第二问")
    assert format_history_for_judge([], "只有一问") == "只有一问"


def test_convert_writes_tsv(tmp_path):
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake")
    records = [
        {
            "id": "rec_1",
            "image": [str(image_path)],
            "conversations": [
                {"from": "human", "value": "<image>\n这是什么？"},
                {"from": "gpt", "value": "一艘船。"},
            ],
        }
    ]
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "out"

    out_path = convert(input_json, out_dir)

    assert out_path.exists()
    header = out_path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "history" in header.split("\t")
