"""评估集 test.jsonl 直读：按轮次拆行、metadata 扁平化、按任务/轮次选取。"""

import json

import pandas as pd
import pytest

from eval_tool.eval_set import load_eval_set, selection_matches, sha256_of
from eval_tool.io import load_truth_dataset, truth_path

INVENTORY = {
    "id": "img1_inventory_locate_23",
    "images": ["a.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\n图中清晰可见的目标都有什么？"},
        {"from": "gpt", "value": "1名人员、3辆卡车。"},
        {"from": "human", "value": "给出那名人员的坐标。"},
        {"from": "gpt", "value": '{"bbox_2d":[472,789,491,857],"label":"人员"}'},
        {"from": "human", "value": "它长什么样？"},
        {"from": "gpt", "value": "深红色车身。"},
    ],
    "metadata": {
        "task_type": "inventory_locate", "difficulty": "medium", "size_bucket": "small",
        "inventory": ["人员x1", "卡车x3"], "bbox_scale": 1000, "label": "人员",
        "source_image": "a.jpg", "n_turns": 3,
    },
}

GROUND = {
    "id": "img2_ground_appearance_0",
    "images": ["b.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\n请给出银灰色的三轮车的边界框。"},
        {"from": "gpt", "value": '{"bbox_2d":[361,897,395,952],"label":"三轮车"}'},
        {"from": "human", "value": "这辆三轮车本身长什么样？"},
        {"from": "gpt", "value": "深红色车身。"},
    ],
    "metadata": {
        "task_type": "ground_appearance", "difficulty": "hard", "size_bucket": "large",
        "describe_kind": "appearance", "bbox_scale": 1000, "label": "三轮车",
        "source_image": "b.jpg", "n_turns": 2,
    },
}


@pytest.fixture
def eval_set(tmp_path):
    path = tmp_path / "eval_set_v1.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in (INVENTORY, GROUND)) + "\n",
        encoding="utf-8",
    )
    return path


def test_multi_turn_records_become_one_row_per_turn(eval_set):
    """一条 inventory_locate 三轮各答一种东西，分别落在计数、单框、描述三个组，
    用三个不同的打分器 —— 整条归一组就没法打分。"""
    frame = load_eval_set(eval_set)
    inventory = frame[frame["task_type"] == "inventory_locate"]
    assert inventory["turn"].tolist() == [1, 2, 3]
    assert inventory["index"].tolist() == [
        "img1_inventory_locate_23__t1",
        "img1_inventory_locate_23__t2",
        "img1_inventory_locate_23__t3",
    ]


def test_history_is_gold_replay_in_the_shape_the_infer_path_expects(eval_set):
    frame = load_eval_set(eval_set)
    turn2 = frame[frame["index"] == "img1_inventory_locate_23__t2"].iloc[0]
    history = json.loads(turn2["history"])
    assert history == [{"q": "图中清晰可见的目标都有什么？", "a": "1名人员、3辆卡车。", "n_img": 1}]
    turn1 = frame[frame["index"] == "img1_inventory_locate_23__t1"].iloc[0]
    assert turn1["history"] == ""


def test_metadata_is_flattened_onto_meta_columns(eval_set):
    """报表要按 task_type / difficulty / size_bucket / describe_kind 拆开看，
    摘掉 metadata 就拆不了。"""
    frame = load_eval_set(eval_set)
    row = frame[frame["task_type"] == "ground_appearance"].iloc[0]
    assert row["meta.difficulty"] == "hard"
    assert row["meta.size_bucket"] == "large"
    assert row["meta.describe_kind"] == "appearance"
    assert row["meta.bbox_scale"] == 1000


def test_nested_metadata_values_survive_intact(eval_set):
    frame = load_eval_set(eval_set)
    row = frame[frame["task_type"] == "inventory_locate"].iloc[0]
    assert row["meta.inventory"] == ["人员x1", "卡车x3"]


def test_image_tag_is_stripped_from_the_question(eval_set):
    frame = load_eval_set(eval_set)
    assert "<image>" not in " ".join(frame["question"].tolist())


def test_select_picks_one_task_and_one_turn(eval_set):
    """A 组要 7 个 ground_* 的轮 1，外加 inventory_locate 的轮 2。"""
    frame = load_eval_set(
        eval_set,
        select=[
            {"task_type": ["ground_appearance"], "turn": 1},
            {"task_type": ["inventory_locate"], "turn": 2},
        ],
    )
    assert sorted(frame["index"]) == [
        "img1_inventory_locate_23__t2",
        "img2_ground_appearance_0__t1",
    ]


def test_select_by_turn_alone_takes_that_turn_of_every_task(eval_set):
    frame = load_eval_set(eval_set, select=[{"turn": 1}])
    assert frame["turn"].unique().tolist() == [1]
    assert len(frame) == 2


def test_a_select_that_matches_nothing_raises_instead_of_returning_empty(eval_set):
    """选空了是配置写错了（任务名拼错、轮次填反）。静默返回空表会在报表里变成一格
    「样本不足」，而那格实际上是 bug。"""
    with pytest.raises(ValueError, match="没有选中任何样本"):
        load_eval_set(eval_set, select=[{"task_type": ["ground_unique"], "turn": 1}])


def test_selection_matches_treats_rules_as_or():
    row = {"task_type": "detect_class", "turn": 1}
    assert selection_matches(row, [{"task_type": ["inventory_locate"]}, {"task_type": ["detect_class"]}])
    assert not selection_matches(row, [{"task_type": ["inventory_locate"]}])
    assert selection_matches(row, None)


def test_images_are_left_unencoded_without_an_image_root(eval_set):
    """代码打分器压根不看图，为了跑一次坐标打分把几个 G 的图读进内存没有道理。"""
    frame = load_eval_set(eval_set)
    assert frame["image"].tolist() == ["", "", "", "", ""]
    assert frame["image_files"].iloc[0] == "a.jpg"


def test_images_are_encoded_when_a_root_is_given(tmp_path, eval_set):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8fake")
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8fake2")
    frame = load_eval_set(eval_set, image_root=tmp_path)
    assert frame["image"].iloc[0]


def test_sha256_is_stable_for_the_same_bytes(eval_set, tmp_path):
    """§13 评估集冻结：不同 checkpoint 之间可比的前提是评的是同一批样本。"""
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(eval_set.read_bytes())
    assert sha256_of(eval_set) == sha256_of(copy)


def test_truth_loading_prefers_jsonl_over_tsv(tmp_path, eval_set):
    eval_set.rename(tmp_path / "eval_set_v1.jsonl")
    pd.DataFrame([{"index": "x", "question": "q", "answer": "a"}]).to_csv(
        tmp_path / "eval_set_v1.tsv", sep="\t", index=False
    )
    assert truth_path(tmp_path, "eval_set_v1").suffix == ".jsonl"
    frame = load_truth_dataset(tmp_path, "eval_set_v1", {"select": [{"turn": 1}]})
    assert len(frame) == 2
    assert "meta.difficulty" in frame.columns


def test_json_array_input_is_accepted_too(tmp_path):
    path = tmp_path / "eval_set_v1.json"
    path.write_text(json.dumps([GROUND], ensure_ascii=False), encoding="utf-8")
    assert len(load_eval_set(path)) == 2
