"""派生评估集：模型历史、双向一致性、问法扰动。

这三件事全部做成「生成一份新的 jsonl，然后当普通数据集跑」，推理引擎一个字符不改 ——
装备和书籍走的是同一条 run_infer，那里面的断点续传改坏了不是报错，是几小时的 GPU
跑到一半续不上。
"""

import json

import pandas as pd
import pytest

from eval_tool.derive import (
    derive_model_history,
    derive_question_perturbation,
    derive_reverse_consistency,
    load_predictions,
    write_jsonl,
)
from eval_tool.eval_set import load_eval_set
from eval_tool.phrase_pool import PhrasePool, format_bbox

GROUND = {
    "id": "img1_ground_appearance_0",
    "images": ["img1.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\n请给出银灰色的三轮车的边界框。"},
        {"from": "gpt", "value": '{"bbox_2d":[100,100,200,200],"label":"三轮车"}'},
        {"from": "human", "value": "这辆三轮车本身长什么样？"},
        {"from": "gpt", "value": "深红色车身，支着遮阳篷。"},
    ],
    "metadata": {"task_type": "ground_appearance", "label": "三轮车", "attribute": "银灰色",
                 "describe_kind": "appearance", "bbox_scale": 1000, "n_turns": 2},
}

INVENTORY = {
    "id": "img2_inventory_locate_1",
    "images": ["img2.jpg"],
    "conversations": [
        {"from": "human", "value": "<image>\n图中清晰可见的目标都有什么？"},
        {"from": "gpt", "value": "1名人员、3辆卡车。"},
        {"from": "human", "value": "给出那名人员的坐标。"},
        {"from": "gpt", "value": '{"bbox_2d":[400,400,450,500],"label":"人员"}'},
        {"from": "human", "value": "它长什么样？"},
        {"from": "gpt", "value": "穿深色外套。"},
    ],
    "metadata": {"task_type": "inventory_locate", "label": "人员",
                 "inventory": ["人员x1", "卡车x3"], "bbox_scale": 1000, "n_turns": 3},
}


@pytest.fixture
def pools(tmp_path):
    ground = tmp_path / "ground_attribute.txt"
    ground.write_text(
        "# 注释\n#! require-any: 框 坐标\n"
        "框出图中{attribute}的{label}。\n"
        "输出{attribute}的{label}的检测框。\n"
        "请给出{attribute}的{label}的边界框。\n"
        "找出图中{attribute}的{label}，给出坐标。\n"
        "定位图中{attribute}的那{mw}{label}。\n",
        encoding="utf-8",
    )
    region = tmp_path / "region_identify.txt"
    region.write_text(
        "# 注释\n#! require-any: 什么\n"
        "图中 {bbox} 区域内的是什么？\n"
        "图中 {bbox} 框内的是什么目标？\n",
        encoding="utf-8",
    )
    return {"ground": PhrasePool.load(ground), "region": PhrasePool.load(region)}


# --------------------------------------------------------- 模型历史（阶段 5）

def test_model_history_replaces_the_gold_history_with_the_models_own_output():
    """第 1 轮只跑一次，gold 历史和 model 历史共用它的结果。换的是**上文**，
    不是真值 —— 目标轮的标准答案保持不变。"""
    predictions = {"img1_ground_appearance_0__t1": '{"bbox_2d":[500,500,600,600]}'}
    derived = derive_model_history([GROUND], predictions)
    assert len(derived) == 1
    record = derived[0]
    assert record["id"] == "img1_ground_appearance_0__mh"
    assert record["conversations"][1]["value"] == '{"bbox_2d":[500,500,600,600]}'
    assert record["conversations"][3]["value"] == "深红色车身，支着遮阳篷。"
    assert record["metadata"]["history_mode"] == "model"
    assert record["metadata"]["derived_from"] == "img1_ground_appearance_0"


def test_model_history_truncates_at_the_target_turn():
    predictions = {
        "img2_inventory_locate_1__t1": "1名人员、2辆卡车。",
        "img2_inventory_locate_1__t2": '{"bbox_2d":[410,405,455,505]}',
    }
    derived = derive_model_history([INVENTORY], predictions)[0]
    assert derived["metadata"]["target_turn"] == 3
    assert len(derived["conversations"]) == 6
    assert derived["conversations"][1]["value"] == "1名人员、2辆卡车。"
    assert derived["conversations"][3]["value"] == '{"bbox_2d":[410,405,455,505]}'


def test_a_record_missing_one_turns_prediction_is_skipped_entirely():
    """用 gold 补一半、model 补一半算出来的衰减率，分子分母不是同一件事。"""
    predictions = {"img2_inventory_locate_1__t1": "1名人员。"}   # 缺轮 2
    assert derive_model_history([INVENTORY], predictions) == []


def test_target_turn_can_be_pinned_per_task():
    predictions = {"img2_inventory_locate_1__t1": "1名人员。"}
    derived = derive_model_history([INVENTORY], predictions,
                                   target_turns={"inventory_locate": 2})[0]
    assert derived["metadata"]["target_turn"] == 2
    assert len(derived["conversations"]) == 4


def test_sampling_is_deterministic_across_runs_and_orderings():
    """评估集加一条样本、记录顺序变了，抽中的那一批也不该跟着变 —— 否则两个
    checkpoint 的链路衰减率算在不同的子集上，没法比。"""
    records = [dict(GROUND, id=f"s{i}",
                    metadata=dict(GROUND["metadata"])) for i in range(200)]
    predictions = {f"s{i}__t1": '{"bbox_2d":[1,2,3,4]}' for i in range(200)}
    first = {r["metadata"]["derived_from"] for r in derive_model_history(records, predictions, sample_ratio=0.3)}
    shuffled = list(reversed(records))
    second = {r["metadata"]["derived_from"] for r in derive_model_history(shuffled, predictions, sample_ratio=0.3)}
    assert first == second
    assert 0.2 < len(first) / 200 < 0.4


def test_single_turn_records_have_no_history_to_replace():
    single = {"id": "x", "images": [], "metadata": {"task_type": "detect_class"},
              "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
    assert derive_model_history([single], {}) == []


# ----------------------------------------------------- 双向一致性（阶段 9.1）

def test_reverse_consistency_asks_about_the_models_own_box(pools):
    """正向问「框出白色面包车」得到框 B，反向拿 B 问「这个框里是什么」。
    不需要任何额外标注 —— 数据集本来就同时有 ground_* 和 region_identify。"""
    predictions = {"img1_ground_appearance_0__t1": '{"bbox_2d":[110,105,205,195]}'}
    derived = derive_reverse_consistency([GROUND], predictions,
                                         question_pool=pools["region"])[0]
    assert format_bbox([110, 105, 205, 195]) in derived["conversations"][0]["value"]
    assert derived["conversations"][1]["value"] == "该区域内的是三轮车。"
    assert derived["metadata"]["task_type"] == "reverse_consistency"
    assert derived["metadata"]["label"] == "三轮车"


def test_reverse_consistency_records_the_forward_iou_without_filtering_on_it(pools):
    """正向框错时反向答「卡车」对那个框来说是对的，但和原来的指代对不上 ——
    那恰恰是要测的失败模式，不能把它筛掉。"""
    predictions = {"img1_ground_appearance_0__t1": '{"bbox_2d":[800,800,900,900]}'}
    derived = derive_reverse_consistency([GROUND], predictions,
                                         question_pool=pools["region"])
    assert len(derived) == 1
    assert derived[0]["metadata"]["forward_iou"] == 0.0


def test_an_unparseable_forward_box_is_dropped(pools):
    """正向根本没框出来，反向问无从问起。它已经被正向的格式合规率记过一次，
    再记一次是重复惩罚。"""
    predictions = {"img1_ground_appearance_0__t1": "图中没有这样的目标。"}
    assert derive_reverse_consistency([GROUND], predictions, question_pool=pools["region"]) == []


def test_a_forward_box_that_clips_to_zero_area_is_dropped(pools):
    """模型的框整个飞出画面（x1、x2 都 > 1000）时，裁剪后两边都贴到边界，退化成零宽。
    拿它去问「这个区域里是什么」问的是一块零面积的地方，模型答什么都没有意义 ——
    而这条样本在正向已经被记过一次错了。实测这种占到 11%。"""
    predictions = {"img1_ground_appearance_0__t1": '{"bbox_2d":[1200,273,1300,324]}'}
    assert derive_reverse_consistency([GROUND], predictions, question_pool=pools["region"]) == []


def test_a_forward_box_that_merely_overflows_is_kept(pools):
    """只是部分越界、裁剪后仍有面积的，照常问 —— 越界只计数不扣分。"""
    predictions = {"img1_ground_appearance_0__t1": '{"bbox_2d":[900,100,1200,300]}'}
    derived = derive_reverse_consistency([GROUND], predictions, question_pool=pools["region"])
    assert len(derived) == 1
    assert derived[0]["metadata"]["forward_box"] == [900.0, 100.0, 1000.0, 300.0]


def test_reverse_consistency_can_be_limited_to_certain_tasks(pools):
    predictions = {
        "img1_ground_appearance_0__t1": '{"bbox_2d":[110,105,205,195]}',
        "img2_inventory_locate_1__t1": '{"bbox_2d":[1,2,3,4]}',
    }
    derived = derive_reverse_consistency([GROUND, INVENTORY], predictions,
                                         question_pool=pools["region"],
                                         tasks=["ground_appearance"])
    assert [r["metadata"]["forward_task_type"] for r in derived] == ["ground_appearance"]


# ------------------------------------------------------- 问法扰动（阶段 9.2）

def test_perturbation_makes_one_record_per_phrasing(pools):
    derived = derive_question_perturbation([GROUND], pools={"ground_appearance": pools["ground"]})
    assert len(derived) == 3
    questions = [r["conversations"][0]["value"] for r in derived]
    assert len(set(questions)) == 3
    assert all(r["metadata"]["perturb_group"] == "img1_ground_appearance_0" for r in derived)
    assert sorted(r["metadata"]["perturb_variant"] for r in derived) == [0, 1, 2]


def test_perturbation_keeps_the_gold_box_as_the_answer(pools):
    derived = derive_question_perturbation([GROUND], pools={"ground_appearance": pools["ground"]})
    assert all(r["conversations"][1]["value"] == GROUND["conversations"][1]["value"] for r in derived)


def test_perturbation_excludes_the_original_phrasing(pools):
    """扰动是「换个说法」。把原句再问一遍，那一次不算扰动。"""
    record = dict(GROUND, conversations=[
        {"from": "human", "value": "<image>\n框出图中银灰色的三轮车。"},
        *GROUND["conversations"][1:],
    ])
    derived = derive_question_perturbation([record], pools={"ground_appearance": pools["ground"]})
    assert "框出图中银灰色的三轮车。" not in [r["conversations"][0]["value"].replace("<image>\n", "")
                                          for r in derived]


def test_templates_needing_an_unavailable_placeholder_are_skipped(pools):
    """量词没给就跳过带 {mw} 的模板，不硬填一个「个」—— 我们测的是模型对正常问法
    的稳定性，不是对病句的容忍度。"""
    derived = derive_question_perturbation([GROUND], pools={"ground_appearance": pools["ground"]})
    assert all("{mw}" not in r["conversations"][0]["value"] for r in derived)


def test_a_pool_too_small_for_the_requested_variants_skips_the_record(tmp_path, pools):
    """凑不满就整条不做：两个变体和三个变体算出来的方差不是同一个量。"""
    small = tmp_path / "small.txt"
    small.write_text("框出图中{attribute}的{label}。\n输出{attribute}的{label}的检测框。\n", encoding="utf-8")
    derived = derive_question_perturbation([GROUND], pools={"ground_appearance": PhrasePool.load(small)},
                                           variants=3)
    assert derived == []


def test_perturbation_phrasings_are_stable_across_runs(pools):
    """问法本身变了，扰动实验前后就不可比了。"""
    first = [r["conversations"][0]["value"] for r in
             derive_question_perturbation([GROUND], pools={"ground_appearance": pools["ground"]})]
    second = [r["conversations"][0]["value"] for r in
              derive_question_perturbation([GROUND], pools={"ground_appearance": pools["ground"]})]
    assert first == second


# --------------------------------------------------------------- 落盘与回读

def test_a_derived_set_reads_back_through_the_normal_eval_set_loader(tmp_path, pools):
    """派生集就是普通评估集 —— 这正是这个方案的全部意义：推理引擎不用知道
    它是派生来的。"""
    predictions = {"img1_ground_appearance_0__t1": '{"bbox_2d":[500,500,600,600]}'}
    path = write_jsonl(derive_model_history([GROUND], predictions), tmp_path / "mh.jsonl")
    frame = load_eval_set(path, select=[{"turn": 2}])
    assert len(frame) == 1
    assert frame.loc[0, "meta.history_mode"] == "model"
    history = json.loads(frame.loc[0, "history"])
    assert history[0]["a"] == '{"bbox_2d":[500,500,600,600]}'


def test_predictions_from_several_files_are_merged(tmp_path):
    """一条 inventory_locate 的三轮分散在三个数据集里，拼模型历史要多轮的预测。"""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame([{"index": "x__t1", "prediction": "一"}]).to_csv(a, index=False)
    pd.DataFrame([{"index": "x__t2", "prediction": "二"}]).to_csv(b, index=False)
    assert load_predictions([a, b]) == {"x__t1": "一", "x__t2": "二"}


def test_conflicting_predictions_for_one_index_are_refused(tmp_path):
    """静默取后一份会让派生集里混进另一次跑的结果。"""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame([{"index": "x__t1", "prediction": "一"}]).to_csv(a, index=False)
    pd.DataFrame([{"index": "x__t1", "prediction": "别的"}]).to_csv(b, index=False)
    with pytest.raises(ValueError, match="index 冲突"):
        load_predictions([a, b])


def test_the_model_history_dataset_selects_the_same_turns_as_the_main_one():
    """派生集是主线的镜像。选的轮次不一样，链路衰减率的分子分母就不是同一件事。

    最容易写错的是拿 ``{"turn": 2}`` 不带任务过滤当简写 —— 那会把
    ``inventory_locate`` 的轮 2（**一个坐标框**）当成描述丢给裁判打分。
    """
    import json
    from pathlib import Path

    config = json.loads(Path("det.example.json").read_text(encoding="utf-8"))
    datasets = config["datasets"]
    assert (datasets["describe_modelhist"]["params"]["select"]
            == datasets["describe"]["params"]["select"])
    # 每条规则都必须点名任务，不能只写 turn
    for rule in datasets["describe"]["params"]["select"]:
        assert rule.get("task_type"), f"select 规则没点名任务：{rule}"
