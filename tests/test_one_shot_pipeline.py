"""一条命令跑完整套：``eval_tool all``。

这一组锁住的是「一次性通路」真的成立所需要的四件事，每一件在补上之前都会让
``all`` 在目标检测这条路上跑不完：

1. 几个数据集共用同一份评估集时，预测要各写各的文件
2. 真值是 jsonl 也算数（以前只找 .tsv）
3. 派生集在两趟推理中间自动造，不用人工跑 derive
4. 关掉裁判（只看代码指标）时报表不能崩
"""

import json

import pandas as pd
import pytest

from eval_tool.aggregate import bootstrap_mean_ci, bootstrap_paired_diff_ci
from eval_tool.artifacts import prediction_stems
from eval_tool.config import ConfigError, load_pipeline_config, parse_derive_plans
from eval_tool.pipeline import PipelineError, run_all


# --------------------------------------------------- 1. 预测文件名不许撞


def test_unique_dataset_names_keep_their_filenames():
    """装备/书籍那些配置一个键对一个评估集，文件名一个字节都不能变。"""
    assert prediction_stems({"mcq": "aero_mcq", "vqa": "aero_vqa"}) == {
        "mcq": "aero_mcq", "vqa": "aero_vqa",
    }


def test_datasets_sharing_one_eval_set_get_distinct_filenames():
    """目标检测八个主线数据集读的都是 eval_set_v1，各自 select 一个子集。

    按名字命名的话它们会全写进同一个 xlsx —— 第一个推完，后面七个看见文件已存在
    就跳过，报表拿到的是**另一个数据集的预测**。数字看着正常，全错。
    """
    stems = prediction_stems({
        "ground_box": "eval_set_v1",
        "region_identify": "eval_set_v1",
        "question_perturbation": "question_perturbation",
    })
    assert stems["ground_box"] == "eval_set_v1__ground_box"
    assert stems["region_identify"] == "eval_set_v1__region_identify"
    # 名字本来就唯一的那个不受影响
    assert stems["question_perturbation"] == "question_perturbation"


# --------------------------------------------------- 2. 关掉裁判时报表不能崩


def test_bootstrap_handles_pandas_na():
    """关了 pointwise 的裁判组打分器把 hit 记成 pd.NA。

    以前这里是 ``astype(float)``，遇到 pd.NA 直接 TypeError —— 而且是在整趟评估的
    最后一步炸，前面几小时的推理全白跑。
    """
    assert bootstrap_mean_ci([1.0, pd.NA, 0.0, None], n_bootstrap=20) == pytest.approx(
        (0.0, 1.0), abs=1.0
    )
    diff, low, high = bootstrap_paired_diff_ci(
        [1.0, pd.NA, 1.0], [0.0, 0.0, pd.NA], n_bootstrap=20
    )
    # 只有第一位两边都有数，差值 1.0；缺值的样本整条丢掉
    assert diff == 1.0


def test_paired_bootstrap_ignores_the_incoming_index():
    """两个序列按**位置**配对。带着各自 DataFrame 行号的 Series 直接相减会按
    索引对齐，错位的那些位置会静默变成 NaN，配对数凭空少一截。"""
    left = pd.Series([1.0, 1.0], index=[7, 9])
    right = pd.Series([0.0, 0.0], index=[3, 5])
    diff, _, _ = bootstrap_paired_diff_ci(left, right, n_bootstrap=20)
    assert diff == 1.0


# --------------------------------------------------- 3. derive 配置


def _datasets():
    return {"ground_box": "eval_set_v1", "describe": "eval_set_v1",
            "describe_modelhist": "describe_modelhist",
            "reverse_consistency": "reverse_consistency"}


def test_derive_plan_infers_source_and_model(tmp_path):
    """能省的都省掉：source 默认取第一个非派生数据集，from 默认取唯一那个非基线模型。"""
    plans = parse_derive_plans(
        {"derive": [{"dataset": "describe_modelhist", "mode": "model-history",
                     "sample_ratio": 0.3}]},
        tmp_path, _datasets(), models=["base", "sft"], baseline="base",
    )
    assert plans[0].source == "ground_box"
    assert plans[0].from_model == "sft"
    assert plans[0].sample_ratio == 0.3


def test_derive_plan_refuses_to_guess_between_two_challengers(tmp_path):
    """派生集要的是**被测模型自己的**输出。两个候选就别猜 —— 猜错了造出来的历史
    是另一个 checkpoint 的，链路衰减率测的就不是这一轮 SFT。"""
    with pytest.raises(ConfigError, match="推不出用哪个模型"):
        parse_derive_plans(
            {"derive": [{"dataset": "describe_modelhist", "mode": "model-history"}]},
            tmp_path, _datasets(), models=["base", "sft_ep2", "sft_ep3"], baseline="base",
        )


def test_question_perturbation_needs_no_predictions(tmp_path):
    """问法扰动只换问法，不看模型答了什么 —— 所以它不需要 from。"""
    pool = tmp_path / "pool.txt"
    pool.write_text("框出图中{label}\n定位{label}\n", encoding="utf-8")
    plans = parse_derive_plans(
        {"derive": [{"dataset": "reverse_consistency", "mode": "question-perturbation",
                     "pools": {"ground_appearance": str(pool)}}]},
        tmp_path, _datasets(), models=["sft"], baseline="sft",
    )
    assert plans[0].needs_predictions is False


@pytest.mark.parametrize("item, message", [
    ({"dataset": "nope", "mode": "model-history"}, "不在 datasets 里"),
    ({"dataset": "describe_modelhist", "mode": "typo"}, "mode 只能是"),
    ({"dataset": "describe_modelhist", "mode": "reverse-consistency"}, "需要 pools.region_identify"),
    ({"dataset": "describe_modelhist", "mode": "question-perturbation"}, "至少要一个 pools"),
])
def test_derive_plan_validation(tmp_path, item, message):
    with pytest.raises(ConfigError, match=message):
        parse_derive_plans({"derive": [item]}, tmp_path, _datasets(),
                           models=["base", "sft"], baseline="base")


def test_derive_source_may_not_be_another_derived_set(tmp_path):
    """从派生集再派生一层，样本的来源就说不清了 —— derived_from 指回的是派生集的 id，
    链路衰减率按它对齐会全部对不上。"""
    with pytest.raises(ConfigError, match="不能是另一份派生集"):
        parse_derive_plans(
            {"derive": [
                {"dataset": "describe_modelhist", "mode": "model-history"},
                {"dataset": "reverse_consistency", "mode": "reverse-consistency",
                 "source": "describe_modelhist", "pools": {"region_identify": "x"}},
            ]},
            tmp_path, _datasets(), models=["base", "sft"], baseline="base",
        )


# --------------------------------------------------- 4. all 端到端


TURNS = [
    {"from": "human", "value": "<image>\n框出图中穿粉色外套的人员"},
    {"from": "gpt", "value": "{\"bbox_2d\":[100,100,300,300],\"label\":\"一般人员\"}"},
    {"from": "human", "value": "这名人员本身有什么特征？"},
    {"from": "gpt", "value": "这是一名身穿粉色外套的人员。"},
]


def _record(index):
    return {
        "id": f"img{index}_ground_appearance_{index}",
        "images": [f"img{index}.jpg"],
        "conversations": TURNS,
        "metadata": {"task_type": "ground_appearance", "bbox_scale": 1000,
                     "label": "一般人员", "attribute": "穿粉色外套",
                     "describe_kind": "appearance", "n_turns": 2,
                     "source_image": f"img{index}.jpg", "difficulty": "easy"},
    }


class _Fake:
    def generate(self, prompt, image_b64=None):
        if any(word in prompt for word in ("框", "定位", "标出")):
            return '{"bbox_2d":[100,100,300,300],"label":"一般人员"}'
        return "一名穿粉色外套的人员。"


def _one_shot_config(tmp_path, *, with_derive=True):
    tsv = tmp_path / "tsv"; tsv.mkdir()
    with (tsv / "eval_set_v1.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(6):
            handle.write(json.dumps(_record(i), ensure_ascii=False) + "\n")
    prompt = tmp_path / "raw.txt"; prompt.write_text("{question}", encoding="utf-8")
    pool = tmp_path / "pool.txt"
    pool.write_text("这个区域 {bbox} 里是什么？\n{bbox} 框住的是什么？\n", encoding="utf-8")

    config = {
        "tsv_dir": "tsv", "work_dir": "work", "out_dir": "out", "cache_dir": "cache",
        "datasets": {
            "ground_box": {"name": "eval_set_v1", "kind": "grounding_single",
                           "params": {"select": [{"task_type": ["ground_appearance"], "turn": 1}]}},
            "describe": {"name": "eval_set_v1", "kind": "describe",
                         "params": {"select": [{"task_type": ["ground_appearance"], "turn": 2}]}},
            "reverse_consistency": {"name": "reverse_consistency", "kind": "object_ident"},
        },
        "enabled_datasets": ["ground_box", "describe", "reverse_consistency"],
        "models": [{"name": "base", "model_path": "fake"}, {"name": "sft", "model_path": "fake"}],
        "baseline_model": "base",
        "infer": {"prompt_file": "raw.txt"},
        "do_pointwise": False, "do_pairwise": False, "bootstrap_n": 20,
    }
    if with_derive:
        config["derive"] = [{"dataset": "reverse_consistency", "mode": "reverse-consistency",
                             "pools": {"region_identify": "pool.txt"}}]
    path = tmp_path / "det.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


def test_all_runs_infer_derive_infer_eval_in_one_call(tmp_path):
    """一条命令：推主线 → 造派生集 → 推派生集 → 出报表。

    派生集必须夹在两趟推理中间 —— 它的内容就是模型自己在第一趟里的输出，
    这不是实现上的将就，是这几个指标的定义。
    """
    config = load_pipeline_config(_one_shot_config(tmp_path))
    result = run_all(config, generator_factory=lambda name: _Fake())

    # 派生集是这一趟自己造出来的，跑之前它并不存在
    assert "reverse_consistency" in result["derive"]
    assert (tmp_path / "tsv" / "reverse_consistency.jsonl").is_file()

    # 两个数据集共用 eval_set_v1，预测各写各的
    files = {path.name for path in (tmp_path / "work" / "sft").glob("*.xlsx")}
    assert "sft_eval_set_v1__ground_box.xlsx" in files
    assert "sft_eval_set_v1__describe.xlsx" in files
    assert "sft_reverse_consistency.xlsx" in files

    # 关着裁判也要出得来报表
    assert "failure_buckets.csv" in result["eval"]
    buckets = pd.read_csv(result["eval"]["failure_buckets.csv"])
    assert set(buckets["model"]) == {"base", "sft"}


def test_all_still_works_without_a_derive_block(tmp_path):
    """没写 derive 就是老行为：推一趟、评一趟，中间不插任何东西。"""
    config = load_pipeline_config(_one_shot_config(tmp_path, with_derive=False))
    config = type(config)(**{**config.__dict__, "enabled_datasets": ["ground_box", "describe"]})
    result = run_all(config, generator_factory=lambda name: _Fake())
    assert result["derive"] == {}
    assert "failure_buckets.csv" in result["eval"]


def test_missing_jsonl_is_reported_as_missing(tmp_path):
    """真值是 jsonl 也算数。以前这里只找 .tsv，目标检测这条路在第一步就被挡住。"""
    path = _one_shot_config(tmp_path)
    (tmp_path / "tsv" / "eval_set_v1.jsonl").unlink()
    config = load_pipeline_config(path)
    with pytest.raises(PipelineError, match="missing TSV/JSONL"):
        run_all(config, generator_factory=lambda name: _Fake())
