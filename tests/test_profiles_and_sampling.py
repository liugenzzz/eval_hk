"""内置 profile、评估集改名、确定性抽样。

这三件事解决的是同一个问题：配置里**只该写这台机器特有的东西**。八个切片按
task_type 怎么分、报表按哪几个维度拆、验收权重各是多少 —— 这些是评估口径，不是环境，
让每个人抄一份等于每次改口径都要追着所有人的配置改。
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from eval_tool.config import (
    ConfigError,
    PROFILE_DIR,
    apply_profile,
    builder_prompt_root,
    load_pipeline_config,
)
from eval_tool.eval_set import load_eval_set, sample_records


# ------------------------------------------------------------------ profile


def test_every_shipped_profile_parses():
    for path in sorted(PROFILE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("datasets"), f"{path.name} 没有 datasets"
        assert data.get("enabled_datasets"), f"{path.name} 没有 enabled_datasets"


def test_profile_supplies_the_bulk_and_config_only_overrides():
    merged = apply_profile({"profile": {"name": "grounding_zh_v1"}, "tsv_dir": "data"})
    assert len(merged["datasets"]) == 11
    assert len(merged["report"]["dims"]) == 11
    assert merged["tsv_dir"] == "data"


def test_config_wins_over_the_profile():
    merged = apply_profile({
        "profile": {"name": "grounding_zh_v1"},
        "enabled_datasets": ["ground_box"],
        "datasets": {"ground_box": {"params": {"iou_gate": 0.75}}},
    })
    # 列表整个替换：想少跑几个数据集，合并的话就减不掉
    assert merged["enabled_datasets"] == ["ground_box"]
    # 字典逐键合并：只写 iou_gate，select 和 kind 还是 profile 的
    assert merged["datasets"]["ground_box"]["params"]["iou_gate"] == 0.75
    assert merged["datasets"]["ground_box"]["kind"] == "grounding_single"
    assert merged["datasets"]["ground_box"]["params"]["select"]


def test_unknown_profile_is_an_error_not_a_silent_empty_run():
    """拼错 profile 名字如果当成「没写」，datasets 就是空的，跑出来是一份空报表 ——
    而配置看着一切正常。"""
    with pytest.raises(ConfigError, match="未知的 profile"):
        apply_profile({"profile": {"name": "grounding_zh_v2"}})


def test_no_profile_means_plain_config():
    raw = {"datasets": {"x": {"name": "x", "kind": "choice"}}}
    assert apply_profile(raw) == raw


# ------------------------------------------------------------------ 评估集改名


def test_eval_set_renames_every_mainline_dataset():
    """评估集文件叫什么都行，不用为了迁就配置去改文件名。"""
    merged = apply_profile({"profile": {"name": "grounding_zh_v1"}, "eval_set": "test"})
    mainline = {k: v["name"] for k, v in merged["datasets"].items()}
    assert mainline["ground_box"] == "test"
    assert mainline["describe"] == "test"
    # 派生集的文件是工具自己造的，名字它自己定，不跟着改
    assert mainline["describe_modelhist"] == "describe_modelhist"
    assert mainline["reverse_consistency"] == "reverse_consistency"


# ------------------------------------------------------------------ 问法池自动解析


def test_pool_paths_come_from_describe_prompt_dir(tmp_path):
    """配置里为了 D 组已经写过构建端的 prompts 路径了，问法池是它的兄弟目录 ——
    再让人把路径抄两遍没有道理，抄错一个字派生集就整个产不出来。"""
    prompts = tmp_path / "builder" / "prompts"
    (prompts / "describe").mkdir(parents=True)
    for name in ("region_identify", "ground_attribute"):
        (prompts / name).mkdir()
        (prompts / name / f"{name}.txt").write_text("问句 {bbox} {label}", encoding="utf-8")

    assert builder_prompt_root({"describe_prompt_dir": str(prompts / "describe")}) == prompts

    config = tmp_path / "det.json"
    config.write_text(json.dumps({
        "profile": {"name": "grounding_zh_v1"},
        "tsv_dir": "data", "work_dir": "w", "out_dir": "o", "cache_dir": "c",
        "dataset_defaults": {"describe_prompt_dir": str(prompts / "describe")},
        "models": [{"name": "base", "model_path": "m"}, {"name": "sft", "model_path": "m"}],
        "baseline_model": "base",
    }), encoding="utf-8")
    plans = {plan.dataset: plan for plan in load_pipeline_config(config).derive_plans}
    assert plans["reverse_consistency"].pools["region_identify"] == (
        prompts / "region_identify" / "region_identify.txt"
    )
    assert plans["question_perturbation"].pools["ground_appearance"] == (
        prompts / "ground_attribute" / "ground_attribute.txt"
    )


# ------------------------------------------------------------------ 抽样


def _records(n):
    return [{"id": f"r{i}"} for i in range(n)]


def test_sampling_is_reproducible_and_seed_controlled():
    first = [r["id"] for r in sample_records(_records(1000), 50, 42)]
    assert first == [r["id"] for r in sample_records(_records(1000), 50, 42)]
    assert first != [r["id"] for r in sample_records(_records(1000), 50, 7)]


def test_adding_a_sample_does_not_reshuffle_the_draw():
    """不用 random.sample 就是为了这个：评估集加了一条，抽中的那批不该跟着换掉 ——
    否则两个 checkpoint 算在不同的子集上，没法比。"""
    before = set(r["id"] for r in sample_records(_records(1000), 50, 42))
    after = set(r["id"] for r in sample_records(_records(1000) + [{"id": "NEW"}], 50, 42))
    assert len(before & after) >= 49


def test_sampling_more_than_available_keeps_everything():
    assert len(sample_records(_records(10), 9999, 42)) == 10
    assert len(sample_records(_records(10), None, 42)) == 10


def _write_set(path, tasks):
    with path.open("w", encoding="utf-8") as handle:
        for i, task in enumerate(tasks):
            handle.write(json.dumps({
                "id": f"s{i}_{task}",
                "images": [f"{i}.jpg"],
                "conversations": [
                    {"from": "human", "value": "<image>\n问"},
                    {"from": "gpt", "value": "答"},
                ],
                "metadata": {"task_type": task, "n_turns": 1},
            }, ensure_ascii=False) + "\n")


def test_every_slice_of_one_eval_set_sees_the_same_sampled_records(tmp_path):
    """抽的是**原始记录**，不是展开后的行。

    在行上抽，八个数据集看到的是八批不同的样本，报表横着对不起来；在记录上抽，
    它们看到的是同一批。
    """
    path = tmp_path / "test.jsonl"
    _write_set(path, ["ground_appearance"] * 50 + ["region_identify"] * 50)
    kept = {
        record["id"]
        for record in sample_records(
            [json.loads(line) for line in path.open(encoding="utf-8")], 20, 42
        )
    }
    for task in ("ground_appearance", "region_identify"):
        frame = load_eval_set(path, select=[{"task_type": [task], "turn": 1}],
                              sample_n=20, sample_seed=42)
        for index in frame["index"]:
            assert str(index).rsplit("__t", 1)[0] in kept


def test_a_slice_emptied_by_sampling_is_skipped_not_fatal(tmp_path, capsys):
    """稀有任务在小抽样里可能一条都没抽到。报错会让「先抽一点跑通」没法做。"""
    path = tmp_path / "test.jsonl"
    _write_set(path, ["ground_appearance"] * 99 + ["count_class"])
    # seed 0 抽的 5 条里正好没有那条 count_class
    frame = load_eval_set(path, select=[{"task_type": ["count_class"], "turn": 1}],
                          sample_n=5, sample_seed=0)
    assert frame.empty
    assert "抽样" in capsys.readouterr().out


def test_selecting_nothing_without_sampling_is_still_an_error(tmp_path):
    """没抽样却选空了，那就是 task_type 拼错或轮次填反 —— 静默返回空表会让报表
    多一格「样本不足」，而那格实际上是 bug。"""
    path = tmp_path / "test.jsonl"
    _write_set(path, ["ground_appearance"] * 5)
    with pytest.raises(ValueError, match="select 没有选中任何样本"):
        load_eval_set(path, select=[{"task_type": ["typo_task"], "turn": 1}])


def test_sample_block_reaches_every_dataset(tmp_path):
    config = tmp_path / "det.json"
    config.write_text(json.dumps({
        "profile": {"name": "grounding_zh_v1"},
        "tsv_dir": "data", "work_dir": "w", "out_dir": "o", "cache_dir": "c",
        "sample": {"n": 500, "seed": 7},
        "models": [{"name": "base", "model_path": "m"}, {"name": "sft", "model_path": "m"}],
        "baseline_model": "base",
    }), encoding="utf-8")
    parsed = load_pipeline_config(config)
    for key in parsed.enabled_datasets:
        assert parsed.dataset_params[key]["sample_n"] == 500
        assert parsed.dataset_params[key]["sample_seed"] == 7


def test_sample_n_must_be_positive(tmp_path):
    config = tmp_path / "det.json"
    config.write_text(json.dumps({
        "profile": {"name": "grounding_zh_v1"}, "tsv_dir": "d",
        "work_dir": "w", "out_dir": "o", "cache_dir": "c",
        "sample": {"n": 0},
        "models": [{"name": "base", "model_path": "m"}], "baseline_model": "base",
    }), encoding="utf-8")
    with pytest.raises(ConfigError, match="sample.n 要是正整数"):
        load_pipeline_config(config)


# ------------------------------------------------------------------ 示例配置


@pytest.mark.parametrize("name", ["det.example.json", "book.example.json",
                                  "pipeline.example.json"])
def test_example_configs_stay_short(name):
    """示例配置里只该有路径、模型和裁判地址。长了就说明口径又被抄进配置了。"""
    lines = sum(1 for _ in Path(name).open(encoding="utf-8"))
    assert lines <= 45, f"{name} 有 {lines} 行，口径应该放进 eval_tool/profiles/"


def test_det_example_still_resolves_to_the_full_eleven_datasets():
    parsed = load_pipeline_config("det.example.json")
    assert len(parsed.datasets) == 11
    assert len(parsed.derive_plans) == 3
    assert len(parsed.report_dims) == 11
    assert parsed.dataset_weights["ground_box"] == 0.40
    assert parsed.do_length_control is False
    # dataset_defaults 的四条路径铺到了每一个数据集
    for key in parsed.enabled_datasets:
        assert parsed.dataset_params[key]["image_root"]
        assert parsed.dataset_params[key]["classes_yaml"]


def test_the_two_exact_constants_agree():
    """object_ident 同时用到 classes 和 synonym 两边的关系常量。

    以前它把两个 ``EXACT`` 都 import 了，后一个静默盖掉前一个 —— 值恰好一样所以没出
    问题，但任何一边改了字面量，「精确命中」就会全判成不中，而且不报错。
    """
    from eval_tool.classes import EXACT as classes_exact
    from eval_tool.synonym import EXACT as synonym_exact

    assert classes_exact == synonym_exact
