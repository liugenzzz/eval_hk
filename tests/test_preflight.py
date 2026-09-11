"""开跑前体检。

整套评估跑完是几个小时，而最常见的失败原因是一条路径写错 —— 尤其 image_root 配错时
推理**不会报错**，模型只是看不见图，几小时之后拿到一份全是废框的报表。这一组锁住
「配错了能在第 10 秒被指出来」。
"""

import json

import pytest

from eval_tool.config import load_pipeline_config
from eval_tool.preflight import preflight, render


def _tree(tmp_path, *, with_pools=True, with_images=True):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "test.jsonl").write_text(json.dumps({
        "id": "s0", "images": ["0.jpg"],
        "conversations": [{"from": "human", "value": "<image>\n框出人员"},
                          {"from": "gpt", "value": "{\"bbox_2d\":[1,1,9,9],\"label\":\"人员\"}"}],
        "metadata": {"task_type": "ground_appearance", "n_turns": 1, "label": "人员"},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("models/base", "models/sft"):
        (tmp_path / name).mkdir(parents=True)
    if with_images:
        (tmp_path / "imgs").mkdir()
    (tmp_path / "classes.yaml").write_text("names: {0: 人员}", encoding="utf-8")
    prompts = tmp_path / "builder" / "prompts"
    (prompts / "describe").mkdir(parents=True)
    (prompts / "describe" / "a.txt").write_text("#! kind: appearance\n#! must-not: 左边\n描述\n",
                                                encoding="utf-8")
    if with_pools:
        for name in ("region_identify", "ground_attribute"):
            (prompts / name).mkdir()
            (prompts / name / f"{name}.txt").write_text("问 {bbox} {label} {attribute}",
                                                        encoding="utf-8")
    config = tmp_path / "det.json"
    config.write_text(json.dumps({
        "profile": {"name": "grounding_zh_v1", "version": "v1"},
        "tsv_dir": "data", "eval_set": "test",
        "work_dir": "w", "out_dir": "o", "cache_dir": "c",
        "dataset_defaults": {
            "image_root": str(tmp_path / "imgs"),
            "classes_yaml": str(tmp_path / "classes.yaml"),
            "describe_prompt_dir": str(prompts / "describe"),
        },
        "models": [{"name": "base", "model_path": "models/base"},
                   {"name": "sft", "model_path": "models/sft"}],
        "baseline_model": "base",
        "judge": {"api_base": "http://127.0.0.1:18180/v1/chat/completions",
                  "model": "qwen3.6-27b"},
    }, ensure_ascii=False), encoding="utf-8")
    return config


def test_a_good_config_passes_clean(tmp_path):
    result = preflight(load_pipeline_config(_tree(tmp_path)))
    assert result.errors == 0, render(result)
    # labels_dir 没配是一处提醒（CHAIR 出不了数），不是错误
    assert result.warnings == 1


def test_a_missing_image_root_is_an_error_not_a_warning(tmp_path):
    """image_root 配错是最贵的一种错：推理不报错，模型看不见图，几小时后才发现
    整批框都是废的。必须拦。"""
    config = _tree(tmp_path, with_images=False)
    result = preflight(load_pipeline_config(config))
    assert result.errors >= 1
    assert any("image_root" in line and "✗" in line for line in result.lines)


def test_a_missing_eval_set_names_the_jsonl_it_wants(tmp_path):
    """truth_path 找不到时会回落到 .tsv，但目标检测读的是 jsonl —— 报 .tsv 会让人
    以为自己该去建一个 tsv。"""
    config = _tree(tmp_path)
    (tmp_path / "data" / "test.jsonl").unlink()
    result = preflight(load_pipeline_config(config))
    line = next(line for line in result.lines if "找不到" in line and "test" in line)
    assert "test.jsonl" in line
    assert "eval_set" in line


def test_a_missing_phrase_pool_says_where_it_looked(tmp_path):
    config = _tree(tmp_path, with_pools=False)
    result = preflight(load_pipeline_config(config))
    line = next(line for line in result.lines if "问法池" in line)
    assert "derive[].pools" in line


def test_labels_dir_is_only_a_warning(tmp_path):
    """不配 labels_dir 只是 CHAIR 幻觉出不了数，别的照跑 —— 不该拦住整趟。"""
    result = preflight(load_pipeline_config(_tree(tmp_path)))
    assert result.errors == 0
    assert any("labels_dir" in line and "!" in line for line in result.lines)


def test_sampling_is_reported(tmp_path):
    config = _tree(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["sample"] = {"n": 500, "seed": 7}
    config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    result = preflight(load_pipeline_config(config))
    assert any("抽 500 条" in line and "种子 7" in line for line in result.lines)


def test_check_exits_nonzero_when_something_is_broken(tmp_path):
    from eval_tool.cli import main

    config = _tree(tmp_path, with_images=False)
    assert main(["check", "--config", str(config)]) > 0


def test_check_exits_zero_on_a_good_config(tmp_path):
    from eval_tool.cli import main

    assert main(["check", "--config", str(_tree(tmp_path))]) == 0


def test_a_config_without_an_infer_block_is_still_a_pipeline_config(tmp_path):
    """profile 里带着 infer，所以配置可以一个 infer 字段都不写。

    is_pipeline_config 读的是原始文件，不套 profile 的话这种配置会被当成老版 eval
    配置走另一条通路 —— 那条路不推理，也不认 derive，而且不会报错。
    """
    from eval_tool.config import is_pipeline_config

    config = _tree(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw.pop("infer", None)
    config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert is_pipeline_config(config)


def test_a_typo_in_the_subcommand_says_so(capsys):
    """敲错子命令（或者代码还没更新）时，老通路会报 "unrecognized arguments: check" ——
    那看起来像参数写错了，实际上是命令不存在。要直说，并列出有哪些。"""
    from eval_tool.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["chekc", "--config", "x.json"])
    message = str(exc.value)
    assert "未知的子命令" in message and "chekc" in message
    assert "check" in message and "git pull" in message


def test_the_legacy_entry_point_still_works():
    """不带子命令、直接 --config 是老装备通路的用法，不能因为上面那个检查坏掉。"""
    from eval_tool.cli import main

    with pytest.raises(SystemExit):
        main(["--config"])          # 缺值，老解析器自己报错


def _multi_model(tmp_path, **extra):
    config = _tree(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["models"] = [{"name": "base", "model_path": "models/base"}] + [
        {"name": f"ckpt_{i}", "model_path": "models/sft"} for i in range(6)
    ]
    raw["derive_from"] = "ckpt_5"
    raw.update(extra)
    config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return config


def test_many_models_warn_about_pairwise_cost(tmp_path):
    """成对判定是「挑战者 × 基线 × 样本 × 两个方向」，评一串 checkpoint 时是乘出来的。"""
    result = preflight(load_pipeline_config(_multi_model(tmp_path, do_pairwise=True)))
    assert result.errors == 0
    assert any("do_pairwise" in line and "6 个非基线模型" in line for line in result.lines)


def test_many_models_warn_that_chain_decay_only_holds_for_one(tmp_path):
    """派生集只由一个模型造，别的 checkpoint 是「接着它的历史答」，不是各自的链路。"""
    result = preflight(load_pipeline_config(_multi_model(tmp_path)))
    assert any("derive_from" in line and "chain_decay" in line for line in result.lines)


def test_two_models_do_not_trigger_the_pairwise_warning(tmp_path):
    result = preflight(load_pipeline_config(_tree(tmp_path)))
    assert not any("do_pairwise" in line for line in result.lines)
