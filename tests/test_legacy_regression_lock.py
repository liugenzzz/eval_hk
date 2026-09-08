"""装备 / 书籍那条老通路的回归锁。

目标检测的东西全部是新增的 kind、新增的模块、新增的报表文件。但 run_eval、report、
judge、infer 是共用的，共用就有改坏的可能 —— 而改坏的表现不一定是报错，可能是装备
评估的某一列悄悄变了值。

这一组测试把老通路的**行为**钉死：走完整的 mcq + judge + vqa 流程，断言分数、关键列、
裁判缓存键都不变。以后任何一次改动碰到它，这里立刻红。

它不检查我们新加的列（合规列是纯新增，不改分），只检查老通路本来就有的东西。
"""

import json

import pandas as pd
import pytest

from eval_tool.cache import JsonlCache
from eval_tool.config import EvalConfig, ModelConfig
from eval_tool.judge import JudgeClient
from eval_tool.run_eval import run


@pytest.fixture
def aero_workspace(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    pd.DataFrame(
        [
            {"index": str(i), "image": "", "question": "这是什么部件？",
             "A": "燃油泵", "B": "滑油泵", "C": "作动筒", "D": "传感器",
             "answer": "B" if i % 2 else "A",
             "category": "P1" if i % 2 else "P3", "l2-category": "零件图",
             "source_id": f"s{i}"}
            for i in range(40)
        ]
    ).to_csv(tsv_dir / "aero_mcq.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"index": str(i), "image": "", "question": "这个说法正确吗？",
             "A": "正确", "B": "错误", "answer": "A" if i % 2 else "B",
             "category": "R1", "l2-category": "结构图", "source_id": f"j{i}"}
            for i in range(40)
        ]
    ).to_csv(tsv_dir / "aero_judge.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"index": str(i), "image": "", "question": f"描述第 {i} 张图",
             "answer": "参考答案", "category": "单轮", "l2-category": "",
             "source_id": f"v{i}"}
            for i in range(40)
        ]
    ).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)
    return tmp_path, tsv_dir


def _aero_config(tmp_path, tsv_dir, preds, **overrides):
    return EvalConfig(
        tsv_dir=tsv_dir,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        datasets={"mcq": "aero_mcq", "judge": "aero_judge", "vqa": "aero_vqa"},
        models=[ModelConfig(name="base", paths=preds)],
        baseline_model="base",
        bootstrap_n=20,
        **overrides,
    )


def _write_preds(tmp_path, name, answers):
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(
        [{"index": str(i), "prediction": answer} for i, answer in enumerate(answers)]
    ).to_csv(path, index=False)
    return str(path)


def test_choice_scoring_is_unchanged(aero_workspace):
    """mcq 走 A/B/C/D，judge 走 A/B —— 这个区分历史上是按数据集键名判的，
    kind 重构之后必须还是同一个结果。"""
    tmp_path, tsv_dir = aero_workspace
    preds = {
        "mcq": _write_preds(tmp_path, "mcq", ["答案是 B" if i % 2 else "答案是 A" for i in range(40)]),
        "judge": _write_preds(tmp_path, "judge", ["正确" if i % 2 else "错误" for i in range(40)]),
    }
    written = run(_aero_config(tmp_path, tsv_dir, preds, enabled_datasets=["mcq", "judge"],
                               do_pointwise=False, do_pairwise=False))
    summary = pd.read_csv(written["report_summary.csv"])
    assert summary.loc[0, "mcq:overall"] == 1.0
    assert summary.loc[0, "judge:overall"] == 1.0
    assert summary.loc[0, "mcq:overall:n"] == 40


def test_choice_detail_columns_are_unchanged(aero_workspace):
    """老通路的 detail 列是下游脚本读的。合规列是新增的，不该顶掉任何一列。"""
    tmp_path, tsv_dir = aero_workspace
    preds = {"mcq": _write_preds(tmp_path, "mcq", ["B"] * 40)}
    run(_aero_config(tmp_path, tsv_dir, preds, enabled_datasets=["mcq"],
                     do_pointwise=False, do_pairwise=False))
    detail = pd.read_excel(tmp_path / "out" / "detail_base_mcq.xlsx")
    for column in ("index", "question", "answer", "prediction", "category", "l2-category",
                   "extracted_choice", "hit", "ambiguous", "no_answer", "missing",
                   "extraction_method", "pred_len"):
        assert column in detail.columns, f"老通路的 {column} 列不见了"


def test_weighted_score_summary_is_unchanged(aero_workspace):
    """total_score 有一道 n>=30 的类别门：n 小的类别照样单独出行，但不驱动总分。
    这条历史行为不能被新报表改掉 —— 装备那边的总分就是这么算出来的。"""
    tmp_path, tsv_dir = aero_workspace
    preds = {"mcq": _write_preds(tmp_path, "mcq", ["答案是 B" if i % 2 else "答案是 A" for i in range(40)])}
    written = run(_aero_config(tmp_path, tsv_dir, preds, enabled_datasets=["mcq"],
                               do_pointwise=False, do_pairwise=False))
    score = pd.read_csv(written["score_summary.csv"])
    assert set(score.columns) >= {"model", "total_score", "total_score_by_n",
                                  "total_score_ungated", "n_total"}
    # P1 / P3 各 20 条，都不到 30，两个都被门在总分之外但各自出行
    assert pd.isna(score.loc[0, "total_score"])
    assert score.loc[0, "total_score_ungated"] == 1.0
    assert score.loc[0, "total_score_by_n"] == 1.0
    assert score.loc[0, "P1"] == 1.0 and score.loc[0, "P1_n"] == 20
    assert "P1(n=20)" in score.loc[0, "categories_excluded_small_n"]


def test_vqa_judge_cache_key_is_unchanged(aero_workspace, monkeypatch):
    """裁判判词缓存键是 (judge_fp, model, dataset, index)。dataset 那一位历史上写死
    "vqa"，重构后由 dataset_key 传入 —— 键名还叫 vqa 时必须产生同一个键，否则装备
    评估攒下来的判词缓存会整批失效，白烧一遍裁判调用。"""
    tmp_path, tsv_dir = aero_workspace
    calls = []

    def fake_pointwise(self, question, reference, prediction, image_b64=None):
        calls.append(question)
        return {"hit": 1, "reason": "ok", "quality_score": 90.0, "param_score": 90,
                "fact_score": 90, "visual_score": 90, "fabrication_score": 90, "style_score": 90}

    monkeypatch.setattr(JudgeClient, "judge_pointwise", fake_pointwise)
    preds = {"vqa": _write_preds(tmp_path, "vqa", ["模型回答"] * 40)}
    run(_aero_config(tmp_path, tsv_dir, preds, enabled_datasets=["vqa"], do_pairwise=False))

    cache_path = tmp_path / "cache" / "judge_cache_pointwise.jsonl"
    records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records, "裁判判词没进缓存"
    keys = [record["key"] for record in records]
    assert all(key["dataset"] == "vqa" for key in keys)
    assert set(keys[0]) == {"judge_fp", "model", "dataset", "index"}

    # 第二次跑必须全部命中缓存，一次裁判都不调
    calls.clear()
    run(_aero_config(tmp_path, tsv_dir, preds, enabled_datasets=["vqa"], do_pairwise=False))
    assert calls == [], "缓存没命中，装备评估会白烧一遍裁判调用"


def test_pairwise_output_columns_are_unchanged(aero_workspace, monkeypatch):
    tmp_path, tsv_dir = aero_workspace

    def fake_pointwise(self, question, reference, prediction, image_b64=None):
        return {"hit": 1, "reason": "ok", "quality_score": 90.0, "param_score": 90,
                "fact_score": 90, "visual_score": 90, "fabrication_score": 90, "style_score": 90}

    monkeypatch.setattr(JudgeClient, "judge_pointwise", fake_pointwise)
    monkeypatch.setattr(JudgeClient, "judge_pairwise", lambda self, *a, **k: ("A", "ok"))

    base_pred = _write_preds(tmp_path, "vqa_base", ["base 回答"] * 40)
    sft_pred = _write_preds(tmp_path, "vqa_sft", ["sft 回答"] * 40)
    written = run(
        EvalConfig(
            tsv_dir=tsv_dir, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
            datasets={"vqa": "aero_vqa"},
            models=[ModelConfig(name="base", paths={"vqa": base_pred}),
                    ModelConfig(name="sft", paths={"vqa": sft_pred})],
            baseline_model="base", enabled_datasets=["vqa"], bootstrap_n=20,
        )
    )
    pairwise = pd.read_csv(written["pairwise_vs_baseline.csv"])
    for column in ("scope", "model", "baseline_model", "n", "win_rate", "tie_rate",
                   "loss_rate", "win_ci_low", "win_ci_high"):
        assert column in pairwise.columns, f"pairwise 的 {column} 列不见了"
    assert (pairwise["model"] == "sft").all()


def test_legacy_config_without_any_new_field_still_runs(aero_workspace):
    """老配置一个新字段都没有：没有 kind、没有 profile、没有 report 块。"""
    tmp_path, tsv_dir = aero_workspace
    preds = {"mcq": _write_preds(tmp_path, "mcq", ["B"] * 40)}
    config = EvalConfig(
        tsv_dir=tsv_dir, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
        datasets={"mcq": "aero_mcq"},
        models=[ModelConfig(name="base", paths=preds)],
        baseline_model="base", enabled_datasets=["mcq"],
        do_pointwise=False, do_pairwise=False, bootstrap_n=20,
    )
    written = run(config)
    assert "report_summary.csv" in written
    assert config.kind_for("mcq") == "choice"


def test_new_report_files_are_additive_only(aero_workspace):
    """新报表文件是新增的，老文件一个都不能少。"""
    tmp_path, tsv_dir = aero_workspace
    preds = {"mcq": _write_preds(tmp_path, "mcq", ["B"] * 40)}
    written = run(_aero_config(tmp_path, tsv_dir, preds, enabled_datasets=["mcq"],
                               do_pointwise=False, do_pairwise=False))
    for name in ("score_summary.csv", "report_summary.csv", "report_summary.json",
                 "report_summary_long.csv", "judge_detail_all.xlsx", "manifest.json"):
        assert name in written, f"老通路的 {name} 不见了"
