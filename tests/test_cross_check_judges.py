"""§15.2 自偏检测：多裁判交叉验证。

需求文档在「已知局限」里自己写了这一条做不到 ——「裁判是 Qwen3.8-27B，被测是
Qwen3-VL-8B，同家族，当前无法做自偏检测（没有异家族裁判）」。这一组测试锁住的就是
补上这一条之后的行为：配一路异家族裁判，明细表多一列 hit__<名字>，报表多两张
诊断表，而主裁判出的 hit（验收用的那一列）一个数都不许变。
"""

import json

import pandas as pd
import pytest

from eval_tool.breakdown import (
    INSUFFICIENT,
    make_judge_agreement,
    make_judge_conclusion,
)
from eval_tool.config import (
    ConfigError,
    EvalConfig,
    JudgeSettings,
    ModelConfig,
    load_config,
    parse_cross_check_judges,
)
from eval_tool.judge import JudgeClient
from eval_tool.run_eval import run


# --------------------------------------------------------------------------- 配置解析


def test_cross_check_judge_inherits_every_field_it_does_not_override():
    """只写不同的字段，其余继承主裁判 —— 尤其是提示词。

    换了提示词就不是在比裁判了，是在比提示词，那样的「不一致」说明不了任何事。
    """
    primary = JudgeSettings(model="qwen3.6-27b", temperature=0.0, pointwise_prompt="RUBRIC")
    judges = parse_cross_check_judges(
        {"cross_check": [{"name": "internvl", "model": "InternVL2-26B"}]}, primary
    )
    assert len(judges) == 1
    name, settings = judges[0]
    assert name == "internvl"
    assert settings.model == "InternVL2-26B"
    assert settings.pointwise_prompt == "RUBRIC"
    assert settings.api_base == primary.api_base
    assert settings.temperature == primary.temperature


def test_cross_check_judge_identical_to_primary_is_rejected():
    """指纹相同 = 同一个裁判判两遍。这不是交叉验证，只是把裁判开销翻倍。"""
    primary = JudgeSettings(model="qwen3.6-27b")
    with pytest.raises(ConfigError, match="指纹相同"):
        parse_cross_check_judges(
            {"cross_check": [{"name": "same", "model": "qwen3.6-27b"}]}, primary
        )


def test_cross_check_judge_may_differ_only_by_temperature():
    """同模型不同温度是合法的一路：它测的是裁判自身的稳定度，不是异家族偏置。"""
    primary = JudgeSettings(model="qwen3.6-27b", temperature=0.0)
    judges = parse_cross_check_judges(
        {"cross_check": [{"name": "hot", "temperature": 0.7}]}, primary
    )
    assert judges[0][1].model == "qwen3.6-27b"
    assert judges[0][1].temperature == 0.7


@pytest.mark.parametrize(
    "item, message",
    [
        ({"model": "x"}, "needs a name"),
        ({"name": "primary", "model": "x"}, "duplicate"),
        ({"name": "a b", "model": "x"}, "只能用字母数字"),
        ({"name": "报表/../etc", "model": "x"}, "只能用字母数字"),
    ],
)
def test_cross_check_judge_names_are_validated(item, message):
    """名字会变成列名 hit__<名字>，还会进 CSV 表头 —— 不许有空格和路径分隔符。"""
    with pytest.raises(ConfigError, match=message):
        parse_cross_check_judges({"cross_check": [item]}, JudgeSettings())


def test_duplicate_cross_check_names_are_rejected():
    with pytest.raises(ConfigError, match="duplicate"):
        parse_cross_check_judges(
            {
                "cross_check": [
                    {"name": "alt", "model": "m1"},
                    {"name": "alt", "model": "m2"},
                ]
            },
            JudgeSettings(),
        )


def test_config_without_cross_check_has_none(tmp_path):
    """没配就是没配 —— 旧配置一个字都不用改，也不会多跑一次裁判。"""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "datasets": {"vqa": {"name": "aero_vqa", "kind": "judge_text"}},
        "judge": {"model": "qwen3.6-27b"},
    }), encoding="utf-8")
    assert load_config(path).cross_check_judges == []


def test_config_reads_cross_check_block(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "datasets": {"vqa": {"name": "aero_vqa", "kind": "judge_text"}},
        "judge": {
            "model": "qwen3.6-27b",
            "cross_check": [{"name": "internvl", "model": "InternVL2-26B"}],
        },
    }), encoding="utf-8")
    judges = load_config(path).cross_check_judges
    assert [name for name, _ in judges] == ["internvl"]
    assert judges[0][1].model == "InternVL2-26B"


# --------------------------------------------------------------------------- 一致性表


def _paired(n, primary, cross, model="sft", dataset="vqa"):
    return [
        {"model": model, "dataset": dataset, "index": f"{model}{i}",
         "hit": primary(i), "hit__alt": cross(i)}
        for i in range(n)
    ]


def test_agreement_reports_delta_and_row_level_disagreement():
    """均分一样不代表两个裁判判得一样。

    这里两个裁判的均分都是 0.5，但它们在**每一行**上都相反 —— delta=0 而
    agree_rate=0。只看均分会得出「两个裁判高度一致」的错误结论。
    """
    details = pd.DataFrame(
        _paired(40, lambda i: float(i % 2), lambda i: float((i + 1) % 2))
    )
    table = make_judge_agreement(details)
    row = table.iloc[0]
    assert row["judge"] == "alt"
    assert row["n"] == 40
    assert row["delta"] == 0.0
    assert row["mad"] == 1.0
    assert row["agree_rate"] == 0.0
    assert row["spearman"] == -1.0
    assert row["status"] == "ok"


def test_agreement_detects_a_lenient_primary_judge():
    """主裁判整批判 1、异家族裁判判 0.6：delta 为负 = 主裁判在送分，
    正是同家族偏爱的典型形状。"""
    details = pd.DataFrame(_paired(30, lambda i: 1.0, lambda i: 0.6))
    row = make_judge_agreement(details).iloc[0]
    assert row["mean_primary"] == 1.0
    assert row["mean_cross"] == 0.6
    assert row["delta"] == -0.4
    # 两边都没有方差，秩相关没有定义。报 nan 而不是 0。
    assert pd.isna(row["spearman"])


def test_agreement_only_uses_rows_both_judges_scored():
    """判失败记 None。两边的失败行不重合，不取交集就是拿两批样本比裁判。"""
    details = pd.DataFrame(_paired(10, lambda i: 1.0, lambda i: 1.0))
    details.loc[0, "hit"] = None
    details.loc[1, "hit__alt"] = None
    row = make_judge_agreement(details, min_n=1).iloc[0]
    assert row["n"] == 8


def test_agreement_is_empty_without_any_cross_judge():
    details = pd.DataFrame([{"model": "sft", "dataset": "vqa", "index": "1", "hit": 1.0}])
    assert make_judge_agreement(details).empty


def test_agreement_flags_small_cells_instead_of_hiding_them():
    details = pd.DataFrame(_paired(5, lambda i: 1.0, lambda i: 0.0))
    assert make_judge_agreement(details).iloc[0]["status"] == INSUFFICIENT


# --------------------------------------------------------------------------- 结论表


def _pair_of_models(base_primary, base_cross, sft_primary, sft_cross, n=40):
    return pd.DataFrame(
        _paired(n, lambda i: base_primary, lambda i: base_cross, model="base")
        + _paired(n, lambda i: sft_primary, lambda i: sft_cross, model="sft")
    )


def test_conclusion_agrees_when_both_judges_see_the_same_gain():
    details = _pair_of_models(0.4, 0.3, 0.8, 0.6)
    row = make_judge_conclusion(details, "base").iloc[0]
    assert row["model"] == "sft"
    assert row["gain_primary"] == pytest.approx(0.4)
    assert row["gain_cross"] == pytest.approx(0.3)
    assert row["verdict"] == "agree"


def test_conclusion_flags_a_sign_flip():
    """主裁判说涨了、异家族裁判说跌了 —— 这份增益多半是裁判认亲，不能报。

    这是整个交叉验证唯一非做不可的产出：逐行一致率再难看，只要结论同号就还站得住；
    符号翻了就是另一回事。
    """
    details = _pair_of_models(0.4, 0.6, 0.8, 0.4)
    row = make_judge_conclusion(details, "base").iloc[0]
    assert row["gain_primary"] > 0
    assert row["gain_cross"] < 0
    assert row["verdict"] == "flipped"


def test_conclusion_needs_enough_samples():
    details = _pair_of_models(0.4, 0.3, 0.8, 0.6, n=5)
    assert make_judge_conclusion(details, "base").iloc[0]["verdict"] == INSUFFICIENT


def test_conclusion_skips_datasets_without_the_baseline():
    details = pd.DataFrame(_paired(40, lambda i: 1.0, lambda i: 1.0, model="sft"))
    assert make_judge_conclusion(details, "base").empty


# --------------------------------------------------------------------------- 端到端


def _vqa_fixture(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    pd.DataFrame([
        {"index": str(i), "image": "", "question": "描述这张图", "answer": "参考答案",
         "category": "单轮", "l2-category": "", "source_id": f"s{i}"}
        for i in range(4)
    ]).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)
    pred = tmp_path / "sft_vqa.xlsx"
    pd.DataFrame(
        [{"index": str(i), "prediction": f"回答{i}"} for i in range(4)]
    ).to_excel(pred, index=False)
    return tsv_dir, pred


def _judges_that_disagree(monkeypatch):
    """两路裁判：主裁判一律判过，交叉裁判一律判不过。

    按 self.settings.model 分派 —— 交叉验证的整个前提就是同一份回答走两个不同的
    客户端，打到同一个 fake 上必须能分辨出是谁在问。
    """
    def fake_pointwise(self, question, reference, prediction, image_b64=None):
        hit = 1 if self.settings.model == "primary-model" else 0
        return {"hit": hit, "reason": self.settings.model, "quality_score": 90.0 * hit,
                "param_score": 90, "fact_score": 90, "visual_score": 90,
                "fabrication_score": 90, "style_score": 90}

    monkeypatch.setattr(JudgeClient, "judge_pointwise", fake_pointwise)


def _config(tmp_path, tsv_dir, pred, cross):
    return EvalConfig(
        tsv_dir=tsv_dir,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        datasets={"vqa": "aero_vqa"},
        models=[ModelConfig(name="sft", paths={"vqa": str(pred)})],
        baseline_model="base",
        judge=JudgeSettings(model="primary-model"),
        cross_check_judges=cross,
        enabled_datasets=["vqa"],
        do_pointwise=True,
        do_pairwise=False,
        bootstrap_n=20,
    )


def test_run_eval_adds_a_column_per_cross_judge(tmp_path, monkeypatch):
    tsv_dir, pred = _vqa_fixture(tmp_path)
    _judges_that_disagree(monkeypatch)
    written = run(_config(
        tmp_path, tsv_dir, pred, [("alt", JudgeSettings(model="alt-model"))]
    ))

    detail = pd.read_excel(written["detail_sft_vqa.xlsx"])
    # 主裁判的 hit 是验收用的那一列，交叉裁判绝不许动它。
    assert list(detail["hit"]) == [1, 1, 1, 1]
    assert list(detail["hit__alt"]) == [0, 0, 0, 0]

    agreement = pd.read_csv(written["judge_agreement.csv"])
    row = agreement.iloc[0]
    assert row["judge"] == "alt"
    assert row["n"] == 4
    assert row["delta"] == -1.0
    assert row["agree_rate"] == 0.0


def test_run_eval_without_cross_judges_writes_no_agreement_table(tmp_path, monkeypatch):
    """没配交叉裁判就不该多出文件，也不该多调一次裁判。"""
    tsv_dir, pred = _vqa_fixture(tmp_path)
    calls: list[str] = []

    def fake_pointwise(self, question, reference, prediction, image_b64=None):
        calls.append(self.settings.model)
        return {"hit": 1, "reason": "ok", "quality_score": 90.0, "param_score": 90,
                "fact_score": 90, "visual_score": 90, "fabrication_score": 90,
                "style_score": 90}

    monkeypatch.setattr(JudgeClient, "judge_pointwise", fake_pointwise)
    written = run(_config(tmp_path, tsv_dir, pred, []))

    assert "judge_agreement.csv" not in written
    assert "judge_conclusion.csv" not in written
    assert set(calls) == {"primary-model"}
    assert "hit__alt" not in pd.read_excel(written["detail_sft_vqa.xlsx"]).columns


def test_a_broken_cross_judge_does_not_take_down_the_main_run(tmp_path, monkeypatch):
    """交叉验证是诊断项。它挂了要记 warning，不能把两小时的主评估一起带走。"""
    tsv_dir, pred = _vqa_fixture(tmp_path)

    def fake_pointwise(self, question, reference, prediction, image_b64=None):
        if self.settings.model != "primary-model":
            raise RuntimeError("交叉裁判服务没起")
        return {"hit": 1, "reason": "ok", "quality_score": 90.0, "param_score": 90,
                "fact_score": 90, "visual_score": 90, "fabrication_score": 90,
                "style_score": 90}

    monkeypatch.setattr(JudgeClient, "judge_pointwise", fake_pointwise)
    written = run(_config(
        tmp_path, tsv_dir, pred, [("dead", JudgeSettings(model="dead-model"))]
    ))

    detail = pd.read_excel(written["detail_sft_vqa.xlsx"])
    assert list(detail["hit"]) == [1, 1, 1, 1]
    # score_vqa 自己会兜住单行的裁判异常并记 hit=None，所以这一列存在但整列为空 ——
    # 无论是整趟抛出还是逐行判失败，交叉列都不许污染主裁判的结论。
    if "hit__dead" in detail.columns:
        assert detail["hit__dead"].isna().all()
    assert "judge_agreement.csv" not in written
