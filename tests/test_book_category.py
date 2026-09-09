"""书籍评估：jsonl 直读 + 按领域分类（七大类）拆报表。

书籍那批评估集和装备走的是同一套代码（同一个 kind、同一套提示词、同一套报表维度），
这里钉死的是它多出来的三件事：

1. 评估集是 OpenAI ``messages`` 形状的 jsonl，每条自带 ``category``；
2. ``params.category_field`` 把那个分类放到报表的分组轴上；
3. ``report.dims`` 加一根 category 维度后，``breakdown.csv`` 每类一行，
   ``metric=quality_score`` 那几行就是「作战应用 80 分」要的数。
"""

import json

import pandas as pd
import pytest

from eval_tool.config import EvalConfig, ModelConfig
from eval_tool.run_eval import run

CATEGORIES = ["作战应用", "拱形基础", "结构强度", "材料工艺", "飞行原理", "维修保障", "任务规划"]


@pytest.fixture
def book_workspace(tmp_path):
    """七大类各 12 条的书籍评估集，外加一条只有 text 的语料行。"""
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    records = [
        {
            "id": f"bk_{i}",
            "images": [],
            "messages": [
                {"role": "user", "content": f"{CATEGORIES[i % 7]} 的问题 {i}？"},
                {"role": "assistant", "content": f"{CATEGORIES[i % 7]} 的参考答案 {i}"},
            ],
            "category": CATEGORIES[i % 7],
        }
        for i in range(84)
    ]
    records.append({"text": "书里的一整段正文。", "category": "作战应用"})
    (tsv_dir / "book_vqa.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return tmp_path, tsv_dir


def _scored_file(tmp_path, tsv_dir, scores):
    """假装裁判已经打过分，省掉端到端测试里的裁判调用。"""
    from eval_tool.io import load_truth_dataset

    truth = load_truth_dataset(tsv_dir, "book_vqa", {"category_field": "category"})
    scored = truth.copy()
    scored["prediction"] = "模型回答"
    scored["quality_score"] = [scores[row] for row in scored["category"]]
    scored["hit"] = [1 if scores[row] >= 70 else 0 for row in scored["category"]]
    path = tmp_path / "scored.xlsx"
    scored.to_excel(path, index=False)
    return str(path)


def _book_config(tmp_path, tsv_dir, scored_path, out_dir=None, **overrides):
    params = {"category_field": "category"}
    return EvalConfig(
        tsv_dir=tsv_dir,
        out_dir=out_dir or tmp_path / "out",
        cache_dir=tmp_path / "cache",
        datasets={"book_vqa": "book_vqa"},
        dataset_kinds={"book_vqa": "judge_text"},
        dataset_params={"book_vqa": params},
        models=[ModelConfig(name="base", scored_paths={"book_vqa": scored_path})],
        baseline_model="base",
        enabled_datasets=["book_vqa"],
        do_pairwise=False,
        do_length_control=False,
        bootstrap_n=20,
        report_dims=[{"key": "category", "from": "category", "min_n": 10}],
        **overrides,
    )


def test_each_class_gets_its_own_0_100_score(book_workspace, tmp_path):
    scores = {"作战应用": 80, "拱形基础": 80, "结构强度": 92, "材料工艺": 64,
              "飞行原理": 71, "维修保障": 88, "任务规划": 55}
    written = run(_book_config(tmp_path, book_workspace[1],
                               _scored_file(tmp_path, book_workspace[1], scores)))

    breakdown = pd.read_csv(written["breakdown.csv"])
    quality = breakdown[
        (breakdown["dim"] == "category") & (breakdown["metric"] == "quality_score")
    ].set_index("value")

    assert set(quality.index) == set(CATEGORIES)
    for category, expected in scores.items():
        assert quality.loc[category, "score"] == pytest.approx(expected)
        assert quality.loc[category, "n"] == 12
        assert quality.loc[category, "status"] == "ok"


def test_the_text_only_line_never_becomes_a_scored_row(book_workspace, tmp_path):
    scores = {category: 80 for category in CATEGORIES}
    written = run(_book_config(tmp_path, book_workspace[1],
                               _scored_file(tmp_path, book_workspace[1], scores)))

    breakdown = pd.read_csv(written["breakdown.csv"])
    per_class = breakdown[
        (breakdown["dim"] == "category") & (breakdown["metric"] == "hit")
    ]
    # 84 条问答记录，语料行不在里面
    assert int(per_class["n"].sum()) == 84


def test_score_summary_also_reports_the_seven_classes(book_workspace, tmp_path):
    scores = {category: 80 for category in CATEGORIES}
    written = run(_book_config(tmp_path, book_workspace[1],
                               _scored_file(tmp_path, book_workspace[1], scores)))

    summary = pd.read_csv(written["score_summary.csv"])

    for category in CATEGORIES:
        assert category in summary.columns
        assert summary.iloc[0][f"{category}_n"] == 12


def test_min_category_n_decides_whether_the_headline_total_exists(book_workspace, tmp_path):
    """每类只有 12 条，默认 30 行的门槛会把 total_score 清空。"""
    scores = {category: 80 for category in CATEGORIES}
    scored_path = _scored_file(tmp_path, book_workspace[1], scores)
    tsv_dir = book_workspace[1]

    gated = run(_book_config(tmp_path, tsv_dir, scored_path))
    lowered = run(_book_config(tmp_path, tsv_dir, scored_path,
                               min_category_n=10, out_dir=tmp_path / "out2"))

    assert pd.isna(pd.read_csv(gated["score_summary.csv"]).iloc[0]["total_score"])
    assert pd.read_csv(lowered["score_summary.csv"]).iloc[0]["total_score"] == 1.0
