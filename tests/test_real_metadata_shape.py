"""真实 test.jsonl 的字段形态回归锁。

`tests/fixtures/eval_set_real_shape.jsonl` 是照着构建端实际产出的样本复刻的十条，
覆盖了七种任务和几个容易踩的形态：

- 拒答类（`exist_negative` / `count_class`）的 `difficulty` / `area_ratio` /
  `equiv_px` / `size_bucket` **四个字段为 null**
- `count_class` 同时带 `count` / `counting` / `polarity` / `hard_negative`
- `detect_describe` **没有 `describe_kind`**，但有 `n_boxes`
- `region_identify` 有 `answer_format: "short"` 的一路（「用一个词回答」）
- 记录 id 里带点和连字符（`frame_47700_jpg.rf.86d6...`、`...-2913-4a5d-...`）
- 类别表里存在上下位关系（`人员` / `一般人员` / `军事人员`）

构建端改了 metadata 的字段名或取值形态，这里会先红 —— 那比评估跑完发现某一列全是
空要早得多。
"""

from pathlib import Path

import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.eval_set import load_eval_set

FIXTURE = Path(__file__).parent / "fixtures" / "eval_set_real_shape.jsonl"
GROUND_TASKS = ["ground_appearance", "ground_contrast", "ground_full", "ground_part"]


@pytest.fixture
def eval_set():
    return load_eval_set(FIXTURE)


def _score(kind, select, predictions, **params):
    frame = load_eval_set(FIXTURE, select=select)
    frame = frame.assign(model="sft", dataset=kind,
                         prediction=[predictions[i] for i in range(len(frame))])
    ctx = scorers.ScoringContext(dataset_key=kind, kind=kind, model_name="sft", params=params)
    return scorers.get(kind).score(frame, ctx)


def test_every_metadata_field_the_report_dimensions_need_is_present(eval_set):
    for column in ("meta.task_type", "meta.difficulty", "meta.size_bucket", "meta.equiv_px",
                   "meta.area_ratio", "meta.label", "meta.describe_kind", "meta.count",
                   "meta.counting", "meta.polarity", "meta.hard_negative",
                   "meta.answer_format", "meta.attribute", "meta.attribute_kind",
                   "meta.n_boxes", "meta.bbox_scale", "meta.source_image"):
        assert column in eval_set.columns, f"metadata 少了 {column}"


def test_refusal_and_counting_rows_carry_null_difficulty_and_size(eval_set):
    """拒答类没有焦点框，四个字段是 null。报表按这两根轴拆时它们不该被算成一个
    叫 'None' 的档。"""
    rows = eval_set[eval_set["task_type"].isin(["exist_negative", "count_class"])]
    assert len(rows) == 2
    for column in ("meta.difficulty", "meta.area_ratio", "meta.equiv_px", "meta.size_bucket"):
        assert rows[column].isna().all(), column


def test_detect_describe_has_no_describe_kind_but_has_n_boxes(eval_set):
    """轮 2 的描述没有 describe_kind —— 它的上游是坐标回指，不属于七种 kind 里的
    任何一种。范围合规因此判不了（scope_ok=NA），这是对的，不是漏判。"""
    row = eval_set[eval_set["task_type"] == "detect_describe"].iloc[0]
    assert pd.isna(row["meta.describe_kind"])
    assert row["meta.n_boxes"] == 2


def test_short_answer_variant_of_region_identify_is_marked(eval_set):
    formats = set(eval_set[eval_set["task_type"] == "region_identify"]["meta.answer_format"])
    assert formats == {"normal", "short"}


def test_ids_with_dots_and_hyphens_survive_the_turn_suffix(eval_set):
    indexes = set(eval_set["index"])
    assert "frame_47700_jpg.rf.86d6075a74413857eea770fa9ed2b9f0_ground_full_11330__t1" in indexes
    assert all(index.count("__t") == 1 for index in indexes)


def test_grounding_scores_across_the_real_size_range():
    """真实数据的 equiv_px 从 29 一路到 496。两把尺子在两端给出的结论完全不同，
    这正是达标率必须按 size_bucket 拆开报的原因。"""
    out = _score("grounding_single", [{"task_type": GROUND_TASKS, "turn": 1}], {
        0: '{"bbox_2d":[196,541,206,580],"label":"一般人员"}',
        1: '{"bbox_2d":[500,400,1000,480],"label":"矿砂船"}',
        2: '```json\n{"bbox_2d":[345,523,362,606]}\n```',
        3: '{"bbox_2d":[386,55,646,958],"label":"铲子"}',
    })
    assert out["pass_dev"].tolist() == [1, 1, 1, 1]
    small = out[out["meta.size_bucket"] == "small"].iloc[0]
    large = out[out["meta.equiv_px"] == 496.0].iloc[0]
    # 同样判「达标」，图幅尺看都很准；目标尺下小目标偏了 12%，大目标是 0。
    assert small["dev_mean4_pct"] < 1.0 and small["dev_obj"] > 0.1
    assert large["dev_obj"] == 0.0


def test_code_fence_from_the_model_does_not_cost_anything():
    out = _score("grounding_single", [{"task_type": GROUND_TASKS, "turn": 1}], {
        0: '{"bbox_2d":[194,539,204,578]}',
        1: '{"bbox_2d":[488,395,999,475]}',
        2: '```json\n{"bbox_2d":[342,521,360,604]}\n```',
        3: '{"bbox_2d":[386,55,646,958]}',
    })
    fenced = out[out["parse_flags"].str.contains("code_fence")].iloc[0]
    assert fenced["pass_dev"] == 1


def test_multi_box_answer_parses_as_two_boxes():
    out = _score("grounding_multi", [{"task_type": ["detect_describe"], "turn": 1}],
                 {0: '[{"bbox_2d":[697,516,884,620],"label":"坦克"},'
                     '{"bbox_2d":[700,521,885,610],"label":"坦克"}]'})
    assert out.loc[0, "n_gt_boxes"] == 2
    assert out.loc[0, "n_matched"] == 2
    assert out.loc[0, "count_correct"] == 1


def test_counting_reads_the_number_out_of_the_builders_answer_wording():
    """构建端的答案写法是「图中一共可以数出 10 名人员」。"""
    out = _score("counting", [{"task_type": ["count_class"], "turn": 1}],
                 {0: "图中一共可以数出 9 名人员。"})
    assert out.loc[0, "gold_count"] == 10
    assert out.loc[0, "pred_count"] == 9
    assert out.loc[0, "count_error"] == -1
    assert out.loc[0, "count_bin"] == "密集"
    assert out.loc[0, "counting"] == "exact"


def test_exist_negative_affirmative_wording_is_read_as_yes():
    """「有，可以看到 2 个坦克。」里含「有」也含「看到」，但不含任何否定词。
    判定顺序反了会把每一条都判成 yes。"""
    out = _score("exist_negative", [{"task_type": ["exist_negative"], "turn": 1}],
                 {0: "有，可以看到 2 个坦克。"})
    assert out.loc[0, "said_yes"] == 1
    assert out.loc[0, "hit"] == 1


@pytest.fixture
def real_class_table(tmp_path):
    """这批数据里真的有 人员 / 一般人员 / 军事人员 这一组（构建端 classes.py 的注释
    点名的就是它），还有大量工业件和武器类。"""
    path = tmp_path / "classes.yaml"
    path.write_text(
        "names:\n" + "".join(f"  - {n}\n" for n in
                             ["人员", "一般人员", "军事人员", "步枪", "坦克", "矿砂船",
                              "铲子", "锤子", "连接器", "蜂鸣器", "电阻器", "电源插座"]),
        encoding="utf-8",
    )
    return str(path)


def _ident_rows():
    return pd.DataFrame([
        {"index": "1", "meta.label": "一般人员", "answer": "这是一般人员。", "prediction": "这是人员"},
        {"index": "2", "meta.label": "人员", "answer": "这是人员。", "prediction": "这是军事人员"},
        {"index": "3", "meta.label": "步枪", "answer": "这是步枪。", "prediction": "这是步枪。"},
    ])


def test_hypernym_classes_in_this_dataset_are_judged_by_direction(real_class_table):
    """金标「一般人员」而模型答「人员」是答粗（上位命中），反过来是答细（下位命中），
    两种病要分开计数。"""
    ctx = scorers.ScoringContext(dataset_key="reg", kind="object_ident", model_name="sft",
                                params={"classes_yaml": real_class_table})
    out = scorers.get("object_ident").score(_ident_rows(), ctx)
    assert out["match_kind"].tolist() == ["hypernym", "hyponym", "exact"]
    assert (out["class_table_authoritative"] == 1).all()


def test_a_fallback_class_table_turns_a_hyponym_into_a_false_exact_hit():
    """**这是必须配 classes_yaml 的硬理由。**

    兜底表由评估集里出现过的 label 拼成，「军事人员」不在其中。模型答「这是军事人员」
    时，最长匹配只能从这句话里抠出表里有的「人员」，于是金标「人员」被判成**精确命中**
    —— 一个答细了（很可能在幻觉一个看不清的属性）的回答拿了满分，E 组主指标偏高。

    配上真实类别表，同一条正确判成下位命中（见上一个测试）。
    """
    ctx = scorers.ScoringContext(dataset_key="reg", kind="object_ident", model_name="sft")
    out = scorers.get("object_ident").score(_ident_rows(), ctx)
    assert out.loc[1, "match_kind"] == "exact"          # 错的，但兜底表看不出来
    assert (out["class_table_authoritative"] == 0).all()  # 报表里能看见这次不可信


def test_describe_scope_check_runs_against_the_builder_word_lists(tmp_path):
    """范围合规的词表读构建端的 prompts/describe/*.txt。这里复刻 appearance 那份。"""
    prompt_dir = tmp_path / "describe"
    prompt_dir.mkdir()
    (prompt_dir / "appearance.txt").write_text(
        "#! kind: appearance\n"
        "#! must-not: 位于 画面 方位 左侧 右侧 上方 下方 旁边 附近 周围 背景\n",
        encoding="utf-8",
    )
    (prompt_dir / "contrast.txt").write_text("#! kind: contrast\n#! must-not:\n", encoding="utf-8")

    frame = load_eval_set(FIXTURE, select=[{"task_type": ["ground_appearance", "ground_contrast"],
                                            "turn": 2}])
    frame = frame.assign(model="sft", dataset="describe", prediction=[
        "这是一名身穿鲜艳粉色外套的人员，位于画面左侧。",   # appearance 跑去说方位
        "这一艘比另外两艘更长，吃水更深。",
    ])
    ctx = scorers.ScoringContext(
        dataset_key="describe", kind="describe", model_name="sft", do_pointwise=False,
        params={"describe_prompt_dir": str(prompt_dir)},
    )
    out = scorers.get("describe").score(frame, ctx)
    assert out["scope_ok"].tolist() == [0, 1]
    assert out.loc[0, "scope_violations"] == "位于,画面,左侧"
    assert out["upstream_form"].tolist() == ["文字指代", "文字指代"]
