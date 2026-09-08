import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.config import ConfigError, EvalConfig, ModelConfig, parse_datasets
from eval_tool.judge import JudgeClient
from eval_tool.run_eval import run


def test_legacy_dataset_keys_keep_their_kind_without_declaring_one():
    config = EvalConfig(
        tsv_dir=".", out_dir=".", cache_dir=".",
        datasets={"mcq": "aero_mcq", "judge": "aero_judge", "vqa": "aero_vqa"},
        models=[],
    )
    assert config.kind_for("mcq") == "choice"
    assert config.kind_for("judge") == "choice"
    assert config.kind_for("vqa") == "judge_text"


def test_new_dataset_key_must_declare_a_kind():
    config = EvalConfig(
        tsv_dir=".", out_dir=".", cache_dir=".",
        datasets={"ground": "eval_set_v1"},
        models=[],
    )
    with pytest.raises(ConfigError, match="没有声明 kind"):
        config.kind_for("ground")


def test_declared_kind_overrides_the_legacy_key_default():
    config = EvalConfig(
        tsv_dir=".", out_dir=".", cache_dir=".",
        datasets={"vqa": "aero_vqa"},
        models=[],
        dataset_kinds={"vqa": "choice"},
    )
    assert config.kind_for("vqa") == "choice"


def test_parse_datasets_accepts_both_the_string_and_object_forms():
    names, kinds, params = parse_datasets(
        {
            "mcq": "aero_mcq",
            "ground": {"name": "eval_set_v1", "kind": "grounding_single", "params": {"iou_gate": 0.5}},
            "plain": {"name": "aero_vqa"},
        }
    )
    assert names == {"mcq": "aero_mcq", "ground": "eval_set_v1", "plain": "aero_vqa"}
    assert kinds == {"ground": "grounding_single"}
    assert params == {"ground": {"iou_gate": 0.5}}


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"mcq": ""},
        {"": "aero_mcq"},
        {"mcq": {"kind": "choice"}},
        {"mcq": {"name": "aero_mcq", "params": []}},
    ],
)
def test_parse_datasets_rejects_broken_shapes(raw):
    with pytest.raises(ConfigError):
        parse_datasets(raw)


def test_unknown_kind_names_the_kinds_that_do_exist():
    with pytest.raises(scorers.UnknownKindError) as exc:
        scorers.get("grounding_single")
    assert "judge_text" in str(exc.value)


def test_kind_cannot_be_registered_twice():
    with pytest.raises(ValueError, match="重复注册"):
        scorers.register("choice")(lambda data, ctx: data)


def test_registered_engines_say_which_scorers_call_a_model():
    assert scorers.get("choice").engine == scorers.CODE
    assert scorers.get("choice").needs_judge is False
    assert scorers.get("judge_text").engine == scorers.JUDGE
    assert scorers.get("judge_text").needs_judge is True
    # 有确定答案的形态不做 pairwise：那是纯浪费裁判调用。
    assert scorers.get("choice").pairwise is False


def _write_choice_dataset(tsv_dir, name, index, answer):
    pd.DataFrame(
        [
            {
                "index": index,
                "image": "img",
                "question": "选哪个？",
                "A": "燃油泵",
                "B": "滑油泵",
                "C": "作动筒",
                "D": "传感器",
                "answer": answer,
                "category": "P3",
                "l2-category": "零件图",
                "source_id": f"s{index}",
            }
        ]
    ).to_csv(tsv_dir / f"{name}.tsv", sep="\t", index=False)


def test_scoring_is_dispatched_by_kind_not_by_dataset_key_name(tmp_path):
    """一个谁也没听说过的数据集键名，只要声明 kind=choice 就照样按选择题打分。

    这就是这次重构要买到的东西：加一套数据（zb / book / 目标检测）只写配置，
    不进 run_eval 加分支。
    """
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    _write_choice_dataset(tsv_dir, "book_set", "1", "B")

    pred = tmp_path / "base_book.csv"
    pd.DataFrame([{"index": "1", "prediction": "答案是 B"}]).to_csv(pred, index=False)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            datasets={"book_mcq": "book_set"},
            dataset_kinds={"book_mcq": "choice"},
            models=[ModelConfig(name="base", paths={"book_mcq": str(pred)})],
            baseline_model="base",
            enabled_datasets=["book_mcq"],
            do_pointwise=False,
            do_pairwise=False,
            bootstrap_n=20,
        )
    )
    summary = pd.read_csv(written["report_summary.csv"])
    assert summary.loc[0, "book_mcq:overall"] == 1.0


def test_choice_style_param_picks_the_binary_option_set(tmp_path):
    """判断题只有 A/B。以前这件事靠数据集键名叫不叫 "judge" 来判，现在
    可以用 params 显式声明，键名叫什么都行。"""
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    pd.DataFrame(
        [{"index": "1", "image": "img", "question": "说法正确吗？", "A": "正确", "B": "错误",
          "answer": "B", "category": "R1", "l2-category": "结构图", "source_id": "s1"}]
    ).to_csv(tsv_dir / "zb_tf.tsv", sep="\t", index=False)

    pred = tmp_path / "base_tf.csv"
    # "错误" 只有在二值口径下才判成 B；四选一口径下抽不出选项。
    pd.DataFrame([{"index": "1", "prediction": "这个说法是错误的"}]).to_csv(pred, index=False)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            datasets={"zb_tf": "zb_tf"},
            dataset_kinds={"zb_tf": "choice"},
            dataset_params={"zb_tf": {"choice_style": "judge"}},
            models=[ModelConfig(name="base", paths={"zb_tf": str(pred)})],
            baseline_model="base",
            enabled_datasets=["zb_tf"],
            do_pointwise=False,
            do_pairwise=False,
            bootstrap_n=20,
        )
    )
    summary = pd.read_csv(written["report_summary.csv"])
    assert summary.loc[0, "zb_tf:overall"] == 1.0


def test_unknown_kind_fails_before_any_judge_call(tmp_path, monkeypatch):
    """跑完两小时推理再发现打分器不存在，代价太大 —— 起跑前就要报出来。"""
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    _write_choice_dataset(tsv_dir, "whatever", "1", "B")

    def _boom(self, *args, **kwargs):
        raise AssertionError("judge must not be constructed before the plan is validated")

    monkeypatch.setattr(JudgeClient, "judge_pointwise", _boom)

    with pytest.raises(scorers.UnknownKindError):
        run(
            EvalConfig(
                tsv_dir=tsv_dir,
                out_dir=tmp_path / "out",
                cache_dir=tmp_path / "cache",
                datasets={"grounding": "whatever"},
                dataset_kinds={"grounding": "bbox_deviation_v9"},
                models=[ModelConfig(name="base", paths={"grounding": "nope.csv"})],
                baseline_model="base",
                enabled_datasets=["grounding"],
                bootstrap_n=20,
            )
        )


def test_enabled_dataset_that_was_never_declared_is_ignored(tmp_path):
    """enabled_datasets 的默认值带着 mcq/judge/vqa 三个键。只声明了其中一个的
    配置不该因为另外两个没数据就往 warnings.log 里灌无关警告。"""
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    _write_choice_dataset(tsv_dir, "aero_mcq", "1", "B")

    pred = tmp_path / "base_mcq.csv"
    pd.DataFrame([{"index": "1", "prediction": "答案是 B"}]).to_csv(pred, index=False)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            datasets={"mcq": "aero_mcq"},
            models=[ModelConfig(name="base", paths={"mcq": str(pred)})],
            baseline_model="base",
            do_pointwise=False,
            do_pairwise=False,
            bootstrap_n=20,
        )
    )
    warn_text = written["warnings.log"].read_text(encoding="utf-8") if "warnings.log" in written else ""
    assert "judge prediction path" not in warn_text
    assert "vqa prediction path" not in warn_text
