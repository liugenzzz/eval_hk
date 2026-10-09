import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.scorers import ScoringContext
from eval_tool.scorers.spatial import parse_relation, score_spatial_relation


@pytest.mark.parametrize("text, axis, expected", [
    # gold answers as they appear in the mbjc test split
    ("戴黑墨镜的那副眼镜位于人员的左侧。", "lr", "左侧"),
    ("位于柜子的左侧。", "lr", "左侧"),
    ("在手表的右侧。", "lr", "右侧"),
    ("窗户在杆子的下方。", "ud", "下方"),
    # the question restated, then answered
    ("左侧还是右侧？是右侧。", "lr", "右侧"),
    ("偏左还是偏右？偏右。", "lr", "右侧"),
    # a negated direction is not the answer
    ("不在柜子的左侧，而在右侧。", "lr", "右侧"),
    # 左右 / 上衣 are not directions on the asked axis
    ("相距十米左右，在右边。", "lr", "右侧"),
    ("穿黄色上衣的人员在桌子的下方。", "ud", "下方"),
    ("It is to the left of the cabinet.", "lr", "左侧"),
    ("The window is below the pole.", "ud", "下方"),
    # long answers: the verdict first, both directions in the analysis (real 8B / 2B outputs)
    ("眼镜在电视的左边。\n\n从图像中可以看到，一位男士站在电视屏幕的左侧……电视屏幕位于他的右侧，因此眼镜在电视的左边。",
     "lr", "左侧"),
    ("带白色标志的那双运动鞋（即右下角那双）相对足球是**偏右**的。\n\n- 足球位于画面的**左侧**，靠近左下角。",
     "lr", "右侧"),
    ("根据图片内容，我们可以看到：\n- 人物穿着白色的短裤。\n\n鞋子在短裤的底部，而短裤覆盖在鞋子之上。\n\n因此，鞋位于短裤的下方。",
     "ud", "下方"),
    # both directions, nothing to choose by: unanswered, not guessed
    ("左侧和右侧都有。", "lr", None),
    ("", "lr", None),
    ("位于上方。", "lr", None),
])
def test_parse_relation(text, axis, expected):
    assert parse_relation(text, axis) == expected


def _data(pred, relation="左侧", axis="lr"):
    return pd.DataFrame({"prediction": pred, "meta.relation": relation, "meta.relation_axis": axis})


def test_scores_against_meta_relation():
    data = pd.DataFrame({
        "prediction": ["在柜子的左侧。", "在右侧。", "说不清。"],
        "meta.relation": ["左侧", "左侧", "左侧"],
        "meta.relation_axis": ["lr", "lr", "lr"],
    })
    ctx = ScoringContext(dataset_key="spatial_relation", kind="spatial_relation", model_name="m")
    scored = score_spatial_relation(data, ctx)
    assert scored["hit"].tolist() == [1, 0, 0]
    assert scored["parse_ok"].tolist() == [True, True, False]


def test_missing_gold_is_an_error_not_a_miss():
    ctx = ScoringContext(dataset_key="spatial_relation", kind="spatial_relation", model_name="m")
    with pytest.raises(ValueError):
        score_spatial_relation(_data(["左侧"], relation=None), ctx)


def test_registered_as_code_kind():
    assert scorers.get("spatial_relation").engine == scorers.CODE
