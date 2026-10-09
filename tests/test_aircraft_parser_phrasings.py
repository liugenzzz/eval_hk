"""Natural phrasings the aircraft code scorers must read correctly.

Each case was a misjudgment (or a crash, for region) before the parsers learned
negation scope, synonym coverage and corner words. A parser that fails these marks
natural-language answers wrong, which penalises base models - they phrase freely -
far more than fine-tuned ones that copy the gold style, and so inflates SFT gains.
"""

import pandas as pd
import pytest

from eval_tool.scorers import ScoringContext
from eval_tool.scorers.aircraft import (LONG_ANSWER, parse_category, parse_presence, parse_region,
                                        score_region)

# Condensed from real Qwen3-VL-8B-Instruct answers on aircraft_eval_v2: a checklist
# that names every defect it rules out, a caveat, and a verdict.
CHECKLIST = ("\n\n**检查依据与细节：**\n1. 螺钉整体结构完整，无明显断裂、变形或缺损。\n"
             "2. 螺纹部分连续、完整，无明显缺牙、乱牙或严重锈蚀。\n3. 表面无明显油污、锈迹或氧化层。\n"
             "4. 电路板表面没有出现明显的刮痕、凹陷、烧焦或物理损伤，没有虚焊或焊点开裂的迹象。\n")
LONG_PRESENCE = [
    ("根据提供的图像，对该蒙皮壁板紧固螺钉进行外观检查，结论如下：\n\n✅ **外观检查结论：合格**" + CHECKLIST
     + "\n**结论：该蒙皮壁板紧固螺钉外观检查合格，可投入使用。**", False),
    ("根据图像分析，该待检部件未发现明显的裂纹或变形。" + CHECKLIST
     + "- 虽然表面有一处微小的凹陷（可能是制造痕迹），但这些并不构成结构性的裂纹或变形。\n\n"
       "结论：该部件在视觉上无裂纹或变形，状态正常。", False),
    ("根据您提供的电路板照片，经过仔细检查，**未发现明显的异常或损坏**。以下是详细分析：" + CHECKLIST
     + "\n**结论**：\n这张照片中的电路板看起来是**正常、完好**的。\n\n"
       "**注意**：照片无法判断内部电气连接或元件参数是否符合设计要求。", False),
    ("根据您提供的图片，对该紧固件套件的外观进行清点，目前未发现明显的不合格项。" + CHECKLIST
     + "**结论**：从外观上看，该紧固件套件无明显缺陷或不合格项，可以认为是合格品。", False),
    ("根据提供的图像，该待检部件存在以下外观缺陷，需记录：\n1. 按钮表面有明显划痕或污渍。" + CHECKLIST
     + "**结论**：该部件存在多个需记录的外观缺陷，主要集中在按钮表面。", True),
]
LONG_CATEGORY = [
    ("根据图像中机身蒙皮壁板的损伤特征，该缺陷可被识别为：\n\n**腐蚀性损伤（Corrosion Damage）**\n\n### 判断依据："
     + CHECKLIST + "图像中可见明显的局部凹陷与孔洞，表面划痕周围颜色发暗。", "corrosion"),
    ("根据图像分析，画面正中那一处缺陷属于**划痕**。\n\n理由如下：" + CHECKLIST
     + "它贯穿了多个螺丝孔的中心区域，形态上具有明显的“划过”痕迹，而非点状、块状或凹坑。", "scratch"),
]


PRESENCE = [
    # a named defect outweighs incidental "normal" words elsewhere in the answer
    ("发现划伤，其余部位完好。", True),
    ("存在裂纹，建议按正常流程复检。", True),
    ("有凹坑，但不影响正常使用。", True),
    ("裂纹没有继续扩展，但仍需处理。", True),
    ("检出紧固件缺失，其他紧固件良好。", True),
    ("否", False),
    ("目前未发现明显的不合格项。", False),
    ("没有虚焊或焊点开裂的迹象。", False),
    ("表面平整，没有出现明显的刮痕、凹陷、烧焦或物理损伤。", False),
    ("无人机表面裂纹明显。", True),
    # examples that spell out a denial are denied with it
    ("该机身蒙皮表面未见明显损伤，如凹陷、撕裂、严重锈蚀或结构性变形。", False),
    ("表面涂层未见明显缺陷（如裂纹、严重锈蚀、凹陷等）。", False),
    ("该机身蒙皮不符合视觉合规性标准，需恢复无瑕疵的外观。", True),
    ("该电路板基本符合外观标准。", False),
    # referring questions: a bare no, and anomalies outside the five classes (real SFT answers)
    ("无", False),
    ("无异物残留", False),
    ("框内有明显老化痕迹。", True),
    # a list member may repeat the head's modifier (real Qwen2-VL-7B answer)
    ("这个检查口盖表面比较平整，没有明显的凹陷、凸起、裂缝或明显的锈蚀等缺陷。", False),
    # one negation scopes over a whole list, and only over that list
    ("未发现腐蚀、裂纹或凹坑。", False),
    ("螺母表面平整无损伤，未见变色或腐蚀迹象。", False),
    ("未见裂纹，但有轻微腐蚀。", True),
    ("未见油污和锈蚀，但有一处凹坑。", True),
    ("没有裂纹，也没有腐蚀，表面完好。", False),
    ("并没有发现什么明显问题。", False),
    ("裂纹：无；腐蚀：无。", False),
    ("紧固件齐全，未缺失。", False),
    ("紧固件没有缺失。", False),
    # look like a negation, are not one
    ("没有修复的裂纹仍然可见。", True),
    ("不是很严重的划伤。", True),
    ("无规则的划痕分布在蒙皮上。", True),
    ("无人机蒙皮上有一道裂纹。", True),
    ("有问题，但外观基本正常。", True),
    # the question restated inside the answer
    ("是否存在缺陷：否。", False),
    ("是否存在缺陷：存在，为腐蚀。", True),
    ("是的", True),
    ("无法判断。", None),
    # English words on letter boundaries: "evident" holds no "dent"
    ("No visible defects.", False),
    ("No staining or corrosion.", False),
    ("Yes, there is corrosion.", True),
    ("It is evident that the panel is normal.", False),
]

CATEGORY = [
    ("该异常是腐蚀锈蚀，不是划伤。", "corrosion"),
    ("不是裂纹，而是划痕。", "scratch"),
    ("排除凹坑，判断为腐蚀。", "corrosion"),
    ("紧固件没有缺失，是腐蚀。", "corrosion"),
    ("表面有锈迹。", "corrosion"),
    ("出现裂口。", "crack"),
    ("紧固件丢失。", "fastener_missing"),
    ("缺少螺钉。", "fastener_missing"),
    ("铆钉脱落。", "fastener_missing"),
    ("这处是紧固件掉了。", "fastener_missing"),
    ("螺丝不见了。", "fastener_missing"),
    ("fastener_missing", "fastener_missing"),
    ("存在crack", "crack"),
]

REGION = [
    ("位于画面左下方。", "lower_left"),
    ("右上方有一处划伤。", "upper_right"),
    ("在左上方。", "upper_left"),
    ("右下方。", "lower_right"),
    ("左侧上方。", "upper_left"),
    ("上方偏右。", "upper_right"),
    ("右侧下方。", "lower_right"),
    ("从左上延伸到右下。", None),
    # a part described by side is not an image region, and must not override 正中
    ("缺陷位于画面正中间，靠近机身左侧的铆钉。", "center"),
    # sides that name the image (real Qwen3-VL-8B answers)
    ("裂纹偏向画面的左侧。", "center_left"),
    # "centre, leaning right" is still the centre cell: that is how the golds read
    ("图中的裂纹大致位于画面的中央偏右区域。", "center"),
    ("裂纹位于画面中心偏左的位置。", "center"),
    # coordinates instead of words: the same 3x3 rule the gold words were made with
    ('```json\n[{"bbox_2d": [484, 416, 660, 598], "label": "凹坑"}]\n```', "center"),
    ("[226.52, 225.88]", "upper_left"),
    ('[{"bbox_2d": [10, 10, 90, 90]}, {"bbox_2d": [900, 900, 990, 990]}]', None),
]


@pytest.mark.parametrize("text, expected", PRESENCE)
def test_presence_phrasings(text, expected):
    assert parse_presence(text) is expected


@pytest.mark.parametrize("text, expected", CATEGORY)
def test_category_phrasings(text, expected):
    assert parse_category(text, "")[0] == expected


@pytest.mark.parametrize("text, expected", LONG_PRESENCE)
def test_long_answers_are_read_by_their_verdict(text, expected):
    assert len(text) > LONG_ANSWER
    assert parse_presence(text) is expected


@pytest.mark.parametrize("text, expected", LONG_CATEGORY)
def test_long_category_answers_are_read_by_their_verdict(text, expected):
    assert len(text) > LONG_ANSWER
    assert parse_category(text, "")[0] == expected


@pytest.mark.parametrize("text, expected", [
    # an explicit conclusion is read first even when the answer is short
    ("结论：未发现异常。\n\n分析：表面平整，未见巡检中常见的裂纹、翘曲或粘接不良等异常。", False),
    # markdown emphasis between the negation and the defect
    ("根据您提供的图片，该航电模块电路板表面**没有**明显的腐蚀现象。", False),
])
def test_conclusion_and_markdown(text, expected):
    assert parse_presence(text) is expected


def test_runaway_list_answer_parses_in_linear_time():
    # Qwen3-VL-2B greedy output on aircraft_eval_v2 that hung an eval run for 14 hours
    # (catastrophic backtracking in the negated-list pattern): a repeated 、-list.
    import time

    items = ["腐蚀", "锈蚀", "破损", "裂纹", "凹陷", "污渍", "划痕", "磨损", "老化", "污垢"]
    runaway = "图中缺陷属于：" + "、".join(items * 120)
    clean = "检查项：" + "、".join(["油污", "灰尘", "水渍", "标签"] * 300) + "都没有问题"
    start = time.time()
    assert parse_category(runaway, "") == (None, "multiple_classes")
    assert parse_presence(runaway) is True
    parse_presence(clean)
    parse_region(runaway)
    assert time.time() - start < 3


def test_short_answers_are_still_read_whole():
    # the second sentence overrides the first; a short answer must not be cut to it
    assert parse_presence("外观基本正常。但右下角有一处裂纹。") is True


def test_category_still_reports_two_named_classes():
    assert parse_category("裂纹和凹坑", "") == (None, "multiple_classes")


@pytest.mark.parametrize("text, expected", REGION)
def test_region_corner_phrasings(text, expected):
    assert parse_region(text) == expected


def test_region_gold_with_corner_word_no_longer_aborts_the_slice():
    # This exact gold is in aircraft_eval_v2; the old parser could not read it, and
    # score_region raises on an unreadable gold, which took down the whole slice.
    gold = "画面右上方那个位置，就是裂纹所在。"
    data = pd.DataFrame({"answer": [gold, gold], "prediction": ["右上角", "左下角"]})
    scored = score_region(data, ScoringContext(dataset_key="aircraft", kind="aircraft_region", model_name="model"))
    assert scored["gold_region"].tolist() == ["upper_right", "upper_right"]
    assert scored["hit"].tolist() == [1, 0]
