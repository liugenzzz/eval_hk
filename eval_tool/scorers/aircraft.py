"""Deterministic aircraft-maintenance recognition and localization scorers."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from ..bbox import BAD_BOX, CODE_FENCE, JSON_REPAIRED, OUT_OF_RANGE, REORDERED, Box, iou, parse_boxes
from ..compliance import BOXES, TEXT
from ..matching import match_boxes
from ..scale import scale_of
from . import CODE, ScoringContext, register


CLASSES = ("fastener_missing", "crack", "corrosion", "dent", "scratch")

# A lost fastener is a fastener word plus a loss word, in either order ("紧固件丢失",
# "铆钉脱落", "缺少螺钉", "紧固件掉了") - too many pairings to list one by one. The gap
# between them may not hold a negation, or "紧固件没有缺失" would read as a loss.
_FASTENER = r"(?:紧固件|螺丝钉?|螺钉|螺栓|螺母|铆钉)"
_LOSS = r"(?:缺失|丢失|缺少|脱落|掉了|掉落|松脱|不见了|没有了|没了)"
# English words match on letter boundaries only: a bare substring test finds "dent"
# in "evident" and "rust" in "thrust". Not \b - Python counts CJK characters as word
# characters, so \b would miss the "crack" in "存在crack".
_L, _R = r"(?<![a-z])", r"(?![a-z])"
CLASS_PATTERNS = {
    "fastener_missing": re.compile(
        rf"{_FASTENER}[^，。；,;！!？?没未无不]{{0,3}}?{_LOSS}"
        rf"|(?:缺失|丢失|缺少|少了|缺)了?[一两几多]?[颗个枚根处]?{_FASTENER}"
        rf"|{_L}fastener[_ ]missing{_R}|{_L}missing\s+(?:fastener|rivet|screw|bolt|nut)s?{_R}"
        rf"|{_L}(?:fastener|rivet|screw|bolt|nut)s?\s+(?:(?:is|are)\s+)?(?:missing|lost|absent){_R}",
        re.I),
    "crack": re.compile(rf"裂纹|裂缝|开裂|裂口|{_L}crack(?:s|ed|ing)?{_R}|{_L}fracture[sd]?{_R}", re.I),
    "corrosion": re.compile(rf"腐蚀|锈蚀|生锈|锈迹|锈斑|{_L}rust(?:y|ed|ing)?{_R}|{_L}corro(?:sion|ded|sive){_R}", re.I),
    "dent": re.compile(rf"凹坑|凹陷|{_L}dent(?:s|ed)?{_R}", re.I),
    "scratch": re.compile(rf"划伤|划痕|刮伤|刮痕|{_L}scratch(?:es|ed)?{_R}", re.I),
}
UNCERTAIN = re.compile(
    r"无法确认|无法确定|无法判断|不能判断|不能确定|难以判断|难以确定|难以确认|不好判断|不确定|说不准|"
    r"看不清|无法辨认|uncertain|not sure|unclear|cannot tell|can't tell|unable to (?:tell|determine)", re.I)
NEGATIVE = re.compile(
    r"未见|未发现|没发现|没看到|没有|没问题|没毛病|不存在|无异常|无缺陷|无损伤|"
    r"正常|完好|良好|合格|通过|不用记录|未检出|不是|齐全|未(?:缺失|丢失|脱落|松脱|掉落)|(?<!是)否(?!则)|"
    r"符合[^，。；]{0,6}?(?:标准|要求|规范)|一致|对得上|无差异|"
    rf"{_L}(?:none|no defects?|normal){_R}", re.I
)
POSITIVE = re.compile(rf"存在|可见|发现|检出|有异常|有缺陷|不正常|不合格|{_L}(?:yes|present){_R}", re.I)
YES = re.compile(r"\s*(?:是|有|是的|有的|对|yes)\s*[。.!！]?\s*", re.I)
# A bare "无" / "否" answering "有无…" is a no.
NO = re.compile(r"\s*(?:无|否|没有|不是|不存在|no|none)\s*[。.!！]?\s*", re.I)

# --- negated mentions -------------------------------------------------------------
# "未见损伤" / "没有腐蚀、裂纹或凹坑" / "不是划伤" name a defect only to deny it. They
# are cut out before anything is counted, so that a negated class neither reports a
# defect nor competes with the real answer ("是腐蚀，不是划伤" is one class, not two).
# Between the negation and the defect only listed modifiers may stand: "没有明显异常"
# is a denial, "没有修复的裂纹" (an unrepaired crack) and "不是很严重的划伤" are not.
# pair_compare answers speak in differences from the reference part: 不一致 is a
# failure, 区别 / 差异 / 偏差 are defect words that can be denied ("没啥区别").
_FAILED = ("不正常", "不合格", "不通过", "没通过", "未通过", "不符合", "不一致")
_DEFECT_WORDS = (r"缺陷|异常|损伤|损坏|破损|瑕疵|故障|问题|毛病|缺失|丢失|脱落|不合格|不正常|不通过|不符合"
                 r"|不一致|区别|差异|偏差|异样"
                 # referring questions also ask after anomalies outside the five classes
                 r"|异物|老化|磨损|变形|松动"
                 rf"|{_L}(?:defects?|damaged?|anomal(?:y|ies|ous)|abnormal(?:it(?:y|ies))?|issues?|problems?){_R}")
_ANY = "|".join([p.pattern for p in CLASS_PATTERNS.values()] + [_DEFECT_WORDS])
_JOIN = rf"\s*(?:、|/|或者|或是|或|和|及|以及|与|也没有|也无|{_L}(?:and|or|nor){_R})\s*"
# One negation covers a whole list, and the list may also name things that are not
# defects: "未见变色或腐蚀迹象" denies the corrosion too. At least one member must be
# a defect, and members are joined by list words only - "没有修复的裂纹" has none.
# A defect member may carry a short noun in front ("没有虚焊或焊点开裂", "物理划伤"),
# but never across 的: that is a relative clause ("修复的裂纹"), not a compound.
#
# No member may contain a list word (、或和及与以也) either. Besides being right, that
# keeps the match linear: when members could swallow the joiners, a long repetitive
# answer ("焊盘/引脚断裂…焊盘/引脚变形…", which greedy decoding produces) left the
# engine trying every way to split the list - one eval run hung on it for 14 hours.
_CJK = r"(?:(?![的、或和及与以也])[一-鿿])"
_DEF = rf"(?:{_CJK}{{0,4}}?(?:{_ANY}))"
# ... and a member is a defect or, only where no defect starts, something else. With
# both allowed, "腐蚀、锈蚀、破损、…" (each ≤4 characters, so also a valid "other")
# parses 2^n ways, and every one is tried when a required continuation fails.
_OTHER = rf"(?:(?!{_DEF})(?:{_CJK}{{1,4}}|{_L}(?!(?:and|or|nor){_R})[a-z]+{_R}))"
# A member may repeat the head's modifier with 的 ("没有明显的凹陷、裂缝或明显的锈蚀").
# The 的 is required: a defect's own short prefix never contains 的, so the two cannot
# both parse the same text (ambiguity is what made the pattern exponential).
_MOD_WORDS = (r"明显|任何|什么|可见|有|发现|看到|看见|检测到|观察到|出现|产生|发生|其他|其它|别的|新的|啥|"
              r"太大|严重|较大|显著|肉眼可见|进一步|相关|类似")
_MEMBER_MOD = rf"(?:(?:{_MOD_WORDS})的\s*)?"
# Lists are bounded: a denial scopes over a handful of items, and an unbounded list is
# re-walked to its end from every start position (quadratic on a runaway answer).
_LIST = rf"(?:{_MEMBER_MOD}{_OTHER}{_JOIN}){{0,6}}{_MEMBER_MOD}{_DEF}(?:{_JOIN}{_MEMBER_MOD}(?:{_DEF}|{_OTHER})){{0,11}}"
_NEG = (r"未见|未发现|未检出|未看到|未观察到|未出现|没发现|没看到|没见到?|没有|不存在|看不到|看不出|找不到"
        r"|(?<!有)无(?!人)|没|不是|并非|而非|而不是|不属于|排除")
_MOD = rf"(?:{_MOD_WORDS})的?"
_NEG_EN = rf"{_L}(?:no|not|without|free\s+of|absence\s+of)(?:\s+(?:any|visible|obvious|apparent|signs?\s+of|evidence\s+of|a|an|the))*\s+"
# Examples that spell out what was denied are denied with it: "未见明显损伤，如凹陷、
# 撕裂、严重锈蚀" / "未见明显缺陷（如裂纹、严重锈蚀、凹陷等）".
_EXAMPLES = (rf"(?:[^，。；,;\n]{{0,6}}?[，,（(]?\s*(?:例如|比如|诸如|包括|如)[:：]?\s*"
             rf"(?:{_DEF}|{_OTHER})(?:{_JOIN}(?:{_DEF}|{_OTHER})){{0,11}})?")
NEGATED_DEFECT = re.compile(rf"(?:(?:{_NEG})\s*(?:{_MOD}\s*){{0,3}}|{_NEG_EN}){_LIST}{_EXAMPLES}", re.I)
# ... and the same denial placed after the defect: "裂纹：无", "腐蚀、裂纹都没有。"
NEGATED_AFTER = re.compile(
    rf"{_LIST}(?:情况|迹象|现象)?\s*[:：]?\s*(?:是|均|都|也)?\s*"
    rf"(?:没有|未见|未发现|没发现|不存在|无|否|none|no)的?(?=\s*(?:[，。；,;！!？?\n]|$))",
    re.I)
# A restated question ("是否存在缺陷：…") is not an answer; drop it before reading.
RESTATED = re.compile(r"(?:是否|有无|有没有)[^，。；,;！!？?:：\n]*")
DEFECT_ASSERTED = re.compile(
    r"(?:有|存在|发现|检出|可见|出现|看到)了?(?:明显的?|一处|一道|多处|少量|轻微的?|些许|一些)?"
    r"(?:异常|缺陷|损伤|损坏|破损|瑕疵|问题|毛病|故障|区别|差异|偏差|异样|异物|老化|磨损|变形|松动)"
    rf"|{_L}(?:defects?|damaged?|anomal(?:y|ies|ous)|abnormal){_R}", re.I)
BAD_FORMAT = {BAD_BOX, CODE_FENCE, JSON_REPAIRED, OUT_OF_RANGE, REORDERED}
OPTION = re.compile(r"^\s*([A-D])\s*[.．、:：]\s*(.+?)\s*$", re.I | re.M)
ANSWER_LETTER = re.compile(r"^\s*(?:答案\s*(?:是|为)?\s*[:：]?\s*)?([A-D])(?:\s*[.．、:：]|\s*$)", re.I)


def _text(value: Any) -> str:
    if value is None or value is pd.NA or isinstance(value, float) and math.isnan(value):
        return ""
    # markdown emphasis sits between words the parser reads together ("**没有**明显的腐蚀")
    return str(value).replace("**", "").strip()


def classes_in_text(value: Any) -> set[str]:
    text = _text(value)
    return {name for name, pattern in CLASS_PATTERNS.items() if pattern.search(text)}


def strip_negated(text: str) -> str:
    """Remove every mention that names a defect only to deny it."""
    return NEGATED_AFTER.sub(" ", NEGATED_DEFECT.sub(" ", text))


# Long answers - base models write a checklist, caveats and a verdict - mention every
# defect word they rule out ("无划痕、无锈蚀…") and hedge ("照片无法判断内部电气连接"),
# so the whole text says little. They are read verdict first: an explicit conclusion,
# then the opening sentence, then the whole text. Short answers, and every gold, are
# read whole, exactly as before.
LONG_ANSWER = 150
_CONCLUSION = re.compile(r"(?:结论|综上所述|综上|综合判断|综合评估|总结|判定结果|检查结论|最终判断)[\s*:：#，,]*([^\n。]+)")


# Longer than any real answer; a runaway generation is cut here before parsing.
MAX_PARSE = 4000


def answer_spans(text: str) -> list[str]:
    text = text[:MAX_PARSE]
    # An explicit conclusion is the model stating its verdict, whatever the length.
    spans = [m.group(1) for m in _CONCLUSION.finditer(text)][::-1]   # the last one is the final word
    if len(text) > LONG_ANSWER:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first = re.split(r"(?<=[。！!？?])", lines[0], maxsplit=1)[0]
        if first.rstrip("* ").endswith(("：", ":")) and len(lines) > 1:
            first += lines[1]               # "该缺陷可归类为：" + "**腐蚀**"
        spans.append(first)
    return [*spans, text]


def canonical_label(value: Any) -> str | None:
    text = _text(value).lower().replace("-", "_").replace(" ", "_")
    if text in CLASSES:
        return text
    found = classes_in_text(value)
    return next(iter(found)) if len(found) == 1 else None


def parse_presence(value: Any) -> bool | None:
    text = _text(value)
    for span in answer_spans(text) if text else []:
        verdict = _presence_of(span)
        # a verdict that itself hedges ("结论：无法确定") ends the reading
        if verdict is not None or UNCERTAIN.search(span):
            return verdict
    return None


def _presence_of(text: str) -> bool | None:
    if not text or UNCERTAIN.search(text):
        return None
    body = RESTATED.sub(" ", text)
    rest = strip_negated(body)
    # A defect that is named and not denied outweighs incidental "normal" words
    # elsewhere: "发现划伤，其余部位完好" and "存在裂纹，建议按正常流程复检" both
    # report a defect. 不合格 counts only when not denied ("未发现不合格项").
    if any(phrase in rest for phrase in _FAILED) or classes_in_text(rest) or DEFECT_ASSERTED.search(rest):
        return True
    if rest != body or NEGATIVE.search(rest) or NO.fullmatch(rest):
        return False
    if POSITIVE.search(rest) or YES.fullmatch(rest):
        return True
    return None


def parse_category(prediction: Any, question: Any) -> tuple[str | None, str]:
    text = _text(prediction)
    if not text:
        return None, "empty"
    result: tuple[str | None, str] = (None, "unrecognized_class")
    for span in answer_spans(text):
        result = _category_of(span, question)
        # a span that names no class says nothing; one that names two is ambiguous
        if result[1] != "unrecognized_class":
            return result
    return result


def _category_of(text: str, question: Any) -> tuple[str | None, str]:
    found = classes_in_text(strip_negated(text))
    if len(found) > 1:
        return None, "multiple_classes"
    text_class = next(iter(found)) if found else None
    letter = ANSWER_LETTER.match(text)
    if letter:
        options = {key.upper(): canonical_label(value) for key, value in OPTION.findall(_text(question))}
        option_class = options.get(letter.group(1).upper())
        if not option_class:
            return None, "unknown_option"
        if text_class and text_class != option_class:
            return None, "letter_text_conflict"
        return option_class, ""
    return (text_class, "") if text_class else (None, "unrecognized_class")


def _attach(data: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    result = data.copy()
    for name in rows[0] if rows else ():
        result[name] = [row[name] for row in rows]
    return result


@register("aircraft_presence", engine=CODE, answer_form=TEXT)
def score_presence(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        status = _text(row.get("meta.image_status"))
        if status not in {"normal", "anomalous"}:
            raise ValueError(f"aircraft_presence missing gold image_status: {status!r}")
        gold = status == "anomalous"
        pred = parse_presence(row.get("prediction"))
        rows.append({
            "gold_present": int(gold), "pred_present": pd.NA if pred is None else int(pred),
            "parse_ok": pred is not None, "hit": int(pred is gold),
            "false_positive": int(pred is True and not gold),
            "false_negative": int(pred is False and gold),
            "unanswered": int(pred is None),
        })
    return _attach(data, rows)


@register("aircraft_category", engine=CODE, answer_form=TEXT)
def score_category(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        gold = _text(row.get("meta.target_type"))
        if gold not in CLASSES:
            raise ValueError(f"aircraft_category missing five-class target: {gold!r}")
        pred, reason = parse_category(row.get("prediction"), row.get("question"))
        rows.append({
            "gold_class": gold, "pred_class": pred or "", "parse_ok": pred is not None,
            "parse_reason": reason, "hit": int(pred == gold),
        })
    return _attach(data, rows)


def _strict_box_format(value: Any, parsed_flags: frozenset[str]) -> bool:
    if parsed_flags & BAD_FORMAT:
        return False
    try:
        items = json.loads(_text(value))
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(items, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("bbox_2d"), list)
        and len(item["bbox_2d"]) == 4
        and canonical_label(item.get("label")) in CLASSES
        for item in items
    )


def _class_aware_matches(gold: tuple[Box, ...], pred: tuple[Box, ...], gate: float) -> list[tuple[int, int]]:
    pairs = []
    gold_by_class: dict[str, list[int]] = defaultdict(list)
    pred_by_class: dict[str, list[int]] = defaultdict(list)
    for index, box in enumerate(gold):
        name = canonical_label(box.label)
        if name:
            gold_by_class[name].append(index)
    for index, box in enumerate(pred):
        name = canonical_label(box.label)
        if name:
            pred_by_class[name].append(index)
    for name in CLASSES:
        gi, pi = gold_by_class[name], pred_by_class[name]
        matching = match_boxes([gold[i] for i in gi], [pred[i] for i in pi], iou_gate=gate)
        pairs.extend((gi[i], pi[j]) for i, j in matching.pairs)
    return pairs


@register("aircraft_grounding", engine=CODE, answer_form=BOXES)
def score_grounding(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    scale, origin = scale_of(data, ctx.params)
    gate = float(ctx.params.get("iou_gate", 0.5))
    rows = []
    for _, row in data.iterrows():
        gold = parse_boxes(row.get("answer"), scale=scale, origin=origin)
        pred = parse_boxes(row.get("prediction"), scale=scale, origin=origin)
        if any(canonical_label(box.label) not in CLASSES for box in gold.boxes):
            raise ValueError("aircraft_grounding gold box lacks a five-class label")
        fmt = _strict_box_format(row.get("prediction"), pred.flags)
        pairs = _class_aware_matches(gold.boxes, pred.boxes, gate)
        matched = len(pairs)
        n_gt, n_pred = len(gold.boxes), len(pred.boxes)
        precision = matched / n_pred if n_pred else 0.0
        recall = matched / n_gt if n_gt else math.nan
        f1 = (2 * precision * recall / (precision + recall)) if n_gt and precision + recall else (0.0 if n_gt else math.nan)
        geometry_hit = matched == n_gt and matched == n_pred
        rows.append({
            "n_gt_boxes": n_gt, "n_pred_boxes": n_pred, "n_matched": matched,
            "n_missed": n_gt - matched, "n_spurious": n_pred - matched,
            "box_precision": precision if n_gt else math.nan,
            "box_recall": recall, "box_f1": f1,
            "mean_iou": sum(iou(gold.boxes[i], pred.boxes[j]) for i, j in pairs) / matched if matched else math.nan,
            "geometry_hit": int(geometry_hit), "format_ok": int(fmt),
            "hit": int(geometry_hit and fmt),
            "false_positive": int(n_gt == 0 and n_pred > 0),
            "parse_flags": pred.flag_text,
        })
    return _attach(data, rows)


REGION_PATTERNS = (
    ("upper_left", ("左上", "上方偏左", "左侧上方", "左侧偏上", "左边上方", "左侧上部")),
    ("upper_right", ("右上", "上方偏右", "右侧上方", "右侧偏上", "右边上方", "右侧上部")),
    ("lower_left", ("左下", "下方偏左", "左侧下方", "左侧偏下", "左边下方", "左侧下部")),
    ("lower_right", ("右下", "下方偏右", "右侧下方", "右侧偏下", "右边下方", "右侧下部")),
    # Bare 左侧/右侧/上部/下部 are deliberately absent: in long answers they describe the
    # part ("机身左侧", "机翼上部"), and as edge patterns they would override a correct 正中.
    # a bare side counts only when it names the image ("画面的左侧"), never a part ("机身左侧")
    ("center_left", ("左侧中部", "左侧中间", "中间偏左", "左中", "中左",
                     "画面左侧", "画面的左侧", "图像左侧", "图像的左侧", "图片左侧", "图片的左侧",
                     "画面左边", "画面的左边", "图像的左边", "图片的左边")),
    ("center_right", ("右侧中部", "右侧中间", "中间偏右", "右中", "中右",
                      "画面右侧", "画面的右侧", "图像右侧", "图像的右侧", "图片右侧", "图片的右侧",
                      "画面右边", "画面的右边", "图像的右边", "图片的右边")),
    ("upper_center", ("中上", "中间偏上", "上方中部", "上部中央", "上方")),
    ("lower_center", ("中下", "中间偏下", "下方中部", "下部中央", "下方")),
    ("center", ("正中", "中央", "中心", "中部", "中间")),
)


def parse_region(value: Any) -> str | None:
    return parse_region_detail(value)[0]


# The gold region words were generated from the defect box: the 3x3 cell its centre
# falls in, by thirds of width and height (Aircraft_Vqa geometry.region_word). An
# answer that gives the location as a box or a point instead of words ("请定位凹坑在
# 图像中的具体区域" invites that) is read by the same rule, on Qwen's 0-1000 scale.
_GRID = (("upper_left", "upper_center", "upper_right"),
         ("center_left", "center", "center_right"),
         ("lower_left", "lower_center", "lower_right"))
_POINT = re.compile(r"^\s*[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]\s*$")


def _cell(x: float, y: float, scale: float) -> str:
    third = scale / 3
    row = 0 if y < third else (1 if y < 2 * third else 2)
    col = 0 if x < third else (1 if x < 2 * third else 2)
    return _GRID[row][col]


def region_from_coords(value: Any, scale: float = 1000) -> str | None:
    text = _text(value)
    point = _POINT.match(text)
    if point:
        return _cell(float(point.group(1)), float(point.group(2)), scale)
    boxes = parse_boxes(text, scale=int(scale)).boxes
    cells = {_cell((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, scale) for b in boxes}
    return cells.pop() if len(cells) == 1 else None


def parse_region_detail(value: Any) -> tuple[str | None, str]:
    """(region, source): source is "words", "coords", or "" when nothing was read."""
    text = _text(value)
    for span in answer_spans(text) if text else []:
        region, ambiguous = _region_of(span)
        if ambiguous:
            return None, ""
        if region is not None:
            return region, "words"
    region = region_from_coords(text) if text else None
    return (region, "coords") if region else (None, "")


def _region_of(text: str) -> tuple[str | None, bool]:
    # Corners are read first and their words removed before the edges are tried, so
    # the 下方 inside "左下方" is not counted a second time as the bottom edge.
    specific = []
    for name, patterns in REGION_PATTERNS[:-1]:
        hits = [pattern for pattern in patterns if pattern in text]
        if hits:
            specific.append(name)
            for pattern in hits:
                text = text.replace(pattern, " ")
    if specific:
        return (specific[0], False) if len(specific) == 1 else (None, True)
    return ("center" if any(pattern in text for pattern in REGION_PATTERNS[-1][1]) else None), False


@register("aircraft_region", engine=CODE, answer_form=TEXT)
def score_region(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        # the gold must be region words; the coordinate reading is for predictions only
        gold, gold_source = parse_region_detail(row.get("answer"))
        if gold is None or gold_source != "words":
            raise ValueError(f"aircraft_region gold region unparseable: {row.get('answer')!r}")
        pred, source = parse_region_detail(row.get("prediction"))
        rows.append({"gold_region": gold, "pred_region": pred or "", "pred_region_source": source,
                     "parse_ok": pred is not None, "hit": int(pred == gold)})
    return _attach(data, rows)


# --- the task types added in aircraft_eval_v2 ------------------------------------

@register("aircraft_region_presence", engine=CODE, answer_form=TEXT)
def score_region_presence(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """referring_region: is there a defect inside the box the question names?

    The truth is about that box, not the image - half the anomalous images have a
    clean box asked about - and the raw data leaves answer_defect_types empty for
    every row, so the gold answer text is read (and must be readable).
    """
    rows = []
    for _, row in data.iterrows():
        gold = parse_presence(row.get("answer"))
        if gold is None:
            raise ValueError(f"aircraft_region_presence gold unreadable: {row.get('answer')!r}")
        pred = parse_presence(row.get("prediction"))
        rows.append({
            "gold_present": int(gold), "pred_present": pd.NA if pred is None else int(pred),
            "parse_ok": pred is not None, "hit": int(pred is gold),
            "false_positive": int(pred is True and not gold),
            "false_negative": int(pred is False and gold),
        })
    return _attach(data, rows)


_JSON_START = re.compile(r"```|\[\s*\{|\[\s*\]")


def _stated_count(value: Any) -> int | None:
    """The count the answer states in words, before any JSON; else its number of boxes."""
    from ..counting import parse_count

    text = _text(value)
    head = _JSON_START.split(text, maxsplit=1)[0]
    count = parse_count(head) if head.strip() else None
    if count is None:
        boxes = parse_boxes(text).boxes
        count = len(boxes) if boxes or re.search(r"\[\s*\]", text) else None
    return count


@register("aircraft_counting", engine=CODE, answer_form=TEXT)
def score_counting(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        gold = _stated_count(row.get("answer"))
        if gold is None:
            raise ValueError(f"aircraft_counting gold count unreadable: {row.get('answer')!r}")
        pred = _stated_count(row.get("prediction"))
        rows.append({"gold_count": gold, "pred_count": pd.NA if pred is None else pred,
                     "parse_ok": pred is not None, "hit": int(pred == gold),
                     "count_abs_error": pd.NA if pred is None else abs(pred - gold)})
    return _attach(data, rows)


# The photo is too blurred / dark to judge; the right answer declines and asks for
# a retake. Any definite verdict - defect or no defect - is a miss.
_DECLINES = re.compile(UNCERTAIN.pattern + r"|模糊|离焦|看不清楚|曝光不足|过暗|光线不足|没法下结论|无法下结论|"
                       r"不能下结论|补拍|重拍|重新拍摄|重新对焦|blurr|too dark", re.I)


@register("aircraft_uncertainty", engine=CODE, answer_form=TEXT)
def score_uncertainty(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        declined = bool(_DECLINES.search(_text(row.get("prediction"))))
        rows.append({"declined": int(declined), "parse_ok": True, "hit": int(declined)})
    return _attach(data, rows)
