from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from .imaging import decode_image_cell
from .prompts import PAIRWISE_JUDGE_PROMPT, VQA_JUDGE_PROMPT


@dataclass(frozen=True)
class JudgeSettings:
    api_base: str = "http://192.168.48.7:18180/v1/chat/completions"
    api_key: str = "sk-local"
    model: str = "qwen3.6-27b"
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 3
    pointwise_prompt: str = VQA_JUDGE_PROMPT
    pairwise_prompt: str = PAIRWISE_JUDGE_PROMPT


def _extract_json_object(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


extract_json_object = _extract_json_object


# Dimension weights for the 1-5 weighted pointwise rubric (judge_equip_pointwise_weighted.txt):
# 参数(param) 最重要 > 事实(fact) > 表达(style)。三者在 1-5 分尺度上加权，
# 再除以 5 归一化到 [0,1]，写入下游沿用已久的 "hit" 列（下游报表/bootstrap
# 只对 hit 取 mean，不关心它是二值还是连续值，所以整条报表管线不用改）。
PARAM_WEIGHT = 0.5
FACT_WEIGHT = 0.35
STYLE_WEIGHT = 0.15

# Dimension weights for the 100-point 5-axis rubric (judge_equip_pointwise_100.txt).
# Each dimension is scored 0-100, weighted-summed to a 0-100 quality score, then
# thresholded to a binary hit — internally graded, externally still 0/1.
PARAM_WEIGHT_100 = 0.35
FACT_WEIGHT_100 = 0.25
VISUAL_WEIGHT_100 = 0.15
FABRICATION_WEIGHT_100 = 0.15
STYLE_WEIGHT_100 = 0.10
PASS_THRESHOLD_100 = 60  # quality_score >= 60 -> hit = 1


def _score_1_to_5(value: object, field: str) -> float:
    score = float(value)  # raises TypeError/ValueError on missing/non-numeric -> caller retries as judge_error
    if not (1 <= score <= 5):
        raise ValueError(f"{field}={value!r} out of range [1,5]")
    return score


def _score_0_to_100(value: object, field: str) -> float:
    score = float(value)  # raises TypeError/ValueError on missing/non-numeric -> caller retries as judge_error
    if not (0 <= score <= 100):
        raise ValueError(f"{field}={value!r} out of range [0,100]")
    return score


def parse_pointwise_judge(content: str) -> dict[str, object]:
    """Parse a pointwise judge response. Supports three formats:
    - 100-point 5-axis rubric: {"param_score", "fact_score", "visual_score",
      "fabrication_score", "style_score", "reasoning"} (0-100 each) -> weighted-summed
      to a 0-100 "quality_score", then thresholded (>=60) to a binary "hit".
    - 1-5 weighted rubric: {"param_score", "fact_score", "style_score", "reasoning"} (1-5 each)
      -> "hit" becomes the weighted, 0-1-normalized quality score.
    - legacy binary: {"correct": true/false, "reason": "..."} -> "hit" is 0 or 1.
    Always returns {"hit", "reason", "quality_score", "param_score", "fact_score",
    "visual_score", "fabrication_score", "style_score"} (fields the format doesn't
    produce are None).
    """
    obj = _extract_json_object(content)
    if "visual_score" in obj or "fabrication_score" in obj:
        param = _score_0_to_100(obj["param_score"], "param_score")
        fact = _score_0_to_100(obj["fact_score"], "fact_score")
        visual = _score_0_to_100(obj["visual_score"], "visual_score")
        fabrication = _score_0_to_100(obj["fabrication_score"], "fabrication_score")
        style = _score_0_to_100(obj["style_score"], "style_score")
        quality = (
            PARAM_WEIGHT_100 * param
            + FACT_WEIGHT_100 * fact
            + VISUAL_WEIGHT_100 * visual
            + FABRICATION_WEIGHT_100 * fabrication
            + STYLE_WEIGHT_100 * style
        )
        return {
            "hit": 1 if quality >= PASS_THRESHOLD_100 else 0,
            "reason": str(obj.get("reasoning", obj.get("reason", ""))),
            "quality_score": round(quality, 2),
            "param_score": param,
            "fact_score": fact,
            "visual_score": visual,
            "fabrication_score": fabrication,
            "style_score": style,
        }
    if "param_score" in obj or "fact_score" in obj or "style_score" in obj:
        param = _score_1_to_5(obj["param_score"], "param_score")
        fact = _score_1_to_5(obj["fact_score"], "fact_score")
        style = _score_1_to_5(obj["style_score"], "style_score")
        weighted = PARAM_WEIGHT * param + FACT_WEIGHT * fact + STYLE_WEIGHT * style
        return {
            "hit": round(weighted / 5.0, 4),
            "reason": str(obj.get("reasoning", obj.get("reason", ""))),
            "quality_score": None,
            "param_score": param,
            "fact_score": fact,
            "visual_score": None,
            "fabrication_score": None,
            "style_score": style,
        }
    return {
        "hit": 1 if obj.get("correct") is True else 0,
        "reason": str(obj.get("reason", "")),
        "quality_score": None,
        "param_score": None,
        "fact_score": None,
        "visual_score": None,
        "fabrication_score": None,
        "style_score": None,
    }


def parse_pairwise_judge(content: str) -> tuple[str, str]:
    obj = _extract_json_object(content)
    winner = str(obj.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B"}:
        winner = "tie"
    return winner, str(obj.get("reason", ""))


class JudgeClient:
    def __init__(self, settings: JudgeSettings):
        self.settings = settings

    def _post(
        self,
        system_prompt: str,
        user_text: str,
        image_b64: str | list[str] | None = None,
        mime: str = "image/jpeg",
    ) -> str:
        images = decode_image_cell(image_b64) if isinstance(image_b64, str) else list(image_b64 or [])
        if images:
            user_content: str | list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for image in images:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}})
        else:
            user_content = user_text
        body = json.dumps(
            {
                "model": self.settings.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": self.settings.temperature,
                "max_tokens": 1024,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            self.settings.api_base,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.settings.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return out["choices"][0]["message"]["content"]

    def judge_pointwise(
        self,
        question: object,
        reference: object,
        prediction: object,
        image_b64: str | list[str] | None = None,
    ) -> dict[str, object]:
        user_text = (
            f"描述任务：{question}\n"
            f"参考描述：{reference}\n"
            f"被测模型回答：{prediction}\n\n"
            "请看着图片，按评分标准打分，只输出 JSON。"
        )
        content = self._post(self.settings.pointwise_prompt, user_text, image_b64=image_b64)
        try:
            return parse_pointwise_judge(content)
        except Exception as exc:
            # Surface the raw judge response in the error so it lands directly in the
            # judge_reason column -- otherwise a parse failure only tells you *that*
            # something was missing/malformed, never *what the judge actually said*,
            # forcing a separate debug round-trip every time.
            raise ValueError(f"{type(exc).__name__}: {exc} | raw_content={content[:500]!r}") from exc

    def judge_raw(self, system_prompt: str, user_text: str, image_b64: str | list[str] | None = None) -> str:
        """Generic call for custom (non pointwise/pairwise) judge prompts, e.g. one-off data curation scripts."""
        return self._post(system_prompt, user_text, image_b64=image_b64)

    def judge_pairwise(
        self,
        question: object,
        reference: object,
        answer_a: object,
        answer_b: object,
        image_b64: str | list[str] | None = None,
    ) -> tuple[str, str]:
        user_text = (
            f"问题：{question}\n"
            f"参考描述：{reference}\n"
            f"回答A：{answer_a}\n"
            f"回答B：{answer_b}\n\n"
            "请看着图片判断 A 更好、B 更好，还是平局，只输出 JSON。"
        )
        content = self._post(self.settings.pairwise_prompt, user_text, image_b64=image_b64)
        return parse_pairwise_judge(content)
