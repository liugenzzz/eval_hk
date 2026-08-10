from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from .imaging import decode_image_cell
# All rubric parsing lives in judge_rubrics: the four current versions (v1/v2/v3/v4) plus
# the two retired formats, so rows scored before the rewrite still read back with the
# semantics they were scored under. Dispatch is on the "rubric" tag the prompts emit,
# with key-presence inference as the fallback for untagged output.
from .judge_rubrics import PASS_THRESHOLD, extract_json_object, parse_pointwise
from .prompts import PAIRWISE_JUDGE_PROMPT, VQA_JUDGE_PROMPT

_extract_json_object = extract_json_object
parse_pointwise_judge = parse_pointwise

# Retained so existing imports keep working; the live weights are in judge_rubrics.
PARAM_WEIGHT = 0.50
FACT_WEIGHT = 0.35
STYLE_WEIGHT = 0.15
PARAM_WEIGHT_100 = 0.35
FACT_WEIGHT_100 = 0.25
VISUAL_WEIGHT_100 = 0.15
FABRICATION_WEIGHT_100 = 0.15
STYLE_WEIGHT_100 = 0.10
PASS_THRESHOLD_100 = PASS_THRESHOLD


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

    @property
    def fingerprint(self) -> str:
        """Identity of this judge configuration.

        Fold this into the cache key and switching rubric versions re-scores instead of
        silently returning verdicts produced by the previous prompt. Without it the cache
        key is only (model, dataset, index), so changing the rubric changes nothing --
        which makes comparing rubric versions impossible.
        """
        blob = "|".join([
            self.model,
            f"{self.temperature:.3f}",
            hashlib.sha256(self.pointwise_prompt.encode("utf-8")).hexdigest()[:12],
            hashlib.sha256(self.pairwise_prompt.encode("utf-8")).hexdigest()[:12],
        ])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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
        return self._post_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )

    def _post_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
    ) -> str:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        body = json.dumps(
            {
                "model": self.settings.model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "max_tokens": max_tokens,
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

    def judge_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
    ) -> str:
        """Send caller-assembled multi-turn/multi-image OpenAI messages."""
        return self._post_messages(messages, max_tokens=max_tokens)

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
