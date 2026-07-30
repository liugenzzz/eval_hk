from eval_tool.judge import JudgeClient, JudgeSettings


class CapturingJudgeClient(JudgeClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = []

    def _post(self, system_prompt, user_text, image_b64=None, mime="image/jpeg"):
        self.calls.append((system_prompt, user_text, image_b64, mime))
        if "pair" in system_prompt:
            return '{"winner": "tie", "reason": "same"}'
        return '{"correct": true, "reason": "ok"}'


def test_judge_client_uses_external_pointwise_prompt():
    client = CapturingJudgeClient(JudgeSettings(pointwise_prompt="external point prompt"))

    result = client.judge_pointwise("q", "ref", "pred", "img")

    assert result["hit"] == 1
    assert result["reason"] == "ok"
    assert client.calls[0][0] == "external point prompt"


def test_judge_client_parses_weighted_pointwise_response():
    class WeightedJudgeClient(JudgeClient):
        def _post(self, system_prompt, user_text, image_b64=None, mime="image/jpeg"):
            return '{"param_score": 5, "fact_score": 3, "style_score": 1, "reasoning": "ok"}'

    client = WeightedJudgeClient(JudgeSettings())

    result = client.judge_pointwise("q", "ref", "pred", "img")

    # 0.5*5 + 0.35*3 + 0.15*1 = 2.5 + 1.05 + 0.15 = 3.7 -> /5 = 0.74
    assert result["hit"] == 0.74
    assert result["param_score"] == 5
    assert result["fact_score"] == 3
    assert result["style_score"] == 1
    assert result["reason"] == "ok"


def test_judge_client_parses_100point_pointwise_response_above_threshold():
    class Weighted100JudgeClient(JudgeClient):
        def _post(self, system_prompt, user_text, image_b64=None, mime="image/jpeg"):
            return (
                '{"param_score": 100, "fact_score": 100, "visual_score": 100, '
                '"fabrication_score": 100, "style_score": 100, "reasoning": "ok"}'
            )

    client = Weighted100JudgeClient(JudgeSettings())

    result = client.judge_pointwise("q", "ref", "pred", "img")

    assert result["hit"] == 1
    assert result["quality_score"] == 100.0
    assert result["visual_score"] == 100
    assert result["fabrication_score"] == 100


def test_judge_client_parses_100point_pointwise_response_below_threshold():
    class Weighted100JudgeClient(JudgeClient):
        def _post(self, system_prompt, user_text, image_b64=None, mime="image/jpeg"):
            return (
                '{"param_score": 40, "fact_score": 40, "visual_score": 40, '
                '"fabrication_score": 40, "style_score": 40, "reasoning": "bad"}'
            )

    client = Weighted100JudgeClient(JudgeSettings())

    result = client.judge_pointwise("q", "ref", "pred", "img")

    assert result["hit"] == 0
    assert result["quality_score"] == 40.0


def test_judge_client_uses_external_pairwise_prompt():
    client = CapturingJudgeClient(JudgeSettings(pairwise_prompt="external pair prompt"))

    winner, reason = client.judge_pairwise("q", "ref", "a", "b", "img")

    assert winner == "tie"
    assert reason == "same"
    assert client.calls[0][0] == "external pair prompt"
