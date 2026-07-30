from eval_tool.score_vqa import merge_pairwise_decisions


def test_merge_pairwise_decisions_returns_model_a_when_both_directions_agree():
    result = merge_pairwise_decisions(
        model_a="sft",
        model_b="base",
        forward_winner="A",
        reverse_winner="B",
    )

    assert result.winner == "sft"
    assert result.outcome_for_a == "win"


def test_merge_pairwise_decisions_returns_model_b_when_both_directions_agree():
    result = merge_pairwise_decisions(
        model_a="sft",
        model_b="base",
        forward_winner="B",
        reverse_winner="A",
    )

    assert result.winner == "base"
    assert result.outcome_for_a == "loss"


def test_merge_pairwise_decisions_returns_tie_when_directions_disagree():
    result = merge_pairwise_decisions(
        model_a="sft",
        model_b="base",
        forward_winner="A",
        reverse_winner="A",
    )

    assert result.winner == "tie"
    assert result.outcome_for_a == "tie"


def test_merge_pairwise_decisions_returns_tie_when_both_directions_tie():
    result = merge_pairwise_decisions(
        model_a="sft",
        model_b="base",
        forward_winner="tie",
        reverse_winner="tie",
    )

    assert result.winner == "tie"
    assert result.outcome_for_a == "tie"
