"""Choice extraction on answers that were misread."""

import pytest

from eval_tool.score_mcq import extract_choice


@pytest.mark.parametrize("text, expected, ambiguous", [
    # "letter followed by the full option": the article A in the option text is not a choice
    ("B. A man opens the door", "B", False),
    ("(C) A dog runs", "C", False),
    ("D：一名男子在奔跑", "D", False),
    ("E. A person is cooking", "E", False),
    # an article at the start is not a leading letter
    ("A man opens the door, so the answer is B", "B", False),
])
def test_leading_letter(text, expected, ambiguous):
    got = extract_choice(text, valid_choices=("A", "B", "C", "D", "E"))
    assert (got.choice, got.ambiguous) == (expected, ambiguous)


@pytest.mark.parametrize("text, expected", [
    ("没错，这个说法是对的。", "A"),
    ("说得不错。", "A"),
    ("这个说法不正确。", "B"),
    ("I don't know, but it is true.", "A"),
    ("That is incorrect.", "B"),
])
def test_judge_words(text, expected):
    assert extract_choice(text, valid_choices=("A", "B"), is_judge=True).choice == expected
