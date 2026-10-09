"""Yes/no reading for exist_negative: natural answers that were misread.

The reader also reads the *gold* answer when a row carries no polarity, so a
misreading here can flip the answer key, not only the prediction.
"""

import pytest

from eval_tool.scorers.counting import _answer_says_yes


@pytest.mark.parametrize("text, expected", [
    ("图中有一架无人机。", True),              # 无人机 holds 无 but is not a negation
    ("Yes, I can see a truck in the snow.", True),  # "no" inside "snow"
    ("有没有货车？有。", True),                # the question restated
    ("是否存在：是。", True),
    ("没看到货车。", False),                   # 看到 inside 没看到
    ("否", False),
    ("没有。", False),
    ("图中无货车。", False),
    ("No, there is none.", False),
    ("无法确定。", None),
    ("看不清。", None),
    ("有，在左上角。", True),
])
def test_answer_says_yes(text, expected):
    assert _answer_says_yes(text) is expected
