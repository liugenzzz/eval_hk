"""xlsx writes survive control characters in questions and model outputs."""

import pandas as pd

from eval_tool.io import excel_safe


def test_control_characters_are_written_as_their_escapes(tmp_path):
    frame = pd.DataFrame({
        "question": ["当 $\x08eta$ 角增大时", "$\x0crac{a}{b}$", "普通文本"],
        "prediction": ["答\x01案", None, 3],
    })
    safe = excel_safe(frame)
    assert safe["question"].tolist() == [r"当 $\beta$ 角增大时", r"$\frac{a}{b}$", "普通文本"]
    assert safe["prediction"].tolist()[0] == "答\\x01案"
    assert safe["prediction"].tolist()[1:] == [None, 3]
    path = tmp_path / "out.xlsx"
    safe.to_excel(path, index=False)        # raised IllegalCharacterError before
    assert pd.read_excel(path)["question"].tolist()[0] == r"当 $\beta$ 角增大时"


def test_clean_frames_are_unchanged():
    frame = pd.DataFrame({"a": ["x\ty\nz"], "b": [1.5]})
    pd.testing.assert_frame_equal(excel_safe(frame), frame)
