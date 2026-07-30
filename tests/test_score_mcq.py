import pandas as pd

from eval_tool.score_mcq import extract_choice, score_choice_dataframe


def test_extracts_explicit_answer_letter_from_free_text():
    result = extract_choice("综合图中标注来看，答案是 C。", valid_choices=("A", "B", "C", "D"))

    assert result.choice == "C"
    assert result.method == "explicit"
    assert result.ambiguous is False
    assert result.no_answer is False


def test_extracts_judge_answer_from_chinese_true_false_words():
    result = extract_choice("这个说法是正确的。", valid_choices=("A", "B"), is_judge=True)

    assert result.choice == "A"
    assert result.method == "judge_word"


def test_extracts_choice_by_unique_option_text_match():
    result = extract_choice(
        "应选择滑油泵，因为它位于系统入口处。",
        valid_choices=("A", "B", "C", "D"),
        options={"A": "燃油泵", "B": "滑油泵", "C": "作动筒", "D": "传感器"},
    )

    assert result.choice == "B"
    assert result.method == "option_text"


def test_marks_ambiguous_when_multiple_distinct_letters_appear():
    result = extract_choice("A 和 C 都可能，但我最终倾向 A。", valid_choices=("A", "B", "C", "D"))

    assert result.choice == "A"
    assert result.ambiguous is True


def test_marks_no_answer_when_choice_cannot_be_extracted():
    result = extract_choice("图中信息不足，无法判断。", valid_choices=("A", "B", "C", "D"))

    assert result.choice is None
    assert result.no_answer is True


def test_scores_choice_dataframe_with_missing_prediction_and_metadata():
    data = pd.DataFrame(
        [
            {
                "index": "1",
                "question": "选哪个？",
                "A": "燃油泵",
                "B": "滑油泵",
                "C": "作动筒",
                "D": "传感器",
                "answer": "B",
                "prediction": "滑油泵",
                "category": "P3",
                "l2-category": "零件图",
            },
            {
                "index": "2",
                "question": "选哪个？",
                "A": "燃油泵",
                "B": "滑油泵",
                "C": "作动筒",
                "D": "传感器",
                "answer": "A",
                "prediction": "",
                "category": "P3",
                "l2-category": "零件图",
            },
        ]
    )

    scored = score_choice_dataframe(data, dataset="mcq")

    assert scored.loc[0, "hit"] == 1
    assert scored.loc[0, "extracted_choice"] == "B"
    assert scored.loc[1, "hit"] == 0
    assert bool(scored.loc[1, "missing"]) is True
    assert bool(scored.loc[1, "no_answer"]) is True
