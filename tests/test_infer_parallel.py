import pandas as pd
import pytest

from eval_tool.infer import chunk_records, merge_indexed_predictions


def test_chunk_records_splits_rows_across_workers_preserving_original_index():
    rows = [{"index": str(i)} for i in range(5)]

    chunks = chunk_records(rows, worker_count=2)

    assert chunks == [
        [(0, {"index": "0"}), (2, {"index": "2"}), (4, {"index": "4"})],
        [(1, {"index": "1"}), (3, {"index": "3"})],
    ]


def test_merge_indexed_predictions_restores_input_order():
    merged = merge_indexed_predictions(
        [
            [(2, "third"), (0, "first")],
            [(1, "second")],
        ],
        total=3,
    )

    assert merged == ["first", "second", "third"]


def test_merge_indexed_predictions_preserves_returned_empty_answer():
    assert merge_indexed_predictions([[(0, "")]], total=1) == [""]


def test_merge_indexed_predictions_rejects_missing_positions():
    with pytest.raises(RuntimeError, match="missing positions: 1"):
        merge_indexed_predictions([[(0, "first")]], total=2)


def test_merge_indexed_predictions_rejects_duplicate_positions():
    with pytest.raises(RuntimeError, match="duplicate position: 0"):
        merge_indexed_predictions([[(0, "first")], [(0, "again")]], total=1)


@pytest.mark.parametrize("position", [-1, 1, "0"])
def test_merge_indexed_predictions_rejects_invalid_positions(position):
    with pytest.raises(RuntimeError, match=f"invalid position: {position}"):
        merge_indexed_predictions([[(position, "answer")]], total=1)
