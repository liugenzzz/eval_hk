import pandas as pd

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
