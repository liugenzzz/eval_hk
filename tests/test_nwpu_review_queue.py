import numpy as np
import pytest

from tools.nwpu_review_queue import build_queue, positive_runs


def test_positive_runs_preserve_distinct_frame_intervals():
    assert positive_runs(np.array([0, 1, 1, 0, 1, 0])) == [(1, 3), (4, 5)]
    with pytest.raises(ValueError, match="binary"):
        positive_runs(np.array([0, 2, 1]))


def test_queue_needs_human_event_labels_and_checks_frame_count(tmp_path):
    videos = tmp_path / "Test"
    videos.mkdir()
    for name in ("a", "b"):
        (videos / f"{name}.avi").touch()
    source = tmp_path / "gt.npz"
    np.savez(source, a=np.array([0, 1, 1, 0]), b=np.zeros(4))
    rows, totals = build_queue(source, videos, probe=lambda _: (2.0, 4))
    assert totals == {"test_videos": 2, "normal_mask_videos": 1, "positive_segments": 1}
    assert rows[0]["start_s"] == 0.5 and rows[0]["end_s"] == 1.5
    assert rows[0]["event_type"] == rows[0]["description"] == ""
    with pytest.raises(ValueError, match="mask has 4 frames"):
        build_queue(source, videos, probe=lambda _: (2.0, 5))
