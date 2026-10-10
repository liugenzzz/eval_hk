"""List frame-labeled NWPU anomaly intervals for human event annotation.

The frame mask identifies when an event may occur. It does not identify the
event type or provide a reference description.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def positive_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return zero-based [start, end) intervals without merging distinct runs."""
    values = np.asarray(mask)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("frame mask must be one-dimensional and binary")
    edges = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def probe_video(path: Path) -> tuple[float, int]:
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened():
            raise ValueError(f"cannot read video: {path}")
        fps = float(reader.get(cv2.CAP_PROP_FPS))
        frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frames <= 0:
            raise ValueError(f"video has no valid fps/frame count: {path}")
        return fps, frames
    finally:
        reader.release()


def build_queue(groundtruth: Path, videos: Path, *, probe=probe_video) -> tuple[list[dict], dict]:
    rows = []
    normal_clips = 0
    with np.load(groundtruth, allow_pickle=False) as masks:
        video_stems = {path.stem for path in videos.glob("*.avi")}
        if set(masks.files) != video_stems:
            raise ValueError(f"mask/video keys differ: masks_only={sorted(set(masks.files)-video_stems)[:5]}, "
                             f"videos_only={sorted(video_stems-set(masks.files))[:5]}")
        for stem in sorted(masks.files):
            path = videos / f"{stem}.avi"
            fps, frames = probe(path)
            mask = masks[stem]
            if len(mask) != frames:
                raise ValueError(f"{stem}: mask has {len(mask)} frames, video has {frames}")
            runs = positive_runs(mask)
            if not runs:
                normal_clips += 1
            for ordinal, (start, end) in enumerate(runs, 1):
                rows.append({
                    "candidate_id": f"{stem}-{ordinal:03d}",
                    "source_clip_id": stem,
                    "video": str(path),
                    "fps": fps,
                    "frame_start": start,
                    "frame_end_exclusive": end,
                    "start_s": round(start / fps, 3),
                    "end_s": round(end / fps, 3),
                    "event_type": "",
                    "description": "",
                    "review_status": "unlabeled",
                })
    return rows, {"test_videos": len(video_stems), "normal_mask_videos": normal_clips,
                  "positive_segments": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, totals = build_queue(args.groundtruth, args.videos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [
            "candidate_id", "source_clip_id", "video", "fps", "frame_start",
            "frame_end_exclusive", "start_s", "end_s", "event_type",
            "description", "review_status"])
        writer.writeheader()
        writer.writerows(rows)
    print(totals)


if __name__ == "__main__":
    main()
