"""Inspect the SAGES CVS Challenge 2024 labels against the frames extracted by the organisers' preprocess_videos.py.

Per video: the frame.csv columns and rows (frame_id present?, votes all 0/1?), and whether the extracted frames
(frame_map.csv, source_frame_id) are exactly the labelled frame_ids. Per extracted labelled frame: image size and
the black-border box (bounding box of the rows and columns whose mean intensity exceeds a threshold). A few frames
per image size are copied for a visual check.

Usage: python -m b2j.inspect_sages <dataset root> <output dir>
Outputs: per_video.csv, per_frame.csv (votes k per criterion, size, border box), samples/*.jpg
"""
from __future__ import annotations

import collections
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CRITERIA = ("c1", "c2", "c3")
RATERS = ("rater1", "rater2", "rater3")
VOTE_COLS = [f"{c}_{r}" for c in CRITERIA for r in RATERS]
BORDER_THRESHOLD = 15  # mean intensity (0-255) above which a row or column counts as image content
SAMPLES_PER_SIZE = 3


def content_box(img: np.ndarray) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the rows/columns whose mean intensity exceeds BORDER_THRESHOLD; bottom/right
    exclusive. A frame with no such row or column returns (0, 0, 0, 0)."""
    gray = img.mean(axis=2)
    rows = np.flatnonzero(gray.mean(axis=1) > BORDER_THRESHOLD)
    cols = np.flatnonzero(gray.mean(axis=0) > BORDER_THRESHOLD)
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, 0, 0)
    return (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def inspect_video(labels_dir: Path, frames_dir: Path) -> tuple[dict, pd.DataFrame | None]:
    df = pd.read_csv(labels_dir / "frame.csv")
    info = {"n_label_rows": len(df), "has_frame_id": "frame_id" in df.columns,
            "missing_vote_cols": ";".join(c for c in VOTE_COLS if c not in df.columns)}
    fmap = pd.read_csv(frames_dir / "frame_map.csv") if (frames_dir / "frame_map.csv").exists() else None
    info["n_extracted"] = 0 if fmap is None else len(fmap)
    if info["missing_vote_cols"]:
        return info, None
    votes = df[VOTE_COLS]
    info["n_nan_votes"] = int(votes.isna().sum().sum())
    info["n_non_binary_votes"] = int((~votes.isin([0, 1]) & votes.notna()).sum().sum())
    if info["has_frame_id"]:
        ids = df["frame_id"].astype(int).tolist()
        info["ids_are_0_150_300"] = ids == list(range(0, 150 * len(ids), 150))
        extracted = set() if fmap is None else set(fmap["source_frame_id"].astype(int))
        info["n_labelled_not_extracted"] = len(set(ids) - extracted)
        info["n_extracted_not_labelled"] = len(extracted - set(ids))
    return info, df


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    per_video, per_frame = [], []
    saved = collections.Counter()
    for split in ("train", "test"):
        meta = pd.read_csv(root / split / "labels" / "metadata.csv")
        names = sorted(p.name for p in (root / split / "labels").iterdir() if p.is_dir())
        frames_root = root / "derived" / split / "frames"
        print(f"{split}: {len(names)} label dirs, {len(meta)} metadata rows, "
              f"{sum(1 for _ in frames_root.iterdir())} frame folders", flush=True)
        for name in names:
            info, df = inspect_video(root / split / "labels" / name, frames_root / name)
            per_video.append({"split": split, "video": name, **info})
            if df is None or not info["has_frame_id"]:
                continue
            fmap = pd.read_csv(frames_root / name / "frame_map.csv").set_index("source_frame_id")
            for row in df.itertuples(index=False):
                r = row._asdict()
                rec = {"split": split, "video": name, "frame_id": int(r["frame_id"]),
                       **{f"k_{c}": sum(r[f"{c}_{x}"] for x in RATERS) for c in CRITERIA}}
                if r["frame_id"] in fmap.index:
                    path = frames_root / name / fmap.loc[r["frame_id"], "file_name"]
                    img = np.asarray(Image.open(path).convert("RGB"))
                    top, bottom, left, right = content_box(img)
                    rec.update({"file": str(path.relative_to(root)), "h": img.shape[0], "w": img.shape[1],
                                "box_top": top, "box_bottom": bottom, "box_left": left, "box_right": right,
                                "mean_intensity": float(img.mean())})
                    size = f"{img.shape[1]}x{img.shape[0]}"
                    if saved[size] < SAMPLES_PER_SIZE and r["frame_id"] == 1200:
                        shutil.copy(path, out / "samples" / f"{size}_{split}_{name[:8]}_{r['frame_id']}.jpg")
                        saved[size] += 1
                per_frame.append(rec)
    pd.DataFrame(per_video).to_csv(out / "per_video.csv", index=False)
    pd.DataFrame(per_frame).to_csv(out / "per_frame.csv", index=False)
    print(f"wrote per_video.csv ({len(per_video)} videos) and per_frame.csv ({len(per_frame)} labelled frames)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
