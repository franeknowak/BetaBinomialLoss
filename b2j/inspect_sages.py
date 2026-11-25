"""Inspect the SAGES CVS Challenge 2024 download: label files, video properties and frame borders.

For every video: the frame.csv columns and rows (frame_id present?, votes all 0/1?, number of labelled frames), and
ffprobe's width, height, frame rate and frame count (does every labelled frame_id exist in the video?). One labelled
frame per video (the middle one) is decoded to measure the black border: the bounding box of the rows and columns
whose mean intensity exceeds a threshold. A few of those frames are saved as PNG per resolution for a visual check.

Usage: python -m b2j.inspect_sages <dataset root> <output dir>
"""
from __future__ import annotations

import collections
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FFMPEG_BIN = Path("/share/apps/ffmpeg-4.2.2/bin")
CRITERIA = ("c1", "c2", "c3")
RATERS = ("rater1", "rater2", "rater3")
VOTE_COLS = [f"{c}_{r}" for c in CRITERIA for r in RATERS]
BORDER_THRESHOLD = 15  # mean intensity (0-255) above which a row or column counts as image content
SAMPLES_PER_RESOLUTION = 3


def probe(video: Path) -> dict:
    out = subprocess.run(
        [str(FFMPEG_BIN / "ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,duration",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": float(num) / float(den),
            "n_frames": int(s["nb_read_packets"]), "duration": float(s.get("duration", "nan"))}


def read_frame(video: Path, frame_id: int) -> np.ndarray:
    png = subprocess.run(
        [str(FFMPEG_BIN / "ffmpeg"), "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{frame_id})",
         "-vsync", "0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        check=True, capture_output=True).stdout
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def content_box(img: np.ndarray) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the rows/columns whose mean intensity exceeds BORDER_THRESHOLD; bottom/right
    exclusive. A frame with no such row returns (0, 0, 0, 0)."""
    gray = img.mean(axis=2)
    rows = np.flatnonzero(gray.mean(axis=1) > BORDER_THRESHOLD)
    cols = np.flatnonzero(gray.mean(axis=0) > BORDER_THRESHOLD)
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, 0, 0)
    return (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)


def inspect_labels(frame_csv: Path) -> tuple[dict, pd.DataFrame | None]:
    df = pd.read_csv(frame_csv)
    info = {"columns": list(df.columns), "n_rows": len(df), "has_frame_id": "frame_id" in df.columns,
            "missing_vote_cols": [c for c in VOTE_COLS if c not in df.columns]}
    if info["missing_vote_cols"]:
        return info, None
    votes = df[VOTE_COLS]
    info["n_nan_votes"] = int(votes.isna().sum().sum())
    info["non_binary_votes"] = int((~votes.isin([0, 1]) & votes.notna()).sum().sum())
    if info["has_frame_id"]:
        ids = df["frame_id"].tolist()
        info["frame_ids"] = ids
        info["ids_step_150_from_0"] = ids == list(range(0, 150 * len(ids), 150))
    return info, df


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    per_video = []
    label_rows = []
    saved = collections.Counter()
    for split in ("train", "test"):
        meta = pd.read_csv(root / split / "labels" / "metadata.csv")
        names = sorted(p.stem for p in (root / split / "videos").glob("*.mp4"))
        label_dirs = sorted(p.name for p in (root / split / "labels").iterdir() if p.is_dir())
        print(f"{split}: {len(names)} videos, {len(label_dirs)} label dirs, {len(meta)} metadata rows; "
              f"videos without labels {len(set(names) - set(label_dirs))}, labels without video "
              f"{len(set(label_dirs) - set(names))}, videos missing from metadata "
              f"{len(set(names) - set(meta.video_name))}", flush=True)
        for i, name in enumerate(names):
            video = root / split / "videos" / f"{name}.mp4"
            rec = {"split": split, "video": name}
            info, df = inspect_labels(root / split / "labels" / name / "frame.csv")
            rec.update({k: v for k, v in info.items() if k != "frame_ids"})
            rec.update(probe(video))
            if df is not None:
                ids = info.get("frame_ids", list(range(0, 150 * len(df), 150)))
                rec["max_frame_id"] = int(max(ids))
                rec["all_ids_in_video"] = bool(max(ids) < rec["n_frames"])
                for j, row in enumerate(df[VOTE_COLS].itertuples(index=False)):
                    r = dict(zip(VOTE_COLS, row))
                    label_rows.append({"split": split, "video": name, "frame_id": ids[j],
                                       **{f"k_{c}": sum(r[f"{c}_{x}"] for x in RATERS) for c in CRITERIA}})
                mid = int(ids[len(ids) // 2])
                img = read_frame(video, mid)
                top, bottom, left, right = content_box(img)
                rec.update({"sample_frame_id": mid, "decoded_h": img.shape[0], "decoded_w": img.shape[1],
                            "box_top": top, "box_bottom": bottom, "box_left": left, "box_right": right,
                            "sample_mean_intensity": float(img.mean())})
                res = f"{img.shape[1]}x{img.shape[0]}"
                if saved[res] < SAMPLES_PER_RESOLUTION:
                    Image.fromarray(img).save(out / "samples" / f"{res}_{split}_{name[:8]}_{mid}.png")
                    saved[res] += 1
            per_video.append(rec)
            if i % 100 == 0:
                print(f"  {split} {i}/{len(names)}", flush=True)
    pd.DataFrame(per_video).to_csv(out / "per_video.csv", index=False)
    labels = pd.DataFrame(label_rows)
    labels.to_csv(out / "frame_votes.csv", index=False)
    print(f"wrote {out / 'per_video.csv'} and {out / 'frame_votes.csv'} ({len(labels)} labelled frames)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
