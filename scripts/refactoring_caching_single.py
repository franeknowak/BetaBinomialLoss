import torch
from PIL import Image
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import shutil 
from typing import List, Tuple

IMAGE_SIZE = (384, 384)
DATASET_MEAN = (0.454315, 0.290313, 0.299898)
DATASET_STD = (0.167318, 0.156652, 0.150197)

EXPECTED_COUNTS = {
    True: 10520,   # remove_black = True
    False: 11090,  # remove_black = False
}

# ==========================================
# ---------- CACHE VALIDATION --------------
# ==========================================
TRANSFORMS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
])
def cache_validation(   dataset_dir: Path,
                        cached_images_path: Path,
                        annotations_path: Path,
                        combine_train_val: bool,
                        remove_black: bool,
                        roi: bool,
                        key_frames_only: bool,
                        force_recache: bool) -> None:
    """
    Validate existing cache; if invalid or forced, recache images.
    """

    # If cache exists and recaching is not forced, verify count.
    if cached_images_path.exists() and not force_recache:
        print("Found cached images, checking validity...")
        count = len(list(cached_images_path.rglob("*.pt")))
        expected = EXPECTED_COUNTS[remove_black]

        if count == expected:
            print("Cached images valid — proceeding.")
            return

        print(f"Cached image count incorrect ({count} vs {expected}). Rebuilding cache...")

    # Rebuild cache.
    print("Rebuilding cached images...")
    shutil.rmtree(cached_images_path, ignore_errors=True)

    cache_images(
        dataset_dir, cached_images_path, annotations_path,
        combine_train_val, remove_black, roi, key_frames_only)

# ==========================================
# ---------- DATA PREPROCESSING -------------
# ==========================================

def refactor_dataset(   annotations_path: Path,
                        remove_black: bool = True,
                        roi: bool = True,
                        key_frames_only: bool = True):
    """
    Load and filter annotations based on removal of black frames,
    keyframes, and ROI requirements.
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    if remove_black:
        black_list = [x['black_ratio'] for x in annotations]
        percentile_index = 0.95
        percentile_value = pd.Series(black_list).quantile(percentile_index)
        annotations = [x for x in annotations if x['black_ratio'] < percentile_value]

    if key_frames_only:
        annotations = [x for x in annotations if x['is_ds_keyframe'] == True]
    
    print(f"Dataset summary - total length: {len(annotations)}. Removed black images? {remove_black}. Kept only key frames? {key_frames_only}. RoI? {roi}.")

    return annotations


def dataset_splits(dataset_dir: Path,
                   annotations: List[dict]) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test based on video ID.
    """  
    train, val, test = [], [], []

    for x in annotations:
        x['label'] = x['ds']
        x['aspect_ratio'] = x['vid_roi_w'] / x['vid_roi_h']

        video_id = int(x['file_name'].split('_')[0])
        split = 'train' if video_id < 121 else 'val' if video_id < 162 else 'test'

        path = dataset_dir / split / x['file_name']
        x['file_path'] = path

        {'train': train, 'val': val, 'test': test}[split].append(x)

    print(f"Dataset split: {len(train)}/{len(val)}/{len(test)}")

    return train, val, test

# ==========================================
# ---------- CACHING FUNCTIONS --------------
# ==========================================

def cache_images(   dataset_dir: Path,
                    cached_images_path: Path,
                    annotations_path: Path,
                    combine_train_val: bool = True,
                    remove_black: bool = True,
                    roi: bool = True,
                    key_frames_only: bool = True) -> None:
    """
    Create folder structure and cache images.
    """
    cached_images_path.mkdir(parents=True, exist_ok=True)

    annotations = refactor_dataset(annotations_path, remove_black, roi, key_frames_only)
    train, val, test = dataset_splits(dataset_dir, annotations)

    if combine_train_val:
        _cache_split(train + val, cached_images_path / "train", roi)
    else:
        _cache_split(train, cached_images_path / "train", roi)
        _cache_split(val, cached_images_path / "val", roi)

    _cache_split(test, cached_images_path / "test", roi)


def _cache_split(dataset: List[dict],
                 out_dir: Path,
                 roi: bool) -> None:
    """Helper function to cache one dataset split."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if roi:
        preprocess_and_cache_roi(dataset, out_dir)
    else:
        preprocess_and_cache_centre_crop(dataset, out_dir)

# ================================================================
# ---------- IMAGE PREPROCESSING (ROI / CENTRE CROP) -------------
# ================================================================

def preprocess_and_cache_roi(dataset: List[dict], cache_dir: Path) -> None:
    """
    Crop using ROI bounding box; resize + normalise; save tensor.
    """
    for dp in tqdm(dataset, desc="Caching ROI images"):
        image = Image.open(dp['file_path']).convert("RGB")

        if dp['aspect_ratio'] >= 1.5:     # Heuristic crop
            img_cropped = image.crop((187, 0, 667, 480))
        else:
            x1, y1 = int(dp['vid_roi_x']), int(dp['vid_roi_y'])
            x2, y2 = x1 + int(dp['vid_roi_w']), y1 + int(dp['vid_roi_h'])
            img_cropped = image.crop((x1, y1, x2, y2))

        tensor = TRANSFORMS(img_cropped)
        _save_tensor(dp, tensor, cache_dir)

    print(f"✓ Cached {len(dataset)} images to {cache_dir}")


def preprocess_and_cache_centre_crop(dataset: List[dict], cache_dir: Path) -> None:
    """Simple fixed crop; resize + normalise; save tensor."""
    for dp in tqdm(dataset, desc="Caching centre-crop images"):
        image = Image.open(dp['file_path']).convert("RGB")
        img_cropped = image.crop((187, 0, 667, 480))
        tensor = TRANSFORMS(img_cropped)
        _save_tensor(dp, tensor, cache_dir)

    print(f"✓ Cached {len(dataset)} images to {cache_dir}")


def _save_tensor(data_point: dict, tensor: torch.Tensor, cache_dir: Path) -> None:
    """Save image + label as `.pt` file."""
    save_name = f"{Path(data_point['file_path']).stem}.pt"
    torch.save({'image': tensor, 'label': torch.tensor(data_point['label'])}, cache_dir / save_name)