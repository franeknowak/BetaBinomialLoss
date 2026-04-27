import torch
from PIL import Image
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import shutil 
from typing import List, Tuple

DATASET_MEAN = (0.454315, 0.290313, 0.299898)
DATASET_STD = (0.167318, 0.156652, 0.150197)
EXPECTED_COUNTS = 11090

def get_transforms(image_size):
    TRANSFORMS = transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),])
    return TRANSFORMS

def cache_validation(   dataset_dir: Path,
                        cached_images_path: Path,
                        annotations_path: Path,
                        temporal: bool,
                        force_recache: bool,
                        image_size) -> None:
    """
    Validate existing cache; if invalid or forced, recache images.
    """

    # If cache exists and recaching is not forced, verify count.
    if cached_images_path.exists() and not force_recache:
        print("Found cached images, checking validity...")
        count = len(list(cached_images_path.rglob("*.pt")))
        if count == EXPECTED_COUNTS:
            print("Cached images valid — proceeding.")
            return

        print(f"Cached image count incorrect ({count} vs {EXPECTED_COUNTS}). Rebuilding cache...")

    # Rebuild cache.
    print("Rebuilding cached images...")
    shutil.rmtree(cached_images_path, ignore_errors=True)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    cache_images(dataset_dir, cached_images_path, annotations, temporal, image_size)

# ==========================================
# ---------- CACHING FUNCTIONS --------------
# ==========================================

def cache_images(   dataset_dir: Path,
                    cached_images_path: Path,
                    annotations: dict,
                    temporal: bool,
                    image_size: int) -> None:
    """
    Create folder structure and cache images.
    """  

    cached_images_path.mkdir(parents=True, exist_ok=True)
    train = annotations['train']
    val = annotations['val']
    test = annotations['test']
    _cache_split(dataset = train,
                 out_dir = cached_images_path / 'train',
                 dataset_dir = dataset_dir / 'train',
                 temporal = temporal,
                 image_size = image_size)
    _cache_split(dataset = val,
                 out_dir = cached_images_path / 'val',
                 dataset_dir = dataset_dir / 'val',
                 temporal = temporal,
                 image_size = image_size)
    _cache_split(dataset = test,
                 out_dir = cached_images_path / 'test',
                 dataset_dir = dataset_dir / 'test',
                 temporal = temporal)


def _cache_split(dataset: dict,
                 out_dir: Path,
                 dataset_dir: Path,
                 temporal: bool) -> None:
    """Helper function to cache one dataset split."""
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess_and_cache_centre_crop(dataset, out_dir, dataset_dir, temporal, image_size)

# ================================================================
# ---------- IMAGE PREPROCESSING (CENTRE CROP) -------------
# ================================================================
def preprocess_and_cache_centre_crop(dataset: dict,
                                     cache_dir: Path,
                                     dataset_dir: Path,
                                     temporal: bool,
                                     image_size: int) -> None:
    TRANSFORMS = get_transforms(image_size)
    """Simple fixed crop; resize + normalise; save tensor."""
    for dp_idx in tqdm(dataset, desc="Caching centre-crop images"):
        dp = dataset[dp_idx]
        if temporal:
            frames= []
            for frame in dp['frames']:
                image_path = dataset_dir / frame
                image = Image.open(image_path).convert("RGB")
                img_cropped = image.crop((187, 0, 667, 480))
                tensor = TRANSFORMS(img_cropped)
                frames.append(tensor)

            frames_tensor = torch.stack(frames, dim=0) 
            _save_tensor(dp, frames_tensor, cache_dir)
        else:
            dp = dataset[dp_idx]
            image_path = dataset_dir / dp_idx
            image = Image.open(image_path).convert("RGB")
            img_cropped = image.crop((187, 0, 667, 480))
            tensor = TRANSFORMS(img_cropped)
            _save_tensor(dp, tensor, cache_dir)

    print(f"✓ Cached {len(dataset)} images to {cache_dir}")


def _save_tensor(data_point: dict, tensors: torch.Tensor, cache_dir: Path) -> None:
    """Save image + label as `.pt` file."""
    save_name = f"{data_point['frames'][-1].split('.')[0]}.pt"
    vid_id = torch.tensor(data_point['annotations']['video_id'])
    frame_id = torch.tensor(data_point['annotations']['frame_id'])
    torch.save({'image': tensors, 'label': torch.tensor(data_point['annotations']['ds']), 'vid_id': vid_id, 'frame_id': frame_id}, cache_dir / save_name)
