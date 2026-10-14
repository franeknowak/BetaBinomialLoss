"""Module c: DINOv3 ViT-B/16 encoder and the embedding cache.

The cache holds the [CLS] embedding (after DINOv3's final LayerNorm) of every image of a dataset, in
LabelTable order, as float32 with the image ids: cache/<dataset>_<tag>.npz. It is computed once per
dataset from the public weights (frozen runs), or from a fine-tuned encoder kept in memory (the H5
re-head), and never contains labels.
"""
from pathlib import Path

import numpy as np
import torch

from b2j.data import INPUT_SIZE, ImageSet, LabelTable

ENCODER = "vit_base_patch16_dinov3.lvd1689m"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def load_encoder(dataset: str):
    """Public DINOv3 ViT-B/16 via timm, CLS-token pooling, no classifier, at the dataset's input size."""
    import timm
    return timm.create_model(ENCODER, pretrained=True, num_classes=0, global_pool="token",
                             drop_path_rate=0.0, img_size=INPUT_SIZE[dataset])


@torch.no_grad()
def embed(encoder, image_set: ImageSet, device, batch_size=64, num_workers=4) -> np.ndarray:
    """Encoder outputs for every image of image_set, in order. float32 [N, D]; encoder in eval mode, no autocast."""
    loader = torch.utils.data.DataLoader(image_set, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=device.type == "cuda")
    was_training = encoder.training
    encoder.eval().to(device)
    out = [encoder(x.to(device, non_blocking=True)).float().cpu() for x, *_ in loader]
    encoder.train(was_training)
    return torch.cat(out).numpy()


def cache_path(dataset: str, tag: str = "frozen") -> Path:
    return CACHE_DIR / f"{dataset}_{tag}.npz"


def build_cache(table: LabelTable, dataset: str, device, tag="frozen", encoder=None, **kw) -> Path:
    """Embed every image of table and write cache_path(dataset, tag). Uses the public encoder if none is given."""
    encoder = encoder if encoder is not None else load_encoder(dataset)
    X = embed(encoder, ImageSet(table, dataset), device, **kw)
    path = cache_path(dataset, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=table.ids, X=X, encoder=ENCODER if tag == "frozen" else f"{ENCODER}:{tag}")
    return path


def load_cache(table: LabelTable, path) -> np.ndarray:
    """Embeddings aligned to table's rows (by id; fails loudly on any missing id)."""
    z = np.load(path, allow_pickle=False)
    pos = {i: r for r, i in enumerate(z["ids"])}
    missing = [i for i in table.ids if i not in pos]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from {path}, e.g. {missing[:3]}")
    return z["X"][[pos[i] for i in table.ids]]


if __name__ == "__main__":   # python -m b2j.embed endoscapes cifar10h mhist
    import argparse
    from b2j.data import load_table
    ap = argparse.ArgumentParser(description="Build the frozen DINOv3 embedding cache per dataset.")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        t = load_table(ds)
        missing = [p for p in t.paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"{ds}: {len(missing)} images missing, e.g. {missing[:2]}")
        p = build_cache(t, ds, dev, batch_size=a.batch_size, num_workers=a.num_workers)
        X = np.load(p)["X"]
        print(f"{ds}: {X.shape} -> {p} | finite {np.isfinite(X).all()} | mean norm {np.linalg.norm(X, axis=1).mean():.2f}", flush=True)
