"""Cache the frozen DINOv3 ViT-B/16 [CLS] embedding of every image of a dataset (PLAN §1.3).

  python -m b2j.embed <dataset> [--batch 128] [--workers 8]

Public weights (timm `vit_base_patch16_dinov3.lvd1689m`), eval mode, full float32 (TF32 off), images via
b2j.data.transform (256², no augmentation). The [CLS] token is token 0 of forward_features, i.e. after the final
LayerNorm (timm's default pooling for this model is the token *average*, so it is not used). Writes
cache/embeddings/<dataset>.npz with emb [N, 768] float32, ids, split, k, n, labels, plus a .json with the settings.
Checks: one row per label-table image, all finite, and the first batch recomputed gives the same embeddings.

--encoder endovit: EndoViT (Batić et al., IJCARS 2024; ViT-B/16 masked autoencoder pretrained on Endo700k), the
authors' Hugging Face release egeozsoy/EndoViT (pytorch_model.bin, Apache-2.0), loaded into timm's VisionTransformer
as in the model card; its own input (224², Endo700k mean / std; same centre crop and bicubic resize as phase 1);
[CLS] after the final LayerNorm, as for DINOv3. Writes cache/embeddings_endovit/<dataset>.npz.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from b2j.data import FERPLUS_EMOTIONS, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, ImageSet, load_table, transform

MODEL = "vit_base_patch16_dinov3.lvd1689m"
REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache" / "embeddings"
ENDOVIT = {"repo": "egeozsoy/EndoViT", "file": "pytorch_model.bin", "size": 224,
           "mean": (0.3464, 0.2280, 0.2228), "std": (0.2520, 0.2128, 0.2093)}    # from the model card
CACHES = {"dinov3": CACHE, "endovit": REPO / "cache" / "embeddings_endovit"}
LABELS = {"endoscapes": ["C1", "C2", "C3"],
          "cifar10h": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
          "mhist": ["SSA"], "amibr": ["atypical"],
          "ferplus": list(FERPLUS_EMOTIONS)}


def load_model(device, pretrained=True):
    import timm
    model = timm.create_model(MODEL, pretrained=pretrained, num_classes=0).to(device).eval()
    cfg = model.pretrained_cfg
    if tuple(cfg["mean"]) != IMAGENET_MEAN or tuple(cfg["std"]) != IMAGENET_STD:
        raise ValueError(f"normalisation mismatch: model {cfg['mean']}/{cfg['std']}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_endovit(device):
    """EndoViT encoder: timm VisionTransformer with the model card's settings, weights from the HF release. Every
    encoder weight must load (only the MAE decoder / mask token may be left over)."""
    from functools import partial
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from timm.models.vision_transformer import VisionTransformer
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)
    sd = torch.load(hf_hub_download(ENDOVIT["repo"], ENDOVIT["file"]), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise ValueError(f"EndoViT: encoder weights missing from the checkpoint: {res.missing_keys}")
    left = [k for k in res.unexpected_keys if not (k.startswith("decoder") or k == "mask_token")]
    if left:
        raise ValueError(f"EndoViT: unexpected non-decoder keys: {left}")
    print(f"EndoViT loaded: {len(sd)} tensors, {len(res.unexpected_keys)} decoder / mask-token tensors unused", flush=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def cls_embedding(model, x):
    """[CLS] after the final LayerNorm: token 0 of forward_features."""
    return model.forward_features(x)[:, 0]


@torch.no_grad()
def embed(model, loader, device):
    out = []
    for img, _, _, _ in loader:
        out.append(cls_embedding(model, img.to(device, torch.float32)).float().cpu())
    return torch.cat(out).numpy()


def main(dataset, batch=128, workers=8, encoder="dinov3"):
    torch.backends.cuda.matmul.allow_tf32 = False             # full float32 (agreed)
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    table = load_table(dataset)
    if encoder == "endovit":
        ds = ImageSet(table, dataset, transform(dataset, ENDOVIT["size"], ENDOVIT["mean"], ENDOVIT["std"]))
        model = load_endovit(device)
    else:
        ds = ImageSet(table, dataset)
        model = load_model(device)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    emb = embed(model, loader, device)
    if emb.shape != (len(table.ids), 768) or not np.isfinite(emb).all():
        raise ValueError(f"bad embeddings: shape {emb.shape}, finite {np.isfinite(emb).all()}")
    first = embed(model, torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(min(batch, len(ds)))),
                                                     batch_size=batch, num_workers=0), device)
    drift = float(np.abs(first - emb[:len(first)]).max())
    if drift > 1e-4:
        raise ValueError(f"recomputed embeddings differ by {drift}")
    out_dir = Path(CACHES[encoder])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{dataset}.npz", emb=emb, ids=table.ids, split=table.split, k=table.k, n=table.n,
             labels=np.array(LABELS[dataset]))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    meta = {"dataset": dataset, "encoder": encoder,
            "model": MODEL if encoder == "dinov3" else f"{ENDOVIT['repo']}/{ENDOVIT['file']}",
            "input_size": INPUT_SIZE[dataset] if encoder == "dinov3" else ENDOVIT["size"],
            "normalisation": [IMAGENET_MEAN, IMAGENET_STD] if encoder == "dinov3" else [ENDOVIT["mean"], ENDOVIT["std"]],
            "token": "cls after final norm",
            "dtype": "float32", "tf32": False, "n_images": int(len(emb)), "labels": LABELS[dataset],
            "splits": {s: int((table.split == s).sum()) for s in ("train", "val", "test")},
            "recompute_max_abs_diff": drift, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "torch": torch.__version__, "commit": commit, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{dataset}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--encoder", choices=list(CACHES), default="dinov3")
    a = ap.parse_args()
    main(a.dataset, a.batch, a.workers, a.encoder)
