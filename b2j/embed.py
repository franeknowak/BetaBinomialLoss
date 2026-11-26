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
--pool mean (EndoViT only): the authors' downstream pooling ("global_pool" in their models_vit.py): mean of the
patch tokens of the last block ([CLS] excluded, before the final LayerNorm), then a fresh LayerNorm (weight 1,
bias 0, eps 1e-6; its learned affine is absorbed by our linear head). Writes cache/embeddings_endovit_mean/.

--encoder surgenetxl: DINOv3 ViT-B/16 initialised from the LVD-1689M weights (= our DINOv3 encoder) and further
pretrained on SurgeNetXL (de Jong et al., MIDL 2026, §3.2), Hugging Face rlpddejong/SurgeNetXL_DINOv1-v3,
DINOv3_ViTb16_size336_SurgeNetXL.pth (weights CC-BY-NC-SA). The authors' pretraining config
(SurgeNetDINO dinov3_configs/config_dinov3_vitb.yaml) differs from LVD-1689M's architecture: no storage (register)
tokens, a learned, unmasked key bias, RoPE base 100. The model is therefore built as in the authors' own downstream
code (SurgeNetDINO segmentation_finetuning/models/vit.py): Hugging Face transformers DINOv3ViTModel with that
config and their key conversion; every encoder weight must load, only the DINO / iBOT heads may be left over, the
stored RoPE periods must match, and the weights must differ from LVD-1689M. (timm's
vit_base_patch16_dinov3.lvd1689m, used by the authors' load_models.py with strict=False, would drop the trained key /
query / value biases and add random register tokens.) Input 336² (its pretraining size), ImageNet mean / std (as in
the authors' downstream code), same centre crop and bicubic resize; [CLS] after the final LayerNorm, as for DINOv3.
Writes cache/embeddings_surgenetxl/<dataset>.npz.

--encoder lemonfm: LemonFM (Che et al., CVPR 2026; ConvNeXt-L, DINO-style self-distillation on the LEMON surgical
video set), Hugging Face visurg/LemonFM, lemonfm.pth (Apache-2.0). Built as in the authors' model_loader.py:
torchvision convnext_large with classifier[2] = Identity, teacher weights with the "backbone." prefix removed; every
weight must load (the authors load with strict=False onto ImageNet weights, which would hide a missing key). The
embedding is the model's output, as in the authors' feature-extraction example: global average pool of the last
stage, then the final LayerNorm (1536-d). Input 224² (pretraining crop size and model card), ImageNet mean / std (the
authors' pretraining and downstream transforms), same centre crop and bicubic resize. Writes
cache/embeddings_lemonfm/<dataset>.npz.
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
SURGENETXL = {"repo": "rlpddejong/SurgeNetXL_DINOv1-v3", "file": "DINOv3_ViTb16_size336_SurgeNetXL.pth", "size": 336}
LEMONFM = {"repo": "visurg/LemonFM", "file": "lemonfm.pth", "size": 224}
DIM = {"dinov3": 768, "endovit": 768, "surgenetxl": 768, "lemonfm": 1536}
CACHES = {"dinov3": CACHE, "endovit": REPO / "cache" / "embeddings_endovit",
          "surgenetxl": REPO / "cache" / "embeddings_surgenetxl",
          "lemonfm": REPO / "cache" / "embeddings_lemonfm",
          "endovit_mean": REPO / "cache" / "embeddings_endovit_mean"}
LABELS = {"endoscapes": ["C1", "C2", "C3"],
          "cifar10h": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
          "mhist": ["SSA"], "amibr": ["atypical"],
          "ferplus": list(FERPLUS_EMOTIONS), "sages": ["C1", "C2", "C3"]}


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


class SurgeNetXLEncoder(torch.nn.Module):
    """HF DINOv3ViTModel with timm's forward_features interface (tokens after the final LayerNorm, [CLS] first)."""
    def __init__(self, hf):
        super().__init__()
        self.hf = hf

    def forward_features(self, x):
        return self.hf(pixel_values=x).last_hidden_state


def surgenetxl_config():
    """The SurgeNetDINO ViT-B/16 pretraining architecture, as in the authors' models/vit.py."""
    from transformers import DINOv3ViTConfig
    return DINOv3ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                           patch_size=16, image_size=SURGENETXL["size"], num_register_tokens=0, key_bias=True,
                           layer_norm_eps=1e-6, rope_theta=100.0, pos_embed_shift=None, pos_embed_jitter=None,
                           pos_embed_rescale=None)


def hf_layer_prefix(hf):
    """Block prefix of HF DINOv3ViTModel: "layer." (transformers 4.x) or "model.layer." (5.x)."""
    return "model.layer." if "model.layer.0.norm1.weight" in hf.state_dict() else "layer."


def convert_surgenetxl(sd, layer_prefix="layer."):
    """DINOv3-repo checkpoint keys -> HF DINOv3ViTModel keys (the authors' load_dinov3_weights; layer_prefix from
    hf_layer_prefix). Returns the converted dict and the stored RoPE periods (not learned; None if absent)."""
    sd = dict(sd)
    periods = sd.pop("rope_embed.periods", None)
    out = {}
    for k, v in sd.items():
        if k == "mask_token":
            k, v = "embeddings.mask_token", v.reshape(1, 1, -1)
        elif k == "cls_token":
            k = "embeddings.cls_token"
        elif k.startswith("patch_embed.proj."):
            k = k.replace("patch_embed.proj.", "embeddings.patch_embeddings.")
        elif k.startswith("blocks."):
            k = layer_prefix + k[len("blocks."):]
            if ".attn.qkv." in k:
                for name, part in zip(("q", "k", "v"), v.chunk(3, dim=0)):
                    out[k.replace("attn.qkv.", f"attention.{name}_proj.")] = part
                continue
            k = (k.replace(".attn.proj.", ".attention.o_proj.").replace(".gamma_1", ".layer_scale1.lambda1")
                 .replace(".gamma_2", ".layer_scale2.lambda1").replace(".mlp.fc1.", ".mlp.up_proj.")
                 .replace(".mlp.fc2.", ".mlp.down_proj."))
        out[k] = v
    return out, periods


def load_surgenetxl(device, base_pretrained=True):
    """SurgeNetXL DINOv3 ViT-B/16 in the authors' architecture at 336². No encoder weight may be missing and nothing
    but the DINO / iBOT heads may be left over. base_pretrained: also load the LVD-1689M weights (timm) and check the
    checkpoint differs from them; prints the cosine similarity per tensor (evidence of the LVD-1689M initialisation)."""
    from huggingface_hub import hf_hub_download
    from transformers import DINOv3ViTModel
    hf = DINOv3ViTModel(surgenetxl_config())
    sd = torch.load(hf_hub_download(SURGENETXL["repo"], SURGENETXL["file"]), map_location="cpu", weights_only=True)
    for key in ("teacher", "model", "state_dict"):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
    conv, periods = convert_surgenetxl(sd, hf_layer_prefix(hf))
    if periods is not None and not torch.allclose(periods.float(), 1 / hf.rope_embeddings.inv_freq, rtol=1e-2):
        raise ValueError("SurgeNetXL: RoPE periods in the checkpoint do not match the model")
    res = hf.load_state_dict(conv, strict=False)
    missing = [k for k in res.missing_keys if k != "embeddings.register_tokens"]   # 0 register tokens: empty
    if missing:
        raise ValueError(f"SurgeNetXL: encoder weights missing from the checkpoint: {missing}")
    left = [k for k in res.unexpected_keys if not k.startswith(("dino_head.", "ibot_head."))]
    if left:
        raise ValueError(f"SurgeNetXL: unexpected checkpoint keys: {left}")
    print(f"SurgeNetXL loaded: {len(sd)} tensors, all encoder weights; unused heads: {len(res.unexpected_keys)}",
          flush=True)
    if base_pretrained:
        import timm
        base = timm.create_model(MODEL, pretrained=True, num_classes=0).state_dict()
        cos = {k: float(torch.nn.functional.cosine_similarity(sd[k].float().flatten(), base[k].float().flatten(), dim=0))
               for k in ("patch_embed.proj.weight", "blocks.0.gamma_1", "blocks.0.attn.qkv.weight",
                         "blocks.11.mlp.fc2.weight", "norm.weight")}
        if all(torch.equal(sd[k].float(), base[k].float()) for k in cos):
            raise ValueError("SurgeNetXL weights are identical to LVD-1689M")
        print(f"SurgeNetXL vs LVD-1689M, cosine similarity per tensor: {cos}", flush=True)
    model = SurgeNetXLEncoder(hf).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_lemonfm(device):
    """LemonFM ConvNeXt-L as in the authors' model_loader.py (no ImageNet weights underneath). Every backbone weight
    must load; only non-backbone entries (DINO head) may be left over."""
    import argparse
    import torchvision
    from huggingface_hub import hf_hub_download
    net = torchvision.models.convnext_large(weights=None)
    net.classifier[2] = torch.nn.Identity()
    torch.serialization.add_safe_globals([argparse.Namespace])
    ck = torch.load(hf_hub_download(LEMONFM["repo"], LEMONFM["file"]), map_location="cpu", weights_only=True)
    sd = ck["teacher"] if isinstance(ck, dict) and "teacher" in ck else ck
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    res = net.load_state_dict(bb, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise ValueError(f"LemonFM: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
    print(f"LemonFM loaded: {len(bb)} backbone tensors, all weights; unused non-backbone: {len(sd) - len(bb)}",
          flush=True)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def output_embedding(model, x):
    """The model's own output (LemonFM: global average pool + final LayerNorm, classifier[2] = Identity)."""
    return model(x)


def cls_embedding(model, x):
    """[CLS] after the final LayerNorm: token 0 of forward_features."""
    return model.forward_features(x)[:, 0]


def mean_patch_embedding(model, x):
    """EndoViT's global_pool: mean of the last block's patch tokens ([CLS] excluded, final LayerNorm skipped), then
    a LayerNorm without affine (= a fresh LayerNorm at its initialisation)."""
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    x = model.blocks(x)[:, model.num_prefix_tokens:].mean(dim=1)
    return torch.nn.functional.layer_norm(x, x.shape[-1:], eps=1e-6)


POOLS = {"cls": cls_embedding, "mean": mean_patch_embedding, "output": output_embedding}


@torch.no_grad()
def embed(model, loader, device, pool="cls"):
    out = []
    for img, _, _, _ in loader:
        out.append(POOLS[pool](model, img.to(device, torch.float32)).float().cpu())
    return torch.cat(out).numpy()


def main(dataset, batch=128, workers=8, encoder="dinov3", pool="cls"):
    if pool == "mean" and encoder != "endovit":
        raise ValueError("--pool mean is implemented for EndoViT only")
    if encoder == "lemonfm":
        pool = "output"                                       # a CNN: no [CLS] token
    elif pool == "output":
        raise ValueError("--pool output is for LemonFM only")
    torch.backends.cuda.matmul.allow_tf32 = False             # full float32 (agreed)
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    table = load_table(dataset)
    if encoder == "endovit":
        ds = ImageSet(table, dataset, transform(dataset, ENDOVIT["size"], ENDOVIT["mean"], ENDOVIT["std"]))
        model = load_endovit(device)
    elif encoder == "surgenetxl":
        ds = ImageSet(table, dataset, transform(dataset, SURGENETXL["size"]))
        model = load_surgenetxl(device)
    elif encoder == "lemonfm":
        ds = ImageSet(table, dataset, transform(dataset, LEMONFM["size"]))
        model = load_lemonfm(device)
    else:
        ds = ImageSet(table, dataset)
        model = load_model(device)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    emb = embed(model, loader, device, pool)
    if emb.shape != (len(table.ids), DIM[encoder]) or not np.isfinite(emb).all():
        raise ValueError(f"bad embeddings: shape {emb.shape}, finite {np.isfinite(emb).all()}")
    first = embed(model, torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(min(batch, len(ds)))),
                                                     batch_size=batch, num_workers=0), device, pool)
    drift = float(np.abs(first - emb[:len(first)]).max())
    if drift > 1e-4:
        raise ValueError(f"recomputed embeddings differ by {drift}")
    out_dir = Path(CACHES[encoder if pool in ("cls", "output") else f"{encoder}_{pool}"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{dataset}.npz", emb=emb, ids=table.ids, split=table.split, k=table.k, n=table.n,
             labels=np.array(LABELS[dataset]))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    meta = {"dataset": dataset, "encoder": encoder,
            "model": {"dinov3": MODEL, "endovit": f"{ENDOVIT['repo']}/{ENDOVIT['file']}",
                      "surgenetxl": f"{SURGENETXL['repo']}/{SURGENETXL['file']}",
                      "lemonfm": f"{LEMONFM['repo']}/{LEMONFM['file']}"}[encoder],
            "input_size": {"dinov3": INPUT_SIZE[dataset], "endovit": ENDOVIT["size"],
                           "surgenetxl": SURGENETXL["size"], "lemonfm": LEMONFM["size"]}[encoder],
            "normalisation": [ENDOVIT["mean"], ENDOVIT["std"]] if encoder == "endovit" else [IMAGENET_MEAN, IMAGENET_STD],
            "token": {"cls": "cls after final norm",
                      "mean": "mean of patch tokens before final norm, then LayerNorm without affine (EndoViT global_pool)",
                      "output": "model output: global average pool + final LayerNorm (ConvNeXt classifier[0])"}[pool],
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
    ap.add_argument("--encoder", choices=["dinov3", "endovit", "surgenetxl", "lemonfm"], default="dinov3")
    ap.add_argument("--pool", choices=list(POOLS), default="cls")
    a = ap.parse_args()
    main(a.dataset, a.batch, a.workers, a.encoder, a.pool)
