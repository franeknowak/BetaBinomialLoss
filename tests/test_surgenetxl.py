"""Tests for the SurgeNetXL (DINOv3 ViT-B/16, SurgeNetXL-pretrained) encoder option. No download: fake checkpoints in
the DINOv3-repo key format, written from a random model in the authors' architecture.
Run: python -m pytest tests/test_surgenetxl.py -q
"""
import pytest
import timm
import torch

from b2j import embed
from b2j.data import IMAGENET_MEAN, IMAGENET_STD, transform

transformers = pytest.importorskip("transformers")


def _hf(seed):
    torch.manual_seed(seed)
    m = transformers.DINOv3ViTModel(embed.surgenetxl_config()).eval()
    with torch.no_grad():                       # non-trivial biases / layer scales, so a dropped one would show
        for name, p in m.named_parameters():
            if name.endswith("bias") or "lambda1" in name:
                p.normal_(0, 0.1)
    return m


def _to_repo_keys(hf):
    """Inverse of embed.convert_surgenetxl: HF keys -> DINOv3-repo keys (fused qkv, gamma_1/2, fc1/fc2)."""
    sd, out, lp = hf.state_dict(), {}, embed.hf_layer_prefix(hf)
    for k, v in sd.items():
        if k == "embeddings.register_tokens":
            continue
        if k == "embeddings.mask_token":
            out["mask_token"] = v.reshape(1, -1)
        elif k == "embeddings.cls_token":
            out["cls_token"] = v
        elif k.startswith("embeddings.patch_embeddings."):
            out[k.replace("embeddings.patch_embeddings.", "patch_embed.proj.")] = v
        elif k.startswith(lp):
            r = "blocks." + k[len(lp):]
            if ".attention.q_proj." in r:
                kind = r.rsplit(".", 1)[1]
                base = r.replace(f".attention.q_proj.{kind}", "")
                parts = [sd[k.replace("q_proj", n)] for n in ("q_proj", "k_proj", "v_proj")]
                out[f"{base}.attn.qkv.{kind}"] = torch.cat(parts)
            elif ".attention.k_proj." in r or ".attention.v_proj." in r:
                continue
            else:
                out[r.replace(".attention.o_proj.", ".attn.proj.").replace(".layer_scale1.lambda1", ".gamma_1")
                     .replace(".layer_scale2.lambda1", ".gamma_2").replace(".mlp.up_proj.", ".mlp.fc1.")
                     .replace(".mlp.down_proj.", ".mlp.fc2.")] = v
        else:
            out[k] = v
    out["rope_embed.periods"] = (1 / hf.rope_embeddings.inv_freq).to(torch.bfloat16)
    return out


def _patch(monkeypatch, tmp_path, sd, base_sd=None):
    p = tmp_path / "w.pth"
    torch.save(sd, p)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: str(p))
    real = timm.create_model

    def create(name, pretrained=False, **kw):           # "pretrained" base = a random timm model, or given weights
        m = real(name, pretrained=False, **kw)
        if pretrained and base_sd is not None:
            m.load_state_dict({**m.state_dict(), **{k: v for k, v in base_sd.items() if k in m.state_dict()}})
        return m
    monkeypatch.setattr(timm, "create_model", create)


def test_loads_every_encoder_weight_and_gives_cls(tmp_path, monkeypatch):
    ref = _hf(0)
    sd = _to_repo_keys(ref)
    sd["dino_head.mlp.0.weight"] = torch.zeros(4, 768)                        # leftover head: ignored
    _patch(monkeypatch, tmp_path, {"teacher": sd})
    m = embed.load_surgenetxl("cpu")
    x = torch.randn(2, 3, 336, 336)
    with torch.no_grad():
        got = embed.cls_embedding(m, x)
        assert torch.allclose(got, ref(pixel_values=x).pooler_output, atol=1e-5)
    assert got.shape == (2, 768)


def test_fails_on_missing_encoder_weights(tmp_path, monkeypatch):
    sd = _to_repo_keys(_hf(0))
    sd.pop("blocks.3.attn.qkv.bias")
    _patch(monkeypatch, tmp_path, sd)
    with pytest.raises(ValueError, match="missing"):
        embed.load_surgenetxl("cpu")


def test_fails_on_unexpected_encoder_keys(tmp_path, monkeypatch):
    sd = _to_repo_keys(_hf(0))
    sd["storage_tokens"] = torch.zeros(1, 4, 768)                              # e.g. a register-token checkpoint
    _patch(monkeypatch, tmp_path, sd)
    with pytest.raises(ValueError, match="unexpected"):
        embed.load_surgenetxl("cpu")


def test_fails_on_wrong_rope_periods(tmp_path, monkeypatch):
    sd = _to_repo_keys(_hf(0))
    sd["rope_embed.periods"] = sd["rope_embed.periods"] * 2
    _patch(monkeypatch, tmp_path, sd)
    with pytest.raises(ValueError, match="RoPE"):
        embed.load_surgenetxl("cpu")


def test_fails_if_identical_to_the_base_weights(tmp_path, monkeypatch):
    sd = _to_repo_keys(_hf(0))
    _patch(monkeypatch, tmp_path, sd, base_sd=sd)
    with pytest.raises(ValueError, match="identical"):
        embed.load_surgenetxl("cpu")


def test_transform_336_imagenet_normalisation():
    from PIL import Image
    import numpy as np
    x = transform("endoscapes", embed.SURGENETXL["size"])(Image.fromarray(np.full((480, 854, 3), 128, np.uint8)))
    assert x.shape == (3, 336, 336)
    assert torch.allclose(x[0, 0, 0], torch.tensor((128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]), atol=1e-4)
