"""Tests for the SurgeNetXL (DINOv3 ViT-B/16, SurgeNetXL-pretrained) encoder option. No download: fake checkpoints.
Run: python -m pytest tests/test_surgenetxl.py -q
"""
import pytest
import timm
import torch

from b2j import embed
from b2j.data import IMAGENET_MEAN, IMAGENET_STD, transform


def _model(seed):
    torch.manual_seed(seed)
    return timm.create_model(embed.MODEL, pretrained=False, img_size=336, num_classes=0).eval()


def _patch(monkeypatch, tmp_path, sd, base_seed=1):
    p = tmp_path / "w.pth"
    torch.save(sd, p)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: str(p))
    real = timm.create_model

    def create(name, pretrained=False, **kw):          # "pretrained" base = a fixed random model (no download)
        if pretrained:
            torch.manual_seed(base_seed)
            return real(name, pretrained=False, **kw)
        return real(name, pretrained=False, **kw)
    monkeypatch.setattr(timm, "create_model", create)


def test_loads_every_encoder_weight_and_gives_cls(tmp_path, monkeypatch):
    ref = _model(0)
    sd = dict(ref.state_dict())
    sd["dino_head.mlp.0.weight"] = torch.zeros(4, 768)                       # leftover head: ignored
    _patch(monkeypatch, tmp_path, sd)
    m = embed.load_surgenetxl("cpu")
    x = torch.randn(2, 3, 336, 336)
    with torch.no_grad():
        got = embed.cls_embedding(m, x)
        assert torch.allclose(got, ref.forward_features(x)[:, 0], atol=1e-5)
    assert got.shape == (2, 768)


def test_fails_on_missing_encoder_weights(tmp_path, monkeypatch):
    sd = dict(_model(0).state_dict())
    sd.pop("norm.weight")
    _patch(monkeypatch, tmp_path, sd)
    with pytest.raises(ValueError, match="missing"):
        embed.load_surgenetxl("cpu")


def test_fails_if_identical_to_the_base_weights(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, {"model": dict(_model(1).state_dict())}, base_seed=1)
    with pytest.raises(ValueError, match="identical"):
        embed.load_surgenetxl("cpu")


def test_transform_336_imagenet_normalisation():
    from PIL import Image
    import numpy as np
    x = transform("endoscapes", embed.SURGENETXL["size"])(Image.fromarray(np.full((480, 854, 3), 128, np.uint8)))
    assert x.shape == (3, 336, 336)
    assert torch.allclose(x[0, 0, 0], torch.tensor((128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]), atol=1e-4)
