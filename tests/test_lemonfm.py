"""Tests for the LemonFM (ConvNeXt-L) encoder option. No download: fake DINO checkpoints written from a random
ConvNeXt (tiny, patched in for convnext_large to keep the test fast).
Run: python -m pytest tests/test_lemonfm.py -q
"""
import argparse

import pytest
import torch
import torchvision

from b2j import embed


def _net(seed):
    torch.manual_seed(seed)
    net = torchvision.models.convnext_tiny(weights=None)
    net.classifier[2] = torch.nn.Identity()
    return net.eval()


def _ckpt(net, drop=None, extra=None):
    sd = {f"backbone.{k}": v for k, v in net.state_dict().items() if k != drop}
    sd["head.mlp.0.weight"] = torch.zeros(4, 768)                          # DINO head: ignored
    sd.update(extra or {})
    return {"teacher": sd, "student": sd, "epoch": 50, "args": argparse.Namespace(arch="convnext_large")}


def _patch(monkeypatch, tmp_path, ck):
    p = tmp_path / "lemonfm.pth"
    torch.save(ck, p)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: str(p))
    monkeypatch.setattr(torchvision.models, "convnext_large", torchvision.models.convnext_tiny)


def test_loads_every_weight_and_gives_pooled_normed_output(tmp_path, monkeypatch):
    ref = _net(0)
    _patch(monkeypatch, tmp_path, _ckpt(ref))
    m = embed.load_lemonfm("cpu")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        got = embed.output_embedding(m, x)
        assert torch.allclose(got, ref(x), atol=1e-5)
        manual = ref.classifier[1](ref.classifier[0](ref.avgpool(ref.features(x))))   # pool -> LayerNorm2d -> flatten
        assert torch.allclose(got, manual, atol=1e-5)
    assert got.shape == (2, 768)                                     # 1536 for ConvNeXt-L
    assert not any(p.requires_grad for p in m.parameters())


def test_missing_weight_fails(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, _ckpt(_net(0), drop="features.0.0.weight"))
    with pytest.raises(ValueError, match="missing"):
        embed.load_lemonfm("cpu")


def test_unexpected_backbone_key_fails(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, _ckpt(_net(0), extra={"backbone.extra.weight": torch.zeros(1)}))
    with pytest.raises(ValueError, match="unexpected"):
        embed.load_lemonfm("cpu")


def test_settings():
    assert embed.LEMONFM["size"] == 224 and embed.DIM["lemonfm"] == 1536
    assert embed.CACHES["lemonfm"].name == "embeddings_lemonfm"
