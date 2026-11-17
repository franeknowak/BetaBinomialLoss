from functools import partial

import pytest
import torch
import torch.nn as nn

from b2j import embed
from b2j.data import transform


def _fake_ckpt(path, extra=None, drop=None):
    from timm.models.vision_transformer import VisionTransformer
    m = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)
    sd = dict(m.state_dict())
    sd.update({"decoder_embed.weight": torch.zeros(512, 768), "mask_token": torch.zeros(1, 1, 512)})
    sd.update(extra or {})
    for k in drop or ():
        sd.pop(k)
    torch.save({"model": sd}, path)
    return m


def test_endovit_loads_encoder_and_ignores_decoder(tmp_path, monkeypatch):
    p = tmp_path / "w.bin"
    ref = _fake_ckpt(p)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: str(p))
    m = embed.load_endovit("cpu")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(embed.cls_embedding(m, x), ref.eval().forward_features(x)[:, 0], atol=1e-5)
    assert embed.cls_embedding(m, x).shape == (2, 768)


def test_endovit_fails_on_missing_encoder_weights(tmp_path, monkeypatch):
    p = tmp_path / "w.bin"
    _fake_ckpt(p, drop=["norm.weight"])
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: str(p))
    with pytest.raises(ValueError, match="missing"):
        embed.load_endovit("cpu")


def test_endovit_transform_size_and_normalisation():
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.full((480, 854, 3), 128, np.uint8))
    x = transform("endoscapes", embed.ENDOVIT["size"], embed.ENDOVIT["mean"], embed.ENDOVIT["std"])(img)
    assert x.shape == (3, 224, 224)
    assert torch.allclose(x[0, 0, 0], torch.tensor((128 / 255 - 0.3464) / 0.2520), atol=1e-4)


def test_endovit_mean_pool_matches_authors_global_pool(tmp_path, monkeypatch):
    # Reference: EndoViT models_vit.py global_pool = fc_norm(mean(blocks(x)[:, 1:])) with fc_norm freshly initialised.
    p = tmp_path / "w.bin"
    ref = _fake_ckpt(p).eval()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, f: str(p))
    m = embed.load_endovit("cpu")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        t = ref.patch_embed(x)
        t = torch.cat([ref.cls_token.expand(2, -1, -1), t], 1) + ref.pos_embed
        for blk in ref.blocks:
            t = blk(t)
        expect = nn.LayerNorm(768, eps=1e-6)(t[:, 1:].mean(1))
        got = embed.mean_patch_embedding(m, x)
        # The pre-norm tokens also give back timm's forward_features after the pretrained final norm.
        assert torch.allclose(ref.norm(t), ref.forward_features(x), atol=1e-5)
    assert got.shape == (2, 768)
    assert torch.allclose(got, expect, atol=1e-5)
