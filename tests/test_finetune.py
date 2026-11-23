"""Tests for the LoRA fine-tuning ablation (b2j/finetune.py). Small random DINOv3 (ViT-S) on CPU; no download.
Run: python -m pytest tests/test_finetune.py -q
"""
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from b2j import finetune as F

SMALL = "vit_small_patch16_dinov3"


def test_lora_starts_as_the_frozen_layer_and_only_changes_q_and_v():
    torch.manual_seed(0)
    base = torch.nn.Linear(12, 36, bias=False)
    lora = F.LoRAqv(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(5, 12)
    torch.testing.assert_close(lora(x), base(x))                       # B = 0 at the start
    with torch.no_grad():
        lora.b_q.weight.normal_(), lora.b_v.weight.normal_()
    d = lora(x) - base(x)
    assert d[:, :12].abs().max() > 0 and d[:, 24:].abs().max() > 0 and d[:, 12:24].abs().max() == 0   # k untouched
    ref_q = (x @ lora.a_q.weight.T @ lora.b_q.weight.T) * 2.0
    torch.testing.assert_close(d[:, :12], ref_q)
    assert not any(p.requires_grad for p in base.parameters())


def test_encoder_all_blocks_wrapped_only_lora_trainable_and_output_unchanged_at_init():
    import timm
    torch.manual_seed(0)
    enc, nb = F.lora_encoder(pretrained=False, model_name=SMALL)
    assert nb == 12 and all(isinstance(b.attn.qkv, F.LoRAqv) for b in enc.blocks)
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all(any(s in n for s in (".a_q.", ".a_v.", ".b_q.", ".b_v.")) for n in trainable)
    assert len(trainable) == 12 * 4
    torch.manual_seed(0)
    ref = timm.create_model(SMALL, pretrained=False, num_classes=0).eval()
    enc.load_state_dict({**enc.state_dict(), **{k.replace("qkv.weight", "qkv.base.weight"): v
                                                for k, v in ref.state_dict().items()}}, strict=False)
    enc.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(enc.forward_features(x), ref.forward_features(x), atol=1e-5, rtol=1e-5)


def test_corner_free_side():
    assert F.corner_free_side(480, 0) == pytest.approx(480)
    assert F.corner_free_side(480, 10) == pytest.approx(480 / (math.cos(math.radians(10)) + math.sin(math.radians(10))))
    assert F.corner_free_side(480, -10) == F.corner_free_side(480, 10)


def test_rotate_crop_leaves_no_empty_corner_and_augmentation_size():
    white = Image.new("RGB", (854, 480), (255, 255, 255))
    tf = F.train_transform("endoscapes")
    rc = F.RotateCrop()
    for s in range(200):
        random.seed(s)
        torch.manual_seed(s)
        out = rc(white.crop((187, 0, 667, 480)))
        assert np.asarray(out).min() >= 250                               # no black fill anywhere
        x = tf(white)
        assert x.shape == (3, 256, 256)


def test_augmentation_has_no_flip():
    names = [type(t).__name__ for t in F.train_transform("endoscapes").transforms]
    assert not any("Flip" in n for n in names)
    assert names == ["CenterCrop", "RotateCrop", "RandomResizedCrop", "ToTensor", "Normalize"]
    rrc = F.train_transform("endoscapes").transforms[2]
    assert rrc.scale == (0.85, 1.0) and rrc.ratio == pytest.approx((0.9, 1 / 0.9))


class _Fake(torch.utils.data.Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 64, 64, generator=g)
        self.k = torch.randint(0, 4, (n, 3), generator=g).float()
        self.table = SimpleNamespace(ids=np.array([f"v_{i}" for i in range(n)]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.k[i], torch.full((3,), 3.0), i


def test_train_run_and_save_final(tmp_path):
    sets = {"train": _Fake(40, 0), "val": _Fake(16, 1), "test": _Fake(16, 2)}
    run, state = F.train_run(sets, ["C1", "C2", "C3"], "b2", 1e-3, 0, "cpu", epochs=1, workers=0, model_name=SMALL,
                             pretrained=False, batch=8)
    assert run["status"] == "ok" and len(run["curve"]) == 4 and run["wd"] == 0.01 and run["lora"]["rank"] == 8
    assert run["checkpoints"]["nll"]["eval"] == run["best_val_loss_eval"]
    assert all(k.startswith("head.") or ".a_" in k or ".b_" in k for k in state)
    F.save_final(tmp_path, run, state, sets, ["C1", "C2", "C3"], "cpu", model_name=SMALL, pretrained=False, workers=0)
    o = json.loads((tmp_path / "outputs_nll.json").read_text())
    assert len(o["images"]) == 32 and o["outputs"] == ["alpha0", "alpha1"]
    r = json.loads((tmp_path / "run.json").read_text())
    assert set(r["checkpoints"]["nll"]["metrics"]) == {"val", "test"}
