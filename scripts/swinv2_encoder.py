import torch
import sys
import torch.nn as nn
import timm

def build_swinv2_encoder(variant: str = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
                         frozen_stages: int = 0):
    
    # Initialise the encoder
    backbone = timm.create_model(   variant,
                                    pretrained=True,
                                    num_classes=0,   # removes classifier head
                                    global_pool=""   # keeps spatial features
                                    )
    
    # Create classifier head
    in_feats = 1024
    backbone.head = nn.Sequential(  SpatialPool(),
                                    nn.Linear(in_feats, 3))
    for p in backbone.parameters():
        p.requires_grad = False

    if frozen_stages is not None:
        num_stages = len(list(backbone.layers))
        if not (0 <= frozen_stages < num_stages):
            raise ValueError(
                f"frozen_stages={frozen_stages} is out of range for a model "
                f"with {num_stages} stages. Valid values: 0 … {num_stages - 1}."
            )
 
        # Unfreeze every stage AFTER the frozen prefix
        for stage_idx in range(frozen_stages, num_stages):
            for p in backbone.layers[stage_idx].parameters():
                p.requires_grad = True
 
        # Also unfreeze the patch-embedding norm inside frozen stages?
        # (Optional — comment in if you need it.)
        # for p in backbone.patch_embed.parameters():
        #     p.requires_grad = True
 
        # Always unfreeze the final encoder LayerNorm
        for p in backbone.norm.parameters():
            p.requires_grad = True
 
        _print_frozen_stages_summary(backbone, frozen_stages)
 
    # ------------------------------------------------------------------
    # 5. Head is always trainable
    # ------------------------------------------------------------------
    for p in backbone.head.parameters():
        p.requires_grad = True
 
    _print_param_summary(backbone)
    return backbone

def _print_frozen_stages_summary(model: nn.Module, frozen_stages: int) -> None:
    counts = [len(list(stage.blocks)) for stage in model.layers]
    num_stages = len(counts)
    lines = ["[SwinV2] Stage freeze summary:"]
    for i, n in enumerate(counts):
        status = "FROZEN" if i < frozen_stages else "trainable"
        lines.append(f"  stage {i}  ({n:2d} blocks, dim={128 * 2**i:4d})  →  {status}")
    print("\n".join(lines))
 
 
def _print_param_summary(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[SwinV2] Params — total: {total:,}  |  "
        f"trainable: {trainable:,}  ({100 * trainable / total:.2f} %)"
    )

class SpatialPool(nn.Module):
    def forward(self, x):
        # x: (batch, 12, 12, 1024)
        return x.mean(dim=(1, 2))  # → (batch, 1024)