import torch
import sys
import torch.nn as nn
import timm

def build_swinv2_encoder(variant: str,
                         frozen_stages: int = 0):
    
    # Initialise the encoder
    backbone = timm.create_model(   variant,
                                    pretrained=True,
                                    num_classes=0,
                                    global_pool="",
                                    drop_path_rate=0.1
                                    )
    
    # Create classifier head
    backbone.head = nn.Sequential(  SpatialPool(),
                                    nn.LayerNorm(backbone.num_features),  # stabilises the pooled features
                                    nn.Dropout(p=0.3),
                                    nn.Linear(backbone.num_features, 3))
    for p in backbone.parameters():
        p.requires_grad = False

    # Freeze certain blocks
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
 
        # Always unfreeze the final encoder LayerNorm
        for p in backbone.norm.parameters():
            p.requires_grad = True

    for p in backbone.head.parameters():
        p.requires_grad = True

    return backbone
 
class SpatialPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2))  