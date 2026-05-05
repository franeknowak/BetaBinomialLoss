import torch
import sys
import torch.nn as nn
import timm

def build_dinov3_encoder(CONFIG):
    
    variant = CONFIG['MODEL']['ENCODER']['NAME']
    frozen_stages = CONFIG['MODEL']['ENCODER']['FROZEN_STAGES'] # NOTE frozen_stages are in fact blocks given the dinov3 architecture. Stages naming convention kept for consistency with swinv2 yaml config.
    drop_path_rate = CONFIG['MODEL']['ENCODER']['DROP_PATH_RATE']
    fc_dropout = CONFIG['MODEL']['CLASSIFIER']['DROPOUT']
    
    # Initialise the encoder
    backbone = timm.create_model(   variant,
                                    pretrained=True,
                                    num_classes=0,
                                    global_pool="token",
                                    drop_path_rate=drop_path_rate)
    
    # Create classifier head
    backbone.head = nn.Sequential(  nn.Dropout(p=fc_dropout),
                                    nn.Linear(backbone.num_features, 3))
    for p in backbone.parameters():
        p.requires_grad = False

    num_blocks = len(backbone.blocks)
    if not (0 <= frozen_stages < num_blocks):
        raise ValueError(f"FROZEN_STAGES={frozen_stages} invalid. Must be 0 to {num_blocks-1}.")

    for i in range(frozen_stages, num_blocks):   # train everything after the frozen prefix
        for p in backbone.blocks[i].parameters():
            p.requires_grad = True

    for p in backbone.norm.parameters():         # always train final norm
        p.requires_grad = True
    for p in backbone.head.parameters():         # always train head
        p.requires_grad = True

    return backbone
 