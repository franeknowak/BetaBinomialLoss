import torch
import sys
import torch.nn as nn

from scripts.lora import inject_lora_into_dinov3_qkv

def build_dino_encoder( repository_path,
                        imnt_weights_path,
                        lora = {'r': 6, 'last_blocks': 6, 'alpha': 12},
                        partial_training_blocks = 6):
    
    assert (lora is None) != (partial_training_blocks is None), "Lora and partial training are mutually exclusive. Exactly one must be None"


    sys.path.insert(0, str(repository_path))

    # Initialise the encoder
    model = torch.hub.load( repository_path,
                            'dinov3_vitb16',
                            source='local',
                            weights=imnt_weights_path)
    
    # Create classifier head
    in_feats = 768
    model.head = nn.Sequential( nn.Linear(in_feats, in_feats),
                                nn.GELU(),
                                nn.Linear(in_feats, 3))
    for p in model.parameters():
        p.requires_grad = False

    if lora is not None:
        num_blocks = len(model.blocks)
        target_layers = list(range(num_blocks - lora['last_blocks'], num_blocks))
        lora_modules = inject_lora_into_dinov3_qkv(model, r=lora['r'], layers=target_layers, verbose=True, alpha = lora['alpha'])
        for module in lora_modules:
            for name, p in module.named_parameters():
                if "linear_a_" in name or "linear_b_" in name:
                    p.requires_grad = True

    if partial_training_blocks is not None:
        K = partial_training_blocks
        num_blocks = len(model.blocks)
        target_layers = list(range(num_blocks - K, num_blocks))

        # Unfreeze last K blocks from the encoder
        for i in target_layers:
            for p in model.blocks[i].parameters():
                p.requires_grad = True

        # Unfreeze final encoder norm
        for p in model.norm.parameters():
            p.requires_grad = True

    # Enable gradients for head
    for p in model.head.parameters():
        p.requires_grad = True
    
    return model