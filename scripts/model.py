import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.encoder_swinv2 import build_swinv2_encoder
class EvidentialHead(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_feats, in_feats),
                                nn.GELU(),
                                nn.Linear(in_feats, 2)) 

    def forward(self, x):
        evidence = F.softplus(self.fc(x)) + 1e-6 
        return evidence + 1

class AdaptiveSpatialPool(nn.Module):
    """Mean-pool every dim between batch and feature.
       Essential for SwinV2 as it's output contains spatial features. 
       [B, H, W, D] -> [B, D]
       [B, N,    D] -> [B, D]
       [B,       D] -> [B, D]   (no-op)
    """
    def forward(self, x):
        if x.ndim <= 2:
            return x
        return x.mean(dim=tuple(range(1, x.ndim - 1)))
    
class GatedPooling(nn.Module):
    def __init__(self, dim, p_dropout):
        super().__init__()
        self.in_drop = nn.Dropout(p_dropout)  
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(dim, 1)
        )
        self.out_dim = dim

    def forward(self, z):                               # z: [B, T, D]
        scores = self.gate(z).squeeze(-1)               # [B, T]
        weights = torch.softmax(scores, dim=1)
        return (z * weights.unsqueeze(-1)).sum(dim=1) # [B, D]
    
class SpatioTemporalModel(nn.Module):
    def __init__(self, encoder, temporal, embed_dim, num_labels=3):
        super().__init__()
        self.encoder      = encoder
        self.spatial_pool = AdaptiveSpatialPool()
        self.temporal     = temporal
        self.heads        = nn.ModuleList([EvidentialHead(embed_dim) for _ in range(num_labels)])
    
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()   # encoder always in eval, regardless of parent mode
        return self

    def forward(self, x):                       # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        with torch.no_grad():                   # encoder is frozen
            z = self.encoder(x)
            z = self.spatial_pool(z)            # [B*T, D]

        z = z.view(B, T, -1)                    # [B, T, D]
        z = self.temporal(z)                    # [B, head_dim]
        return [head(z) for head in self.heads]


def _build_encoder(CONFIG):
    if 'swinv2' in CONFIG['MODEL']['ENCODER']['NAME']:
        encoder = build_swinv2_encoder(CONFIG)
        ft_state = torch.load(CONFIG['MODEL']['ENCODER']['FT_WEIGHTS'], map_location='cpu')
        encoder.load_state_dict(ft_state)

    elif 'dinov3' in CONFIG['MODEL']['ENCODER']['NAME']:
        raise NotImplementedError(f"Dino encoder not yet implemented.")

    else:
        raise ValueError(f"Unknown encoder: {CONFIG['MODEL']['ENCODER']['NAME']}")

    encoder.head = nn.Identity()

    return encoder

def _build_temporal(in_dim, CONFIG):
    if CONFIG['MODEL']['TEMPORAL']['NAME'] == 'gated_pooling':
        return GatedPooling(in_dim, p_dropout=CONFIG['MODEL']['TEMPORAL']['TEMPORAL_DROPOUT'])

    elif CONFIG['MODEL']['TEMPORAL']['NAME'] == 'lstm':
        raise NotImplementedError(f"LSTM temporal aggregator not yet implemented.") 
    
    else:
        raise ValueError(f"Unknown temporal aggregator: {CONFIG['MODEL']['TEMPORAL']['NAME'] }") 

def build_model(CONFIG):
    encoder = _build_encoder(CONFIG)

    for p in encoder.parameters():
        p.requires_grad = False

    temporal   = _build_temporal(encoder.num_features, CONFIG)

    return SpatioTemporalModel(encoder, temporal, temporal.out_dim)