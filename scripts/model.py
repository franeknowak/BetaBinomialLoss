import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
# Support functions
def _apply_frozen_stages(encoder, CONFIG):

    frozen_stages = CONFIG['MODEL']['ENCODER'].get('FROZEN_STAGES', 0)
    if frozen_stages == 0:
            return

    name = CONFIG['MODEL']['ENCODER']['NAME']

    # Freeze everything, then selectively unfreeze from frozen_stages onwards.
    # This preserves patch_embed and positional embeddings as frozen, matching
    # the intent that early-stage frozen means the full early network is fixed.
    for p in encoder.parameters():
        p.requires_grad = False

    if 'swinv2' in name:
        stages = list(encoder.layers)
        if not (0 < frozen_stages < len(stages)):
            raise ValueError(f"FROZEN_STAGES={frozen_stages} out of range [0, {len(stages)-1}] for {name}.")
        for stage in stages[frozen_stages:]:
            for p in stage.parameters():
                p.requires_grad = True

    elif 'dinov3' in name:
        blocks = list(encoder.blocks)
        if not (0 < frozen_stages < len(blocks)):
            raise ValueError(f"FROZEN_STAGES={frozen_stages} out of range [0, {len(blocks)-1}] for {name}.")
        for block in blocks[frozen_stages:]:
            for p in block.parameters():
                p.requires_grad = True

    else:
        raise ValueError(f"FROZEN_STAGES not supported for encoder: {name}")

    for p in encoder.norm.parameters():
        p.requires_grad = True

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
    

# Classifiers
class EvidentialHead(nn.Module):
    def __init__(self, in_feats, p_dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, in_feats//4),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_feats//4, 2),
        )

    def forward(self, x):
        evidence = F.softplus(self.fc(x)) + 1e-6
        return evidence + 1
    
class BCEHead(nn.Module):
    def __init__(self, in_feats, p_dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, in_feats//4),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_feats//4, 1),
        )

    def forward(self, x):
        return self.fc(x)

# Temporal Aggregators
class GatedPooling(nn.Module):
    def __init__(self, dim, p_dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//4),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(dim//4, 1)
        )
        self.out_dim = dim

    def forward(self, z):                               # z: [B, T, D]
        scores = self.gate(z).squeeze(-1)               # [B, T]
        weights = torch.softmax(scores, dim=1)
        return (z * weights.unsqueeze(-1)).sum(dim=1) # [B, D]

class LSTMTemporal(nn.Module):
    def __init__(self, in_dim, hidden_size, num_layers, p_dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size   = in_dim,
                            hidden_size  = hidden_size,
                            num_layers   = num_layers,
                            dropout      = p_dropout if num_layers > 1 else 0.0,
                            batch_first  = True)
        self.dropout = nn.Dropout(p = p_dropout) 
        self.out_dim = hidden_size

    def forward(self, z):                   # z: [B, T, D]
        out, _ = self.lstm(z)               # z: [B, T, D]
        return self.dropout(out[:, -1, :])  # [B, hidden_size]

class NoTemporal(nn.Module):
    """Pass-through for single-frame inputs — no learnable parameters."""
    def __init__(self, in_dim):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, z):   # z: [B, 1, D]
        return z[:, -1, :]  # [B, D] — takes the (only) frame

# Full Model Class    
class SpatioTemporalModel(nn.Module):
    def __init__(self, encoder, temporal, heads, freeze_encoder):
        super().__init__()
        self.encoder      = encoder
        self.spatial_pool = AdaptiveSpatialPool()
        self.temporal     = temporal
        self.heads        = heads
        self.freeze_encoder = freeze_encoder
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self                             

    def forward(self, x):                       # x: [B, T, C, H, W]
        if x.ndim == 4:                         # If not temporal model
            x = x.unsqueeze(1)                  # [B, C, H, W] → [B, 1, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder(x)
                z = self.spatial_pool(z)
        else:
            z = self.encoder(x)
            z = self.spatial_pool(z)

        z = z.view(B, T, -1)                    # [B, T, D]
        z = self.temporal(z)                    # [B, head_dim]
        return [head(z) for head in self.heads]


def _build_encoder(CONFIG):
    name = CONFIG['MODEL']['ENCODER']['NAME']
    if 'swinv2' in name:
        drop_path_rate = 0.1
        global_pool    = ""
    elif 'dinov3' in name:
        drop_path_rate = 0.0
        global_pool    = "token"
    else:
        raise ValueError(f"Designated encoder not supported: {name}")
    encoder = timm.create_model(    name,
                                    pretrained    = True,
                                    num_classes   = 0,
                                    global_pool   = global_pool,
                                    drop_path_rate= drop_path_rate)
    return encoder

def _build_temporal(in_dim, CONFIG):
    if CONFIG['MODEL']['TEMPORAL']['NAME'] == 'gated_pooling':
        temporal_gp = GatedPooling( in_dim,
                                    p_dropout = CONFIG['MODEL']['TEMPORAL']['GATED_POOLING']['DROPOUT'])
        return temporal_gp

    elif CONFIG['MODEL']['TEMPORAL']['NAME'] == 'lstm':
        temporal_lstm = LSTMTemporal(in_dim,
                                     hidden_size = CONFIG['MODEL']['TEMPORAL']['LSTM']['HIDDEN_SIZE'],
                                     num_layers  = CONFIG['MODEL']['TEMPORAL']['LSTM']['NUM_LAYERS'],
                                     p_dropout   = CONFIG['MODEL']['TEMPORAL']['LSTM']['DROPOUT'])
        return temporal_lstm         
    
    elif CONFIG['MODEL']['TEMPORAL']['NAME'] is None:
        return NoTemporal(in_dim)
    
    else:
        raise ValueError(f"Unknown temporal aggregator: {CONFIG['MODEL']['TEMPORAL']['NAME'] }")

def _build_heads(in_dim, CONFIG, num_labels=3):
    loss = CONFIG['TRAIN']['LOSS']
    fc_dropout = CONFIG['MODEL']['CLASSIFIER']['DROPOUT']

    if loss == 'bbl':
        return nn.ModuleList([EvidentialHead(in_dim, fc_dropout) for _ in range(num_labels)])

    elif loss == 'bce':
        return nn.ModuleList([BCEHead(in_dim, fc_dropout) for _ in range(num_labels)])
    
    else:
        raise ValueError(f"Provided loss not implemented: {loss}")
 

def build_model(CONFIG):
    encoder = _build_encoder(CONFIG)
    freeze_encoder = CONFIG['TRAIN']['FREEZE_ENCODER']

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        _apply_frozen_stages(encoder, CONFIG)

    temporal   = _build_temporal(encoder.num_features, CONFIG)

    heads    = _build_heads(temporal.out_dim, CONFIG)

    return SpatioTemporalModel(encoder, temporal, heads, freeze_encoder)
