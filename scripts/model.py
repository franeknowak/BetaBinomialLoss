import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.encoder_swinv2 import build_swinv2_encoder
from scripts.encoder_dinov3 import build_dinov3_encoder
class EvidentialHead(nn.Module):
    def __init__(self, in_feats, p_dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, in_feats),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_feats, 2),
        )

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
    
class SpatioTemporalModel(nn.Module):
    def __init__(self, encoder, temporal, heads):
        super().__init__()
        self.encoder      = encoder
        self.spatial_pool = AdaptiveSpatialPool()
        self.temporal     = temporal
        self.heads        = heads
    
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
        ft_weights_path = CONFIG['MODEL']['ENCODER']['FT_WEIGHTS']

        if ft_weights_path is not None:
            ft_state = torch.load(ft_weights_path, map_location='cpu')
            encoder.load_state_dict(ft_state)
            print(f"\nSuccessfully loaded {ft_weights_path} ft weights to {CONFIG['MODEL']['ENCODER']['NAME']} encoder.\n")
        else: print("\nEncoder initialised with IMTN weights!\n")

    elif 'dinov3' in CONFIG['MODEL']['ENCODER']['NAME']:
        encoder = build_dinov3_encoder(CONFIG)
        ft_weights_path = CONFIG['MODEL']['ENCODER']['FT_WEIGHTS']

        if ft_weights_path is not None:
            ft_state = torch.load(ft_weights_path, map_location='cpu')
            encoder.load_state_dict(ft_state)
            print(f"\nSuccessfully loaded {ft_weights_path} ft weights to {CONFIG['MODEL']['ENCODER']['NAME']} encoder.\n")
        else: print("\nEncoder initialised with IMTN weights!\n")
    else:
        raise ValueError(f"Unknown encoder: {CONFIG['MODEL']['ENCODER']['NAME']}")

    encoder.head = nn.Identity()

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
    
    else:
        raise ValueError(f"Unknown temporal aggregator: {CONFIG['MODEL']['TEMPORAL']['NAME'] }")

def _build_heads(in_dim, CONFIG, num_labels=3):
    loss = CONFIG['TRAIN']['LOSS']

    if loss == 'bbl':
        fc_dropout = CONFIG['MODEL']['CLASSIFIER']['DROPOUT']
        return nn.ModuleList([EvidentialHead(in_dim, fc_dropout) for _ in range(num_labels)])

    elif loss == 'bce':
        fc_dropout = CONFIG['MODEL']['CLASSIFIER']['DROPOUT']
        return nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Dropout(p=fc_dropout),
                nn.Linear(in_dim, 1),
            ) for _ in range(num_labels)
        ])

    else:
        raise ValueError(f"Unknown loss: {loss}")
 

def build_model(CONFIG):
    encoder = _build_encoder(CONFIG)

    # E2E training is not currently in development
    for p in encoder.parameters():
        p.requires_grad = False

    temporal   = _build_temporal(encoder.num_features, CONFIG)

    heads    = _build_heads(temporal.out_dim, CONFIG)

    return SpatioTemporalModel(encoder, temporal, heads)



######## JUST FOR ENCODER FT ########
def build_ft_model(CONFIG):
    name = CONFIG['MODEL']['ENCODER']['NAME']
    if 'swinv2' in name:
        return build_swinv2_encoder(CONFIG)
    elif 'dinov2' in name or 'dinov3' in name:
        return build_dinov3_encoder(CONFIG)
    else:
        raise ValueError(f"Unknown encoder: {name}")