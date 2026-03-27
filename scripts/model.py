import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.dino_encoder import build_dino_encoder
class EvidentialHead(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_feats, in_feats),
                                nn.GELU(),
                                nn.Linear(in_feats, 2)) 

    def forward(self, x):
        evidence = F.softplus(self.fc(x)) + 1e-6 
        alpha = evidence + 1
        return alpha
    
class GatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, z):  # z: [B, T, D]
        scores = self.gate(z).squeeze(-1)      # [B, T]
        weights = torch.softmax(scores, dim=1)
        pooled = (z * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return pooled
    
class SequenceEvidentialModel(nn.Module):
    def __init__(self, frozen_encoder, embed_dim=768, num_labels=3):
        super().__init__()
        self.encoder = frozen_encoder        # DINOv3 (frozen)
        self.pool = GatedPooling(embed_dim)
        self.heads = nn.ModuleList(
            [EvidentialHead(embed_dim) for _ in range(num_labels)]
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        with torch.no_grad():
            z = self.encoder(x)              # [B*T, D]

        z = z.view(B, T, -1)                 # [B, T, D]
        z_seq = self.pool(z)                 # [B, D]

        return [head(z_seq) for head in self.heads]
    

def build_model(backbone_dict,
                classifier_dict):
    if backbone_dict['dino'] is not None:
        encoder = build_dino_encoder(**backbone_dict['dino']['kwargs'])
        pretrained_encoder_weights = torch.load(backbone_dict['dino']['backbone_weights'], map_location='cpu')
        encoder.load_state_dict(pretrained_encoder_weights)

    encoder.head = nn.Identity()

    for p in encoder.parameters():
        p.requires_grad = False
    
    if classifier_dict['temporal_processing'] == 'gated_pooling' and classifier_dict['head'] == 'evidential':
        model = SequenceEvidentialModel(encoder)

    return model