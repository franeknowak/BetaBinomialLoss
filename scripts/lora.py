import math
import torch
import torch.nn as nn
class LoRA(nn.Module):
    """LoRA for DINOv3 qkv: adds low-rank updates to Q and V and K. Includes scaling factor."""
    def __init__(self, qkv: nn.Module,
                 linear_a_q: nn.Linear, linear_b_q: nn.Linear,
                 linear_a_k: nn.Linear, linear_b_k: nn.Linear,
                 linear_a_v: nn.Linear, linear_b_v: nn.Linear,
                 alpha: int = 1,
                 r: int = 3):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.alpha = alpha
        self.r = r
        self.scaling = alpha / r

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        # dim is the per-projection dimension (out_features = 3 * dim)
        self.dim = self.out_features // 3

        # sanity check
        assert self.out_features == 3 * self.dim, "qkv out_features must be 3 * dim"

        # freeze base qkv params (as in the original)
        for p in self.qkv.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original qkv
        qkv = self.qkv(x)  # (B, N, 3*dim) or (any_batch_shape..., 3*dim)

        # LoRA low-rank deltas (Q and V only)
        new_q = self.scaling * self.linear_b_q(self.linear_a_q(x))
        new_k = self.scaling * self.linear_b_k(self.linear_a_k(x))
        new_v = self.scaling * self.linear_b_v(self.linear_a_v(x))

        # Add to Q and V slices
        qkv[..., : self.dim]      = qkv[..., : self.dim]      + new_q
        qkv[..., self.dim : 2*self.dim] = qkv[..., self.dim : 2*self.dim] + new_k
        qkv[..., -self.dim : ]    = qkv[..., -self.dim : ]    + new_v
        return qkv
    
def _make_lora_pair(in_dim: int, r: int, device, dtype):
    A = nn.Linear(in_dim, r, bias=False, device=device, dtype=dtype)
    B = nn.Linear(r, in_dim, bias=False, device=device, dtype=dtype)
    return A, B

def _reset_lora_params(w_as, w_bs):
    for w_a in w_as:
        nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
    for w_b in w_bs:
        nn.init.zeros_(w_b.weight)

def inject_lora_into_dinov3_qkv(model: nn.Module, r: int = 3, layers: list[int] | None = None, verbose: bool = True, alpha: int = 12):
    """
      - Inject LoRA into EVERY block.attn.qkv
      - LoRA on Q, K and V
    Returns a list of the newly created LoRA modules (for later saving if needed).
    """
    lora_modules = []
    w_as, w_bs = [], []

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("Model has no .blocks; cannot inject LoRA as requested.")

    target_indices = range(len(blocks)) if layers is None else layers

    for i in target_indices:
        blk = blocks[i]
        qkv = blk.attn.qkv
        if not hasattr(qkv, "in_features") or not hasattr(qkv, "out_features"):
            raise RuntimeError(f"blocks[{i}].attn.qkv does not look like a Linear module.")

        in_dim  = qkv.in_features
        out_dim = qkv.out_features
        assert out_dim % 3 == 0, "qkv.out_features must be divisible by 3"

        device = qkv.weight.device if hasattr(qkv, "weight") else next(blk.parameters()).device
        dtype  = qkv.weight.dtype  if hasattr(qkv, "weight") else next(blk.parameters()).dtype

        # Create LoRA A/B for Q and V (exactly like the reference)
        w_a_q, w_b_q = _make_lora_pair(in_dim, r, device, dtype)
        w_a_k, w_b_k = _make_lora_pair(in_dim, r, device, dtype)
        w_a_v, w_b_v = _make_lora_pair(in_dim, r, device, dtype)
        w_as.extend([w_a_q, w_a_k, w_a_v])
        w_bs.extend([w_b_q, w_b_k, w_b_v])      

        # Wrap qkv with LoRA
        lora_qkv = LoRA(qkv, w_a_q, w_b_q, w_a_k, w_b_k, w_a_v, w_b_v, alpha = alpha, r=r)
        blk.attn.qkv = lora_qkv
        lora_modules.append(lora_qkv)

    # Reset LoRA params to match the reference (A=kaiming, B=zeros)
    _reset_lora_params(w_as, w_bs)

    if verbose:
        print(f"[LoRA] Injected into qkv of {len(lora_modules)} blocks (rank={r}).")

    return lora_modules
