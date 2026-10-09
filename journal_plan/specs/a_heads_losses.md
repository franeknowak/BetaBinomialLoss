# Module a — heads, per-sample (k, n) losses, arm registry

Code: `b2j/losses.py`, `b2j/heads.py`, `b2j/arms.py`. Tests: `tests/test_module_a.py` (`python -m pytest tests/test_module_a.py -q`).
New package; the MICCAI code in `scripts/` is untouched and used only as a regression reference.

## Interface
- Labels: `k, n` float tensors `[B, K]`, one column per binary head (K = 3 CVS, 10 CIFAR-10H one-vs-rest, 1 MHIST). n may vary per sample (CIFAR-10H, subsampling).
- `MultiHead(in_dim, K, out_dim, head_type)` → `[B, K, out_dim]`. MLP = D → 256 → out, GELU, dropout 0.1 (plan §2; no LayerNorm, unlike MICCAI's head). Linear = D → out.
- `arms.build(arm, in_dim, k_train, n_train, eps=None)` → `(head, PanelLoss)`. Class weights and KL priors are computed from training labels only.
- `PanelLoss(out, k, n)` = Σ_heads mean_batch(per-sample loss). `per_sample` returns `[B, K]`.
- Everything runs on the input's device (no CPU round trip, unlike MICCAI). Gradient clipping (1.0) belongs to the trainer (module c).

## Losses
| kind | target / likelihood | out_dim |
|---|---|---|
| hard_bce | BCE on majority y = 1[2k > n] (strict; ties → negative) | 1 |
| soft_bce | BCE on k/n, **no pos_weight**, samples not weighted by n | 1 |
| b2 | full Beta-Binomial NLL of k | n under Beta(α₁, α₀), incl. log C(n,k) | 2 (α₀ logit, α₁ logit) |
| dup | BCE on unbiased 3-rater split prob. 1 − [C(k,3)+C(n−k,3)]/C(n,3) (= 1[0<k<3] at n = 3); n ≥ 3 | 1 |

- Positivity (B²-r): α = softplus(logit) + ε, ε = 1e-3; `eps=0.1` override is the R0 fallback. B²-M: softplus + 1e-6 + 1.
- Weighting (`weighted`): per-sample w± = N/(2N±) by majority label, same for hard BCE-w and B²-r-w (not `pos_weight`).
- KL (`kl_lambda`): analytic KL(Beta(α₁,α₀) ‖ Beta(p₁,p₀)).
  - B²-r+KL prior ("moment"): π = mean k/n; S by method of moments, Var(k/n) = π(1−π)[ρ + (1−ρ)E(1/n)], ρ = 1/(S+1), clamped.
  - B²-M prior ("miccai"): π = majority-positive prevalence, ν = 2 (reproduces the MICCAI PI values = 1/(2w+)).
- Diagnostic `stuck_share`: per head, share of k ≥ 1 samples with α₁ logit < −7 (ε fallback rule: > 5% or any non-finite loss).

## Arms (plan §4.1, trained ones)
hard_bce, hard_bce_w, soft_bce, b2r, b2r_w, b2r_kl (λ 0.05), b2m (legacy: +1, weights, KL λ 0.05 to MICCAI prior), b2r_linear, dup (n_train ≥ 3). Post-hoc arms (Global-S, S\*(m), S\*(m_B²), c, ensemble) are module d.

## Tests cover
Beta-Binomial vs scipy; KL vs torch; split target vs hypergeometric; strict majority; weight formula; moment prior recovers a simulated Beta (π 0.2, S 6); soft-BCE target; decomposition (large S → minimiser at mean k/n); **B²-M equals the MICCAI `total_bb_loss`**; positivity floors; finite gradients at logits ±30; stuck share; head shapes; one SGD step per arm; DUP refusal at n < 3; ε override spares B²-M; GPU residency (CUDA only).

## Choices made here (flag if wrong)
- MLP head without input LayerNorm, hidden 256 (plan text), vs MICCAI's LayerNorm + D/4.
- Soft BCE is not n-weighted on CIFAR-10H (n 47–63); B² is n-weighted by construction.
