# Module a — heads, per-sample (k, n) losses, registry of model variants

Code: `b2j/losses.py`, `b2j/heads.py`, `b2j/variants.py`. Tests: `tests/test_module_a.py` (`python -m pytest tests/test_module_a.py -q`).
New package; the MICCAI code in `scripts/` is untouched and used only as a regression reference.

## Interface
- Labels: `k, n` float tensors `[B, K]`, one column per binary head (K = 3 CVS, 10 CIFAR-10H one-vs-rest, 1 MHIST). n may vary per sample (CIFAR-10H, subsampling).
- `MultiHead(in_dim, K, out_dim, head_type)` → `[B, K, out_dim]`. MLP = D → 256 → out, GELU, dropout 0.1 (plan §2; no LayerNorm, unlike MICCAI's head). Linear = D → out.
- `variants.build(name, in_dim, k_train, n_train, eps=None)` → `(head, PanelLoss)`. Class weights and KL priors are computed from training labels only.
- `PanelLoss(out, k, n)` = Σ_heads mean_batch(per-sample loss). `per_sample` returns `[B, K]`.
- Runs on the input's device, except `torch.lgamma`/`digamma`, which fail on CUDA on some UCL nodes (GitHub issue #22; pilot job 7522356): the B² NLL uses no lgamma (integer votes → exact sums of logs on the GPU); log C(n, k) and the KL are evaluated on the CPU (`losses._on_cpu`). Tested: equals the closed form to 1e-9, and every lgamma/digamma call is on a CPU tensor. The B² NLL and KL are computed in **float64** and cast back (user decision 2026-10-14): float32 lgamma loses precision at large S. Gradient clipping (1.0) belongs to the trainer (module c).

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

## Trained model variants (plan §4.1)
hard_bce, hard_bce_w, soft_bce, b2r, b2r_w, b2r_kl (λ 0.05), b2m (legacy: +1, weights, KL λ 0.05 to MICCAI prior), b2r_linear, dup (n_train ≥ 3). Post-hoc comparisons (Global-S, S\*(m), S\*(m_B²), c, ensemble) are module d.

## Tests cover
Beta-Binomial vs scipy; KL vs torch; split target vs hypergeometric; strict majority; weight formula; moment prior recovers a simulated Beta (π 0.2, S 6); soft-BCE target; decomposition (large S → minimiser at mean k/n); **B²-M equals the MICCAI `total_bb_loss`**; positivity floors; finite gradients at logits ±30; stuck share; head shapes; one SGD step per variant; float32 inputs exact at S ~ 1e4; DUP refusal at n < 3; ε override spares B²-M; GPU residency (CUDA only).

## Decisions (2026-10-14, user + judgement)
- B²-M (the MICCAI recipe) is kept as one legacy row (user).
- No input LayerNorm in the head: with timm `global_pool='token'`, DINOv3 already applies its final LayerNorm (`norm`, learned affine) before taking the CLS token (`fc_norm` is off), so a second one is redundant. Hidden 256, dropout 0.1 kept.
- Soft BCE is not weighted by n on CIFAR-10H (standard soft-label practice; n varies only 47–63). B² is n-weighted by construction; noted in the paper.
- Code says "variants" (user finds "arms" ambiguous).
