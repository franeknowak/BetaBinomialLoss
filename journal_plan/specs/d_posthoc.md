# Module d — post-hoc quantities (validation only)

Code: `b2j/posthoc.py` (its spread step was started in module c). Tests: `tests/test_module_d.py` (+ the spread tests in `tests/test_module_c.py`).
Everything here is fitted on validation and applied unchanged to test; nothing is resampled in the bootstrap (plan §4.3).

## From a model's output to a panel prediction
- `mean_spread(variant, out, eps)` → m (predicted share of "yes" votes) and the raw spread S0 (B²: α₀+α₁; BCE: ∞).
- `panel(variant, out, scale)` applies the variant's spread step (module c, plan amendment 2026-10-14 #5): B² → S = c·S0; hard/soft BCE → Global-S (one S\* per criterion); B²-M native. The independent-raters plug-in is `S = ∞` (supplementary row).
- `fit_s_of_m(m, k, n)` / `s_of_m(fit, m)`: **S\*(m)**, a spread that depends only on m (plan §2). Per criterion: 5 equal-size bins of validation m (by rank), one maximum-likelihood S per bin, knots at each bin's median logit m, **S interpolated linearly in logit m**, flat beyond the outer knots. Fitted on soft BCE's m (baseline row) and on B²-r's own m → S\*(m_B²), the per-sample comparator for H4.
  - Interpolating S itself (not log S) follows the plan, and it matters: on the plan's steep example (S = 1 at m = 0.15 → 30 at 0.3), log-S interpolation makes P_maj monotone and would hide the effect R3 must check.
- `pmaj_monotone(fit, n)`: plan §3 point 2 check, per criterion, on a 2,001-point grid of m. If false, S\*(m) is kept as an extra U1 comparator.

## Panel probabilities (from the vote distribution)
- `vote_pmf(m, S, n)`: P(k of n say yes), Beta-Binomial (Binomial at S = ∞); n may differ per image (CIFAR-10H).
- `p_majority(pmf, n)` = P_maj = P(k > n/2). **Strict majority; at even n a tie counts as "not yes"** (same as the hard-BCE label). n = the panel whose majority is the target: Endoscapes 3, MHIST 7, CIFAR-10H the image's full panel (47–63).
- `p_split3(m, S)` = P(a 3-rater panel is not unanimous) = 3m(1−m)·S/(S+1); the same 3-rater question on every dataset (as DUP's target).
- `p_unanimous(pmf, n)`, `agree_with_decision(P_maj, decision)` (P(the majority agrees with a yes/no decision)).
- Soft-BCE ensemble (5 seeds): `mixture_pmf` = mean of the members' vote distributions; `mixture_p_split3` = mean of their P(split). Discrimination score = mean m.

## Thresholds (per criterion, on validation)
- `threshold_max_bacc(score, y)`: maximises validation BACC of `score ≥ t` against the majority label; candidates are midpoints between distinct scores; ties → lowest threshold (favours sensitivity). Used for ACC, BACC, F1. A criterion with one class in validation → 0.5.
- `threshold_at_recall(score, y, 0.90)`: the largest threshold with validation recall ≥ 0.90 (H6 secondary: FCR at a validation-chosen operating point, with the realised test recall reported).

## Tests cover
- pmf against scipy, with varying n;
- **the plan's boundary table (§3 point 3) and P_maj ranges (§3 point 4) reproduced to ±0.0015**;
- P_maj > 0.5 ⟺ m > 0.5 for every S; even-n ties;
- P(split) = 1 − P(unanimous) at n = 3; ensemble mixtures;
- S\*(m) recovers a constant S and a step in S; linear interpolation and flat ends; n = 1 → ∞;
- monotonicity check (slow S\*(m) passes, the plan's steep example fails);
- BACC threshold against brute force with tied scores; recall threshold is the largest that meets 0.90.

## Not in this module
Metrics, bootstrap and verdicts (module e); which checkpoint each quantity is read from (A for discrimination, B for panel endpoints) is applied by the analysis scripts (module f).
