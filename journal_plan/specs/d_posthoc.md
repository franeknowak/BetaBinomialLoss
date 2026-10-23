# Module d — post-hoc quantities (validation only)

> **Changed 2026-10-16 (module h, 05_plan amendment #3):** S\*(m), S\*(m_B²) and `pmaj_monotone` were removed. Concentrations are compared as a ladder: plug-in (S = ∞) → Global-S (soft BCE's m) → B²-r. Whether S carries information beyond m is shown by ranking B²-r frames by P_maj vs by m (F5, module f).

Code: `b2j/posthoc.py` (its concentration step was started in module c). Tests: `tests/test_module_d.py` (+ the concentration tests in `tests/test_module_c.py`).
Everything here is fitted on validation and applied unchanged to test; nothing is resampled in the bootstrap (plan §4.3).

## From a model's output to a panel prediction
- `mean_concentration(variant, out, eps)` → m (predicted share of "yes" votes) and the raw concentration S0 (B²: α₀+α₁; BCE: ∞).
- `panel(variant, out, scale)` applies the variant's concentration step (module c, plan amendment 2026-10-14 #5): B² → S = c·S0; hard/soft BCE → Global-S (one S\* per criterion); B²-M native. The independent-raters plug-in is `S = ∞` (supplementary row).

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
- BACC threshold against brute force with tied scores; recall threshold is the largest that meets 0.90.

## Not in this module
Metrics, bootstrap and verdicts (module e); which checkpoint each quantity is read from (A for discrimination, B for panel endpoints) is applied by the analysis scripts (module f).
