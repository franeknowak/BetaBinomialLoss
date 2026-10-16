# Module e — metrics, two-level bootstrap, verdicts

> **Changed 2026-10-16 (module h, 05_plan amendment #1):** no equivalence margins. `margin()`, `CAPS` and the equivalence / non-inferiority verdicts were removed; readings are superiority or difference. The bootstrap's random draws and scalar results are unchanged; it now also accepts array-valued metrics (pointwise CI, for F5).

Code: `b2j/metrics.py`, `b2j/stats.py`. Tests: `tests/test_module_e.py`.

## Targets (from the evaluation panel k, n)
- Majority y = 1[k > n/2] (strict; even-n ties are "not yes").
- 3-rater split target: 1[0 < k < 3] at n = 3; for larger panels the unbiased probability that a random 3-rater sub-panel is not unanimous (identical to the DUP training target, `losses.split_target`).
- Soft targets (the split target on CIFAR-10H/MHIST) enter AUROC and AP as weighted rows: each image counts as a positive with weight t and a negative with weight 1 − t. For binary targets this equals the usual metric (tested).

## Metrics (`metrics.py`; per head, then mean over heads)
| Function | Plan §4.3 / §4.2 item |
|---|---|
| `discrimination(score, y, thr)` | ACC, BACC, F1 at the validation-BACC threshold; mAP; AUROC (secondary) |
| `vote_nll(m, S, k, n)` | vote NLL per frame-criterion and per vote (CIFAR-10H) |
| `ece(p, y)` | ECE of P_maj, 10 equal-mass bins |
| `brier`, `theta_mse` | Brier of P_maj / P(split); θ-MSE against the full panel |
| `fcr_at_recall(score, y, 0.90)` | **FCR = 1 − precision at test recall 0.90**, read off the PR curve (first point, descending score, whose recall ≥ 0.90; tied scores enter together) — threshold-free primary (H6, U1-a) |
| `fcr_at_threshold(score, y, thr)` | secondary FCR at the validation-chosen recall-0.90 threshold, with the realised recall |
| `aurc(p_maj, y)` | U1-b: confidence max(P_maj, 1 − P_maj), risk = decision (P_maj ≥ 0.5) disagrees with the majority; mean selective risk over all coverages |
| `split_rate_at_coverage(p_split, t, 0.8)` | U1-c: split rate among the 80% of frames with the lowest P(split) |
| `triage(p_split, t)` | U2: AP, AUROC of P(split); catch rate = share of the split mass in the top 20% |
| `split_reliability(p_split, t)` | P(split) reliability: 10 equal-mass bins → mean prediction, mean observed, jackknife SE (noise floor) |

## Statistics (`stats.py`)
- `two_level_bootstrap(metric, models, combine, clusters, strata, paired, B=2000)`:
  - level 1: test clusters resampled with replacement, **shared by every model** in the comparison (Endoscapes: videos; MHIST: images; CIFAR-10H: images within class via `strata`);
  - level 2: seeds resampled per model; a model's value = per-seed metric averaged over the drawn seeds. Models in a `paired` group (two scores of the same runs, e.g. P_maj vs m within B²-r for S6a) share the seed draw;
  - `combine` maps the per-model values to the statistic (a Δ, or a difference of advantages for S4b/S5c);
  - the metric may return an array (e.g. precision on a recall grid); the CI is then pointwise (`Result.estimate/lo/hi` are arrays);
  - estimate = all rows, all seeds; 95% percentile CI. Validation-fitted quantities are inputs, never refitted.
  - Ensemble: pass it as one model with one "seed" (resampled over test clusters only).
- `verdict(result, kind, direction)` — plan §5.1 (no margins):
  - superiority (`direction` = predicted sign): success if the CI excludes 0 in the predicted direction; fail if in the other; else inconclusive;
  - difference (`direction` = the sign that means the first model is better: +1 for BACC/mAP/AP, −1 for NLL/FCR): "better" / "worse" if the CI excludes 0; else "no evidence of a difference". Never "equivalent".
- Metrics: `precision_at_recalls(score, y, grid)` uses the same operating-point rule as `fcr_at_recall` (first point whose recall reaches each grid value; ties enter together), mean over heads.

## Tests cover
ACC/BACC/F1/mAP/AUROC against scikit-learn; weighted soft targets reduce to the usual metrics; split target equals the DUP target; FCR at recall against sklearn's PR curve and a hand case with ties; AURC, coverage and catch rate by hand; ECE ≈ 0 when calibrated and 0.4 for a constant 0.9; jackknife SE equals the SE of a mean; NLL by hand; bootstrap: identical models give exactly 0, a cluster spanning strata is refused, **coverage ≈ 95% in simulation** (shared cluster effects, 5 seeds), paired seed draws remove seed noise from within-model contrasts; every reading branch (no margin functions left); an array-valued metric gives the same pointwise CI as the scalar runs; precision on a grid agrees with FCR at recall 0.90.

## Real-data check
Run on the smoke-test validation predictions of the new models (module f), instead of MICCAI-era predictions (plan amendment 2026-10-14 #6).
