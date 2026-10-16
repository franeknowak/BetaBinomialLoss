# Module f — run specs, launcher, smoke test

> **Changed 2026-10-16 (module h, 05_plan amendments #2, #3):** the `:S*(m)` rows were removed; F2 plots the advantage of B²-r over Global-S with bootstrap CIs; F5 shows P_maj-vs-m precision–recall curves with the difference band.

> **Changed 2026-10-16 (05_plan amendment #6):** the `rehead` regime and `part` A/B are removed (`RunSpec.from_json` returns None for old runs of those designs); `train_frac` keeps a nested share of the training clusters (seed-specific order, the same rows for every variant). Block `r1` = main 135 + full-panel check 10 + subsampling 95 + size sweep 40 = 280 frozen runs. Supplementary figure `figS_train_fraction`: Δ (B²-r − soft BCE) on mAP and BACC against the training fraction, with bootstrap CIs. The smoke block lists 34 runs (the re-head and part-B runs of job 7519912 are history).

Code: `b2j/runs.py`, `b2j/blocks.py`, `b2j/launch.py`, `b2j/report_smoke.py`. Tests: `tests/test_module_f.py`. Job scripts live on the cluster only (`BetaBinomialLoss/jobs/*.qsub.sh`, gitignored).

## Runs
- `RunSpec(dataset, variant, seed, regime, n_sub, val_full_panel, part, lr, wd, encoder_lr, epochs, eps, block)`; its name encodes every field that differs from the defaults. Output: `results/journal/<block>/<dataset>/<name>/` (cluster; gitignored) with `spec.json` + the trainer's files (module c).
- `tables(spec)`: training and validation votes subsampled to `n_sub` raters with the run's seed (validation keeps the full panel if `val_full_panel`); **test always the full panel** and passed to the trainer without labels.
- Regimes:
  - `frozen`: head on `cache/<dataset>_frozen.npz`;
  - `finetune`: encoder (public DINOv3) + head on images;
  - `rehead` (H5): fine-tune B²-r on part A (80% of training clusters, drawn per seed), embed every image with the checkpoint-B encoder held in memory (`features_ftB.npz`, ≈ 31 MB; the encoder is never saved), train a fresh frozen head on part B. Comparator: a `frozen` run with `part="B"` (same part B for the same seed).
- `execute(spec)` skips a run whose `summary.json` exists, so a resubmitted block resumes.
- `python -m b2j.launch <block> [--task i --ntasks N]`: runs every N-th spec from i (SGE job arrays); one failing run does not stop the block; the exit code reports failures.

## Smoke block (plan §4.4 module f; 36 runs, 1 seed, default lr 3e-4 / wd 1e-2)
- every frozen variant × Endoscapes, CIFAR-10H, MHIST (27);
- 1 fine-tuning epoch per loss (hard BCE, soft BCE, B²-r) on Endoscapes, encoder lr 1e-5 (3);
- one re-head pass on CIFAR-10H + its frozen part-B comparator (2);
- code paths: CIFAR-10H n = 3, n = 3 with full-panel validation, soft BCE n = 1, MHIST DUP n = 3 (4).

## Smoke report (`python -m b2j.report_smoke smoke` → `results/journal/smoke/report.md`; validation only)
- One row per run: evaluation epoch of A and B, validation mAP (A), validation vote NLL per vote (B), DUP loss, non-finite steps, stuck share at the last evaluation (max over heads), train/val median-S ratio, time.
- **ε fallback rule** (plan §2): triggered if any B²-r-family run (not B²-M) has a non-finite loss or a stuck share > 5% → rerun with ε = 0.1 for every B²-r-family run (user informed first).
- **Module e on real validation predictions** (replaces the MICCAI-era test): Endoscapes frozen B²-r vs soft BCE — discrimination at the validation-BACC threshold, vote NLL, ECE of P_maj, FCR at recall 0.90 for m and P_maj, AURC, triage, split rate at 80% coverage — and a 500-resample bootstrap over validation videos. A pipeline check, not a result.

## Bug caught by these tests
Endoscapes ids were pandas object arrays, which `np.load` refuses without pickle: the first Endoscapes cache (job 7519881) and every prediction file would have been unreadable. Ids are now plain strings (test in `test_module_b.py`); the smoke job rebuilds the Endoscapes cache first.

## Tables and figures (`b2j/analysis.py`)
`python -m b2j.analysis tables|figures <block> [--split val]`. **`--split test` raises unless `--open-test` is passed (R3 only)**; tested.
- Rows per dataset: every trained variant (A for discrimination, B for panel and use cases) + post-hoc rows `<variant>:plugin` (S = ∞; soft and hard BCE; step 1 of the spread ladder plug-in → Global-S → B²-r) and the soft-BCE ensemble (mean m; mixture of the members' Binomial vote distributions). All post-hoc fits from validation. Every prediction is carried as a vote distribution, so mixtures need no special case.
- T1 discrimination (ACC, BACC, F1, mAP, AUROC; post-hoc rows left out because they share their model's m); T2 panel (vote NLL per frame or per vote, ECE and Brier of P_maj, P(split) AUROC and Brier, θ-MSE); T3 use cases (FCR at recall 0.90 for m and P_maj, AURC, U1-c split rate at 80% coverage, U2 AP and catch rate). Mean ± sd over seeds; CIs and verdicts come from the verdict script (written with the R1 blocks).
- Figures: F1 theory (no data); F2 H4 advantage = vote NLL of soft BCE + Global-S − vote NLL of B²-r (checkpoint B; per vote on CIFAR-10H; positive = B²-r better) vs training raters n, with two-level bootstrap 95% CIs (different runs: shared test clusters, seeds resampled per model; numbers in `F2_advantage_<ds>.json`); F3 S vs m per B² variant; F4 validation S and fitted c per evaluation, frozen vs fine-tuned; F5 (a) precision vs recall for B²-r ranked by P_maj and by m, soft BCE's m faint (precision on a 0.5–1.0 recall grid, mean over heads then seeds), (b) the precision difference P_maj − m within B²-r with its bootstrap band (paired seeds), (c) risk–coverage; descriptive, the claim is only at recall 0.90 (numbers in `F5_pr_<ds>.json`). F6 (frames ranked by P(split)) needs the images: to be run on the cluster in R3.
- Checked on the smoke validation outputs (pulled to the Mac without any test file): tables match the smoke report; figures rendered and inspected. F3 shows B²-M's `+1` floor on real data.

## Still to do in module f
Verdict script (S0 … S6b with bootstrap CIs and the §5.1 readings; no margins) and the R1/R2 blocks, after module g; F6 on the cluster.
