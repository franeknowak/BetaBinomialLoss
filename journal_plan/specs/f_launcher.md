# Module f — run specs, launcher, smoke test

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

## Still to do in module f
Table and figure scripts (T1–T3, F1–F6) written against validation outputs, once the smoke outputs exist; R1/R2 blocks after module g.
