# Journal plan — STATE

## Current step
Round 0, module b: Endoscapes DONE and tested; CIFAR-10H and MHIST loaders written and unit-tested on synthetic data, real data not yet on disk (2026-10-09).

## Done
- Planning complete (steps 1–4, deliverables 04/05/06, 2026-10-09 amendments). See HANDOFF.md.
- 2026-10-09: branch `journal-extension` created from main, plan + CLAUDE.md committed and pushed to origin.
- 2026-10-09: module a — `b2j/{losses,heads,arms}.py`, spec `journal_plan/specs/a_heads_losses.md`, tests `tests/test_module_a.py` (25 pass, CUDA test skipped on Mac). B²-M reproduces MICCAI `total_bb_loss` exactly.

- 2026-10-09: module b — `b2j/data.py`, spec `journal_plan/specs/b_loaders.md`, tests `tests/test_module_b.py` (32 pass in total). Endoscapes exact k from per-rater columns; split sizes and prevalences match the plan; U2 feasible on all criteria.
- Finding: strong systematic rater effects on Endoscapes (C1 positive rate 0.033 / 0.160 / 0.279 by rater) → limitation paragraph (spec b).

## Cluster (verified 2026-10-09)
- Host comic2; SGE (`/opt/gridengine/bin/lx-amd64/qsub`), no Slurm. Queues incl. gpu.q, fastgpu.q, test.gpu.q; gpu.q h_rt up to 1000 h.
- Job template: `/cluster/project7/FranekLapChole/BetaBinomialLoss/ft_experiments.qsub.sh` — `-l gpu=true -l gpu_type=a6000 -pe gpu 1 -l tmem=24G -l h_rt=24:00:00 -R y`; sources `/share/apps/source_files/python/python-3.11.9.source`, `cuda-11.8.source`, `gcc-9.2.0.source`; venv `../MICCVS/dino_env`; `HF_HOME=.../BetaBinomialLoss/.hf_cache`. (venv python fails on the login node without the python source file; fine in jobs.)
- Endoscapes: `/cluster/project7/FranekLapChole/SwinCVS/endoscapes/` (train/val/test jpgs + `all_metadata.csv` with per-rater columns). Local copy of the csv in `dataset/endoscapes/` (gitignored).
- Cluster repo `/cluster/project7/FranekLapChole/BetaBinomialLoss` is on `main` with untracked `scripts/bbkl_loss_old.py`; not yet switched to `journal-extension`.
- CVS Challenge: NOT downloaded (only the two download scripts exist). Needed for R4 only.
- SSH: ControlMaster must be opened by the user interactively (`ssh comic true` in their terminal; key passphrase + password). Persists 4 h.

## Exact next action
1. With user OK: download CIFAR-10H counts (github jcpeterson/cifar-10h, `data/cifar10h-counts.npy`) + CIFAR-10 test images (torchvision) on the cluster into `/cluster/project7/FranekLapChole/datasets/cifar10h/`; export PNGs `{id:05d}.png`; run the CIFAR loader on real data and add a real-data test (counts 10000×10, n 47–63).
2. MHIST: when access arrives, verify the vote column header and patient grouping, add a real-data test.
3. Module c: embedding cache + trainer (checkpoints A/B, silent test writing).

## Open questions (user)
- MHIST research-use request: has it been sent? (R0 day 1)
- Keep/drop the legacy B²-M row (implemented; costs one arm).
- Permission to download CIFAR-10H (+ CIFAR-10 test images) on the cluster.
- Module a choices: MLP head without LayerNorm (plan's 768→256→out); soft BCE not n-weighted on CIFAR-10H. See spec.
