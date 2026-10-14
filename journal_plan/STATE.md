# Journal plan — STATE

## Current step
Round 0. Module c (feature cache + one trainer) WRITTEN and unit-tested on the Mac (2026-10-14), not yet run on the cluster (needs timm/DINOv3 → first real run in module f smoke test). MHIST downloaded and verified. CIFAR-10H download job running.

(Previous:) Round 0. Module a DONE. Module b DONE for Endoscapes; CIFAR-10H and MHIST loaders written and unit-tested on synthetic data, real data not yet downloaded. Next: dataset download job (needs user go-ahead), then module c. (Updated 2026-10-14; handing over to Claude CLI.)

## Read first
HANDOFF.md → this file → `05_plan.md` (the 2026-10-09 and **2026-10-14 amendments** at the top override the body; 2026-10-14 adds a validation-only method-development round D, a storage policy and a corrected statement of the soft-BCE decomposition).

## User instructions (standing)
- **Dataset layout (2026-10-14):** `<repo>/../datasets/<name>` on BOTH Mac and cluster (`b2j.data.DATA_ROOT`). Mac: done 2026-10-14 (`python_code/datasets/` + compatibility symlink `dataset -> datasets` for older projects). Repo-internal `dataset/` copy is obsolete (gitignored; user may delete).
- **Git layout (2026-10-14):** all code on `journal-extension`. Job scripts (`*.qsub.sh`, gitignored), results and weights live ONLY on the cluster, never in git. Local drafts of job scripts sit in `jobs/` (ignored) and are copied to the cluster with `ssh comic 'cat > …' < file`.
- **No jobs submitted until the user says go** (2026-10-14: "don't run any jobs at the moment").
- Cluster etiquette: look-ups (ls, head, grep, cat, git, qstat, quota) are fine on the comic login node. Downloads, big unzips, heavy python, training → a script submitted with `qsub`.
- Don't say "arms" to the user: say "model variants" / "training configurations", or name them. Define abbreviations: **B²-M = the MICCAI recipe** (softplus+1, class weights, KL to prior); **B²-r = the revised recipe** (softplus+ε, unweighted, no KL). Explain actions in plain language; module letters need a one-line description.
- The user still aims to beat baseline scores, not only analyse (→ round D, 05_plan amendment 2026-10-14 #1).
- Checkpoints: keep storage small (05_plan amendment 2026-10-14 #3). Tell the user if more space is needed.
- Downloads APPROVED: CIFAR-10H and MHIST. MHIST access GRANTED: the user has time-limited Dropbox links (email) for `MD5SUMs.txt`, `annotations.csv`, `images.zip` (332.8 MB). Links are NOT stored in the repo (public GitHub). Ask the user for fresh links when the download job is ready; verify MD5.

## Done
- Planning (steps 1–4, deliverables 04/05/06). Branch `journal-extension` created and pushed 2026-10-09.
- Module a (model heads + loss functions for every trained model variant): `b2j/{losses,heads,variants}.py`, spec `specs/a_heads_losses.md`, tests `tests/test_module_a.py`. B²-M reproduces MICCAI `total_bb_loss` exactly. 2026-10-14: renamed arms→variants; B² NLL/KL computed in float64; decisions in the spec (keep B²-M; no LayerNorm because DINOv3's CLS output is already layer-normalised; soft BCE not n-weighted).
- Module b (label loading): `b2j/data.py`, spec `specs/b_loaders.md`, tests `tests/test_module_b.py`. Endoscapes exact k from per-rater columns; splits 6960/2331/1799; U2 feasible. 33 tests pass, 1 skipped (CUDA) on the Mac.
- Finding: strong systematic rater effects on Endoscapes (C1 positive rate 0.033 / 0.160 / 0.279 by rater; κ 0.15–0.65) → limitation paragraph; also a candidate direction for round D.
- 2026-10-14: user corrected "B² learns m exactly like soft BCE". It is only a population (infinite-data) statement; on finite heterogeneous data the means differ with unknown sign (toy sim `analysis/finite_sample_mean_sim.py`: gap ≈ 20% of estimation error at every N 100–5000). H0 is now an open two-sided question (05_plan amendment 2026-10-14 #2). **Never reason from the asymptotic equality.**
- Cluster cleanup by the user (2026-10-14) does not change the plan. MICCAI-era results JSONs and 9 weight files remain in `BetaBinomialLoss/results` and `weights` (module e may regenerate MICCAI-era validation predictions from them).

## Cluster (verified 2026-10-14; layout changed since 2026-10-09)
- Host comic2; SGE (`qsub`), no Slurm. Queues incl. gpu.q, fastgpu.q, test.gpu.q, cpu.q; gpu.q h_rt up to 1000 h.
- Project `/cluster/project7/FranekLapChole/`: now `backup/ BetaBinomialLoss/ datasets/ dinov3/` + download scripts. **`MICCVS/` and `SwinCVS/` no longer exist.**
- Endoscapes: `/cluster/project7/FranekLapChole/datasets/endoscapes/` (train/val/test jpgs, `all_metadata.csv`). Local copy of the csv in `dataset/endoscapes/` (gitignored).
- Python env: `/cluster/project7/FranekLapChole/BetaBinomialLoss/dino_env` (python 3.11, timm 1.0.26); copy in `backup/dino_env`. Old job scripts point at `../MICCVS/dino_env` → use `./dino_env`.
- Job template: `BetaBinomialLoss/ft_experiments.qsub.sh` — `-l gpu=true -l gpu_type=a6000 -pe gpu 1 -l tmem=24G -l h_rt=24:00:00 -R y`; sources `/share/apps/source_files/python/python-3.11.9.source`, `cuda/cuda-11.8.source`, `gcc-9.2.0.source`; `HF_HOME=.../BetaBinomialLoss/.hf_cache`. The venv python only works after sourcing the python file (fine in jobs).
- Storage: quota 750 GB, 24 GB used; file limit 500k, 205k used (watch file counts: CIFAR PNGs +10k, MHIST +3.2k, CVS Challenge frames later).
- Cluster repo `BetaBinomialLoss` is on `journal-extension` since 2026-10-14 (update: `git fetch -q origin && git merge --ff-only origin/journal-extension`). Cluster git is 1.8.3 (old syntax: `git stash save`, no `stash push --`). The old MICCAI configs' one-line DATASET_DIR edits are kept in `stash@{0}` (on main). Untracked there: `dino_env/`, `scripts/bbkl_loss_old.py`. Job scripts in `BetaBinomialLoss/jobs/` (gitignored); `jobs/download_datasets.qsub.sh` copied, not submitted.
- CVS Challenge not downloaded (R4 only).
- SSH: the user opens the ControlMaster interactively (`ssh comic true`: key passphrase + password); persists 4 h. With BatchMode, a dead master gives "Connection closed by UNKNOWN port 65535" → ask the user to re-open it.

## Exact next action
1. Downloads (user green light 2026-10-14). Job script `BetaBinomialLoss/jobs/download_datasets.qsub.sh` on the cluster (gitignored; local draft in `jobs/`), arg `cifar10h|mhist`, logs in `BetaBinomialLoss/jobs/logs/`.
   - CIFAR-10H: job **7519817** submitted 13:14 → check `jobs/logs/dl_datasets.o7519817`, then `ls datasets/cifar10h/images | wc -l` = 10000.
   - MHIST: DONE (jobs 7519820 download + 7519821 md5 OK/unzip). `datasets/mhist/{annotations.csv, MD5SUMs.txt, images/MHIST_*.png}` (3152). Vote column `Number of Annotators who Selected SSA (Out of 7)`; loader verified on real file: train/val/test 1740/435/977, SSA majority 990. Mac has `../datasets/mhist/annotations.csv` only.
2. After download: real-data tests for CIFAR-10H (counts 10000×10, n 47–63) and MHIST (verify the vote column header and patient grouping).
3. Module c DONE on the Mac: `b2j/{embed,train,posthoc}.py`, spec `specs/c_cache_trainer.md`, `tests/test_module_c.py`. Decisions to confirm with the user are in the spec (soft BCE checkpoint B via Global-S; per-vote selection NLL; DUP selection; default epochs 50 frozen / 10 fine-tune). After CIFAR-10H lands: real-data CIFAR-10H test (counts 10000×10, n 47–63), then a cache-building job per dataset (GPU, short) — needs user go-ahead per job. Next module: d (BACC thresholds, S*(m) on either m; c and Global-S already in posthoc.py).
4. Then modules d–g per 05_plan §4.4. Round D (method development on validation) after R1; log in `journal_plan/iterations.md`.

## Open questions (user)
- 2026-10-14 review of module c choices (reviewer agent, 3 rounds; proposals NOT yet implemented, awaiting user):
  - checkpoint B for every one-number model (hard AND soft BCE) via Global-S; plug-in as supplementary row;
  - user reports Endoscapes overfits by epoch 2–3 → evaluate ~4×/epoch, short fixed budgets (frozen ~10–15 ep, fine-tune ~5 ep), no early stop, A/B chosen from the logged evaluations; optional 3-point smoothing (keep the real centre state, window fixed in advance);
  - report A/B epochs + an A↔B swap table; optional full-panel-validation sensitivity check at n = 3;
  - reviewer claim "S matters little at n≈51" is WRONG (plan §3 pt 4: S matters more with more raters) — do not adopt.
- Go-ahead to submit the download job (and fresh MHIST links at that time).
- Optional training-set-size sweep (10/30/100% of training images, soft BCE vs B²-r, frozen, 5 seeds, Endoscapes + CIFAR-10H, 60 cheap runs): add to R1 or not?
