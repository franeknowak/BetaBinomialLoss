# Journal plan — STATE

## Current step
Round 0. Module d (post-hoc quantities) DONE on the Mac 2026-10-14. Feature-cache GPU job 7519881 running (user go-ahead). Next: check cache, then module e (metrics, bootstrap, verdicts).

(Previous:) Round 0. Module a DONE. Module b DONE for Endoscapes; CIFAR-10H and MHIST loaders written and unit-tested on synthetic data, real data not yet downloaded. Next: dataset download job (needs user go-ahead), then module c. (Updated 2026-10-14; handing over to Claude CLI.)

## Read first
HANDOFF.md → this file → `05_plan.md` (the 2026-10-09 and **2026-10-14 amendments** at the top override the body; 2026-10-14 adds a validation-only method-development round D, a storage policy, a corrected statement of the soft-BCE decomposition, and (#5) training length + checkpoint selection).

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
   - CIFAR-10H: DONE (job 7519817, md5 OK). `datasets/cifar10h/{cifar10h-counts.npy, test_labels.npy, images/00000..09999.png}`; n 47–63; human plurality = CIFAR label 0.9921; split 6000/2000/2000 (600/200/200 per class); 1211 ambiguous images (plurality < 0.9). Mac has the two .npy files only. Real-data test in `tests/test_module_b.py`.
   - MHIST: DONE (jobs 7519820 download + 7519821 md5 OK/unzip). `datasets/mhist/{annotations.csv, MD5SUMs.txt, images/MHIST_*.png}` (3152). Vote column `Number of Annotators who Selected SSA (Out of 7)`; loader verified on real file: train/val/test 1740/435/977, SSA majority 990. Mac has `../datasets/mhist/annotations.csv` only.
2. After download: real-data tests for CIFAR-10H (counts 10000×10, n 47–63) and MHIST (verify the vote column header and patient grouping).
3. Module c DONE on the Mac: `b2j/{embed,train,posthoc}.py`, spec `specs/c_cache_trainer.md`, `tests/test_module_c.py`. Review decisions APPROVED and implemented 2026-10-14 (spec §Decisions): Global-S for hard+soft BCE at checkpoint B; 4 evals/epoch; budgets 15 frozen / 5 fine-tune; no early stop; 3-evaluation smoothing for selection; A/B epochs + swap table reported; full-panel-validation sensitivity check at CIFAR-10H n = 3 (add to R1 block in module f). All three datasets are on the cluster and verified. Next: a cache-building job per dataset (GPU, short) — needs user go-ahead per job. Next module: e.
4. Feature cache: `jobs/build_cache.qsub.sh` (cluster only) → job **7519881** (submitted 13:59, A6000) runs `python -m b2j.embed endoscapes cifar10h mhist` → `BetaBinomialLoss/cache/<dataset>_frozen.npz`; log `jobs/logs/build_cache.o7519881`. Check shapes (11090/10000/3152 × 768), finite, and time per dataset.
5. Module d DONE: `b2j/posthoc.py` (S*(m) with S linear in logit m as the plan says — log-S would hide the non-monotone P_maj; panel probabilities; ensemble; thresholds; pmaj_monotone), spec `specs/d_posthoc.md`, `tests/test_module_d.py` (reproduces plan §3 table and ranges). 67 tests pass.
6. Then modules e–g per 05_plan §4.4. Round D (method development on validation) after R1; log in `journal_plan/iterations.md`.

## Open questions (user)
- Optional training-set-size sweep (10/30/100% of training images, soft BCE vs B²-r, frozen, 5 seeds, Endoscapes + CIFAR-10H, 60 cheap runs): add to R1 or not?
