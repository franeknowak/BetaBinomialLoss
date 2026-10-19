# Module b — loaders, subsampling, ambiguous subset

Code: `b2j/data.py`. Tests: `tests/test_module_b.py` (Endoscapes test runs on real metadata when `dataset/endoscapes/all_metadata.csv` exists).

## LabelTable
One row per image: `k, n` float `[N, K]`, `cluster` (bootstrap unit), `split`, `paths`, optional raw `counts [N, C]` (CIFAR-10H). `take(split)`, `subset(mask)`.

## Endoscapes2023 (CVS201)
- Source: `all_metadata.csv` (cluster: `/cluster/project7/FranekLapChole/datasets/endoscapes/` (moved from SwinCVS/ by 2026-10-14)), keyframes only (`is_ds_keyframe`, 11,090).
- k per criterion = sum of `cvs_annotator_{1,2,3}` binary votes; checked against the averaged C1–C3 columns (max error 0.0). n = 3.
- Split by video id: train 1–120 / val 121–161 / test 162–201 → 6,960 / 2,331 / 1,799 frames (plan values). Cluster = video.
- Majority prevalence train 0.156/0.112/0.179 (= MICCAI PI values), val 0.163/0.125/0.167, test 0.235/0.168/0.278.
- Split frames (0<k<3) per criterion: train 1899/1094/2103, val 544/307/530, test 701/374/613 → U2 feasible (≥ 30) for all three criteria.
- Images `{split}/{vid}_{frame}.jpg`; centre-crop 480 (as MICCAI) → 384², ImageNet normalisation, no augmentation.
- Per-rater table (`endoscapes_rater_table`, train+val): positive rates C1 0.033/0.160/0.279, C2 0.096/0.119/0.151, C3 0.269/0.112/0.237; pairwise κ 0.15–0.65. **Strong systematic rater effects** (fixed panel) → limitation paragraph: part of the "spread" on Endoscapes is rater bias, not image ambiguity.

## CIFAR-10H
- Input `counts [10000, 10]` (`cifar10h-counts.npy`) + CIFAR-10 test labels for stratification. K = 10 one-vs-rest heads, k_c = votes for c, n = image total (47–63). Cluster = image.
- Split 6,000 / 2,000 / 2,000 stratified by true class, `default_rng(seed=0)`.
- Images: 32×32 PNG `{id:05d}.png` → 224² bicubic. **Not yet downloaded** (needs user OK).

## MHIST
- `annotations.csv`: vote column matched by name containing "ssa" and "number"/"vote" (header UNVERIFIED; fails loudly otherwise), cross-checked against `Majority Vote Label` (k > 3.5 ⟺ SSA). n = 7. Official `Partition` train/test; val = 20% of train by image (seed 0). Cluster = image. 224² native. **Awaiting access.**

## AMi-Br (round D, 2026-10-19)
- `AMI-BR.csv` (GitHub DeepMicroscopy/AMi-Br, commit 20a217b): k = atypical votes of 3 pathologists, n = 3; checked against `majority_atypical`. 3,720 mitoses from 202 slides; votes 0/1/2/3 = 2496/392/420/412. Split 60/20/20 by slide within source (MIDOG21/TUPAC16), seed 0. Cluster = slide. 128 px → 224 bicubic. Real-data check passed on the cluster (job 7559142).

## FER+ (round D, 2026-10-19)
- Labels `fer2013new.csv` (Microsoft FERPlus repo): 10 crowd votes per FER2013 face over 8 emotions + unknown + NF. Kept: unique image name (173 unnamed rows and 4 names occurring twice are dropped) and plurality vote = an emotion (drops 399 unknown/NF faces) → 35,481 faces.
- K = 8 one-vs-rest heads; n = the face's emotion votes (unknown/NF votes are not part of the panel), 4–10, < 10 for 33%; 61% ambiguous (plurality < 0.9). `counts` set → joint subsampling (only n_sub ≤ 4 is possible on the full set).
- Official split: Training / PublicTest / PrivateTest → train / val / test = 28,383 / 3,553 / 3,545. Cluster = image.
- Images: Kaggle copy arnabkumarroy02/ferplus (112 px, class folders, training set augmented); `b2j.prepare_ferplus` unpacks only the originals named in the csv, skipping names that map to different files. Grayscale → RGB → 224 bicubic.

## Subsampling
`subsample(table, n_sub, seed)`: without replacement, fixed per seed. CIFAR-10H draws jointly from the 10 category counts (multivariate hypergeometric) so the one-vs-rest heads stay consistent (Σ_c k_c = n_sub); binary tables draw per head. Applied to train and val only; test is always scored against the full panel (caller's responsibility, module c/f).

## Ambiguous subset
Full-panel plurality share < 0.9: CIFAR-10H max category share; binary tables, any head.

## Tests cover
real Endoscapes split sizes, vote range, MICCAI prevalence, no video leak; stratified split sizes/balance/determinism; one-vs-rest shape; joint subsample sums, ≤ full counts, seed determinism; hypergeometric unbiasedness; ambiguous mask; split-frame counts; MHIST parsing + inconsistency guard; image tensor sizes.
