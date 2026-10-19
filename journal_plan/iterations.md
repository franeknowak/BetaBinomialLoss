# Round D — method iterations (validation only)

Rules: 05_plan amendment 2026-10-14 #1. Each entry is written **before** its runs: diagnosis → change → expected effect; the validation outcome is added afterwards. Kept and dropped iterations all go to the supplement. Test predictions stay unopened until R3. Any tuning budget spent on a new B² version is also spent on soft BCE and hard BCE.

User direction (2026-10-19, after the R1 report): the R1 gains are statistically clear but too small to matter to readers; make the improvement *useful*. Endoscapes results may stay flat (reason known: 3 fixed raters with strong rater effects). Other datasets may be added.

## Where R1 left us (validation, frozen encoder)

| | CIFAR-10H (≈50 crowd raters) | MHIST (7 pathologists) | Endoscapes (3 surgeons) |
|---|---|---|---|
| vote NLL, soft BCE + Global-S → B²-r | 0.0120 → 0.0108 per vote (−10%) | 1.528 → 1.486 per image (−2.8%) | 0.684 → 0.679 (−0.7%) |
| triage AP (find images the panel splits on), soft BCE → B²-r / DUP | 0.196 → 0.200 / 0.206 | **0.550 → 0.600** / 0.601 | 0.373 → 0.375 / 0.374 |
| mAP, B²-r − soft BCE | 0.000 | −0.006 | 0.000 |

Reading: B²-r's per-image spread helps where expert disagreement is common and genuine (MHIST); on images that are mostly easy (CIFAR-10H) or with fixed raters and rater effects (Endoscapes) there is little to gain. Discrimination is unchanged, as the decomposition predicts on large training sets (but the size sweep hints at a gain with little data: Endoscapes Δ mAP +0.06 at 10%, CI includes 0).

Training-subset curves (tune2): B²-r validation NLL bottoms out early (CIFAR-10H epoch ≈2.5, Endoscapes ≈2–6) and then rises while training NLL keeps falling → the spread head memorises the training vote patterns. Checkpoint B stops before this, but it also caps how much the spread can learn.

## D0 — diagnostics (no training), planned 2026-10-19

- **Diagnosis:** we do not yet know where B²-r's gain comes from (which images), nor how much headroom a per-image spread has on each dataset.
- **Analysis (validation predictions of R1, pulled to the Mac):**
  1. Gain per image (NLL Global-S − NLL B²-r) by ambiguity level (vote fraction bins) and by m bins.
  2. How much predicted S varies across images; rank correlation of predicted S with the observed disagreement on validation.
  3. Ceiling: on CIFAR-10H (≈50 raters) the per-image vote fraction is close to the truth, so the NLL of a 3-rater panel under the best possible per-image Beta can be approximated; compare B²-r and Global-S against it.
- **Expected:** tells whether to work on the spread head (D1/D2) or on data (more experts per image / other datasets).
- **Outcome:** (pending)

## Candidate iterations (to be fixed after D0)

- **D1 — mean–spread parameterisation.** Head outputs logit m and log S directly (α₁ = mS, α₀ = (1 − m)S) instead of two softplus α's. Standard in Beta regression; decouples the mean from the spread so the spread can be regularised on its own.
- **D2 — keep the spread from memorising.** Shrink log S towards the dataset-level spread (penalty on (log S − log S_global)²), or train the spread part with its own stronger weight decay / dropout. Target: the validation NLL keeps falling past epoch 2–3.
- **D3 — seed ensemble of B²-r** (mixture of 5 Betas) against the soft-BCE ensemble + Global-S. Same compute for both.
- **Data:** add datasets with several raters per image and genuine disagreement.
  - **AMi-Br — APPROVED 2026-10-19.** Breast-cancer mitoses, normal vs atypical, 3 pathologists' individual votes, 3,720 crops (128 px) from 202 slides; votes 0/1/2/3 atypical = 2496/392/420/412 (22% split). Public GitHub (DeepMicroscopy/AMi-Br; no licence file; cite Bertram et al., BVM 2025). Loader `data.amibr_labels` (slide-level 60/20/20, stratified by source MIDOG21/TUPAC16, seed 0; the authors' split is unseeded). Download on the cluster only: `jobs/download_datasets.qsub.sh amibr`.
  - Non-medical shortlist (user asked 2026-10-19). **2026-10-19: user approved Galaxy Zoo DECaLS and FER+.** Not needed for D0–D2 (they run on R1 validation predictions); download in the background, use once D0 says whether spread carries information. Constraints: GZ DECaLS images = 4 zips, 104 GB, 314k PNGs → would break the 500k file limit (205k used) if unzipped → embed straight from the zips / subset; FER+ images (fer2013.csv) are on Kaggle behind a login → user downloads it (we never handle credentials).
    - **Galaxy Zoo DECaLS** (Walmsley et al., MNRAS 2022; Zenodo 4573248): 314k galaxies, volunteer vote counts per question (e.g. smooth vs featured, bar, edge-on), n ≈ 5 to 40+ → variable n is native to B². Note: the authors already train with a Dirichlet-Multinomial loss — closest prior work, must be cited and positioned against.
    - **FER+** (Microsoft FERPlus): FER2013 faces (48 px, ~35k) with 10 crowd votes per image over 8 emotions; images need the Kaggle FER2013 download. Highly ambiguous classes.
    - **So2Sat LCZ42 validation votes**: 10 expert votes per polygon for 10 European cities (subset only); EO uncertainty benchmarks at gitlab.lrz.de/ai4eo/WG_Uncertainty.
    - Text (other encoder needed): ChaosNLI (100 votes per NLI item), LeWiDi shared-task sets (5+ annotators).
