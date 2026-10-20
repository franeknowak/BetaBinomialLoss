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
- **Outcome (2026-10-19, `results/journal/r1/diag_d0_val.md`; validation):**
  1. *Where the gain is.* CIFAR-10H: all of it on images with a small dissenting minority (≤ 15% of ≈50 votes: 13% of image–class pairs, +131% of the total gain); B²-r is slightly *worse* on unanimous pairs (85%, −38%). MHIST (7 votes): the gain is spread evenly and is *smallest* on the genuinely split images (30–50% minority: +3% of the gain; unanimous + near-unanimous: +81%). Endoscapes (3 votes): all of the gain on unanimous images; split images ≈ 0.
  2. *Does S say anything beyond m?* No. Within deciles of m, the rank correlation of B²-r's predicted overdispersion ρ = 1/(S+1) with "the panel split" is ≈ 0 on every head (CIFAR-10H −0.07 to +0.04; Endoscapes −0.08 to −0.01) and of the wrong sign on MHIST (−0.14). S does vary across images (10th–90th percentile ≈ 0.4–0.6 decades), but that variation does not track disagreement.
  - *Reading:* B²-r's R1 advantage comes from better-calibrated vote probabilities (mostly in the tails, near-unanimous images), not from knowing which images split. So the MHIST triage-AP gain (0.55 → 0.60) must come through m, not S; the reading in the R1 table above is wrong on that point. A per-image spread that carries information is what B² does not yet deliver → targets D1/D2.
  3. *Ceiling (CIFAR-10H).* First run invalid: the reference (Jeffreys-smoothed Binomial from the other ≈47 votes) was beaten by every model (0.030 vs 0.034 per vote), so it was not a ceiling. Replaced by the empirical-Bayes posterior predictive (d3001d5; job 7559168). Reference 0.0232 per vote on a fresh 3-vote panel vs 0.0299 for both soft BCE + Global-S and B²-r trained on all votes → the best models are ≈ 29% above the reference, and B²-r closes 1% of that gap (n = all), 7–11% when trained on 3–10 votes per image (where B²-r 0.0304–0.0308 beats soft BCE 0.0310–0.0318). The reference is nearly a Binomial with the image's true vote share, so the headroom is mostly in the **mean** m, not in the spread.
  - *Conclusion.* (a) B²-r's spread is not informative per image; its gain is calibration of near-unanimous images. (b) Its advantage is largest with few votes per training image (CIFAR-10H n = 3–10), not with full panels. (c) Most headroom is in m. → D1/D2 test whether a regularised spread can carry information (if not, the method story moves to (b): few raters per image); a better m is the size-sweep / few-votes question.

## AMi-Br baseline (block d_base, job 7559187, validation; 2026-10-19)
- Frozen DINOv3 is weak on 128 px mitoses: mAP ≈ 0.48–0.50, AUROC ≈ 0.70–0.72. B²-r vs soft BCE: Δ mAP +0.004 [−0.007, +0.016], Δ BACC +0.004 [−0.014, +0.022]; vote NLL 0.979 vs 0.980; split AP (u2_ap) 0.337 vs 0.317, catch-20% 0.323 vs 0.285. Weighted variants and B²-M worse. Chosen checkpoints late in the 15-epoch budget (B ≈ epoch 13–14) → the frozen budget may be short here; fine-tuning likely matters more than the loss on this dataset.

## D1 + D2 — regularised mean–spread head (planned 2026-10-20, before any run)
- **Diagnosis (D0):** S does not track disagreement beyond m (within-m-decile Spearman ≈ 0) and memorises training votes (tune2 curves: B²-r val NLL rises after epoch 2–6).
- **Change:** D1 = `b2ms` (head outputs logit m and log S directly). D2 = `b2ms` + λ (log S − log S_prior)², S_prior = the dataset-level spread of the moment-matched prior. λ ∈ {0.03, 0.3, 3}, **all reported** (a dose–response, nothing selected): λ → ∞ is B² with one dataset-wide spread, so if validation NLL keeps improving with λ the per-image spread is noise.
- **Runs (block `d12`, frozen, 5 seeds, full panels, Endoscapes / CIFAR-10H / MHIST / AMi-Br):** b2ms, b2ms_shrink × 3 λ, plus soft BCE and B²-r rerun in the same block as baselines (identical specs to R1/d_base → also a reproducibility check); lr / wd inherited from the B²-r family (no new tuning, so no budget to match). 24 runs per dataset, 120 in total, ≈ 1–2 GPU-h.
- **Read-out (validation):** vote NLL and split AP vs soft BCE + Global-S and vs B²-r (difference + bootstrap CI); D0 within-m-decile Spearman of ρ = 1/(S+1) for every B² variant; NLL against λ.
- **Expected:** if regularisation lets S carry information, Spearman > 0 and NLL / split AP improve over B²-r on MHIST and AMi-Br (the datasets with genuine disagreement). If not, NLL rises monotonically towards B² with a global spread and Spearman stays ≈ 0 → drop the per-image spread claim and centre the paper on the few-raters regime.
- **Outcome (2026-10-20; jobs 7561609 + 7561610, 120/120 runs ok; `results/journal/d12/`; validation):**
  - *Reproducibility:* soft BCE and B²-r in d12 reproduce R1 / d_base to the printed digit.
  - *D1, B²-ms (no shrinkage), vote NLL vs B²-r:* MHIST 1.464 vs 1.486 (better; soft BCE + Global-S 1.528); CIFAR-10H 0.0106 vs 0.0108 per vote (better, but P_maj ECE 0.0031 vs 0.0018 worse, and 0.0299 = B²-r on the fresh 3-vote ceiling panel); Endoscapes 0.687 vs 0.679 (worse, also worse than soft BCE 0.684); AMi-Br 0.984 vs 0.979 (worse). Discrimination unchanged. MHIST gain again on unanimous / near-unanimous images; split images (30–50%) slightly worse than soft BCE.
  - *Spread beyond the mean:* within-m-decile Spearman of ρ with "panel split" still ≈ 0 or negative for every B² variant (CIFAR-10H −0.11 to −0.01; MHIST −0.22 to −0.14; Endoscapes −0.07 to −0.01; AMi-Br −0.00 to +0.08). B²-ms spreads S much wider (log10 S p10–p90 up to 0.4–2.3) without tracking disagreement.
  - *D2 INVALID (design error, not a finding):* the shrinkage target S_prior is the moment-matched prior's S = spread of the vote share **across images** (marginal, ignores x; CIFAR-10H ≈ 0.5), not a within-image spread (≈ 10–100). Shrinking towards it flattens the likelihood in m: CIFAR-10H theta_mse 0.0025 → 0.0057 / 0.0153 / 0.0175 and mAP 0.997 → 0.990 as λ grows; vote NLL worsens monotonically on CIFAR-10H and MHIST. So λ → ∞ is not "B² with one dataset-wide spread", and the dose–response does not answer the question. Fix proposed to the user (not run): shrink log S towards one learnable log S per head (λ → ∞ = B² with a single learned spread, the B² analogue of Global-S); rerun only the 3 shrink variants (60 runs).
  - *Reading so far:* D1 does not make the spread informative and is not consistently better than B²-r (2 of 4 datasets better, 2 worse).


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
