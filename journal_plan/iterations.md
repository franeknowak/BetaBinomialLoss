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

## D2b — shrink the spread towards its batch mean (planned 2026-10-20, before any run; user: "rerun to fix mistakes")
- **Change:** penalty λ (log S_i − mean over the batch of log S)², i.e. λ × the batch variance of log S per head; the overall level of S is still learned from the votes (the penalty's gradient sums to zero). λ → ∞ = B² with one learned spread per head, the B² analogue of soft BCE + Global-S. λ ∈ {0.03, 0.3, 3}, all reported. Variants `b2ms_var{λ}`; the invalid `b2ms_shrink*` runs are moved to `results/journal/d12_invalid_shrink/` (kept, not analysed).
- **Also added (same block):** FER+ (35,481 faces, 8 emotions, 4–10 crowd votes per face, 61% ambiguous) → tuning (18 runs), the R1-style baseline block `d_base` (45 runs), and the d12 block (30 runs). Lr / wd from FER+'s own tuning.
- **Runs:** d12 now = soft BCE, B²-r, B²-ms, B²-ms var × 3 λ on 5 datasets × 5 seeds; already finished runs are skipped → 60 new D2b runs on the 4 old datasets + 30 FER+ runs.
- **Expected:** if the per-image spread is noise, vote NLL improves (or stays) as λ grows and Spearman stays ≈ 0; if it carries information, NLL is best at a small or zero λ, with Spearman > 0. FER+ is where an informative spread is most plausible (crowd panels, variable n, much genuine ambiguity).
- **Outcome (2026-10-20, job chain 7561886–88, `results/journal/d12/`; validation, vote NLL mean of 5 seeds):**

  | | Endoscapes | CIFAR-10H | MHIST | AMi-Br | FER+ |
  |---|---|---|---|---|---|
  | soft BCE + Global-S | 0.6842 | 0.0120 | 1.528 | 0.9796 | 0.6856 |
  | B²-r | **0.6792** | 0.0108 | 1.486 | **0.9793** | **0.6761** |
  | B²-ms (λ = 0) | 0.6873 | **0.0106** | **1.464** | 0.9842 | 0.6871 |
  | B²-ms var λ = 0.03 / 0.3 / 3 | 0.6838 / 0.6836 / 0.6832 | 0.0107 / 0.0112 / 0.0115 | 1.481 / 1.516 / 1.523 | 0.9824 / 0.9798 / 0.9796 | 0.6819 / 0.6825 / 0.6838 |

  Shrinking the spread to its batch mean narrows S as intended (FER+ 10th–90th percentile of log10 S: 1.7 decades at λ = 0 → 0.05 at λ = 3), and moves B²-ms towards soft BCE + Global-S. It helps where B²-ms was worse than B²-r (Endoscapes, AMi-Br, FER+), hurts where B²-ms was best (CIFAR-10H, MHIST), and never beats the better of B²-r and B²-ms. **Dropped.** Every λ keeps discrimination (mAP) within noise of B²-ms.
- **FER+ baseline (block d_base, validation, 5 seeds):** vote NLL soft BCE + Global-S 0.6856 → B²-r 0.6761 (−1.4%), B²-r + KL 0.6734, B²-M 0.752, hard BCE 0.914. First dataset where B²-r also improves discrimination: mAP 0.758 → 0.768 (sd 0.005 each), accuracy 0.919 → 0.931. θ-MSE and split AUROC unchanged (0.0123 → 0.0122; 0.855 → 0.855).

## D0b — the D0 spread test had the wrong sign (found 2026-10-20 while reading D2b)
- **Mistake:** at a fixed mean m, a smaller S (larger ρ = 1/(S+1)) moves probability to the unanimous panels (P(3-rater split) = 3m(1−m)·S/(S+1)). An informative per-image S therefore gives a **negative** Spearman(ρ, split) within m deciles. D0 read ≈ 0 / negative as "uninformative"; the observed values are mostly negative (CIFAR-10H B²-ms −0.01 to −0.12, MHIST −0.14 / −0.20, Endoscapes −0.03 to −0.10, FER+ −0.00 to −0.09), i.e. weakly in the informative direction. The test is also confounded by m varying within a decile. D0 conclusion (a) is withdrawn until D0b reports.
- **Change (analysis only):** spread ablation per B² model: its own m with its per-image S replaced by one S per head fitted on the same validation votes (as the models' own spread step is). Vote NLL and split AUROC (score 1 − P(unanimous)) with and without the per-image S, on all d12 variants and 5 datasets.
- **Expected:** if ΔNLL and ΔAUROC are ≈ 0, the per-image S adds nothing beyond m and the D0 reading stands; if positive, the spread is informative and the direction for D4 is to strengthen it (not shrink it).
- **Outcome (2026-10-20, job 7569075, `results/journal/d12/diag_d0_val.md`; validation, mean ± sd over 5 seeds; ΔNLL per (image, head), positive = per-image S helps):**

  | | votes per image | B²-r ΔNLL (relative) | B²-r ΔAUROC | B²-ms ΔNLL (relative) | B²-ms ΔAUROC |
  |---|---|---|---|---|---|
  | CIFAR-10H | ≈ 50 | +0.024 ± 0.001 (4.3%) | +0.002 | +0.059 ± 0.003 (10.8%) | +0.012 |
  | MHIST | 7 | +0.046 ± 0.003 (3.1%) | +0.021 | +0.068 ± 0.001 (4.7%) | +0.044 |
  | FER+ | 4–10 | +0.006 ± 0.001 (0.9%) | −0.000 | +0.003 ± 0.002 | +0.000 |
  | Endoscapes | 3 | +0.001 ± 0.000 | +0.000 | −0.003 ± 0.002 | +0.002 |
  | AMi-Br | 3 | −0.001 ± 0.001 | +0.006 | −0.006 ± 0.001 | −0.018 |

  Shrinkage (λ) removes the information monotonically, as it should. **Reading:** the per-image spread *is* informative beyond m where panels are larger (CIFAR-10H, MHIST; B²-ms more than B²-r), and is noise or slightly harmful with 3 raters per image (Endoscapes, AMi-Br): with 3 votes, the overdispersion of one image is barely identifiable from training data. FER+ (crowd, ≤ 10 votes) sits in between. D0 conclusion (a) is replaced by this; (b) and (c) stand. This matches the R1 pattern (MHIST triage gain; Endoscapes flat) and gives a mechanistic account of when B²'s spread is useful.

## D3 — seed ensembles with spreads (planned 2026-10-20, before any result; analysis only, no training)
- **Diagnosis:** the only ensemble row is the pre-registered soft-BCE ensemble of Binomials (S = ∞), a weak comparator (AMi-Br vote NLL 1.27 vs 0.98 for a single soft BCE + Global-S). D0 says most headroom is in m, and averaging seeds is the standard way to improve m.
- **Change:** new rows (the pre-registered "ensemble" row is unchanged): `ensemble_gs` = 5 soft-BCE seeds, each with its own Global-S, mixed; `ensemble_<v>` = 5 seeds of a B² variant (b2r, b2ms, b2ms_var λ), each with its own c-scaled S, mixed. Same compute on both sides (5 trained heads each).
- **Read-out (validation, d12 block):** vote NLL, P_maj calibration, split AP: ensemble_b2r vs ensemble_gs, and each ensemble vs its single models.
- **Expected:** both ensembles beat their single models; if B²'s advantage is calibration rather than spread, the gap between the two ensembles should shrink relative to the single-model gap.
- **Outcome (2026-10-20, d12 block, validation, vote NLL):**

  | | Endoscapes | CIFAR-10H | MHIST | AMi-Br | FER+ |
  |---|---|---|---|---|---|
  | soft-BCE ensemble (pre-registered, S = ∞) | 0.7877 | 0.0164 | 1.586 | 1.266 | 0.7277 |
  | soft-BCE ensemble + Global-S | 0.6834 | 0.0118 | 1.522 | 0.9793 | 0.6689 |
  | B²-r ensemble | **0.6764** | 0.0106 | 1.481 | **0.9789** | **0.6630** |
  | B²-ms ensemble | 0.6824 | **0.0104** | **1.458** | 0.9832 | 0.6693 |
  | single soft BCE + Global-S → single B²-r | 0.6842 → 0.6792 | 0.0120 → 0.0108 | 1.528 → 1.486 | 0.9796 → 0.9793 | 0.6856 → 0.6761 |

  Ensembling helps both losses, and the B² advantage survives it: the gap between the B²-r and soft BCE + Global-S ensembles (Endoscapes 0.0070, CIFAR-10H 0.0012, MHIST 0.041, AMi-Br 0.0004, FER+ 0.0059) is about the single-model gap (0.0050, 0.0012, 0.042, 0.0003, 0.0095), not smaller. On FER+ the B²-r ensemble also has the best mAP (0.776) and split AUROC (0.861). Expectation "gap shrinks" not met. Ensembles are the best panel predictors on every dataset → **kept** as a reported configuration (both sides, same compute); the pre-registered S = ∞ ensemble is clearly the weakest comparator and is reported as such.


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
