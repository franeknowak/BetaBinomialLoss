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
- **Data:** add a dataset with several experts per image and genuine disagreement (shortlist in STATE, user decision).
