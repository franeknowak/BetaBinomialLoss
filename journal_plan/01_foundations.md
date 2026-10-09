# Step 1 — Foundations report

*Journal extension of "B² Loss for Evidential Classification of CVS" (MICCAI 2026).
Inputs read: paper PDF, reviews + rebuttal + meta-reviews, derivation notes (project doc), repo `main@2f59ef9` (all of `scripts/`, `main.py`, configs incl. git history, `results/results_26_05_26`).
Every number below is reproducible from `journal_plan/analysis/` (scripts + outputs). `checkJ.py` is superseded by `checkJ2.py` (the saved result JSONs store rounded labels; `checkJ2` re-joins the true vote counts from the annotation file).*

> **Read with 01b_feedback_and_step2_brief.md**, which supersedes this file where they differ: softplus stays and only the `+1` is in question; no Wilcoxon requirement; per-video evaluation out of scope; MICCAI encoder was majority-vote-fine-tuned SwinV2 (not to be used); the annotator 'bias' is sequential/hierarchical marking.

---

## 0. Summary — what matters for the plan

1. **The loss, the P_maj formula and the KL are implemented correctly** (code = closed form to 1e-13). The derivation in the paper and notes is right. The problems are in the *modelling choices around it*, not the algebra.
2. **Soft BCE is the large-S limit — confirmed exactly:** `L_BB(k; m, S) = 3·BCE(k/3, m) − log C(3,k) − c(k,m)/S + O(S⁻²)`, and `E[c] = 0` when the mean is right and θ is homogeneous. The *only* thing B² adds over soft BCE is S, and S is learned only from the second moment of the votes (3 annotator pairs per image).
3. **S played no role in any MICCAI classification number.** P_maj > 0.5 ⟺ E[θ] > 0.5 for every S (0 violations on a dense grid), so thresholded metrics at 0.5 are identical with or without S; AP(P_maj) and AP(E[θ]) agree to 3 decimals in all saved runs. The accuracy gains must be attributed to the mean term (soft-label, weighted) — and partly to the **baselines being weighted differently** (B² used a 5.4:1 positive:negative ratio, BCE 3.2:1 for C1).
4. **The S the trained models report is mostly an artefact**, from three mechanisms the data cannot override:
   - the head's `softplus + 1` floor forces α₁, α₀ ≥ 1, so **S ≥ 1/min(m, 1−m)**. On the test set 23–58% of C2/C3 predictions sit on that floor and median S is within 8–28% of it (ρ(S, m) down to −0.94);
   - the **majority-vote class weights** reshape the vote distribution being fitted: they roughly double the mean of θ and make S finite where the data say infinite (and the reverse);
   - the **KL term** caps S at ≈ 7–10 wherever the data would push it higher (λ = 0.05, hard-coded).
   As a result, **a single constant S ≈ 1–2 per criterion gives a lower held-out vote NLL than the per-image S the model predicts, in all 4 runs and all 3 criteria**, and S separates split votes from unanimous ones almost no better than the mean does (AUROC within mean-quintiles 0.37–0.65).
5. **With 3 annotators, S is barely identifiable.** One image carries Fisher information about log S of 0.014 (n=3, S=10) against 0.35 (n=50). Pinning S down to within ×1.6 takes ≈290 *indistinguishable* images at n=3 against ≈12 at n=50; at S=50 it takes ≈4000 against ≈33. Reported S values above ~10 at n=3 come from regularisation, not from the data.
6. **S is not "evidence" or epistemic uncertainty.** In this likelihood it is the predicted spread of θ among images the model treats as alike (heterogeneity), plus any annotator correlation, *minus* annotator heterogeneity (a biased annotator makes the votes **under**-dispersed, which pushes S up). With a learnable encoder S grows every epoch (E2E val S 5 → 16 in 4 epochs), so the S you see depends on when early stopping fired.
7. **The decision quantity and the reported uncertainty should change** (Step 2 inputs): drop the +1 floor, remove or post-hoc correct the label-dependent weights, report P(split vote) and P(unanimous) instead of 2/S, and evaluate them with vote-level proper scores on held-out data.
8. **Reproducibility gap:** `FT_WEIGHTS` loading is not implemented (the config validator says so), λ is not configurable, and the paper's encoder-adaptation stage cannot be run from `main`. The 26/05 reference runs (frozen SwinV2, ImageNet weights) reach test BACC ≈ 64%, against 72.4% in the paper. The journal paper does not need to reproduce MICCAI, but it must not reuse those numbers.
9. **Synthetic evaluation: not found** (repo, every branch, git history, project). Section 7 lists the likely causes this analysis predicts, each with a one-line check.

---

## 1. Loss re-derivation and three-way check (paper / notes / code)

**Derivation.** k | θ ~ Bin(n, θ) and θ ~ Beta(α₁, α₀) give
p(k | α) = C(n,k) · B(k+α₁, n−k+α₀) / B(α₁, α₀), so
L_BB = −log C(n,k) − log B(k+α₁, n−k+α₀) + log B(α₁, α₀).
P(k ≥ 2 | α) for n=3 is 3E[θ²] − 2E[θ³], with E[θʲ] the rising-factorial ratios (paper Eq. 8). All three agree with each other and with `scipy.stats.betabinom` (`checkA.out`: max |Δ| = 1.4e-13 for the NLL, 2.6e-14 for P_maj). The Dirichlet KL in the code equals the closed-form Beta KL (1.1e-13), with the prior ordered correctly as (α₀ᵖ, α₁ᵖ). Soft labels stored in float32 round back to the right k.

**Unstated assumptions (none of them in the paper):**

| # | Assumption | Where it enters | Consequence |
|---|---|---|---|
| A1 | The 3 votes are conditionally iid given θ (exchangeable, equally skilled, independent) | Binomial likelihood, P_maj formula | See §6. Endoscapes uses fixed raters, not a random draw from a population |
| A2 | Images the network maps to the same output share one Beta | Identifies S at all (§3) | S means "spread among look-alikes", not per-image uncertainty |
| A3 | α ≥ 1 (head = softplus + 1) | Model family | Excludes U-shaped and low-concentration Betas; S ≥ 2 and S ≥ 1/min(m, 1−m). The data need α < 1: the moment-matched marginal S per criterion is 2.3 / 1.0 / 1.5 (C1/C2/C3, train) |
| A4 | Label-dependent weights do not change what θ means | Weighted loss | False: they change the fitted distribution (§5) |
| A5 | The KL prior is reachable | Prior α₁ᵖ = πν = 0.31, 0.22, 0.36 < 1 | Unreachable under A3, so the KL pushes α₁ to its floor of 1 |
| A6 | π = prevalence of θ | Prior mean | The code uses **majority-vote** prevalence (C3: 0.179) where the **mean vote fraction** is what fits a prior on θ (C3: 0.215) |

**Mismatches between paper, notes and code:**

| Item | Paper | Notes | Code |
|---|---|---|---|
| λ (KL weight) | not reported | λ free | **hard-coded 0.05** (`build_loss.py` never passes it) |
| ν | "concentration parameter" | "∈ ~2–10" | 2 |
| Encoder | DINOv3, last 6 blocks adapted on majority votes, then frozen | — | `FT_WEIGHTS` "not currently implemented"; frozen runs use raw timm weights |
| Class weights | not mentioned | w± = N/(2N±), by majority vote | same as notes for B²; **BCE uses `pos_weight` = N/(2N₊) with negative weight 1** → ratio 3.2 against B²'s 5.4 (C1) |
| Uncertainty | "u = 2/S, weak evidential support in representation space" (Discussion) | "if encoder is good, uncertainty is mainly representation quality" | 2/S |
| "Unlike cross-entropy, it does not assume deterministic ground truth" | claim | — | Soft-label CE does not assume it either; the mean term *is* soft CE (§4) |
| Stats | one-sided Wilcoxon, 5 seeds, "significant" | — | With n = 5 the smallest one-sided p is 1/32 = 0.031, so "significant" only means all 5 seed pairs moved the same way, and seed pairing across different models is arbitrary |

---

## 2. What S determines

S does not enter the location of θ. It enters:
- the **spread** of θ: Var θ = m(1−m)/(S+1);
- the **predicted vote distribution**. Its useful functionals are P(split) = 1 − P(k=0) − P(k=3) and P(unanimous). For two random annotators, the disagreement probability is 2m(1−m)·S/(S+1);
- **P_maj only slightly**: at m = 0.3, P_maj moves from 0.259 (S = 3.4) to 0.216 (S → ∞). P_maj > 0.5 ⟺ m > 0.5 holds for all S (`checkA.out`, §C).

So S cannot change any hard decision made at 0.5 by P_maj. It changes rankings only when S varies a lot across images at similar m, and in the saved runs it does not: AP(P_maj) − AP(m) ≤ 0.002 everywhere (`checkJ.out`/`checkJ2.out`).

---

## 3. Where S comes from: data, KL, architecture; n = 3 against many

### 3.1 How S is identified
For a single image, with any number of annotators, the NLL is minimised at S → ∞: a lone vote count is fitted best by a Binomial (§4, the O(1/S) term). **A finite S can only be learned when the model is forced to give the same α to images whose θ differs.** S then estimates the heterogeneity of θ within that equivalence class. The information about it comes from annotator **pairs**: n(n−1)/2 pairs per image, so 0 for n = 1 (unidentifiable), 3 for n = 3, 1225 for n = 50.

Fisher information about log S from one image (`checkE.out` §E):

| m = 0.15 | n=1 | n=2 | n=3 | n=5 | n=10 | n=50 |
|---|---|---|---|---|---|---|
| S = 2 | 0 | .032 | .066 | .122 | .214 | .423 |
| S = 10 | 0 | .005 | .014 | .036 | .096 | .346 |
| S = 50 | 0 | .0003 | .0010 | .0031 | .012 | .120 |

Look-alike images needed for SE(log S) = 0.5: n=3 → 61 / 289 / 4011 (S = 2 / 10 / 50); n=50 → 9 / 12 / 33.
Finite-sample MLE, 30 look-alikes, true S = 10: at n=3 the 10–90% range is 2.9–∞ and 27% of fits diverge; at n=50 it is 7.2–18.3 (§H).
**Conclusion:** at n=3, S is identifiable only below ~5–10, and only when the model pools hundreds of similar images. Any per-image S the network reports above ~10 is set by the regularisers and the architecture.

### 3.2 Data against KL against weights against the α ≥ 1 floor (population optimum, `checkE.out` §F/G)
Population optimum for a group of look-alike images, using the C1 settings (λ = 0.05, ν = 2, π = 0.156, w = 3.20/0.59):

| truth (m, S_true) | pure NLL | + α≥1 | + weights | + KL | **code (all three)** |
|---|---|---|---|---|---|
| 0.15, ∞ (P_maj 0.061) | 0.150, ∞ | 0.150, ∞ | 0.261, 5.5 | 0.150, 7.6 | **0.253, 3.9 (P_maj 0.204)** |
| 0.33, ∞ (0.255) | 0.330, ∞ | 0.330, ∞ | 0.533, ∞ | 0.322, 8.7 | **0.520, 12.7 (0.528)** |
| 0.50, ∞ (0.500) | 0.5, ∞ | 0.5, ∞ | 0.672, ∞ | 0.485, 9.3 | **0.662, 35.2 (0.729)** |
| 0.15, 20 (0.073) | 0.150, 20 | 0.150, 20 | 0.280, 4.0 | 0.150, 6.7 | **0.272, 3.7 (0.226)** |
| 0.02, 1.5 (0.015) | 0.020, 1.5 | 0.020, **49.6** | 0.069, 14.4 | 0.025, 39.9 | **0.076, 13.2** |
| 0.15, 1.5 (0.127) | 0.150, 1.5 | 0.156, 6.4 | 0.394, 2.5 | 0.157, 6.4 | **0.379, 2.6 (0.355)** |

Readings:
- **The KL alone** brings every homogeneous group to S ≈ 7–10 and pulls m toward π (0.95 → 0.92). Data can only push S *below* that, when S_true ≲ 5. In practice the KL decides S for every image that is "easy".
- **The α ≥ 1 floor** turns true low-S, low-m groups (the typical CVS frame: S_true ≈ 1–2, m ≈ 0.02–0.15) into S ≈ 6–50, which is maximally *over*-confident exactly where the annotators disagree most.
- **The weights** create an apparent S that tracks m and not the data: they make homogeneous groups look over-dispersed at low m (S = 5.5) and under-dispersed at mid m (S = ∞). They also explain the paper's Fig. 3 remark that positives have higher concentration: with the code, unanimous positives settle at S ≈ 35–50 and negatives at ≈ 4–30, a pure weighting effect.

### 3.3 What the trained models actually output (`checkJ2.out`, `checkK.out`; test split unless stated)
- S lies in roughly [2.3, 12]; medians 3.2–9. The val-set S rises at **every** epoch in every run, including no-KL (frozen SwinV2: 2.5 → 5.1 over 7 epochs; E2E: 5 → 16 over 4), so the S reported depends on the early-stopping epoch.
- C2/C3: 20–36% of images have α₁ ≈ 1 (on the floor); median S·min(m, 1−m) = 1.08–1.28. **S is close to 1/m by construction.**
- No-KL against KL: median S 9.1 against 7.7 (C1), almost unchanged for C2/C3 → in these early-stopped runs, **the floor and the weights dominate, not the KL**.
- **A constant S beats the predicted S** on held-out vote NLL in every run and criterion (e.g. frozen SwinV2, val C2: predicted 0.921, constant S = 1.0 gives 0.782; test C1: 1.151 against 1.115 at S = 2.2).
- Detecting split votes (k ∈ {1, 2}) from low S: AUROC 0.45–0.76, but the mean alone gives 0.52–0.80. Within quintiles of m, S manages 0.37–0.65. **S adds essentially no information beyond the mean.**
- Mean predicted θ ≈ 0.29–0.47 against an observed vote fraction of 0.13–0.29: the weights shift the mean upward, as predicted in §5.

### 3.4 Frozen against learnable encoder
- **Frozen encoder + small head:** the head is a smooth function of fixed embeddings, so look-alike groups exist and S has a defined meaning: *the heterogeneity of annotator response among images that are near each other in this embedding*. That mixes genuine perceptual ambiguity with representation inadequacy (images the frozen features cannot tell apart), and the two cannot be separated without a better encoder. This is the setting where S is closest to interpretable.
- **Fine-tuned (full or LoRA):** the representation can isolate training images, and each isolated image has optimum S → ∞. S grows with training, is bounded only by the KL and early stopping, and on test data it becomes a mixture of "distance from training" and "how long we trained". It is not calibrated to anything. The note in the derivation doc ("uncertainty mainly meaningful when trained end-to-end") has it the wrong way round for this loss. The opposite conclusion follows, unless S is fitted on **held-out** data (e.g. a separate dispersion head or post-hoc S calibration on validation), which is a Step-2 candidate.
- LoRA sits in between: lower capacity to memorise, so S stays finite longer. That is still a regularisation effect, not identification.

---

## 4. Soft BCE as the large-S limit — confirmed, with what it implies

Expanding log(mS + j) for large S (verified numerically: residual·S → −c to 3 d.p. by S = 10⁴, `checkA.out` §B):

> L_BB(k; m, S) = **n·BCE(y = k/n, m)** − log C(n,k) − **c(k,m)/S** + O(S⁻²),
> c(k,m) = k(k−1)/(2m) + (n−k)(n−k−1)/(2(1−m)) − n(n−1)/2.

E_k[c] = (n(n−1)/2) · Var_G(θ)/(m(1−m)) for k drawn from a mixture G of θ with mean m. It is zero for a homogeneous group with the right mean and positive under heterogeneity. The O(1/S) term is the *only* place the loss differs from soft BCE, and it is driven by pairwise co-voting.

**Implications for what the method adds:**
1. The mean m is learned by the same objective as soft BCE (×3), so **any accuracy or AP claim against soft BCE has to come from S**, and §2 shows S cannot affect thresholded metrics and barely affects AP. Prediction: **B² and properly tuned soft BCE will tie on BACC/AP/AUROC**. The journal paper should not claim classification gains over soft BCE.
2. What B² can claim is a **calibrated predictive distribution over the panel's votes**: P(split), P(unanimous), P_maj accounting for spread. It can win only on proper scores for votes (vote NLL, P(split) calibration), and only where the data identify S (§3.1). This is a narrower but honest contribution, and it is testable.
3. Soft BCE has its own dispersion-free analogue: a **soft BCE + post-hoc constant S** fitted on validation. That is the natural strong baseline for claim 2, and the current models already lose to it (§3.3).
4. Against hard BCE, the advantage comes from soft labels, not from the Beta-Binomial structure.

(Side note: soft BCE with `pos_weight` w has optimum m* = w·ȳ/(w·ȳ + 1 − ȳ), a per-vote prior shift. B²'s majority-label weights produce a different, k-dependent distortion. The MICCAI soft-BCE baseline used `pos_weight` on soft targets, so the two were distorted differently as well.)

---

## 5. Effect of the majority-vote class weights on the learned θ

Weighting by w(y_maj(k)) makes the expected loss the NLL of the **reweighted vote distribution** q′(k) ∝ q(k)·w(k). Votes k ≥ 2 are up-weighted by N₋/N₊ = 5.4 / 7.9 / 4.6 (C1/C2/C3). Consequences (§3.2 table, §3.3):
- **θ is no longer "probability a random annotator says visible"**: for a true m of 0.15 → 0.25, 0.33 → 0.52, 0.50 → 0.66. P_maj inflates 3.3× at m = 0.15, and m = 0.33 images cross the 0.5 threshold. That is the intended effect for BACC, but it destroys calibration and the paper's own interpretation of θ and P_maj.
- **q′ is not a Beta-Binomial**: the jump at k = 2 manufactures over-dispersion at low m and under-dispersion at mid m, so S is corrupted as a function of m.
- It probably contributes to the **video-level over-prediction** at threshold 0.5 (Table 3): inflated frame P_maj combined with max-aggregation over all frames. The size of this contribution has not been measured (the per-video tables were not recomputed).
- It is a label-dependent prior shift, so it is **invertible**: q(k) ∝ q′(k)/w(k) recovers the unweighted vote distribution at inference. This, or unweighted training plus threshold selection on validation, is the obvious fix (a Step-2 hypothesis). The MICCAI comparison was also **unequally weighted** (B² 5.4:1 against BCE 3.2:1 for C1), which on its own shifts BACC at 0.5.

---

## 6. Independent, equally skilled annotators: where it enters and what failure does

It enters at (i) the Binomial likelihood (votes iid given θ), (ii) the P_maj formula, and (iii) the meaning of θ ("a randomly selected expert"). Endoscapes uses a **fixed panel**, so θ is the *average of three individual rates*, not a population quantity, and "random annotator" sampling is fiction. The reviewer (R2) asked about exactly this, and the rebuttal only restated the assumption.

Simulation (`checkI.out`; three fixed annotators, θ ~ Beta(m·S_true, (1−m)·S_true) across images):

| mechanism | effect on vote dispersion | fitted S (S_true = 5, m = .3) | P(k≥2) fitted vs true |
|---|---|---|---|
| iid (assumed) | — | 5.0 | .250 vs .250 |
| one biased annotator (+1.5 logit) | **under**-dispersed (Poisson-binomial) | **7.2** | .364 vs .359 |
| skill spread (−1, 0, +1 logit) | under-dispersed | **9.4** | .254 vs .254 |
| shared image-level shock (correlated) | over-dispersed | **1.8** | .316 vs .314 |
| (S_true = ∞, biased annotator) | | ∞ | .381 vs .368 |

- **Annotator heterogeneity (bias or skill) inflates S**: it hides real ambiguity and makes the model look more certain. **Correlated annotators deflate S.** S therefore cannot be read as "image ambiguity" without per-annotator labels.
- P(k ≥ 2) stays roughly right *for that panel*, because it is a functional of the vote distribution. But "majority" then means *that panel's* majority. With one biased rater, the majority-vote "ground truth" inherits the bias (for m = 0.3, P(k ≥ 2) moves from 0.25 to 0.36).
- **The suspected biased annotator in Endoscapes** is therefore a problem for every method evaluated against majority vote, not only for B². It is a reason to *model* annotators (rater-specific confusion, Dawid–Skene / Tanno et al. 2019 style), or to evaluate on a dataset with many raters, rather than a reason to drop Endoscapes. Testing it requires rater-identified labels. The reformatted annotation file holds only the mean (`ds`), and the raw dataset is not in the connected folder. **Unverified**: whether Endoscapes' metadata keeps rater identities.

---

## 7. Why the synthetic evaluation went badly — diagnosis without the code (UNVERIFIED)

The code and results were not found. From the analysis above, likely causes ranked by prior probability, each with a one-line check:

| # | Likely cause | Kind | Prediction / quick check |
|---|---|---|---|
| 1 | **Truth was per-image θ with S_true = ∞ given x** (θ a deterministic function of the input). The correct answer is then S → ∞ everywhere, and any metric comparing predicted S with a "true" per-image S, or expecting low S on hard images, is ill-posed | evaluation design | Did the generator have a latent variable not visible in x? If not, S has no target |
| 2 | **n = 3 votes per synthetic image** → S unidentifiable unless thousands of look-alikes exist (§3.1) | design | S-recovery error should fall sharply when n is raised to 10–50 with everything else fixed |
| 3 | **α ≥ 1 head** cannot represent S_true < 1/min(m, 1−m) → systematic over-estimation of S for low-m or low-S truths (§3.2) | method/code | Error concentrated at extreme m and small S_true |
| 4 | **Class weights left on** → mean θ biased upward, S distorted as a function of m | method/code | Plot fitted m against true θ: slope/offset above the diagonal for θ < 0.5 |
| 5 | **KL (λ = 0.05) caps S ≈ 7–10** → S compressed whenever S_true > ~7 | method | Fitted S saturates; try λ = 0 |
| 6 | **Model selection by BACC + early stopping** → S depends on the stopping epoch (it rises every epoch) | evaluation/code | Do S curves still change at the selected epoch? |
| 7 | **Comparing on BACC/AP against soft BCE** → a tie is the *correct* outcome (§4); calling it "poor results" would be a design problem, not a method failure | design | Expected; switch the primary metric to vote NLL / P(split) calibration |

A correctly specified synthetic check is cheap (a 1-D or 2-D input, θ = f(x) + latent nuisance z hidden from the model, n ∈ {3, 10, 50}, true S_true(x) known). It belongs in Step 2.

---

## 8. Code and reproducibility issues (to fix before any journal run)

1. `λ` not configurable (`build_loss.py` → default 0.05).
2. `FT_WEIGHTS` ignored → the paper's two-stage encoder cannot be run from `main`.
3. Head floor `softplus + 1` (A3) — remove, or make it a configurable `α_min`.
4. BCE `pos_weight` = N/(2N₊) while B² weights are N/(2N±) → unequal baselines. The soft-BCE baseline also applies `pos_weight` to soft targets.
5. Prior π uses majority prevalence instead of mean vote fraction; the prior α₁ᵖ < 1 is unreachable.
6. Saved result JSONs store **rounded** labels, so k is lost for any later analysis (Fig. 3 must have been made from another source; **unverified**).
7. Model selection on val BACC at 0.5 → E2E B² selected epoch 1, with train BACC 57%.
8. Val/test shift: test prevalence is 1.5× val for C1 (0.235 against 0.163) and C3 (0.279 against 0.167). Video-level thresholds tuned on 41 val videos to precision = 1.0 are therefore fragile, and the CIs in Table 3 show it.
9. Minor: the loss and KL run on CPU (`.cpu()` in the forward loss), which is slow but correct; no augmentation; reference results are single-seed.

---

## 9. Previous reviewers' concerns — status

| Concern (who) | Rebuttal / camera-ready | Status for journal |
|---|---|---|
| Framing of "annotator agreement" vs anatomical truth (R1) | Intro clarified | Resolved in wording; §6 shows the framing still needs the fixed-panel caveat |
| Single dataset (R1, MR) | Argued none other available; CVS Challenge "added to codebase" (not on `main`) | **Unresolved**: needs ≥2 more datasets |
| Single baseline (R1, MR) | SwinCVS only | **Unresolved**: needs soft BCE, hard BCE, and a dispersion-aware baseline, equally tuned |
| Mixed per-criterion results; metrics where the baseline wins were discounted (R1) | Argued metrics are misaligned | **Unresolved**; pre-specify metrics (§4 predicts ties on discrimination) |
| No video-level baseline (R1) | Declined | **Unresolved** |
| Per-video variance (R2) | Stratified bootstrap CIs added | Resolved, but CIs at the precision = 1.0 threshold are wide |
| Pred:GT metric (R2) | Replaced by specificity | Resolved |
| Seed variance is not the variance of interest (R2) | "Standard practice" | **Unresolved**: need video-clustered bootstrap over test data *and* seeds |
| Paired t-test inappropriate (R2) | Switched to one-sided Wilcoxon, n = 5 | **Unresolved** (min p = 0.031; arbitrary pairing) |
| Binomial-fit limitations; annotator sampling mechanism (R2, MR) | Restated exchangeability | **Unresolved** — §6 shows it matters and is testable |
| Class imbalance at video level (R2) | Argued indirect effect via calibration | **Unresolved**; §5 shows the weights *hurt* calibration |
| KL not substantiated (R2) | "Principled Bayesian regulariser" | **Unresolved** — §3.2: the KL mostly sets S, and its prior is unreachable |
| Majority-vote encoder adaptation may bias (R4/R3, MR) | Claimed tested, not reported | **Unresolved** and currently non-reproducible (`FT_WEIGHTS`) |
| Code availability (R1, R4) | Public repo | Resolved |
| Clinical interpretation of the uncertainty (MR) | — | **Unresolved** — the central weakness (§3, §4) |
| Fig. 3 font (R2) | — | Cosmetic |

---

## 10. Inputs this gives Step 2 (proposer brief, not yet proposals)

- The contribution has to move from "better classifier" to **"calibrated prediction of what an expert panel would say, with a dispersion that is identified and validated"**, measured by vote-level proper scores. Datasets with many raters (CIFAR-10H, ~50 per image) are where S is identifiable, so they belong in the core, not the margin.
- Method levers already justified above: free α (no floor); unweighted likelihood with weights handled by a prior-shift correction or threshold choice on validation; a KL that is optional and reachable (or replaced by a held-out S calibration); a reported quantity that is directly checkable (P(split), P(unanimous)); and optionally rater-aware modelling when identities exist.
- Baselines must be weight-matched and tuned on the same budget. "Soft BCE + constant S fitted on validation" is the minimal dispersion-aware baseline.
- A correct synthetic check with known S_true and n ∈ {3, 10, 50}, plus a hidden nuisance variable.

## Not verified
- Synthetic-evaluation code/results (not found).
- Whether Endoscapes keeps rater identities; whether one rater is biased.
- Which encoder and weights produced the MICCAI tables (the current code cannot load the adapted weights).
- How Fig. 3 was produced, given that the saved labels are rounded.
- Literature novelty: not yet searched. To check in Step 2: Beta-/Dirichlet-multinomial likelihoods for crowd labels, Peterson et al. 2019 (CIFAR-10H), Uma et al. 2021 (learning from disagreement survey), Tanno et al. 2019 and Raykar et al. 2010 (annotator models), Bengs et al. 2022 (pitfalls of evidential/second-order losses).
