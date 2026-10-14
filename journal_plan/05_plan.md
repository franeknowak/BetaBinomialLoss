# 05 — Final plan: journal extension of the B² (Beta-Binomial) CVS paper

*Final proposer output, 2026-10-08, after two critique rounds (`03_critique_*_r0.md`, `03_critique_*_r1.md`). Supersedes `02_proposals_r0.md` and `02_proposals_r1.md`. Nothing has been run. **UNVERIFIED** marks facts not confirmed on the web. Response tables are in the appendix.*

> **Amendments, 2026-10-09 (user decisions; these override anything below or in the appendix):**
> 1. The `+1` floor is removed (not ablated). Softplus + ε stays.
> 2. DINOv3 ViT-B/16 is the only encoder.
> 3. Every configuration runs **5 seeds**, frozen or fine-tuned.
> 4. The operating point is **recall 0.90**.
> 5. Four datasets: Endoscapes2023, CIFAR-10H, MHIST, and the CVS Challenge last (R4; licence to publish confirmed by the user).
> 6. All code lives on GitHub branch `journal-extension`. Experiments run on the UCL cluster through a local Claude Code session (see `HANDOFF.md`).
> The pre-amendment version is in `analysis/05_plan_pre_amendment_2026-10-09.md`.

> **Amendments, 2026-10-14 (user feedback; these override anything below):**
> 1. **Method development loop (new round D, between R1 and R2).** The user wants an iterative process of improving the method from results, and still aims to beat the baselines' scores, not only analyse. Rules that keep this compatible with honest testing:
>    - Iterations use **validation only**; test predictions stay unopened until R3.
>    - Each iteration is logged in `journal_plan/iterations.md` before it runs: the diagnosis (which validation result motivated it), the change, the expected effect, then the validation outcome. Kept and dropped changes are both logged and reported in the paper (supplement).
>    - Candidate directions, prioritised from R1 diagnostics: how S is parameterised (e.g. predict m and S directly, or log S); regularising S across similar images; the training length and learning rate for the spread; using the Endoscapes per-rater structure; combining the B² term with a soft-BCE term for the mean.
>    - Fairness: any tuning budget spent on the new B² version is also spent on soft BCE and hard BCE (same grid size).
>    - Budget: up to 3 iterations, each a frozen-encoder R1-sized block on validation (≈ 1 user-day + ≈ 5–15 GPU-h each). The best validated version becomes "B²-final"; R1's B²-r stays in the paper as the pre-specified version.
>    - The §5 criteria are committed and dated at the end of R0 (for B²-r) as planned, and **re-committed for B²-final at the end of D, before R2 and before any test prediction is opened**. Both versions are scored in R3.
>    - Improvements are judged on Endoscapes **and** CIFAR-10H validation, to avoid tuning to the 41 Endoscapes validation videos.
> 2. **Soft BCE and B² do NOT learn the same mean in practice** (user correction, 2026-10-14; supersedes §1 "The mean m is learned exactly like soft BCE" and "Why it is not an accuracy paper", and the corresponding wording in 01_foundations). The decomposition only says both losses have the same **population** target, E[θ | x]. That is an infinite-data statement: with finite, heterogeneous training sets (medical imaging) there is no guarantee the two ever coincide, and nothing tells us how many images "enough" would be. B² also weights each image's contribution to m through S (split votes count less when S is small), so the two estimators differ on any real dataset, with a sign we cannot predict. Do not reason from the asymptotic equality when designing or interpreting experiments.
>    - Toy evidence (`analysis/finite_sample_mean_sim.py`; logistic head, 3 raters, extra per-image heterogeneity, 200 repeats per N): averaged over repeats the two are equally accurate, but on any single training set their m differ by about 20% of the estimation error, at every N from 100 to 5,000. The toy has one global S and a well-specified model; real data (per-sample S, MLP head, misspecification) can make the gap larger and systematic.
>    - What B² adds explicitly is S, which separates rater-sampling noise (finite n) from genuine disagreement (3-rater example: soft BCE's Binomial implies P(split) 0.632 against an observed 0.415; B² gets 0.415).
>    - **Consequence for H0:** H0 is no longer predicted to be "equivalent". It is an open, two-sided empirical question with three readings, all reportable: B²-r better on discrimination (a finite-sample benefit; supports the aim to beat baselines), equivalent within ±δ, or worse. Same test, same margins; only the prediction and the framing change.
>    - **Optional, user to decide:** a training-set-size sweep (10%, 30%, 100% of training images; soft BCE vs B²-r; frozen encoder; 5 seeds; Endoscapes + CIFAR-10H = 60 cheap runs) to measure the finite-sample gap on real data directly.
> 3. **Storage policy.** Project quota is 750 GB (24 GB used on 2026-10-14) and **500k files** (205k used). Keep: test/val prediction files for checkpoints A and B (small), head weights (≈ 1 MB each), embedding caches (≈ 35 MB per dataset). Do **not** keep full fine-tuned encoder weights (≈ 350 MB each, ≈ 45 GB for R2 + R4 at 2 per run): write predictions, cache the part-B features needed by the re-head, then delete. Never write per-epoch checkpoints.
> 4. Decisions: B²-M row kept; no LayerNorm in the head (DINOv3's CLS output is already layer-normalised); soft BCE not weighted by n; the Beta-Binomial loss is computed in float64; code uses "variants", not "arms".
> 5. **Training length and checkpoint selection** (user decisions after a 3-round review of module c; supersedes §2 "Checkpoints" and the hard-BCE row of "Scored quantity per arm" where they differ; details in `specs/c_cache_trainer.md`):
>    - Endoscapes overfits by epoch 2–3 (user's MICCAI experience, frozen and fine-tuned). So: **4 validation evaluations per epoch**, short fixed budgets (**15 epochs frozen, 5 fine-tuned**; module g checks they cover the optimum), **no early stopping** (stopping is separated from selection; the full curve also serves H5/F4).
>    - Checkpoints A and B are chosen on a **centred moving average over 3 evaluations** of the selection criterion, so one lucky evaluation on a small, correlated validation set cannot decide. The kept weights are the real state at the chosen evaluation, never averaged.
>    - Checkpoint B for **every one-number model (hard and soft BCE) uses Global-S** (one S\* per criterion, refitted at each evaluation). The independent-raters Binomial plug-in stays as a supplementary row and an H4 comparator, not a selection rule. Hard BCE's m is pushed to 0/1 by majority-label training; this is reported as its property, not corrected. DUP is selected on its own validation loss for both A and B.
>    - One checkpoint per run, chosen on the mean over heads. Reported: the A and B evaluation of every run, and a supplementary A↔B swap table (discrimination at B, panel metrics at A).
>    - Added to R1: CIFAR-10H n = 3 with **full-panel validation** (sensitivity check separating selection noise from missing rater information). Main design keeps validation votes subsampled like training.

## 1. Summary (one page)

**Paper.** *Predicting the expert panel: when does a Beta-Binomial head add to soft labels? An analysis with application to the Critical View of Safety.* The paper is a substantial departure from MICCAI: one common single-image model, no spatiotemporal model, no majority-vote encoder adaptation, no video-level evaluation, and no MICCAI numbers reused.

**Why it is not an accuracy paper.** B² = n·BCE(k/n, m) + const − c(k,m)/S + O(S⁻²). The mean m is learned exactly like soft BCE. The only thing B² adds is the spread S, and S matters only where (i) the data identify it (annotator *pairs*; classical overdispersion) and (ii) a decision depends on it. At threshold 0.5 it never does. At the low thresholds a BACC- or safety-tuned classifier uses (≈ 0.15–0.3), a per-sample S moves the decision boundary in m by up to ≈ 0.09, and the effect grows with panel size.

**Contributions (3).**
1. **Analysis:** the soft-BCE decomposition; how and where S enters decisions; and identifiability of a per-sample spread from n raters (classical overdispersion theory applied to deep heads), quantified by vote subsampling.
2. **A corrected recipe (B²-r) and a traps ablation:** majority-vote class weights and a per-sample KL each set S instead of the data. The `+1` floor is removed outright, not ablated: a Beta needs only α > 0, so the `+1` has no justification in a likelihood model.
3. **A CVS evaluation with matched baselines:** S-aware confirmation, referral and annotation triage against hard and soft BCE, a global spread, a mean-dependent spread S\*(m), a direct disagreement predictor (DUP) and a soft-BCE ensemble.

**Datasets.** Endoscapes2023 (3 raters, CVS) and CIFAR-10H (≈ 51 raters) are the core: every criterion can be decided on them. **MHIST** (7 pathologists) is the third dataset. **The CVS Challenge** (licence to publish confirmed by the user) is the fourth, run **last** (R4) as a confirmatory second CVS dataset.

**Hypotheses (5).**
- **H0:** the mean ties soft BCE on discrimination.
- **HT:** each trap biases S; removing it costs no BACC.
- **H4 (core):** B²-r's per-sample S beats the best spread that depends only on its own m, but only with many raters.
- **H5:** fine-tuning turns S into a training-length artefact; a frozen encoder or a held-out head keeps its meaning.
- **H6:** S-aware decisions lower false confirmations exactly where H4 holds.

**Rounds.**

| Round | What it does | User-days | GPU-h |
|---|---|---|---|
| R0 | Build 6 modules, smoke test, and run a validation-only pilot that sets the equivalence margins; freeze the criteria | 13–14 | ≈ 4 |
| R1 | Frozen encoder: 9 arms × 5 seeds × 3 datasets, plus vote subsampling | 0.5 | ≈ 15 |
| R2 | Full fine-tuning: 3 losses × 5 seeds × 2 datasets, plus a held-out re-head | 1 | ≈ 40 |
| R3 | Open the test predictions once and write down every verdict | 5–6 | 0 |
| R4 | CVS Challenge (last): loader + the R1 frozen arms + R2 fine-tuning losses, 5 seeds; test opened once | 3–4 | ≈ 35 |
| **Total before writing** | | **≈ 23–26** | **≈ 95** |

Writing adds about 15 user-days.

**Expected outcome: B.** The per-sample spread helps on CIFAR-10H and is equivalent to S\*(m) on Endoscapes. If MHIST also shows the many-rater effect (**B+**), target **MedIA**; otherwise **TMLR**. Outcome A, a CVS gain on Endoscapes, would also go to MedIA. Outcome C, no per-sample information, would be a TMLR analysis paper. Every outcome yields a paper.

---

## 2. Method changes

**Architecture (every arm).** DINOv3 ViT-B/16 from public weights → [CLS] embedding → MLP head (768 → 256 → output, GELU, dropout 0.1). Input 384² for surgical frames and 224² for CIFAR-10H and MHIST. No augmentation. One model per frame or image. All code goes on branch `journal-extension`, never `main`.

**The B²-r recipe (proposed), against the MICCAI recipe B²-M.**

| Component | B²-M (MICCAI) | B²-r | Why |
|---|---|---|---|
| Positivity | softplus + 1 + 10⁻⁶ | **softplus + ε, ε = 10⁻³** (softplus stays) | a Beta needs only α > 0; the `+1` (EDL's uniform-prior convention) forces S ≥ 1/min(m, 1−m) and is removed, not ablated |
| Fallback | — | ε = 0.1 if the R0 smoke test finds a non-finite loss, or more than 5% of k ≥ 1 training samples with an α₁ logit < −7 | the risk is saturation, not explosion: \|∂L/∂f\| ≲ 1 |
| Class handling | majority-vote weights (and unequal BCE `pos_weight`) | **unweighted**; the threshold is chosen on validation | weights distort m and S (HT-w); BACC is optimal at P(y_maj = 1 \| x) > π |
| KL | λ = 0.05 toward an unreachable prior | **none** | it caps S and pulls m toward π (HT-KL) |
| Spread level | — | **post-hoc c** per criterion: Beta(c·α₁, c·α₀), fitted on validation by vote likelihood. Applied to every B² arm except the legacy B²-M row | a consistent one-parameter fit for the weakest-identified quantity |
| Loss device | CPU | GPU, gradient clipping at 1.0 (all arms) | speed |
| Reported uncertainty | 2/S | **P_maj, P(majority agrees with the decision), P(split), P(unanimous)**; 2/S only descriptively | probabilities on a decision scale |

**Scored quantity per arm (fixed).**

| Arm | Discrimination (rank / threshold) | P_maj, P(split), vote NLL |
|---|---|---|
| hard BCE | its output p | P_maj = p; vote NLL Binomial(n, p), reported for completeness; no P(split) |
| soft BCE | m | Binomial plug-in (S = ∞) |
| Global-S / S\*(m) | m | Beta-Binomial with the fitted spread |
| B² arms | m = α₁/S | Beta-Binomial (× c) |
| DUP | — | P(split) only |
| Soft-BCE ensemble (5 members) | mean m | mixture of the members' pmfs |

**Post-hoc steps (validation only, never resampled in the bootstrap).**
- Thresholds: maximise validation BACC (for ACC, BACC, F1).
- c: for every B² arm.
- S\* (Global-S): one value per criterion.
- **S\*(m)**: 5 equal-mass bins of m, MLE of S per bin, piecewise-linear interpolation in logit m. Fitted on **soft BCE's m** (baseline row) **and on B²-r's own m**, written S\*(m_B²). The second is the per-sample comparator, because it holds the mean fixed.

**Checkpoints (same rules for every arm).**
- **A** = best validation mAP. Used for discrimination: table T1, H0 and HT-w.
- **B** = best validation vote NLL after the post-hoc spread step. Used for every panel endpoint (HT-KL, H4, H5) and for **every within-model comparison (H6, U1, U2): all scores come from checkpoint B**.
- The trainer keeps both running-best checkpoints, fitting c, S\* and S\*(m) on validation at each epoch (cheap in the frozen regime). It writes test predictions only for those two checkpoints and never prints a test metric.

---

## 3. Hypotheses

**Shared maths** (re-checked with scipy):
1. P_maj > 0.5 ⟺ m > 0.5 for every S, so S never matters at t = 0.5.
2. At a fixed S, P_maj is strictly increasing in m. The same holds for a **slowly varying** S(m), but **not** for a steep one: S = 1 for m ≤ 0.15 rising to 30 at m = 0.3 makes P_maj non-monotone. So only a per-sample S, or a steep S\*(m), can reorder frames. R3 checks P_maj's monotonicity under the fitted S\*(m) numerically; if it is non-monotone, S\*(m) is kept as an extra U1 comparator.
3. The boundary in m at n = 3, for S = 0.5 / 1 / 2 / 5 / ∞:

   | t | boundary in m |
   |---|---|
   | 0.15 | 0.156 / 0.165 / 0.182 / 0.207 / 0.244 |
   | 0.20 | 0.206 / 0.216 / 0.232 / 0.255 / 0.287 |
   | 0.30 | 0.306 / 0.313 / 0.326 / 0.342 / 0.363 |

4. At m = 0.3, P_maj covers 0.216–0.286 over S ∈ [1, ∞) for n = 3, against 0.001–0.273 for n = 51.
5. P(split, n = 3) = 3m(1−m)·S/(S+1).

Verdicts are three-way (success / fail / inconclusive; §5.1). Margins δ = 2h come from the R0 pilot. Effect sizes quoted below are expectations, not criteria.

**H0 (anchor): the mean is learned like soft BCE.**
- *Change:* none; B²-r against soft BCE, both scored on m, at checkpoint A.
- *Justification:* the decomposition. At the small S of CVS data, the Beta-Binomial down-weights split votes when fitting m, so a small difference of unknown sign is possible.
- *Prediction:* Δ BACC and Δ mAP are equivalent within ±δ on Endoscapes and on CIFAR-10H.
- *Refuted if:* the CI is disjoint from ±δ. If δ exceeds its cap, H0 is "CI reported, no claim".

**HT: three traps set S instead of the data.** Single-factor ablations off B²-r. The primary rows have c; the secondary rows without c show the level damage.
- **HT-w** (per-sample weights w± = N/(2N±), identical for B²-r-w and hard-BCE-w).
  - *Justification:* the BACC optimum is P(y_maj = 1 | x) > π, which is robust to label shift. Weighting equals thresholding only for a well-specified model; the MLP's hidden layer makes the difference "expected small, measured". For B², weights fit q′(k) ∝ q(k)w(k), which is not a Beta-Binomial.
  - *Prediction:* Δ BACC (weighted − unweighted + threshold) equivalent within ±δ for each loss.
  - *Diagnostics:* ECE, mean m against the validation vote fraction, ρ(S, m).
- **`+1` floor: removed, not tested** (user decision, 2026-10-09). A Beta needs only α > 0. Softplus (kept) guarantees that smoothly. The `+1` is the uniform-prior convention of evidential deep learning and has no role in a likelihood model, so it is dropped rather than ablated. The only safeguard is numerical: ε = 10⁻³, or 0.1 if the R0 smoke test trips the fallback rule. Diagnostic kept: the stuck share (k ≥ 1 with an α₁ logit < −7). The MICCAI recipe (which includes the `+1`) survives only as the single legacy row B²-M.
- **HT-KL** (corrected KL: π = mean vote fraction, ν = moment-matched S, λ = 0.05; with c).
  - *Justification:* a per-sample KL never fades with more data; it caps S and pulls m toward π.
  - *Prediction:* non-inferiority of no-KL, i.e. the upper bound of Δ NLL (B²-r − KL) < +δ.
  - *Refuted if:* the CI excludes 0 in favour of the KL.

**H4 (core): does the per-sample S carry information beyond m?**
- *Arms, all at checkpoint B:* B²-r against **S\*(m_B²)** (the per-sample test); S\*(m) from soft BCE and Global-S (secondary rows); soft BCE with the Binomial plug-in.
- *Design:* CIFAR-10H with training (and validation) votes subsampled to n ∈ {1, 3, 5, 10, all ≈ 51}, test scored against the full panel; MHIST with n ∈ {1, 3, 5, 7}; Endoscapes at n = 3.
  - n = 1 is an anchor: S is unconstrained there and the tie-break c = S\* = ∞ makes the margin zero by construction.
  - The endpoint is test vote NLL **per vote**, on all images and on the **ambiguous subset** (full-panel plurality share < 0.9).
- *Justification:* the information about S scales with the number of rater pairs, n(n−1)/2. B²-r nests S\*(m_B²) in the population sense. Comparing at the same m removes the mean-quality confound.
- *Predictions:*
  - (a) CIFAR-10H, n = all: B²-r beats S\*(m_B²), and S\*(m_B²) beats Global-S.
  - (b) margin(all) − margin(3) > 0.
  - (c) Endoscapes: B²-r is equivalent to S\*(m_B²) within ±δ_NLL.
  - (d) The plug-in loses to every spread arm.
- *Refuted if:* (a) fails (§5.1 superiority rule) → outcome C.

**H5: what S means under a frozen vs a fine-tuned encoder.**
- *Arms:* frozen (R1, 5 seeds) against full fine-tuning (R2, 5 seeds), for hard BCE, soft BCE and B²-r, on Endoscapes and CIFAR-10H.
  - Memorisation check: a B²-r linear-head arm, and the ratio of median S on training images to median S on validation images.
  - Held-out re-head (CIFAR-10H): B²-r fine-tuned on 80% of the training images (part A); a fresh B²-r head trained on the cached fine-tuned features of the other 20% (part B).
  - **Its comparator:** a frozen-encoder B²-r head trained on the **same part B**.
  - Every margin uses S\*(m) fitted on that arm's own m.
- *Justification:* a trainable encoder can isolate training images, whose optimum is S → ∞, so S tracks training length (01_foundations §3.4). A frozen head can also memorise, which the memorisation check tests.
- *Predictions:*
  - (c, decisive) CIFAR-10H: frozen margin − fine-tuned margin > 0. Secondary: re-head margin > fine-tuned margin; re-head against frozen-part-B reported.
  - Descriptive: fitted c is lower under fine-tuning (c is not resampled, so there is no CI); validation S rises over epochs under fine-tuning.
  - (d) train/val S ratio ≤ 1.5 when frozen. If it exceeds 1.5, the H4 primary is read from whichever frozen head (linear or MLP) has the lower validation NLL.
- *Refuted if:* (c) fails.

**H6: panel probabilities at a fixed operating point (tests U1, U2).**
- *All scores from checkpoint B.*
- *Operating-point rule, the same for every dataset:* false-confirmation rate **FCR = 1 − precision at test recall 0.90**, read off each score's test PR curve. This is threshold-free and compares rankings at identical sensitivity. Secondary: FCR at a validation-chosen threshold for sensitivity 0.90, with the realised test sensitivity.
- *Justification:* shared maths points 1–5, and Chow's rule for referral.
- *Predictions (linked to H4):*
  - (a) Endoscapes, where H4c predicts equivalence: Δ FCR (P_maj − m within B²-r) equivalent within ±δ_FCR, and U2 AP of B²-r P(split) equivalent to the best m-only score.
  - (b) CIFAR-10H, full-panel-majority target: Δ FCR < 0.
  - (c) Descriptive: DUP, the ensemble, 2/S.
- *Refuted if:* (b) fails (the spread is identified but changes no decisions). An Endoscapes gain in (a) moves the paper toward outcome A.

---

## 4. Experiments

### 4.1 Baselines (equal encoder, head, tuning budget, checkpoint and threshold rules; unweighted except the HT-w arms)

| Arm | Target | Spread | Role |
|---|---|---|---|
| Hard BCE (+ hard BCE-w) | majority vote | — | standard practice; HT-w |
| Soft BCE | k/n, no `pos_weight` | plug-in | H0 twin |
| Global-S | soft BCE | one S\* per criterion | post-hoc |
| S\*(m) / **S\*(m_B²)** | soft-BCE m / B²-r's m | spread as a function of m | post-hoc; S\*(m_B²) is the per-sample test |
| **DUP** | BCE on the split indicator 1[0 < k < 3] (CVS); on CIFAR-10H and MHIST, the unbiased 3-rater split estimate as a soft target; **only for n_train ≥ 3** | — (direct P(split)) | Raghu-style comparator for U1-c and U2 |
| Soft-BCE ensemble | 5 soft-BCE seeds | mixture | U1/U2 only; bootstrapped over test clusters only |
| B²-r, + linear head, + part-B head | vote counts | per sample × c | proposed; H5 |
| B²-r-w, B²-r + KL | vote counts | per sample × c | HT |
| B²-M | MICCAI recipe | native (no c) | legacy row |

### 4.2 Use-case endpoints (post-hoc, all scores from checkpoint B)

- **U1, S-aware CVS confirmation with referral.** User: an intraoperative or review tool whose costly error is a false confirmation. Each score is paired with the risk it is optimal for:
  - **U1-a (primary):** FCR at test recall 0.90, comparing P_maj and m within B²-r, and soft-BCE m.
  - **U1-b:** AURC with score max(P_maj, 1 − P_maj); risk = disagreement with the majority.
  - **U1-c:** split rate among accepted frames at 80% coverage, with score P(split).
  - Comparators: plug-in, Global-S, S\*(m_B²), DUP (U1-c), ensemble.
- **U2, annotation triage.** User: dataset curators.
  - Endpoints: AP and AUROC of P(split) for the observed split, and the catch rate in the top 20%. On CIFAR-10H and MHIST, both B²-r and DUP are scored against the same 3-rater unbiased split target.
  - Comparators: the m-only score 3m(1−m) (ranks identically to Global-S), S\*(m_B²), DUP.
  - Feasibility: test split frames are counted per criterion in R0; fewer than 30 → descriptive only.
- Calibrated P_maj for audit and teaching is covered by the metrics below and gets one sentence in the discussion.

### 4.3 Metrics and statistics

- **Primary metrics:**
  - ACC, BACC and F1 at the validation-BACC threshold on the discrimination score, plus mAP;
  - vote NLL, per frame-criterion on CVS and MHIST and **per vote** on CIFAR-10H against the full panel;
  - ECE of P_maj;
  - P(split) AUROC;
  - FCR at test recall 0.90;
  - AURC.
- **Secondary metrics:**
  - AUROC;
  - Brier of P_maj and P(split);
  - θ-MSE against the full panel (CIFAR-10H, MHIST);
  - U1-c split rate; U2 AP and catch rate;
  - **P(split) reliability**, as a higher-order calibration check in the sense of Ahdritz et al.: 10 equal-mass bins, observed split frequency per bin (on CIFAR-10H the bin mean of the unbiased estimate), with a jackknife noise floor.
- **Diagnostics:** c, S\*, the stuck share, ρ(S, m), validation S per epoch, and the train/val S ratio.
- **Seeds:** 5 per configuration in every regime, frozen or fine-tuned (user decision).
- **Two-level bootstrap**, 2,000 resamples.
  - Test clusters are resampled and shared across arms: videos (Endoscapes), images (MHIST), images stratified by class (CIFAR-10H).
  - Seeds are resampled independently per arm; the statistic is the per-seed metric averaged over the resampled seeds.
  - Validation-fitted quantities are not resampled. Clustering is only a resampling unit; there is no per-video evaluation.
- **Margins:**
  - The R0 pilot (tuned soft BCE and B²-r, 5 seeds each, validation only) gives the bootstrap CI half-width h for each margin metric, and **δ = 2h**. That gives a 95% chance of declaring equivalence when the true difference is 0, and it matches the 5 seeds of the main runs.
  - Caps: 3 pp (BACC, mAP), 5 pp (FCR), 0.05 (AP), 1% of the soft-BCE plug-in validation NLL (vote NLL). A δ above its cap turns that test into "CI reported, no claim", decided before R1.
- **Primary comparison families (5):** H0, HT, H4, H5c, H6. Everything else is secondary and descriptive. No p-values.
- **Claim rule:** a claim needs the predicted verdict on **both Endoscapes and CIFAR-10H**, unless the hypothesis names one dataset (H4a/b, H5c and H6b on CIFAR-10H; H4c and H6a on Endoscapes). **MHIST is confirmatory**: it is reported and it can upgrade B to B+ (§5.2), but it neither creates nor removes a core claim.

### 4.4 Rounds

**R0: build, smoke test, pilot, freeze (13–14 user-days; ≈ 4 GPU-h).** Each module gets a one-page spec and a unit test, and is committed before the next starts.

| Module | Content | Days |
|---|---|---|
| a | K heads, per-sample (k, n); loss on GPU; ε, λ, weighting, linear head and DUP as config; no `pos_weight` | 1.5 |
| b | Loaders: Endoscapes exact k + split-frame counts (+ per-rater table ≤ ½ d); CIFAR-10H counts, hypergeometric subsampling, ambiguous subset; MHIST (½–1 d, access form on day 1) | 2–2.5 |
| c | Embedding cache (including from a fine-tuned checkpoint, for the re-head) + **one** trainer for embeddings or images; running-best checkpoints A and B; silent test writing | 2.5 |
| d | Post-hoc: BACC thresholds, c, S\*, S\*(m) on either m | 1.25 |
| e | Metrics (FCR at recall, AURC, reliability with jackknife), two-level bootstrap, three-way verdict function; unit-tested on MICCAI-era **validation** predictions | 2.5 |
| f | Launcher, aggregation, all table/figure scripts written against validation; **smoke test covering every frozen arm, 1 fine-tuning epoch per loss and one re-head pass** (it also decides the ε fallback) | 2.5 |
| g | Pilot on validation only: tuning grid per loss family (head lr ∈ {1e-4, 3e-4, 1e-3} × wd ∈ {1e-4, 1e-2}, 1 seed, val mAP; inherited by the trap, linear and DUP arms), then 5 seeds of soft BCE and B²-r → h → δ = 2h; commit and date §5 with the δ values filled in | 0.5 |

**R1: frozen encoder (0.5 user-day; ≈ 15 GPU-h ≈ 4 h on 4 GPUs).** Supports H0, HT, H4, H5d, H6; tables T1–T3; figures F2, F3, F5.

| Block | Datasets | Arms | Seeds | Runs |
|---|---|---|---|---|
| Main | Endoscapes, CIFAR-10H, MHIST | hard BCE, hard BCE-w, soft BCE, B²-r, B²-r-w, B²-r + KL, B²-M, B²-r linear, DUP | 5 | 135 |
| H5 comparator | CIFAR-10H | B²-r frozen, part B only | 5 | 5 |
| Subsampling | CIFAR-10H n ∈ {1, 3, 5, 10}; MHIST n ∈ {1, 3, 5} | soft BCE, B²-r (all n); DUP (n ≥ 3) | 5 | 55 + 40 |
| Post-hoc | all | Global-S, S\*(m), S\*(m_B²), c, ensemble | – | – |

**R2: full fine-tuning (1 user-day; ≈ 40 GPU-h).** Supports H5 and secondary rows of T1/T2; figure F4. It can launch right after R1.

| Block | Datasets | Arms | Seeds | Runs |
|---|---|---|---|---|
| Tuning (5 epochs) | Endoscapes, CIFAR-10H | hard BCE, soft BCE, B²-r × 2 encoder lr | 1 | 12 |
| Main | Endoscapes, CIFAR-10H | hard BCE, soft BCE, B²-r | 5 | 30 |
| Re-head | CIFAR-10H | B²-r fine-tuned on part A + fresh head on part-B features | 5 | 5 + 5 |

**R3: lock and analyse (5–6 user-days; no training).** Open the test predictions once and write down the verdicts.
- **Tables.** T1 discrimination; T2 panel prediction and the HT ablation; T3 use cases.
- **Figures.**
  - F1: theory (boundary curves; P_maj against S for n = 3/11/51);
  - F2: H4 margin against n (CIFAR-10H, MHIST);
  - F3: S against m per trap arm;
  - F4: validation S per epoch and c by regime;
  - F5: FCR and risk–coverage;
  - F6: qualitative CVS frames ranked by P(split).
- **Pre-declared overrun rule:** F6 and the without-c rows go to the supplement, with no rescoping.

**R4: CVS Challenge, last (3–4 user-days; ≈ 35 GPU-h).** Confirmatory second CVS dataset, with raters drawn per clip from a pool. Built and launched only after R3, so it never delays the core.
- *Loader:* 3 votes per frame and criterion; split 490/105/105 of the 700 labelled training clips, by clip, stratified by country (the official test labels are withheld).
- *Runs:* the 9 R1 frozen arms × 5 seeds (45 runs), plus fine-tuning of hard BCE, soft BCE and B²-r × 5 seeds (15 runs), with the R2 hyperparameters reused (no new tuning).
- *Endpoints:* the Endoscapes endpoints (S0, ST-w, ST-KL, S4c, S6a), reported as replication. CVS Challenge cannot create or remove a core claim. Its test predictions are opened once, after R3.
- *Not used* as a test of fixed panel against rater pool (confounded by institution, format and prevalence).

---

## 5. Success criteria and outcomes (committed with the δ values at the end of R0)

### 5.1 Verdict definitions
- **Superiority:** success if the 95% CI excludes 0 in the predicted direction; fail if it excludes 0 in the other direction; otherwise inconclusive.
- **Equivalence:** success if the CI ⊂ [−δ, δ]; fail if the CI is disjoint from [−δ, δ]; otherwise inconclusive.
- **Non-inferiority:** success if the upper bound is < δ; fail if the CI excludes 0 against the prediction; otherwise inconclusive.

| ID | Endpoint (test, frozen unless stated) | Type | Datasets |
|---|---|---|---|
| S0 | Δ BACC, Δ mAP on m, B²-r − soft BCE (checkpoint A) | equivalence | Endoscapes + CIFAR-10H |
| ST-w | Δ BACC, weighted − (unweighted + threshold), for B²-r and hard BCE | equivalence | Endoscapes + CIFAR-10H |
| ST-KL | Δ vote NLL, B²-r − KL (with c) | non-inferiority | Endoscapes + CIFAR-10H |
| S4a | Δ vote NLL per vote, B²-r − S\*(m_B²), n = all; all images and the ambiguous subset | superiority (< 0) | CIFAR-10H |
| S4b | margin(all) − margin(3) | superiority (> 0) | CIFAR-10H |
| S4c | Δ vote NLL, B²-r − S\*(m_B²) | equivalence | Endoscapes |
| S5c | frozen margin − fine-tuned margin | superiority (> 0) | CIFAR-10H |
| S6a | Δ FCR at test recall 0.90, P_maj − m within B²-r | equivalence | Endoscapes |
| S6b | same, full-panel-majority target | superiority (< 0) | CIFAR-10H |

MHIST runs the S4a/b and S6b endpoints in parallel, as confirmatory evidence (n_train ∈ {3, 5, 7} for S4b).

### 5.2 Outcome → story → venue (decided after R3, with no new experiments)

| Outcome | Condition | Story | Venue (proposer's view; the orchestrator decides) |
|---|---|---|---|
| **A** | S4a success **and, on Endoscapes**, either S4c favours B²-r or S6a shows a gain (FCR lower, CI excluding 0) | "A per-sample panel spread is learnable from 3 raters under a frozen foundation encoder, and it reduces false CVS confirmations" | MedIA |
| **B+** | S4a success; Endoscapes equivalent or inconclusive; MHIST shows the S4a and S6b effects | "A spread needs more than 3 raters: shown on natural and pathology images. For 3-rater CVS data, use B²-r with a mean-dependent spread" | MedIA |
| **B** (expected) | as B+, but MHIST null or inconclusive | the same story, with natural-image evidence only | TMLR (MedIA borderline) |
| **C** | S4a fail or inconclusive | "B² = soft BCE + a mean-dependent spread: theory, the traps, and how many raters a spread needs" | TMLR |

H0 and HT are reportable under every outcome. An H0 fail is reported as a finding and does not change the letter. TMI and TPAMI are not recommended: the paper is a loss analysis with half of its evidence non-medical, and the general technique is prior art (§7).

---

## 6. Datasets and logistics

| | **Endoscapes2023 (CVS201)**, core | **CIFAR-10H**, core | **MHIST**, default third (confirmatory) |
|---|---|---|---|
| Content | 201 LC videos; 11,090 CVS-labelled frames, one every 5 s | the 10,000 CIFAR-10 test images, 32×32 → 224² | 3,152 H&E colorectal-polyp images, 224×224; HP vs SSA |
| Raters | 3, fixed panel | mean 51 (47–63); 2,571 MTurk workers; 511,400 labels | **7** board-certified GI pathologists |
| (k, n) | k = 3·`ds` per criterion; recomputed from the source annotations, because the saved MICCAI JSONs are rounded | 10 one-vs-rest heads, k_c = votes for c, n = the image's total. "Majority positive" = more than half the panel chose c | k = SSA votes, from the majority label and the agreement level (4/7–7/7) or from the SSA-count column. The page confirms the count is in `annotations.csv`; the exact header is UNVERIFIED. n = 7 |
| Split | official 120 / 41 / 40 videos (6,960 / 2,331 / 1,799 frames) | 6,000 / 2,000 / 2,000, stratified by class, fixed seed | official train/test 2,175 / 977 (HP/SSA 1,545/630 and 617/360, per the science critic's check of the paper); validation = 20% of train by image |
| Subsampling | — | n ∈ {1, 3, 5, 10, all}: hypergeometric draws from the counts, fixed per seed | n ∈ {1, 3, 5, 7}, same method |
| Access | public, CC BY-NC-SA 4.0; installed | public (GitHub `jcpeterson/cifar-10h`, TFDS); licence type UNVERIFIED | research-use agreement + request form; download links expire after 4 h. **Send on R0 day 1.** If access has not arrived by the R1 launch, R1 runs on the two core datasets, which decide every criterion |
| Known issues, as stated in the paper | Step 1 analysed this test split with MICCAI-era models: **disclosed**, and no Endoscapes-only claim is made. Val/test prevalence shift (C1 0.163 → 0.235; C3 0.167 → 0.279). Fixed panel with sequential marking: one limitation paragraph and a per-rater table (≤ ½ day, else one sentence) | mostly near-unanimous, hence the ambiguous subset (plurality share < 0.9). One-vs-rest heads are not jointly coherent, so no multiclass quantities are computed | patient grouping UNVERIFIED: a train/val leak would bias model selection, c and S\*(m). Acceptable for a confirmatory dataset; stated |

**CVS Challenge: fourth dataset, run last (R4).** The user holds a licence to publish. It adds a second CVS dataset in which the 3 raters are drawn per clip from a pool (1,000 clips, 18 frames each). The loader costs about 2 user-days (frame extraction; timestamp alignment UNVERIFIED), so it is built after R3 and never blocks the core. It is confirmatory, like MHIST. It is not used as a test of fixed panel against rater pool, because that comparison is confounded by institution, format and prevalence. The cluster already holds download scripts (`download_cvschallenge*.qsub.sh`).

**Logistics.**
- Branch `journal-extension`; each module is committed with its spec.
- The success criteria are committed and dated at the end of R0.
- Test predictions are opened once, in R3.
- Code and configs are released with the paper.

---

## 7. Novelty and positioning

**New relative to MICCAI** (one paragraph in the introduction):
- a common single-image model, with no spatiotemporal model and no majority-vote adaptation;
- the theory, the corrected recipe and the traps ablation;
- two new datasets, and equally tuned and equally weighted baselines, including S\*(m) and DUP;
- new statistics and use cases;
- no video-level evaluation, and no MICCAI numbers reused.

The MICCAI paper's implicit claim that the loss is new is corrected.

**Why frame-level only** (one paragraph). The object is the panel's response to a frame. Video-level confirmation adds an aggregation rule that is independent of the loss question. The user fixed this scope.

**Claims the paper can make (conditional on §5):**
- discrimination is equivalent to soft BCE on m;
- the effect of each trap;
- how many raters a per-sample spread needs, on real data;
- what S means under frozen and fine-tuned encoders;
- whether S-aware decisions lower false confirmations at a fixed recall.

**Claims it cannot make:**
- that the loss is new, or that B² is a better classifier or state of the art;
- that S is epistemic uncertainty or an OOD signal;
- that S measures image ambiguity independently of the panel;
- anything at video level or about clinical safety;
- that the traps generalise beyond second-order heads trained on vote counts.

**Prior art the paper must cite and position against** (r0 searches plus the critics' verified additions):

| Area | Work | Position |
|---|---|---|
| Vote-count likelihoods with deep nets | Walmsley et al. 2020, Binomial likelihood plus active vote allocation ([1905.07424](https://arxiv.org/abs/1905.07424)); Walmsley et al. 2022 and `zoobot`, Dirichlet-Multinomial ([2102.08414](https://arxiv.org/abs/2102.08414)); Hechinger et al., Bayesian Dirichlet-Multinomial, not deep ([2311.08874](https://arxiv.org/abs/2311.08874)) | B² is the two-category case; the loss is not claimed |
| Classical overdispersion | Williams 1975 (*Biometrics*), Crowder 1978 (*JRSS-B*), Prentice 1986 (*JASA*); ρ = 1/(S+1) | contribution 1(c) applies this to deep heads. References from memory; titles to be checked |
| Soft labels / disagreement | Peterson et al. 2019 ([1908.07086](https://arxiv.org/abs/1908.07086)); Uma et al. 2021 ([JAIR](https://jair.org/index.php/jair/article/view/12752)); Singh et al., soft-label training ([2511.14117](https://arxiv.org/abs/2511.14117)); Raykar et al. 2010; Collins et al. 2022, Baan et al. 2022, Malinin et al. 2019, Tanno et al. 2019 (from the critic's list; details UNVERIFIED) | support H0 and the soft-label premise |
| Rater subsampling | Kohli 2026, annotation saturation on ChaosNLI ([2605.29797](https://arxiv.org/abs/2605.29797)) | first-order saturation; ours is identifiability of a per-sample *second-order* spread against S\*(m_B²) |
| Direct disagreement prediction | Raghu et al. 2019 ([ICML](https://proceedings.mlr.press/v97/raghu19a.html)) | a baseline (DUP) |
| Second-order uncertainty | Bengs et al. 2022 ([2203.06102](https://arxiv.org/abs/2203.06102)); Ahdritz et al. 2024, higher-order calibration ([2412.18808](https://arxiv.org/abs/2412.18808)) | with several votes per sample the Beta-Binomial NLL is a proper score for the counts; no epistemic claim. Ahdritz et al. is used as an **evaluation** (P(split) reliability) |
| Calibration under ambiguity | Tao et al. 2026 ([2603.22879](https://arxiv.org/abs/2603.22879)) | **first-order** post-hoc calibrators; Global-S, S\*(m) and c are the second-order analogue |
| Medical multi-rater | mostly segmentation (e.g. [2202.07550](https://arxiv.org/pdf/2202.07550)); no CVS work predicting the vote distribution was found (not exhaustive) | — |

**What remains novel:**
1. the three traps in second-order heads trained on vote counts, with ablations;
2. a controlled real-data identifiability study of a per-sample spread against the number of raters, at fixed m, on natural and pathology images;
3. the decision analysis: S changes decisions only through per-sample variation at fixed m, mostly at low thresholds, growing with panel size, tested by FCR at fixed recall;
4. the encoder-regime analysis;
5. the CVS application with matched baselines.

This is enough for TMLR, and for MedIA under outcome A or B+.

---

## 8. Decisions (taken 2026-10-09) and remaining open items

**Taken:**
- the `+1` floor is removed (not ablated);
- DINOv3 is the only encoder;
- 5 seeds everywhere;
- recall 0.90;
- datasets Endoscapes + CIFAR-10H (core), MHIST (third) and CVS Challenge (fourth, R4, last);
- scope defaults approved: frozen against full fine-tuning only, no LoRA, no augmentation, one-vs-rest CIFAR-10H, no synthetic check, no ChaosNLI;
- code on GitHub branch `journal-extension`, executed on the UCL cluster through a local Claude Code session (`HANDOFF.md`).

**Still open:**
1. Send the MHIST research-use request on R0 day 1.
2. Keep or drop the single legacy B²-M row (the MICCAI recipe as published). It costs one arm and documents what changed since MICCAI.
3. Venue: after R3, by the §5.2 rule (04_target_journal.md).
4. Non-blocking checks: the MHIST column header and patient grouping; the Endoscapes per-rater column names; the CIFAR-10H licence type; the classical overdispersion reference titles.

## Appendix: responses to the critics

### A.1 Round-1 response (r0 → r1)

Marks: A = accept, R = revise, C = cut, — = not reviewed by that critic. "Action" refers to r1; r2 changes in A.2 take precedence.

| Item | Science | Scope | Action | Reason |
|---|---|---|---|---|
| POS | R | R | adopted | 1(b) restated for thresholds other than 0.5. Traps claim limited to second-order heads trained on vote counts. Identifiability recast as classical overdispersion theory with citations. 3 contributions, short title. Written for outcome B as the expected story. Added "new relative to MICCAI" and "why frame-level only" |
| U1 | R | R | adopted | Rebuilt around the operating threshold t ≈ π, where S moves the decision (§2). Three endpoints, each score matched to its own risk: false-confirmation rate at matched sensitivity (primary), AURC for P(majority agrees), split rate among accepted for P(split). Merged with H6 |
| U2 | R | R | adopted | Vote-allocation simulation **cut** (leak + straw-man comparator; Walmsley 2020 cited instead). DUP baseline added. Kept: P(split) AP/AUROC and top-20% catch rate |
| U3 | C | C | adopted (cut) | Its endpoints moved to §4.4 metrics; one sentence in the discussion |
| H0 | R | R | adopted | Every arm is thresholded and ranked on **m** (primary); P_maj is secondary and belongs to H6. Primary metrics BACC + mAP. Margins sized by a validation-only pilot (R0). Three-way verdict |
| H1 | R | R | adopted (merged into HT) | "Impossible with a frozen encoder" replaced by "expected small, measured" (the head has a hidden layer). ECE and the mean-θ check (against the *validation* vote fraction) are diagnostics. soft-BCE-w and the prior-shift readout cut |
| H2 | R | R | partly | Gradient argument corrected: \|∂L/∂f\| ≲ 1, the risk is saturation. Diagnostic changed to the share of pre-softplus logits < −7. Reported with and without c. **One fallback (ε = 0.1), decided in the R0 smoke test**; the mean–precision head F2 is dropped (scope: new code). Science asked to keep the ladder; time wins, and the terminal fallback needs no code (§3 HT) |
| H3 | R | R | adopted (merged into HT) | c applied to every B² arm, trap effects also reported without c. c ranges and "m pulled to π" are now descriptive |
| H4 | R | A | adopted | S\*(m) baseline added; "B²-r beats S\*(m)" is now the per-sample claim. n = 1 is an anchor with a pre-set tie-break (c = S\* = ∞). n_train = 5 added. Ambiguous-image subset reported. NLL per vote on CIFAR-10H |
| H5 | R | R | partly | LoRA cut (not wired; scope). Frozen vs full fine-tuning on Endoscapes + CIFAR-10H, 3 seeds. Held-out spread arm moved to **CIFAR-10H** and built from existing parts (a fresh head on frozen fine-tuned features of held-out images), not a new head type. Memorisation check: linear-head arm + train–val S gap (zero new code). The cross-fitted S-head is rejected (new code). Augmentation confound removed: no augmentation in either regime |
| H6 | R | R | adopted | Merged with U1. Old (a) is now an equivalence claim with a pilot margin, linked to H4: an S-aware decision gain is predicted exactly where B²-r beats S\*(m). 2/S is reported descriptively. Soft-BCE ensemble only |
| D-ENDO | R | A | adopted | Disclose that Step 1 analysed this test split. Endoscapes conclusions need CIFAR-10H agreement. Count test split-frames per criterion from labels in R0. Per-rater table capped at ½ day |
| D-CVSC | R | R | adopted | Off the critical path: email on R0 day 1, decision at the R1 launch or 2026-11-06, whichever is first. All criteria satisfiable without it. "Test clips replace our split" deleted. Not presented as a test of panel structure |
| D-C10H | R | A | partly | Ambiguous subset and the "> half the panel chose c" definition adopted. Confusable-pair sensitivity task rejected (adds a table, tests no hypothesis) |
| D-MHIST (new) | suggested | — | partly | Adopted as the **replacement third dataset** if CVS Challenge is not cleared by the decision date (orchestrator rule 5). Facts verified (§4.1). My view: MHIST is more informative for H4 than CVS Challenge (§8) |
| D-SYNTH | A | C | cut | User time. CIFAR-10H and MHIST subsampling answer identifiability on real data. Revisit only as a TMLR-revision item |
| D-CHAOSNLI | A | C | cut | No new hypothesis; one sentence on modality-agnosticism |
| BL-HARD | A | A | kept | — |
| BL-SOFT | A | A | kept | Soft BCE without `pos_weight` on soft targets (the repo applies it today) |
| BL-GLOBALS | R | A | adopted | Kept, and S\*(m) added next to it (post-hoc, minutes) |
| BL-ENS | A | R | adopted (scope) | Soft-BCE 5-member ensemble only; its predictive is the mixture of member pmfs; bootstrapped over test clusters only (stated) |
| BL-B2M | A | R | adopted | One legacy row, native output; prior-shift readout cut |
| BL-DUP (new) | suggested | suggested | adopted | Frozen head, BCE on the split indicator; minutes per run, reuses the soft-BCE code |
| MET | R | R | adopted | Scored quantity fixed per arm (§4.4 table). Primary list cut to the minimum + 4 panel metrics. NLL per vote. P(split) reliability added as the higher-order calibration check (Ahdritz et al.). "Exact" renamed "unbiased estimate". acc@90 and P(split-among-3) MSE cut |
| STAT | R | R | adopted | 10 seeds in the frozen regime (minutes each), 3 under fine-tuning. Three-way verdicts. ≤ 5 primary comparison families. Claim rule written for Endoscapes + CIFAR-10H; the third dataset is confirmatory |
| SEL | R | A | adopted | Per-epoch val and test predictions written silently. Two checkpoints per run: best val mAP (discrimination) and best val vote NLL after post-hoc spread (panel endpoints), same rule for every arm. Test silence enforced in the trainer |
| R0 | R | R | adopted | Re-estimated at 11–12 user-days, 6 modules each with a one-page spec and a unit test. Validation-only pilot sizes the margins. Licence rule fixed |
| R1 | R | R | adopted | Arms per §4.6 (10 arms × 10 seeds), plus subsampling n ∈ {1, 3, 5, 10, all}. ≈ 25 GPU-h |
| R2 | R | R | adopted | Full fine-tuning only, Endoscapes + CIFAR-10H, 3 seeds; held-out arm on CIFAR-10H; no augmentation. ≈ 25 GPU-h |
| R3 | A | R | adopted | 4–5 user-days; figure and table scripts written in R0 against validation |
| SUCC | R | R | adopted | Rebuilt: pilot margins, three-way verdict, every row decidable on Endoscapes + CIFAR-10H |
| OUTC | R | A | adopted | Outcome A now also needs a pre-specified U1/U2 gain on CVS. "Inconclusive on CVS" maps to the B framing |
| JOUR | R | A | partly | Decision rule in §6. MedIA if outcome A, or B with a positive many-rater medical result (MHIST); TMLR otherwise. The orchestrator makes the final call |
| NOV | R | A | adopted | Added classical overdispersion, Walmsley 2020, Kohli 2026, Singh et al., Hechinger et al.; corrected the Tao et al. description (first-order calibrators); Ahdritz et al. now used as an evaluation |

### A.2 Round-2 response (r1 → this plan)

| Item | Science r1 | Scope r1 | Action | Reason |
|---|---|---|---|---|
| STAT / SUCC (margins) | R | A | adopted | δ = 2h: P(equivalence success \| true difference 0) = 0.95, against 0.67 at 1.5h; caps kept |
| H4 / SUCC (per-sample test) | R | A | adopted | S\*(m) is also fitted on B²-r's own m; S4a–c use S\*(m_B²), which removes the mean-quality confound. The H4 fail rule is aligned with §5.1 (no special override) |
| H5 | A | R | adopted | Frozen B²-r head on the same 20% part B as the re-head's comparator; every margin uses S\*(m) on that arm's m; the c ordering is descriptive (no CI possible) |
| U1 / H6 | R | A | adopted | FCR at test recall 0.80, one rule for every dataset; "t ≈ 0.1" deleted; validation-threshold FCR is secondary |
| POS (monotonicity) | R | A | adopted | "Slowly varying S(m)"; the steep counterexample is stated; R3 checks the fitted S\*(m) numerically |
| OUTC | R | A | adopted | Only Endoscapes can trigger outcome A; MHIST can upgrade B to B+ |
| SEL | A | R | adopted | Checkpoint B for every score in within-model comparisons; test predictions stored only for the two selected checkpoints |
| R0 | A | R | adopted | 13–14 user-days; the smoke test covers 1 fine-tuning epoch per loss and one re-head pass |
| R1 | A | R | adopted | DUP only at n_train ≥ 3; one config list with MHIST fixed |
| R3 | A | R | adopted | 5–6 user-days; F6 and the without-c rows go to the supplement on overrun (pre-declared) |
| D-CVSC | A | C | partly | MHIST is the default third dataset. The CVS Challenge is an optional add-on only with written permission before R1, never on the critical path. Not fully cut, because the user proposed it: flagged as the user's decision (§8) |
| U2 (minor) | A | A | adopted | On CIFAR-10H and MHIST, B²-r and DUP are scored against the same 3-rater unbiased split target |
| BL-B2M (minor) | A | A | adopted | Native output, no c |
| D-MHIST facts (minor) | A | A | adopted | Split and class counts added; a possible patient leak would bias c and S\*(m) as well, which is stated |

**Totals:** user-days ≈ 20–22 before writing (R0 13–14, R1 0.5, R2 1, R3 5–6); GPU ≈ 56 h.
