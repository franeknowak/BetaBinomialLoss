# 02 — Proposals, round 1 (proposer, revised)

*Written 2026-10-08 by the proposer after `03_critique_science_r0.md` and `03_critique_scope_r0.md`. This is a complete, standalone plan; it replaces `02_proposals_r0.md`. Inputs as for r0, plus both critiques. Nothing has been run. Facts not confirmed on the web are marked **UNVERIFIED**. Maths re-checked for this round with scipy: the threshold table in §2 and the monotonicity of P_maj in m at fixed S.*

**Shorthand.** k = positive votes; n = votes on the sample; m = α₁/S (Beta mean); S = α₀ + α₁; t = decision threshold.
- **B²-r**: the revised recipe. softplus + ε (ε = 10⁻³), unweighted Beta-Binomial NLL, no KL, then one post-hoc spread scalar c per criterion fitted on validation.
- **B²-M**: the MICCAI recipe (+1 floor, majority-vote weights, KL λ = 0.05, ν = 2), kept as a legacy row.
- **Global-S**: soft BCE plus one S* per criterion fitted on validation.
- **S\*(m)**: soft BCE plus a spread that depends only on m, fitted on validation (new).
- **DUP**: a direct split predictor, i.e. a head trained by BCE to predict whether the panel splits (Raghu-style; new).

## R. Response to the critics

Marks: A = accept, R = revise, C = cut, — = not reviewed by that critic. "Action" refers to this r1 plan.

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

---

## 1. Positioning

**Working title.** *Predicting the expert panel: when does a Beta-Binomial head add to soft labels? An analysis with application to the Critical View of Safety.*

**The constraint.** B² = n·BCE(k/n, m) − log C(n,k) − c(k,m)/S + O(S⁻²) (01_foundations §4). The mean is learned by the soft-BCE objective, so the paper makes no accuracy claim against soft BCE. The only thing B² can add is the spread S. S matters only where (i) the data identify it and (ii) a decision depends on it.

**Expected story (outcome B, §4.8), with A and C both reachable.** On many-rater data (CIFAR-10H; MHIST if used) a per-sample spread is learnable and changes decisions. On 3-rater CVS data, the corrected recipe matches or beats soft BCE on panel prediction, but a per-sample spread adds little over a spread that depends only on the mean. The paper says *how many raters* that takes, and which implementation choices silently set S instead of the data.

**Contributions (3).**
1. **Analysis.** (a) The soft-BCE decomposition above. (b) How S enters decisions. At t = 0.5 it never does (symmetry). With a constant S, or an S that depends only on m, P_maj is a strictly increasing function of m (checked numerically), so the decisions equal mean-thresholding at a re-tuned threshold. A per-sample S changes decisions only through its variation among samples with similar m, and that effect is large at the low thresholds a BACC-tuned or safety-tuned classifier uses (t ≈ 0.15–0.3, §2), and grows with panel size. (c) Identifiability: S is identified only from within-sample annotator pairs. This is classical Beta-Binomial overdispersion theory (intra-class correlation ρ = 1/(S+1), design effect 1 + (n−1)ρ; Williams 1975, Crowder 1978, Prentice 1986), applied here to deep second-order heads and quantified on real data by vote subsampling.
2. **A corrected recipe and a traps ablation** for second-order heads trained on vote counts: the `+1` floor, label-dependent class weights and an unreachable per-sample KL each set S instead of the data. Each is tested by a single-factor ablation (HT).
3. **CVS evaluation with matched baselines**: S-aware confirmation at the operating threshold, referral, and annotation triage, against hard/soft BCE, Global-S, S\*(m), a direct split predictor and a soft-BCE ensemble, all with identical class handling.

**New relative to the MICCAI paper** (one paragraph in the introduction, as MedIA requires). The architecture changes: one common single-frame model instead of the spatiotemporal one, and no majority-vote encoder adaptation. The theory, the recipe and the traps ablation are new. There are two or three new datasets, all baselines are equally tuned and weighted, the statistics and use cases are new, and video-level evaluation is dropped. MICCAI numbers are not reused. The MICCAI paper's implicit claim that the loss is new is corrected: Dirichlet-/Beta-Multinomial likelihoods on vote counts already exist (§7).

**Why frame-level only** (one paragraph in the paper). The object of study is the panel's response to a single frame. Video-level confirmation adds an aggregation rule (e.g. max over frames) that is independent of the loss question, and it was a separate contribution of the MICCAI paper. The user has fixed frame-level evaluation.

**Claims the paper CAN make (conditional on the results).**
- B²-r and soft BCE rank and classify frames equivalently on m, within a margin sized in advance (H0).
- Each trap moves S (and m, for the weights) in a predicted direction; removing it does not hurt discrimination and improves vote NLL (HT).
- How many raters per training image a per-sample spread needs, on real data (H4).
- What S means under a frozen vs a fine-tuned encoder (H5).
- Whether S-aware decisions at the operating threshold reduce false confirmations on CVS and on many-rater data (H6), with a CI and a three-way verdict.

**Claims the paper CANNOT make.**
- That the Beta-Binomial loss is new, or that B² is a better classifier or state of the art on CVS.
- That S is epistemic uncertainty, "evidence", or an OOD signal (consistent with Bengs et al. 2022).
- That S measures image ambiguity independently of the panel. Endoscapes has a fixed panel with sequential marking; the paper gives one limitation paragraph and does not compare panel structures across datasets, since that comparison is confounded by institution, format and prevalence.
- Anything at video level or about the clinical safety of a deployed system.
- That the traps generalise beyond second-order heads trained on vote counts.

**Test-set disclosure.** Step 1 (01_foundations §3.3) analysed MICCAI-era predictions on the Endoscapes **test** split: floor shares, constant-S NLL and the prevalence shift. Some predictions below were informed by those numbers. The paper discloses this, and no Endoscapes-only result is used as confirmation: every claim also needs CIFAR-10H agreement (§4.5).

---

## 2. Use cases: when (α₀, α₁) changes a decision

**The maths that decides everything here** (re-checked with scipy for this round).
1. At t = 0.5: P_maj > 0.5 ⟺ m > 0.5 for every S. S cannot change a decision.
2. At fixed S, P_maj is strictly increasing in m (checked for S = 0.3, 1, 3, 30). So **Global-S** and **S\*(m)** (if S\*(m) is smooth) give the same *ranking* as m, and the same decisions after the threshold is re-tuned on validation. Only a **per-sample** S that varies among frames of similar m can change a ranking or a decision.
3. At the low thresholds the protocol actually uses, a per-sample S moves the decision boundary substantially, even at n = 3:

| t on P_maj (n = 3) | boundary in m for S = 0.5 / 1 / 2 / 5 / ∞ |
|---|---|
| 0.15 | 0.156 / 0.165 / 0.182 / 0.207 / 0.244 |
| 0.20 | 0.206 / 0.216 / 0.232 / 0.255 / 0.287 |
| 0.30 | 0.306 / 0.313 / 0.326 / 0.342 / 0.363 |

   At t = 0.2, a frame with m = 0.25 is confirmed if the model says S = 2 and not if it says S = ∞. Lower S pulls P_maj toward m.
4. For a larger panel the effect grows: at m = 0.3, P_maj covers 0.216–0.286 over S ∈ [1, ∞) for n = 3, but 0.001–0.273 for n = 51.
5. P(split, n = 3) = 3m(1−m)·S/(S+1). This is also monotone in 3m(1−m) at a constant S, so for *ranking* by split risk, Global-S is identical to the mean-only plug-in.

So both use cases reduce to one question: **does B²-r's per-sample S carry information beyond m?** H4 tests this on vote likelihood (B²-r against S\*(m)); H6 tests it on decisions. The plan predicts that the two agree.

**Deployment condition:** a frozen public foundation encoder (DINOv3 ViT-B/16) with a light MLP head, the realistic setting for small surgical datasets. Full fine-tuning is the contrast (H5).

### U1. S-aware CVS confirmation with referral (primary use case; tested by H6)
- **Who.** An intraoperative CVS assistant or a video-review tool. The user is the surgeon or reviewer. The costly error is a **false confirmation** (saying a criterion is met when the expert majority says it is not).
- **Decision.** Per frame and criterion: *confirm* if P_maj > t; *refer* if the predicted panel would plausibly split; otherwise *not met*. Each score is evaluated against the risk it is optimal for:
  - **U1-a (primary).** Confirmation at matched sensitivity. For every score, the threshold is set on validation to give sensitivity 0.80 against the majority vote (one rule for all). Endpoint: test **false-confirmation rate** FCR = FP / (TP + FP). Comparison: P_maj(B²-r) against m(B²-r), which isolates the S effect within one model, and against soft-BCE m.
  - **U1-b.** Referral by Chow's rule: score max(P_maj, 1 − P_maj); risk = disagreement with the majority; endpoint **AURC**.
  - **U1-c.** Referral of contested frames: score P(split); risk = the split rate among accepted frames; endpoint **split rate among accepted at 80% coverage**.
- **Comparators.** Mean-only plug-in (S = ∞), Global-S, S\*(m), DUP (for U1-c), and the soft-BCE 5-member ensemble. All are post-hoc on saved predictions.
- **Datasets.** Endoscapes and the third dataset (CVS Challenge or MHIST). On CIFAR-10H the target is "more than half the full panel chose class c", which is the large-panel contrast.

### U2. Annotation triage (secondary)
- **Who.** Curators of multi-rater datasets, deciding where to spend expert time.
- **Decision.** Send frames with the highest predicted P(split) for extra raters or a consensus discussion.
- **Endpoints.** AP and AUROC of P(split) for the observed split (k ∉ {0, n}); **catch rate** = the share of all split frames among the top 20% by score.
- **Comparators.** Mean-only 3m(1−m) (which, by point 5, ranks identically to Global-S), S\*(m), and DUP. DUP predicts disagreement directly (Raghu et al. 2019); without it a reviewer would call the Beta an indirect route.
- **Feasibility check (R0, labels only).** Count split frames per criterion in each test split. If any criterion has fewer than 30 split frames, report its AUROC as descriptive only.
- **Cut.** The CIFAR-10H vote-allocation simulation. It leaked the reference votes, and its fair comparator, Neyman allocation by m, needs only the mean. Active vote allocation with a Binomial likelihood is also prior art (Walmsley et al. 2020).

**Folded into the metrics (former U3).** Calibration of P_maj and P(split) supports audit and teaching. Those endpoints are in §4.4, and the discussion gives one sentence on that use.

---

## 3. Method hypotheses (5)

Common to all: one single-image architecture (encoder → [CLS] → MLP 768 → 256 → output, GELU, dropout 0.1). **Every B² arm gets the same post-hoc c** (fitted on validation by vote likelihood). Every verdict is three-way: **success / fail / inconclusive** (§4.5). The margins δ are fixed by the R0 pilot before R1 (§4.5); below they appear as symbols. Expected effect sizes are given as descriptive expectations, not as criteria.

### H0 (anchor). The mean is learned like soft BCE
- **Change.** None. B²-r against soft BCE.
- **Scored quantity.** **m for every arm**: thresholds (validation-BACC rule) and rankings (mAP) both use m. P_maj belongs to H6, not here, so that S cannot leak into this comparison.
- **Justification.** B² = n·BCE(k/n, m) + const − c(k,m)/S + O(S⁻²), so for large S the gradient in m is n × the soft-BCE gradient. At the small S of CVS data the expansion is loose: the Beta-Binomial then down-weights split votes relative to unanimous ones when fitting m. A small difference is possible, with no predicted sign.
- **Prediction.** Δ BACC and Δ mAP (B²-r − soft BCE) are equivalent within ±δ_BACC and ±δ_mAP on Endoscapes and on CIFAR-10H (third dataset confirmatory). Descriptive: both beat hard BCE on mAP on the CVS data.
- **Verdict.** Success: CI ⊂ [−δ, δ]. Fail: CI wholly outside [−δ, δ]. Inconclusive: otherwise. If the pilot gives δ above the cap (3 pp for BACC or mAP), H0 is pre-declared "reported as a CI, no equivalence claim".

### HT. Three implementation traps set S instead of the data (merges r0's H1–H3)
Three single-factor ablations, each adding one trap to B²-r. One primary endpoint per trap; primary rows **with c**, secondary rows without c (the without-c rows show the "level" damage that c partly repairs).

| Trap | Arm | Primary endpoint and success rule | Diagnostics (descriptive) |
|---|---|---|---|
| **HT-w** weights | B²-r-w and hard-BCE-w: per-sample weight w± = N/(2N±) by majority label, identical for both (never `pos_weight`) | Δ BACC (weighted − unweighted + val threshold), per loss: equivalent within ±δ_BACC | ECE of P_maj; mean m on validation against the validation mean vote fraction; Spearman ρ(S, m); vote NLL |
| **HT-floor** `+1` | B²-r + floor (softplus + 1) | Δ vote NLL (B²-r − floor) < 0, CI excluding 0 | share of α₁ within 0.01 of 1 (floor arm); share of "stuck" positives in B²-r (below) |
| **HT-KL** | B²-r + corrected KL: π = mean vote fraction, ν = moment-matched marginal S, λ = 0.05 | Non-inferiority of no-KL: CI upper bound of Δ vote NLL (B²-r − KL) < +δ_NLL. Fail if the CI excludes 0 in favour of the KL | fitted c per arm; \|m − m_noKL\| against m |

**Justifications.**
- **HT-w.** BACC is maximised by thresholding P(y_maj = 1 | x) at the majority prevalence π. This rule also survives pure label shift, so it transfers across the Endoscapes val/test prevalence shift. With w₊/w₋ = (1−π)/π, a well-specified hard-BCE model's 0.5 threshold moves exactly to η > π, so weighting and thresholding coincide in that limit. With a finite MLP head (a 256-unit hidden layer) and any misspecification they differ. The difference is **expected to be small and is measured**. For B² and soft BCE, majority-label weights fit q′(k) ∝ q(k)·w(y_maj(k)), which is not a Beta-Binomial: m is inflated and S becomes a function of m (01_foundations §5). The ECE drop is near-certain by construction, so it is a diagnostic, not a criterion.
- **HT-floor.** The strongest evidence is empirical (Step 1): 20–36% of C2/C3 predictions sat on α₁ = 1, and a constant S beat the predicted S on held-out vote NLL in every run. Theory: α ≥ 1 forces S ≥ 1/min(m, 1−m). The marginal moment-matched S (1.0–2.3) is only supporting evidence, since the conditional S given x is larger. **Stability (corrected).** With α₁ = softplus(f) + ε, ∂L/∂α₁ ≈ −1/α₁ for k ≥ 1, but softplus′(f) = σ(f) ≈ softplus(f) as f → −∞, so |∂L/∂f| ≲ 1. Examples: f = −5 gives −0.86, f = −10 gives −0.043, f = −15 gives −3·10⁻⁴. There is no explosion; the risk is **saturation**, where a logit pushed far negative cannot be pulled back by later positive votes. Diagnostic: the share of training samples **with k ≥ 1** whose α₁ logit is < −7, i.e. α₁ < 2ε. Low α₁ on k = 0 samples is legitimate and is not counted. Gradient clipping at norm 1.0 is applied to every arm.
- **Fallback (single, decided in the R0 smoke test before anything else is launched).** If any pilot seed gives a non-finite loss, or the stuck share exceeds 5% at the last epoch, set ε = 0.1 for every B² arm. If ε = 0.1 also trips the rule, HT-floor is reported as "not testable with this head" and B²-r keeps ε = 0.1. No new head is built (the mean–precision head of r0 is dropped).
- **HT-KL.** The KL is applied per sample to the predicted Beta, so it never fades with more data. With the MICCAI settings its prior was unreachable under the floor; it caps S at ≈ 7–10 and pulls m toward π (01_foundations §3.2). Post-hoc c is a one-parameter maximiser of a strictly proper held-out score, so it is consistent for the best level. The KL arm also gets c, so the comparison is "KL against no KL" at equal post-processing.
- **Descriptive expectations (not criteria).** c is near 1 in the frozen regime; the floor costs more vote NLL on C2/C3 than on C1; the weighted B² mean m on validation sits about 0.1 above the vote fraction.

### H4 (core). Does B²-r's per-sample spread carry information beyond m?
- **Arms (all spreads fitted on validation).** B²-r (with c); **S\*(m)**: soft-BCE m plus a spread fitted on validation in 5 equal-mass bins of m (MLE per bin, piecewise-linear interpolation in logit m); **Global-S**; soft BCE with the Binomial plug-in (S = ∞).
- **Design.** CIFAR-10H: training votes subsampled to n_train ∈ {1, 3, 5, 10, all ≈ 51}, validation subsampled to the same n, test scored against the full panel. MHIST (if it is the third dataset): n_train ∈ {1, 3, 5, 7}. Endoscapes (and CVS Challenge): n = 3 as given.
- **n = 1 is an anchor, not evidence.** At n = 1 the Beta-Binomial is Bernoulli(m). S is not constrained: it drifts with the optimiser rather than staying at its initial value, and c, S\* and S\*(m) are unidentified (the validation likelihood is flat). Pre-set tie-break: c = S\* = S\*(m) = ∞, so all arms become Binomial(m) and the margin is zero by construction. The growth test uses n ≥ 3.
- **Endpoint.** Test vote NLL **per vote** (n varies 47–63 on CIFAR-10H), on all images and on the **ambiguous subset**, fixed now as images whose full-panel plurality share is < 0.9. That keeps the one-vs-rest dilution (most head–image pairs have k ≈ 0) from hiding the effect.
- **Justification.** Fisher information about log S per sample is 0 at n = 1 and grows with the number of rater pairs n(n−1)/2 (0.014 at n = 3 against 0.35 at n = 50 for S = 10; 01_foundations §3.1). This is the classical overdispersion picture (§1). B²-r nests Global-S and S\*(m) in the population sense, so its optimum is never worse; in finite samples its extra variance can lose. Because P_maj and P(split) depend only on (m, S), a spread that tracks only m adds no information about x beyond m. Hence the per-sample claim is "B²-r beats S\*(m)", and "S\*(m) beats Global-S" is the weaker "spread varies" claim.
- **Predictions.** (a) CIFAR-10H, n = all: B²-r beats S\*(m) (CI below 0) on all images and on the ambiguous subset, and S\*(m) beats Global-S. (b) The B²-r − S\*(m) margin grows with n: the CI of [margin(all) − margin(3)] lies above 0. (c) Endoscapes, frozen: B²-r and S\*(m) are equivalent within ±δ_NLL; S\*(m) beats Global-S. (d) The soft-BCE plug-in loses to every spread arm on every dataset. Descriptive: a per-image gain on CIFAR-10H of the order of 0.02 nats.
- **Verdict.** (a) decides H4: success / fail (CI favouring S\*(m) or including 0 with the point estimate ≥ 0) / inconclusive (point estimate < 0 but the CI includes 0). If (a) fails, the story is outcome C (§4.8).

### H5. What S means under a frozen vs a fine-tuned encoder
- **Arms.** Frozen (R1, 10 seeds) against **full fine-tuning** (R2, 3 seeds), on Endoscapes and CIFAR-10H, for hard BCE, soft BCE and B²-r. No augmentation in either regime, which removes the augmentation confound. LoRA is cut. Two cheap additions:
  - **Memorisation check (frozen).** B²-r with a linear head (a config change), plus the ratio of median S on training images to median S on validation images, for both heads.
  - **Held-out re-head (CIFAR-10H only, where an effect is predicted).** Fine-tune B²-r on 80% of the training images (part A). Cache features of the other 20% (part B) from that fine-tuned encoder, and train a fresh B²-r head on part B with the existing frozen-head trainer, selecting on validation.
- **Justification.** With a trainable encoder each training image can be isolated; its optimum is S → ∞, so S grows with training and its test value reflects training length, not panel heterogeneity (01_foundations §3.4). A frozen MLP head can also memorise: Step 1 saw frozen S rise from 2.5 to 5.1 over 7 epochs, with the floor and weights on. So the frozen "look-alike" reading of S is tested here, not assumed. A head trained on images the encoder never saw restores the frozen meaning.
- **Predictions.** (a) Fitted c is lower under fine-tuning than frozen (CI of the log-c difference below 0). (b) Median validation S rises over epochs under fine-tuning in all 3 seeds. (c) CIFAR-10H: the H4 margin (S\*(m) NLL − B²-r NLL) is smaller under fine-tuning than frozen (CI of the difference above 0), and the held-out re-head's margin is larger than the fine-tuned margin. (d) Frozen: the train/val median-S ratio is ≤ 1.5 for the MLP head.
- **Pre-specified reading of (d).** If the ratio exceeds 1.5, frozen S is partly memorisation. The H4 primary is then read from whichever frozen head (linear or MLP) has the lower validation vote NLL, and the paper says so.
- **Verdict.** (c) decides H5: fail if the fine-tuned margin is ≥ the frozen margin (CI excluding the predicted order); inconclusive if the CI covers 0.

### H6. Panel probabilities, used at the operating threshold (tests U1/U2)
- **Change.** The model reports P_maj, P(majority agrees with the decision), P(split) and P(unanimous), never 2/S. Decision scores and risks are as defined in U1-a/b/c and U2.
- **Justification.** §2 points 1–5: S changes decisions only through per-sample variation at fixed m, and mostly at low t. Chow's rule makes max(P_maj, 1 − P_maj) the optimal referral score for majority-disagreement risk.
- **Prediction (linked to H4, which makes it refutable both ways).** (a) Where B²-r and S\*(m) are equivalent on vote NLL (expected: Endoscapes), the U1-a ΔFCR (P_maj against m, within B²-r) is equivalent within ±δ_FCR, and the U2 AP of B²-r P(split) is equivalent to the best m-only score within ±δ_AP. (b) Where H4a succeeds (CIFAR-10H, full-panel-majority target, t ≈ 0.1): ΔFCR < 0 with the CI excluding 0. Descriptive: AURC of the P_maj-based confidence about 10% below the mean-only confidence. (c) Descriptive: DUP against B²-r P(split) on U2 (no predicted sign), the soft-BCE ensemble, and 2/S.
- **Verdict.** (b) fails → the spread is identified but does not change decisions. (a) shows a CVS gain (CI excluding 0 in favour of P_maj) → upgrade toward outcome A. Both would be reported.

---

## 4. Validation plan

### 4.1 Datasets: two fixed, one by rule

**Core:** Endoscapes2023 and CIFAR-10H. **Third dataset (confirmatory):** the CVS Challenge if written permission to publish arrives by the **decision date: the R1 launch or 2026-11-06, whichever comes first**. Otherwise **MHIST**. Both requests (the CVS Challenge email and the MHIST research-use form) go out on R0 day 1, so whichever is needed is ready. Every success criterion can be decided on the two core datasets alone. All samples are single images with vote counts available for training.

| | **Endoscapes2023 (CVS201)** | **CIFAR-10H** | **CVS Challenge** (if cleared) | **MHIST** (otherwise) |
|---|---|---|---|---|
| Content | 201 LC videos; 11,090 CVS-labelled frames, one every 5 s | the 10,000 CIFAR-10 test images, 32×32 | 1,000 clips of 90 s, 54 institutions, 24 countries; 18 frames per clip at 5-s intervals | 3,152 H&E colorectal-polyp images, 224×224; binary HP vs SSA |
| Votes per sample | 3, fixed panel | mean 51 (range 47–63); 2,571 MTurk workers; 511,400 labels | 3 per clip, drawn from a pool of 20 qualified surgeons | **7** board-certified GI pathologists |
| Label → (k, n) | k = 3·`ds` per criterion (exact; recompute from the source, since the saved MICCAI JSONs are rounded) | 10 one-vs-rest heads: k_c = votes for class c, n = the image's total. "Majority positive" means more than half the panel chose c, which is not the CIFAR plurality label | per-rater binary labels per frame → k, n = 3. File format and column names UNVERIFIED | k = number of pathologists choosing SSA (the annotation file holds this count; exact column header UNVERIFIED), n = 7, positive = SSA |
| Split | official 120 / 41 / 40 videos | 6,000 / 2,000 / 2,000, stratified by class, fixed seed | 490 / 105 / 105 of the 700 training clips, stratified by country and clip-level status; no test-clip substitution | official train/test (2,175 / 977 according to the science critic; my fetch did not show the sizes); validation = 20% of train by image. Patient-level grouping UNVERIFIED, so train/val leakage affects model selection only |
| Vote subsampling (H4) | — (n = 3) | n_train ∈ {1, 3, 5, 10, all}: hypergeometric draws from the counts, fixed per seed | — | n_train ∈ {1, 3, 5, 7}, same method |
| Access / licence | public, CC BY-NC-SA 4.0 (installed) | public (GitHub `jcpeterson/cifar-10h`, TFDS); licence type UNVERIFIED | Grand Challenge page: "not publicly available", no use or publication outside the challenge. Release status UNVERIFIED, **blocker** | Dataset Research Use Agreement + request form; download links expire after 4 h |
| Loader work | small | ½ day | ≈ 2 days (frame extraction; timestamp alignment UNVERIFIED) | ½ day |
| Known issues | test split analysed in Step 1 (disclosed, §1); val/test prevalence shift (C1 0.163 → 0.235, C3 0.167 → 0.279); fixed panel with sequential marking (one limitation paragraph; per-rater table capped at ½ day); split frames per criterion in test counted in R0 (U2 feasibility) | mostly near-unanimous, hence the ambiguous subset (plurality share < 0.9, fixed now); one-vs-rest heads are not jointly coherent (Σ m_c ≠ 1), so no multiclass quantities are computed from them | ~7% robotic; annotator confidence not used; not compared with Endoscapes as a "panel structure" test (confounded) | single-centre (UNVERIFIED); SSA prevalence to be counted in R0 |

**Why MHIST is a good fallback, and arguably better.** It is the only *medical* dataset here with n > 3. With n = 7 it can test H4's growth with n (n_train ∈ {1, 3, 5, 7}) on medical data, which is exactly the result a MedIA reviewer would ask for if the CVS evidence is null. The loader is ½ day and the fixed Beta-Binomial model applies unchanged.

**Cut:** the synthetic check and ChaosNLI (§5).

### 4.2 Common protocol

- **Encoder.** DINOv3 ViT-B/16 from public weights; no majority-vote adaptation. Input 384² (surgical) or 224² (CIFAR-10H, MHIST). **No augmentation in either regime.**
- **Frozen regime.** Cached embeddings; 50 epochs; minutes per run; **10 seeds**.
- **Full fine-tuning.** 10 epochs, AdamW, cosine schedule; **3 seeds**. LoRA is cut.
- **Hyperparameters.** Tuned per loss family (hard BCE / soft BCE / B²) in the R0 pilot on validation: head lr ∈ {1e-4, 3e-4, 1e-3} × weight decay ∈ {1e-4, 1e-2}, 1 seed, selected by validation mAP. Encoder lr ∈ {1e-5, 3e-5} for fine-tuning. Trap arms, the linear head and DUP inherit their family's setting.
- **Checkpoints, same rule for every arm.** Validation and test predictions are written silently at **every epoch**. Two checkpoints are chosen on validation: **best val mAP**, used for discrimination (H0, HT-w); and **best val vote NLL after the post-hoc spread step** (c, S\*, S\*(m)), used for every panel endpoint (HT-floor, HT-KL, H4–H6). Validation S per epoch is logged for H5.
- **Post-hoc, all on validation.** Thresholds (validation-BACC rule; for U1-a, matched sensitivity 0.80); c for every B² arm; S\* and S\*(m) for soft BCE.
- **Test silence.** The trainer never prints or aggregates test metrics. Test prediction files are opened once, in R3.
- **Code.** Branch `journal-extension`; never `main`. Each R0 module gets a one-page spec and a unit test, and is committed before the next starts, so a cut session can resume from the spec.

### 4.3 Baselines

| Arm | Target | Spread | Notes |
|---|---|---|---|
| Hard BCE | majority vote | — | weighted variant only in HT-w |
| Soft BCE | k/n | Binomial plug-in (S = ∞) | no `pos_weight` |
| Global-S | soft BCE | one S\* per criterion (val) | post-hoc |
| **S\*(m)** | soft BCE | S as a function of m: 5 equal-mass bins on val, interpolated | post-hoc, minutes; the per-sample test (H4) |
| **DUP** | BCE on the split indicator 1[0 < k < 3] (CVS); for CIFAR-10H and MHIST, the unbiased 3-rater split estimate from the training votes as a soft target (U2 there is defined for a 3-rater panel) | — (predicts P(split) directly) | frozen head, minutes; Raghu et al. 2019 comparator for U1-c/U2 |
| Soft-BCE ensemble | 5 soft-BCE seeds | mixture of member pmfs | U1/U2 only; bootstrapped over test clusters only |
| **B²-r** | vote counts | per sample, × c | proposed |
| B²-M | MICCAI recipe | per sample, × c | legacy row, native output |

All arms share the encoder, head, tuning budget, checkpoint rules and threshold rules, and all are unweighted except the HT-w arms.

### 4.4 Metrics

**Scored quantity per arm (fixed):**

| Arm | Discrimination (rank / threshold) | P_maj | P(split) | Vote NLL |
|---|---|---|---|---|
| hard BCE | p (its output) | p | — | Binomial(n, p), reported for completeness only |
| soft BCE | m | Binomial tail at m | 3m(1−m) (n = 3) | Binomial(n, m) |
| Global-S, S\*(m) | m | Beta-Binomial tail | Beta-Binomial | Beta-Binomial |
| B² arms | m = α₁/S | Beta-Binomial tail (× c) | Beta-Binomial (× c) | Beta-Binomial (× c) |
| DUP | — | — | its output | — |
| ensemble | mean m | mixture | mixture | mixture |

**Primary** (per criterion/head, then macro):
- the minimum set ACC, BACC, F1 (at the validation-BACC threshold on the discrimination score) and mAP;
- vote NLL: per frame-criterion on CVS and MHIST, **per vote** on CIFAR-10H, against the full panel;
- ECE of P_maj (15 equal-mass bins, against the majority);
- P(split) AUROC;
- FCR at matched sensitivity (U1-a);
- AURC (U1-b).

**Secondary:** AUROC; Brier of P_maj and P(split); θ-MSE against the full panel (CIFAR-10H, MHIST); split rate among accepted at 80% (U1-c); top-20% catch rate and P(split) AP (U2); **P(split) reliability**. The last is the higher-order calibration check in the sense of Ahdritz et al.: 10 equal-mass bins of predicted P(split), observed split frequency per bin (on CIFAR-10H, the bin mean of the unbiased 3-rater estimate; its noise floor is reported by jackknife), and the weighted mean |predicted − observed|.

**Diagnostics:** c, S\*, the share of α₁ on the floor, the stuck share, ρ(S, m), median S per epoch, and the train/val S ratio.

### 4.5 Statistics

- **Seeds.** 10 in the frozen regime and 3 under fine-tuning. The fine-tuning comparisons are secondary and their CIs will be wide; this is stated.
- **Two-level bootstrap** (2,000 resamples). Test clusters are resampled with replacement: videos for Endoscapes, clips for the CVS Challenge, images for MHIST, and images stratified by class for CIFAR-10H. These are shared across arms, so the comparison is paired on test data. Seeds are resampled independently per arm, because seed pairing across arms is arbitrary. The statistic is the per-seed metric averaged over the resampled seeds. The ensemble is resampled over test clusters only. Validation-fitted quantities (thresholds, c, S\*, S\*(m)) are not resampled. Clustering is a resampling unit only; there is no per-video evaluation.
- **Three-way verdicts.** *Superiority:* success if the CI excludes 0 in the predicted direction; fail if it excludes 0 in the other direction; inconclusive otherwise. *Equivalence:* success if the CI ⊂ [−δ, δ]; fail if the CI is disjoint from [−δ, δ]; inconclusive otherwise. *Non-inferiority:* success if the upper bound is < δ; fail if the CI excludes 0 against the prediction; inconclusive otherwise.
- **Margins from a validation-only pilot (R0).** The tuned soft BCE and B²-r, 5 seeds each, are run on validation, and the two-level bootstrap on validation gives the CI half-width h for each margin metric (BACC, mAP, vote NLL, FCR, P(split) AP). Then δ = 1.5·h, rounded up, with caps: 3 pp (BACC, mAP), 5 pp (FCR), 0.05 (AP), and 1% of the soft-BCE plug-in validation NLL (vote NLL). A δ above its cap turns that equivalence test into "CI reported, no claim", decided before R1. Validation and test have similar sizes (41/40 videos; 2,000/2,000 images), so h transfers.
- **Primary comparison families (5).** F1 H0; F2 HT (three traps); F3 H4 (a and c); F4 H5c; F5 H6 (U1-a). Everything else is secondary and descriptive. No p-values.
- **Claim rule.** A claim needs the predicted verdict on **both Endoscapes and CIFAR-10H**, except where a hypothesis names one dataset (H4a/b, H5c, H6b: CIFAR-10H; H4c, H6a: Endoscapes). The third dataset is confirmatory: it is reported, and any disagreement with the core datasets is stated, but it neither creates nor removes a claim.

### 4.6 Rounds, user-days and GPU hours

User-days follow the scope critic's module estimates, plus the cost of this round's additions. Each round is one batch from one config list. A bug means fixing it and re-running the whole round, which stays affordable because R1 takes about 7 GPU-hours.

#### R0: build, pilot, freeze (≈ 11.5–12.5 user-days; ≈ 4 GPU-h)
| Module | Content | User-days |
|---|---|---|
| a | K heads with per-sample (k, n); loss on GPU; ε, λ, weighting mode, linear/MLP head and DUP head as config; no `pos_weight` on soft targets | 1.5 |
| b | Endoscapes exact k + test split-frame counts (+ per-rater table ≤ ½ day); CIFAR-10H counts + hypergeometric subsampling + ambiguous subset | 1.5–2 |
| c | Embedding cache + **one** trainer (embeddings or images); per-epoch silent prediction files; both checkpoint rules | 2–2.5 |
| d | Post-hoc: thresholds (BACC rule, matched sensitivity), c, S\*, **S\*(m)** | 1–1.25 |
| e | Metrics incl. P(split) reliability; two-level bootstrap; three-way verdict function; unit-tested on MICCAI-era **validation** predictions after recovering exact k | 2 |
| f | Launcher, aggregation, all figure/table scripts written against validation; smoke test of every arm (decides the ε fallback) | 2 |
| g | **Pilot**: tuning grid + 5 seeds of soft BCE and B²-r on validation → margins δ; commit and date §4.7 with the margins filled in | 0.5 |
| Day 1, in parallel | CVS Challenge permission email; MHIST research-use request | — |
| + third dataset | MHIST loader (½) **or** CVS Challenge loader (2), after the decision date | 0.5 or 2 |

#### R1: frozen encoder (≈ ½ user-day; ≈ 26 GPU-h ≈ 7 h on 4 GPUs). Supports H0, HT, H4, H5d, H6
| Block | Datasets | Arms | Seeds | Runs |
|---|---|---|---|---|
| Main | Endoscapes, CIFAR-10H, third | hard BCE, hard BCE-w, soft BCE, B²-r, B²-r-w, B²-r + floor, B²-r + KL, B²-M, B²-r linear head, DUP = 10 | 10 | 300 |
| Subsampling | CIFAR-10H (n ∈ {1, 3, 5, 10}); MHIST if used (n ∈ {1, 3, 5}) | soft BCE, B²-r, DUP | 10 | 120 (+ 90) |
| Post-hoc | all | Global-S, S\*(m), c, soft-BCE ensemble | – | – |

#### R2: full fine-tuning (≈ ½–1 user-day; ≈ 25 GPU-h ≈ 1 day of wall-clock). Supports H5, and H0/H4 under fine-tuning (secondary)
| Block | Datasets | Arms | Seeds | Runs | GPU-h |
|---|---|---|---|---|---|
| Tuning (5 epochs) | Endoscapes, CIFAR-10H | hard BCE, soft BCE, B²-r × 2 encoder lr | 1 | 12 | ≈ 6 |
| Main | Endoscapes, CIFAR-10H | hard BCE, soft BCE, B²-r | 3 | 18 | ≈ 18 |
| Held-out re-head | CIFAR-10H | B²-r fine-tuned on 80%, fresh head on the cached features of the other 20% | 3 | 3 (+ 3 head runs) | ≈ 1.5 |

R2 needs only validation results from the R0 pilot, so it can launch right after R1, or in parallel on spare GPUs.

#### R3: lock and analyse (≈ 4–5 user-days; no training)
Open the test predictions once, run the scripts written in R0, and write down the verdicts. Outputs:
- **Tables.** T1 discrimination (all arms; frozen and fine-tuned rows); T2 panel prediction plus the HT ablation (with and without c); T3 use cases (FCR, AURC, split rate among accepted, AP, catch rate).
- **Figures.** F1 theory (the boundary table as a curve; P_maj against S for n = 3/11/51), from existing scripts; F2 the H4 margin against n_train (CIFAR-10H, plus MHIST if used); F3 S against m per trap arm; F4 validation S per epoch and c by regime; F5 risk–coverage and FCR on CVS and CIFAR-10H; F6 a qualitative panel of CVS frames ranked by predicted P(split).

**Totals.** User days before writing:

| | MHIST as third dataset | CVS Challenge as third dataset |
|---|---|---|
| R0 + loader | 11.5–12.5 + 0.5 | 11.5–12.5 + 2 |
| R1 | 0.5 | 0.5 |
| R2 | 0.5–1 | 0.5–1 |
| R3 | 4–5 | 4–5 |
| **Total** | **≈ 17–19.5** | **≈ 18.5–21** |

That is about 1–1.5 user-days above the scope critic's trimmed 15–18 (or 17–20 with the CVS Challenge). The excess is S\*(m), DUP, the pilot and the two-checkpoint rule, each ≤ ½ day and each needed to make a central claim refutable (H4, U2, H0/H6, the panel endpoints).

GPU: ≈ 4 (R0) + 26 (R1) + 25 (R2) ≈ **55 GPU-h**, down from ≈ 175 in r0.

### 4.7 Success criteria (to be committed, with δ values filled in, at the end of R0)

| ID | Endpoint (test; frozen unless stated) | Type | Datasets |
|---|---|---|---|
| S0 (H0) | Δ BACC, Δ mAP on m, B²-r − soft BCE | equivalence ±δ | Endoscapes + CIFAR-10H |
| ST-w | Δ BACC, weighted − (unweighted + threshold), for B²-r and for hard BCE | equivalence ±δ_BACC | Endoscapes + CIFAR-10H |
| ST-floor | Δ vote NLL, B²-r − floor (with c) | superiority (< 0) | Endoscapes + CIFAR-10H |
| ST-KL | Δ vote NLL, B²-r − KL (with c) | non-inferiority (+δ_NLL) | Endoscapes + CIFAR-10H |
| S4a | Δ vote NLL per vote, B²-r − S\*(m), n = all; all images and the ambiguous subset | superiority | CIFAR-10H |
| S4b | margin(all) − margin(3) | superiority (> 0) | CIFAR-10H |
| S4c | Δ vote NLL, B²-r − S\*(m) | equivalence ±δ_NLL | Endoscapes |
| S5c | frozen margin − fine-tuned margin | superiority (> 0) | CIFAR-10H |
| S6a | Δ FCR at sensitivity 0.80, P_maj − m within B²-r | equivalence ±δ_FCR | Endoscapes |
| S6b | same, full-panel-majority target | superiority (< 0) | CIFAR-10H |

### 4.8 Outcome → story (decided after R3)

| Outcome | Condition | Story | Venue view (§6) |
|---|---|---|---|
| **A** | S4a success **and** on a CVS dataset either the H4 margin favours B²-r or S6a shows a gain (FCR lower, CI excluding 0) | "A per-sample panel spread is learnable from 3 raters under a frozen foundation encoder, and it reduces false CVS confirmations." | MedIA |
| **B+** | S4a success; CVS equivalent or inconclusive; **MHIST** S4a-type and S6b-type results succeed | "The spread needs more than 3 raters: shown on natural and pathology images. For 3-rater CVS data, use the corrected recipe with S\*(m)." Medical many-rater evidence included | MedIA |
| **B** (expected) | S4a success; CVS equivalent or inconclusive; no medical many-rater success | Same as B+, but the many-rater evidence is natural-image only | TMLR (MedIA borderline) |
| **C** | S4a fail or inconclusive | "B² = soft BCE + a mean-dependent spread. Theory, the traps, and how many raters a spread needs." | TMLR |

H0 and HT are reportable under every outcome. An H0 fail (a discrimination difference) is reported as a finding and does not change the outcome letter.

---

## 5. Non-image modality and the synthetic check: cut

There is no non-image modality in the plan, and the synthetic check is cut. Neither tests a hypothesis the three datasets do not, and both cost user time: ChaosNLI ≈ 1–2 days, the synthetic check ≈ 1 day. The paper says in one sentence that the loss does not depend on the input type. If a TMLR reviewer asks, the synthetic check (01b §C) is a 1-day revision item, and ChaosNLI (100 votes per item, 4,645 items; EMNLP 2020; verified in r0) is the natural text dataset; Kohli 2026 already subsamples it.

---

## 6. Target journal: decision rule (the orchestrator makes the final call)

| Outcome (§4.8) | Venue | Reason |
|---|---|---|
| A | **MedIA** | A non-null CVS result plus theory and a corrected recipe: a medical-imaging methods paper with clinical relevance |
| B+ | **MedIA** | The many-rater evidence includes a medical dataset (MHIST), and the CVS application has matched baselines. Medical readers get both the rule ("how many raters") and a medical demonstration |
| B | **TMLR** (MedIA possible, but borderline) | The positive result is on CIFAR-10H only. MedIA reviewers would ask why the medical result is null. TMLR judges correctness and evidence, and a well-argued "when does the spread help" paper is in scope |
| C | **TMLR** | An analysis paper with a careful negative result |

Not recommended under any outcome:
- **TMI**: a loss analysis, half the evidence non-medical.
- **TPAMI**: the general technique is prior art (§7), so the general-ML novelty is too thin.

**My view.** Outcome B is the most likely, so **TMLR is the most likely venue unless MHIST delivers B+**. That is one more reason to prefer MHIST as the third dataset (§8). Experiments are identical for every venue up to writing.

---

## 7. Novelty (r0's 8 searches, plus the science critic's verified additions)

**Prior art the paper must cite and position against**
1. **Vote-count likelihoods with deep networks are not new.**
   - Walmsley et al. 2020 (MNRAS; [arXiv 1905.07424](https://arxiv.org/abs/1905.07424)): a **Binomial** likelihood on volunteer vote counts with Bayesian CNNs, plus active learning of which galaxies to label (prior art for vote allocation, hence the U2 cut).
   - Walmsley et al. 2022 (Galaxy Zoo DECaLS; [arXiv 2102.08414](https://arxiv.org/abs/2102.08414)) and the `zoobot` library ([losses](https://zoobot.readthedocs.io/en/latest/autodoc/pytorch/training/losses.html)): a **Dirichlet-Multinomial** NLL on vote counts. B² is its two-category case.
   - Hechinger et al. ([arXiv 2311.08874](https://arxiv.org/abs/2311.08874)): a Bayesian Dirichlet-Multinomial model of annotation distributions (not a deep head).
2. **Classical overdispersion.** Beta-Binomial models for clustered binary data, the intra-class correlation ρ = 1/(S+1), and identification from within-cluster pairs: Williams 1975 (*Biometrics*), Crowder 1978 (*JRSS-B*), Prentice 1986 (*JASA*). Cited from memory; exact titles to be checked when writing. Contribution 1(c) is this theory applied to deep second-order heads, and is presented as such.
3. **Soft labels and disagreement.**
   - Peterson et al. 2019 (CIFAR-10H; [arXiv 1908.07086](https://arxiv.org/abs/1908.07086)).
   - Uma et al. 2021 (JAIR survey; [link](https://jair.org/index.php/jair/article/view/12752)).
   - Singh et al. ("Distributions In, Distributions Out: The Case for Soft-Label Training", [arXiv 2511.14117](https://arxiv.org/abs/2511.14117)): soft-label training matches hard-label accuracy and tracks annotator entropy on ChaosNLI, POPQUORN and CIFAR-10H, with no second-order model. Supports H0.
   - Raykar et al. 2010 ([JMLR](https://www.jmlr.org/beta/papers/v11/raykar10a.html)) and coupled cross-entropy ([arXiv 2306.03288](https://arxiv.org/pdf/2306.03288)): annotator models.
   - Collins et al. 2022 (CIFAR-10S), Baan et al. 2022 ("Stop measuring calibration when humans disagree"), Malinin et al. 2019 (Dirichlet likelihood on ensemble outputs), Tanno et al. 2019: cited from the science critic's list, not re-read (UNVERIFIED details).
4. **Rater subsampling.** Kohli 2026 ("Metric-Dependent Annotation Saturation", [arXiv 2605.29797](https://arxiv.org/abs/2605.29797)) subsamples ChaosNLI's 100 annotators. First-order metrics such as KL saturate by about 10 annotators; disagreement ranking needs 20–50. This overlaps our vote-subsampling idea. Ours differs: it concerns the identifiability of a **per-sample second-order spread** in a deep head, judged against S\*(m).
5. **Predicting disagreement directly.** Raghu et al. 2019 ([ICML](https://proceedings.mlr.press/v97/raghu19a.html)) show that direct prediction of doctor disagreement beats classifier-derived uncertainty. It is now a **baseline** (DUP), not only a citation.
6. **Second-order uncertainty.**
   - Bengs et al. 2022 ([NeurIPS](https://arxiv.org/abs/2203.06102)): second-order learners trained on single labels give unfaithful epistemic uncertainty. With several votes per sample, the Beta-Binomial NLL is a proper score for the counts, so our setting is not the one they criticise. We still make no epistemic claim.
   - Ahdritz et al. 2024 ([arXiv 2412.18808](https://arxiv.org/abs/2412.18808)): higher-order calibration with k-snapshots. **Used as an evaluation** (P(split) reliability, §4.4), not only cited.
7. **Calibration under ambiguity.** Tao et al. 2026 ([arXiv 2603.22879](https://arxiv.org/abs/2603.22879)) propose **first-order** post-hoc calibrators (Dirichlet-Soft, MCTS, LS-TS) fitted against annotator distributions, on CIFAR-10H, ChaosNLI, and ISIC 2019 / DermaMNIST with *synthetic* annotations. r0 overstated their closeness. Global-S, S\*(m) and c are the **second-order** analogue (they fit the spread, not the first-order probabilities), and are positioned that way.
8. **Medical multi-rater uncertainty** exists mainly for segmentation (e.g. [arXiv 2202.07550](https://arxiv.org/pdf/2202.07550)). The searches found no CVS work that predicts the panel's vote distribution (not exhaustive).

**What remains novel**
1. The diagnosis of **three implementation traps** in second-order heads trained on vote counts, with single-factor ablations and measured effects.
2. A **controlled real-data identifiability study of a per-sample spread** against the number of raters, judged against a mean-conditional spread (distinct from first-order saturation studies), on natural and, if MHIST is used, pathology images.
3. The **decision analysis**: S changes decisions only through per-sample variation at fixed m, mostly at low thresholds, with the effect growing with panel size, tested with an S-aware FCR at the operating threshold.
4. The **encoder-regime analysis** (frozen against fine-tuned, with a memorisation check and a held-out re-head).
5. **The CVS application** with equally tuned, equally weighted baselines, including S\*(m) and a direct disagreement predictor.

Enough for TMLR. Enough for MedIA under outcome A or B+.

---

## 8. Open items and remaining disagreements

**Blocking, with decision rules**
1. ★ **CVS Challenge permission.** Email on R0 day 1; decide at the R1 launch or 2026-11-06, whichever comes first; otherwise use MHIST.
2. ★ **MHIST access** (research-use form, R0 day 1).

**Non-blocking checks:** MHIST split sizes, patient grouping and column header; Endoscapes per-rater column names; the CIFAR-10H licence type; the exact titles of the classical overdispersion references.

**Where I still differ from a critic**
- **MHIST vs CVS Challenge as the third dataset.** I follow the orchestrator's rule (CVS Challenge if cleared). My view is that MHIST is worth more to this paper even if the CVS Challenge clears. It is the only medical dataset with n > 3, it tests H4's growth in n on medical data, it costs ½ day instead of 2, and it is the route to MedIA under the most likely outcome. If the user agrees, make MHIST the third dataset outright and keep the CVS Challenge as future work.
- **Held-out spread arm.** The scope critic would cut it and the science critic would move it. I kept it on CIFAR-10H only, built from existing parts (≈ ¼ day, ≈ 1.5 GPU-h), because without it H5's mechanism claim cannot be tested where an effect is predicted.
- **Fallback ladder.** The science critic would keep the mean–precision head; I dropped it on time grounds. The risk is that HT-floor becomes "not testable" if ε = 0.1 also saturates. I judge that unlikely, given |∂L/∂f| ≲ 1 and clipping.
- **Cross-fitted S-head** (science): rejected as new code. The linear-head arm and the train/val S ratio answer the same memorisation question at no cost.
- **Seeds** (scope: 5): 10 in the frozen regime, because it costs minutes of GPU and no user time, and the equivalence tests need it.
