# 01b — User feedback on Step 1, corrections, synthetic-generator review, and the Step 2 brief

*Written 2026-10-08 after the user's review of 01_foundations.md. This file supersedes 01_foundations.md wherever they disagree.*

## A. User comments and how they change the plan

| # | User comment | Consequence |
|---|---|---|
| 0 | There is no synthetic code. A synthetic check would be written from scratch, and it is optional | Synthetic study is optional. If included it must be small: one script, CPU-scale, testing only the maths claims. Never on the critical path |
| 1 | Rater identities exist. The "bias" is procedural: one annotator never marked a criterion unless another criterion was already reached, so images/sequences were not marked independently. User does not see it as a problem for the loss | Endoscapes stays in. The dependence is mostly **across criteria** (C1–C3 hierarchy for one rater) and **across frames** (sequential marking), not a within-frame skill difference. Treat it as a stated limitation plus at most one cheap check, not a modelling programme. Rater-specific models are out of scope unless they cost almost nothing |
| 2 | MICCAI used SwinV2 fine-tuned on majority-vote labels. Wrong, do not use | No majority-vote pre-adaptation anywhere. Encoders start from public pretrained weights only. MICCAI numbers are not reused |
| — | Ignore the Wilcoxon request (overeager reviewer). The paper's CIs were not good. "One-sided Wilcoxon, n = 5" can be changed | Statistics redesigned from scratch (section E.6). No Wilcoxon requirement |
| — | If S's meaning is undefined, still find a use for (α₀, α₁): uncertainty, or anything that removes cases where the model may be wrong. Don't over-rely on S | Primary use case becomes **selective prediction / referral** (abstain or refer when the predicted panel response is unreliable), and the test is whether the Beta adds anything over the mean. See D |
| — | Think about use cases and users (CVS in particular, others allowed). Alphas are probably useful under conditions such as a frozen encoder. "Just output the mean" is not an option we take | Use-case section D. The proposer must name users, conditions and the decision each output supports |
| — | Check that softplus is not removed by accident | **Confirmed: softplus stays.** Only the `+ 1` after softplus is in question (see B.1) |
| — | Make a new branch before touching the repo; never push to main | No repo code changes in this planning stage. `journal_plan/` is an untracked folder; nothing has been committed. Before any code work: `git checkout -b journal-extension` |
| — | "Dispersion-aware baseline" was unclear | Renamed and defined in E.4 as **"soft BCE + one global concentration"** |
| — | Metrics: at least ACC, BACC, mAP, F1, plus others per task | Section E.5 |
| — | Ignore per-video evaluation completely | Removed from the plan. Frame/sample-level only |
| — | Don't overthink assumption violations: the world isn't perfect, and real data always carries bias | Given to the science critic as an explicit instruction: judge assumption violations by whether they change a conclusion, not by purity |
| — | Synthetic-generator notes attached; review them independently | Section C |

## B. Corrections to 01_foundations.md

1. **Softplus.** 01_foundations §0.7 and §8.3 said "drop the +1 floor". To be explicit: the head is `α = softplus(f(x)) + 1e-6 + 1`.
   - **Softplus is kept.** It guarantees α > 0 with smooth, non-vanishing gradients. `exp` overflows for large logits and makes the lgamma terms explode. `ReLU` gives exactly-zero α (lgamma(0) = ∞) and dead units.
   - **The `+1` is what is questioned.** It comes from Sensoy et al.'s EDL, where α = evidence + 1 encodes a uniform Dirichlet prior added to the counts. In a likelihood model it is an arbitrary restriction to α ≥ 1. The proposed change is `α = softplus(f(x)) + ε` with a small ε (1e-3; the user's synthetic notes already use this). It is a hypothesis to test, not an assumed fix: dropping the floor may destabilise training when α → 0, because lgamma(α) ≈ −log α. The proposer must give a fallback (e.g. a lower ε such as 0.1, or a floor on S only).
2. **Encoder for MICCAI** was SwinV2 fine-tuned on majority votes (user). The S-floor and weight findings in §3.3 used the 26/05 runs (frozen SwinV2 / DINOv3 from public weights and an E2E SwinV2). They do not depend on that encoder.
3. **Statistics row (§1 table, §9).** The Wilcoxon criticism stands as a fact about n = 5, but it is no longer a requirement. Statistics are redesigned (E.6).
4. **Annotator dependence (§6).** The user's finding is sequential/hierarchical marking, which the iid-Binomial simulation in §6 did not cover. Its likely effects:
   - on the **criteria**: C2/C3 positives are less likely when C1 was not reached, by that rater. This creates cross-criterion correlation in votes; the three independent heads ignore it;
   - on **S**: a rater who systematically withholds positives behaves like a negatively biased rater. That makes the votes *under*-dispersed, which pushes S up a little.
   It does not invalidate the per-frame Beta-Binomial as a working model. Pragmatic stance: one sentence of limitation, and optionally one check (per-rater positive rates per criterion, conditional on the other criteria).

## C. Independent review of the user's synthetic-generator design (attached notes)

*Reviewed as a design; no code exists. Verdict: the idea of a known-θ test bench with a Bayes reference is sound, but as designed it mostly tests recovery of the mean. Most of the advertised S tests are ill-posed for this loss. A much smaller generator answers the maths questions.*

| Component | Assessment |
|---|---|
| θ = sigmoid(η/τ), τ set by clarity = dᵀu, z = Uu + ε (D = 1024, m = 16, σ_z = 0.7) | θ is a **deterministic function of u**, and u is almost exactly recoverable from 1024 noisy linear features. The posterior sd of u is ≈ σ_z/‖U column‖, tiny if U has O(1) entries. So in the **independent regime S_true = ∞ almost everywhere**: the correct output is a Binomial. "Contested" frames have θ near 0.5. That is **mean** information, which soft BCE learns equally well. The only place a finite S is correct is the narrow band at the hard clarity threshold, where posterior error flips τ and makes θ | z bimodal |
| Hard τ threshold | Makes θ(z) discontinuous, so an MLP must learn a step. Errors there are mean errors, and smoothing them shows up as spurious finite S. A smooth τ(clarity) is simpler and removes an artefact |
| Correlated regime p_eff ~ Beta(θρ, (1−θ)ρ) | Good: the data are then exactly Beta-Binomial with **S_true = ρ, constant**. That tests whether S is identifiable as n varies, but S is constant, so a single global S (E.4 baseline) is already optimal there. To test *per-sample* S, ρ must **vary with x**, e.g. ρ(x) = f(clarity) |
| Calibration to the Endoscapes C2 histogram | Adds four fitted parameters and the note's own warning (p_clear ≈ 0.94: almost no contested frames). It has no bearing on the maths claims. **Cut** for a theory check |
| OOD shift orthogonal to a and d, expecting 2/S to rise | **Ill-posed for this loss.** Nothing in the Beta-Binomial likelihood makes S fall away from the training data: S models heterogeneity of θ, not novelty (01_foundations §3). The outcome is set by how the MLP extrapolates, and would be "poor" by construction. Test OOD with ensembles or feature density, if at all; not as a claim for this loss |
| Variable n per frame (1..n) | Legitimate and useful, but note: the correct comparison is **count-weighted soft BCE** (Binomial NLL = n·BCE(k/n) + const), which also tells 1/1 from 3/3. The win is over *unweighted* soft BCE, not over the Binomial likelihood. Frames with n = 1 carry zero information about S |
| Bayes reference via posterior over u | Good, and cheap. For the B² model the predictive vote NLL is already integrated (the Beta-Binomial pmf), so comparing it to the Bayes predictive NLL is like-for-like. The note's caveat applies only to plug-in Binomial baselines |
| KL prior mean = E[k/n], weights from P(k ≥ 2) | Agrees with 01_foundations A6 |
| Evidential head with softplus + 1e-3 (α < 1 allowed) | Agrees with B.1 |

**Likely reasons the user's synthetic results looked poor** (consistent with 01_foundations §7):
(i) S_true = ∞ in the default regime, so there was no S signal to recover;
(ii) the OOD test expected behaviour the loss does not produce;
(iii) the comparison with soft BCE on mean metrics should tie;
(iv) Endoscapes-calibrated settings leave ~6% contested frames;
(v) if class weights or the KL were on, the mean was biased and S compressed.

**Recommended minimal synthetic check (optional):** x ∈ ℝ² observed; θ(x) smooth; a hidden nuisance h ~ N(0,1) not given to the model, entering θ = sigmoid(g(x) + s(x)·h), so S_true(x) varies with x and is known in closed form or by quadrature; n ∈ {1, 3, 10, 50}; N ∈ {1k, 10k}; MLP heads; Bayes reference by quadrature. CPU only, about a day of the user's time. It answers: (a) the large-S equivalence to soft BCE; (b) identifiability of S against n and N; (c) the effects of the +1 floor, the weights and the KL. Include it only if the plan needs a theory figure.

## D. Use cases for (α₀, α₁) — seeds for the proposer, not conclusions

The question is: under what conditions is the full Beta (not just its mean) worth outputting, who uses it, and for what decision?

1. **Selective prediction / referral** (main candidate; the user's "remove cases where the model may be wrong"). From (α₀, α₁) we get, in closed form:
   - P(the panel's majority agrees with the model's decision) = max(P_maj, 1 − P_maj);
   - P(split panel) = 1 − P(k = 0) − P(k = n);
   - P(unanimous agreement).
   Refer the frames where these are low. The mean-only versions (S → ∞) exist as well, so the test is whether the Beta-based score gives a better risk–coverage curve than mean-only scores and than plain soft-BCE / hard-BCE confidence. CVS user: an intraoperative assistant that *declines to confirm* a criterion when experts would plausibly disagree. That is safer than a confident yes/no, and it is the natural reading of "flag doubtful views".
2. **Frozen foundation-model encoder + light head** (the user's hunch; 01_foundations §3.4 agrees). Here look-alike groups exist, so S has a defined meaning: experts disagree on frames like this. This is the realistic deployment setting for small surgical datasets (DINOv3 / surgical FMs, linear or MLP probes). Prediction: per-sample α carry more information than a global concentration under a frozen encoder than under full fine-tuning.
3. **Annotation triage / adaptive number of annotators** (dataset builders, e.g. CVS dataset curators). Use the predicted vote distribution to decide which frames need a 4th/5th expert or a consensus meeting (high P(split)), and which can be single-annotated (P(unanimous) high). Concrete payoff: annotation cost saved at a fixed label quality.
4. **Label auditing / posterior predictive check.** Frames where the observed k is improbable under the predicted Beta-Binomial point to label errors or protocol artefacts, e.g. the sequential-marking pattern the user found. Users: dataset maintainers. Cheap to evaluate if a known artefact exists.
5. **Variable or small n.** When the number of votes differs per sample, the likelihood weights samples correctly. Its fair rival is count-weighted soft BCE (C), so the method's specific gain is only in combination with S.
6. **Other domains with many raters** (a non-image modality is optional): radiology/dermatology/retinal grading with several graders; subjective NLP labels with many raters per item (e.g. toxicity datasets with variable annotator counts); sleep staging with several scorers. The proposer decides whether any adds value beyond CIFAR-10H, and verifies the facts about each dataset before relying on them.

## E. Step 2 brief (given to the proposer)

**E.1 Fixed (user):** votes ~ Binomial(n, θ), θ ~ Beta; the model outputs the Beta parameters; training data must carry multiple votes per sample; softplus positivity stays; no majority-vote pre-adaptation; no per-video evaluation; experiments go on a new git branch.

**E.2 Open:** sample weighting and its implementation; encoder choice and frozen / full fine-tune / LoRA; use and form of the KL; interpretation of the uncertainty (it must be useful in practice); the `+1` floor; everything else.

**E.3 Datasets:** CVS Challenge, Endoscapes2023 (installed; keep, see A.1), CIFAR-10H, optionally a minimal synthetic check (C). For each: download, dataloader work, label format, annotators per sample, licence/access. Verify the facts (web) and mark anything unverified. Default sample = single image; a common architecture across losses.
- CIFAR-10H is a multi-class dataset with ~50 votes per image. Propose a binary formulation consistent with the fixed Beta model, and a train/val/test split *within* the 10k CIFAR-10H images, because training must carry votes.
- **Vote subsampling (n_train ∈ {1, 3, 5, 10, all}, evaluated against held-out full-panel θ)** is the obvious real-data identifiability experiment; the proposer may adopt or reject it.

**E.4 Baselines (equally tuned, same budget, same model selection):**
- hard BCE (majority vote);
- soft BCE (target k/n; count-weighted if n varies);
- **soft BCE + one global concentration**: train soft BCE, then fit a single number S* per criterion on validation, so that Beta(m·S*, (1−m)·S*) maximises the validation vote likelihood. This gives the same kinds of outputs as B² (P(split), P_maj, …) with no per-sample concentration. If B² cannot beat it, the per-sample α carry no extra information;
- optionally: a deep ensemble of soft-BCE models as a non-evidential source of uncertainty for selective prediction (cost: training extra seeds).
Class handling must be identical across losses (see the weighting finding, 01_foundations §5).

**E.5 Metrics (pre-specified):**
- Minimum, against the majority vote: ACC, BACC, mAP, F1. Thresholds are chosen on validation by a stated rule, the same rule for all methods.
- Plus, per task: AUROC; vote-level NLL; Brier / ECE of P_maj; P(split) AUROC/Brier; θ-MSE against held-out full-panel θ (CIFAR-10H); risk–coverage curves / AURC and accuracy at fixed coverage for selective prediction.

**E.6 Statistics (redesigned):** ≥ 5 seeds per configuration. Report mean ± sd across seeds, *and* a paired difference to the reference loss with a 95% CI from a bootstrap that resamples **videos** (Endoscapes/CVS) or images (CIFAR-10H) jointly with seeds. Pre-specify the primary comparisons and a small set of success criteria. No reliance on per-test p-values.

**E.7 Constraints:** the user's time and Claude usage are scarce; GPU is not (3–4 A6000; ~1–2 h per 10-epoch run). Prefer a few large, pre-specified rounds over iteration. One journal paper. Model selection on validation only; test sets used once.

**E.8 Output format required from the proposer:**
- Method hypotheses H1…Hk, each with: the change; the mathematical justification; the prediction; the result that would refute it.
- Use cases (who, under which conditions, which decision).
- The validation plan (datasets with logistics, baselines, metrics, seeds, runs).
- An opinion on the non-image modality.
- A candidate target journal (MedIA, TMI, TPAMI, TMLR) with reasons.
