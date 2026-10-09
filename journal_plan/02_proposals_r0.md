# 02 — Proposals, round 0 (proposer)

*Written 2026-10-08 by the Step-2 proposer. Inputs: `01_foundations.md`, `01b_feedback_and_step2_brief.md` (wins on conflict; §E is the spec), the MICCAI paper and reviews, the repo at `main` (read-only). Nothing here has been run. Every number marked "prediction" is a prediction, not a result. Facts that could not be verified on the web are marked **UNVERIFIED**.*

**Shorthand used throughout.** k = positive votes, n = votes on the sample, m = α₁/S (Beta mean), S = α₀ + α₁. "B²-r" = the revised B² recipe proposed here (softplus + ε, unweighted likelihood, no KL, one post-hoc spread scalar fitted on validation, threshold chosen on validation). "B²-M" = the MICCAI recipe (softplus + 1, majority-vote class weights, KL λ = 0.05, ν = 2), kept only as a legacy reference arm. "Global-S" = soft BCE + one global spread S* per criterion fitted on validation (01b §E.4).

---

## 1. Positioning (what the journal paper is)

**Working title.** *Predicting the expert panel, not the majority: Beta-Binomial learning from multiple annotators, when the spread is identified, and what it is good for (with application to the Critical View of Safety).*

**The problem we have to be honest about.** 01_foundations §4 shows that the B² loss is n·BCE(k/n, m) − log C(n,k) − c(k,m)/S + O(S⁻²). The mean m is therefore learned by the soft-BCE objective. A "better classifier than soft BCE" claim is not available, and the MICCAI accuracy gain must be attributed to soft labels and to unequal class weights (01_foundations §0.3, §5). The journal paper cannot rest on accuracy.

**The contribution, in one sentence.** A model that outputs a Beta over the panel's response rate is a *predictive model of what a panel of experts would say*, including whether they would split. That is useful only where (i) the spread is identified by the data and (ii) someone makes a decision that depends on the spread. The paper gives the theory for (i), a corrected recipe that makes the spread learnable, and tests (ii) on two CVS datasets and one many-rater benchmark.

**Contributions (planned).**
1. **Analysis (theory, cheap, already mostly derived).** (a) B² = soft BCE + an O(1/S) pairwise co-voting term, so the mean is learned identically and S is learned only from annotator *pairs*. (b) Thresholded decisions on P_maj at 0.5 do not depend on S; the dependence of the majority decision on S grows with panel size (n = 3: weak; n → ∞: P_maj → P(θ > 0.5), which depends strongly on S). (c) Fisher information about S per sample against n, and what that means for 3-rater medical datasets. (d) Three implementation traps that set S instead of the data: the α ≥ 1 floor, label-dependent class weights, and an unreachable KL prior. These generalise to any evidential/second-order head trained on soft labels.
2. **A corrected recipe (B²-r)** that removes the three traps, with each change tested as a pre-registered hypothesis (§3).
3. **A controlled identifiability experiment on real data**: CIFAR-10H with the training votes subsampled to n_train ∈ {1, 3, 10, all}, evaluated against the full held-out panel. This shows how many raters per training image a per-sample spread needs.
4. **Use-case evaluation on CVS** (Endoscapes2023, CVS Challenge): selective confirmation ("decline to confirm when the panel would plausibly split") and annotation triage, against equally tuned hard BCE, soft BCE, Global-S and a seed ensemble.

**Claims the paper CAN make (if the predictions hold).**
- B²-r and soft BCE are *equivalent* on discrimination (ACC/BACC/mAP/F1/AUROC) within a pre-set margin; that equivalence is predicted by the theory and reported as such, not as a failure.
- B²-r gives a better *predictive distribution over votes* (vote NLL, P(split) calibration) than soft BCE and than Global-S where the spread is identified, with the size of the gain tied to n and to the encoder regime.
- Correct class handling (unweighted likelihood + threshold chosen on validation) loses no BACC and restores calibration of P_maj.
- The +1 floor and the KL bias S in a predictable direction; removing them improves vote NLL.
- Under a frozen foundation-model encoder, per-sample (α₀, α₁) carry information that a global spread does not (claim conditional on H4/H5).
- Whether Beta-derived scores improve risk–coverage for CVS confirmation over mean-only confidence, with a CI (either answer is publishable; the sign is not assumed).

**Claims the paper CANNOT make.**
- That the Beta-Binomial loss itself is new: Dirichlet-Multinomial likelihoods on vote counts are already used for Galaxy Zoo (§7).
- That B² is a better classifier than soft BCE, or state of the art on CVS. MICCAI numbers are not reused.
- That S or 2/S is epistemic uncertainty, "evidence", or an out-of-distribution signal. It is the predicted heterogeneity of the panel's response among look-alike inputs (01_foundations §3, §6).
- That S measures image ambiguity independently of the panel. With a fixed panel (Endoscapes; sequential, hierarchical marking per 01b §B.4), θ is that panel's average rate; the paper says so in one limitation paragraph.
- Anything at video level or about clinical safety of a deployed system. Frame-level only; the "decline to confirm" use case is evaluated as a frame-level selective-prediction proxy.
- Results for the spatiotemporal MICCAI model. All comparisons use one common single-image architecture.

---

## 2. Use cases: who needs (α₀, α₁) rather than the mean

**What the Beta adds, stated exactly.** From (α₀, α₁) we get, in closed form, the predicted distribution of the panel's votes. For n = 3:
- P(split) = 1 − P(k=0) − P(k=3) = 3m(1−m) · S/(S+1);
- P(unanimous) = 1 − P(split);
- P_maj = P(k ≥ 2) and P(majority agrees with the model's decision) = max(P_maj, 1 − P_maj);
- for a large panel, P(majority of the panel positive) → P(θ > 0.5) = 1 − I₀.₅(α₁, α₀), the Beta survival function at 0.5.

The mean-only versions are the S → ∞ limits (e.g. P(split) = 3m(1−m)). **The Beta changes a decision only when S varies between samples that have similar m, and the size of that effect depends on the panel size.** Numerical check done for this file (scipy, `betabinom`): at m = 0.3, as S runs over [1, ∞), P_maj for a 3-rater panel moves only between 0.286 and 0.216; for an 11-rater panel between 0.277 and 0.078; for a 51-rater panel between 0.273 and 0.001. So for 3-rater CVS data the spread mostly affects P(split) (through the factor S/(S+1), which matters only for S ≲ 5), and barely affects the majority decision. For many-rater data the spread drives the majority decision. Every use case below is chosen so that this prediction can be checked.

**Default deployment condition for all use cases:** a frozen public foundation-model encoder (DINOv3 ViT-B/16) with a light MLP head. This is the realistic setting for small surgical datasets, and it is the setting in which "images the model treats as alike" exist, so S has a defined meaning: *how much a panel's response varies among frames that look like this to the encoder* (01_foundations §3.4). LoRA and full fine-tuning are evaluated as contrasts (H5), not as the default.

### U1. Selective CVS confirmation: "decline to confirm when the panel would plausibly split" (primary)
- **Who.** An intraoperative CVS assistant, or a video-based assessment tool, that must say *criterion met / not met / refer to a human*. The user is the operating surgeon or a reviewer.
- **Decision.** Per frame and criterion: abstain when the predicted panel response is unreliable. Two pre-specified abstention scores: (a) Chow's rule with respect to the panel majority, abstain on the lowest P(majority agrees) = max(P_maj, 1 − P_maj); (b) abstain on the highest P(split).
- **Why the Beta might matter.** (b) uses S directly; (a) uses it weakly at n = 3 (above). The honest expectation is a small gain on CVS and a large one on many-rater data (H6).
- **Evaluation (Round 3, frame level).** Risk–coverage curves and AURC with risk = disagreement with the majority vote; accuracy at 80% and 90% coverage; and the **split rate among accepted frames at 80% coverage** (how often the system commits on a frame where the experts in fact split). Endoscapes2023 and CVS Challenge. CIFAR-10H with target = full-panel majority serves as the large-panel contrast. Compared with soft-BCE confidence |p − 0.5|, Global-S, the B²-r mean-only plug-in, and 5-seed ensembles.

### U2. Annotation triage for CVS dataset builders (secondary)
- **Who.** Curators of multi-rater surgical datasets (the CVS Challenge and Endoscapes teams are the obvious examples), deciding where to spend expert time.
- **Decision.** Which frames get a 4th rater or a consensus discussion (high P(split)), and which could be single-annotated (high P(unanimous)).
- **Evaluation (Round 3).** (a) Endoscapes and CVS Challenge: AP and AUROC of predicted P(split) for the observed split (k ∈ {1, 2}), and the **catch rate**: the share of all split frames found when the top 20% of frames by P(split) are sent for review. (b) CIFAR-10H, trained on 3 votes per image (n_train = 3, H4): mean squared error between predicted P(split among 3 raters) and its exact value under the full ~51-rater panel (a hypergeometric probability computable from the counts), plus a small **vote-allocation simulation**: with a fixed budget of extra votes, giving them to the images with the highest predicted P(split) against giving them uniformly, and measuring the error of the resulting θ estimate against the full panel. Analysis-only, no extra training.

### U3. Panel-calibrated probabilities for audit and teaching (secondary)
- **Who.** Post-operative CVS documentation review and surgical education, where a reviewer looks at a frame and wants "how likely would an expert panel confirm this criterion, and would they agree with each other?".
- **Decision.** Report P_maj and P(split) as probabilities, and flag frames for human review. The requirement here is calibration, not ranking.
- **Evaluation.** ECE and Brier of P_maj against the majority vote; held-out vote NLL; Brier of P(split); on CIFAR-10H, θ-MSE against the full panel. These are the endpoints of H1–H4.

### Seeds from 01b §D that are deliberately not used as claims
- **Label auditing / posterior-predictive checks (01b D.4).** The known Endoscapes artefact is per-rater and cross-criterion (sequential, hierarchical marking; 01b §B.4). A per-criterion Beta for the panel total cannot see it, and there is no ground truth for "label errors" at frame level. At most one exploratory figure (frames with the lowest predicted p(k_obs)); no claim.
- **Variable n (01b D.5).** None of the three datasets has materially variable n (CVS: 3 throughout; CIFAR-10H: 47–63). The question is covered more cleanly by controlled vote subsampling (H4).
- **OOD detection.** Out of scope: the Beta-Binomial likelihood gives S no reason to fall away from the training data (01b §C).

---

## 3. Method hypotheses

Seven hypotheses. H0 is the anchor the theory predicts; H1–H3 fix the three traps found in Step 1; H4–H6 test whether the per-sample spread is worth having and for what. The recipe B²-r = H1 + H2 + H3 applied together. Every arm uses the same single-image architecture: encoder → [CLS] embedding → MLP head (768 → 256 → 2 per criterion, GELU, dropout 0.1). Heads differ only in the output layer and loss.

### H0 (anchor). The mean is learned like soft BCE, so discrimination ties
- **Change.** None. This is the comparison B²-r against soft BCE under identical class handling, threshold rule and model selection.
- **Justification.** L_BB(k; m, S) = n·BCE(k/n, m) − log C(n,k) − c(k,m)/S + O(S⁻²) (01_foundations §4). For large S the gradient with respect to m is n times the soft-BCE gradient. Caveat: CVS spreads are small (moment-matched marginal S = 1.0–2.3), and at small S the expansion is loose: the Beta-Binomial then down-weights split votes relative to unanimous ones when fitting m. A small difference in m is possible, with no predicted sign.
- **Prediction.** ACC, BACC, F1, mAP and AUROC (macro over criteria) of B²-r and soft BCE differ by ≤ 1 pp, with the 95% CI of the difference inside ±1.5 pp, on all three datasets (frozen encoder). Both beat hard BCE on mAP/AUROC on the CVS datasets (soft labels carry information), by an unknown margin.
- **Refuted if** the CI of any primary discrimination difference lies wholly outside ±1.5 pp on any dataset. If B²-r wins, the O(1/S) term acts as a useful reweighting, which is a finding to report, not to hide.

### H1. Class handling: unweighted likelihood, threshold on validation
- **Change.** Every loss is trained unweighted. Each criterion's decision threshold is chosen on validation by one rule for every method: the threshold that maximises validation BACC. The ACC, BACC and F1 numbers are all reported at that threshold. The weighted arms use one identical sample weight across all losses (w± = N/(2N±) by majority label, applied as a per-sample weight, never as `pos_weight`), so H1 also removes the MICCAI imbalance between baselines (5.4:1 against 3.2:1). For the legacy B²-M arm only, the closed-form prior-shift correction q(k) ∝ q′(k)/w(y_maj(k)) is applied at inference as a secondary readout.
- **Justification.** (i) With label-dependent weights the expected loss is the NLL of q′(k) ∝ q(k)·w(y_maj(k)), which is not a Beta-Binomial: θ is shifted (true m 0.15 → 0.25; 0.33 → 0.52) and S becomes a function of m (01_foundations §5, §3.2). (ii) Balanced accuracy, ½(TPR + TNR), is maximised pointwise by predicting positive iff P(y_maj = 1 | x) > π, the majority prevalence. A calibrated unweighted model plus a threshold therefore reaches the BACC optimum without distorting θ; weighting with w₊/w₋ = (1 − π)/π moves the 0.5 threshold of a hard-BCE model exactly to p > π. The two routes coincide for a well-specified head. They can differ only when the weights change the learned *features*, which is impossible with a frozen encoder and possible under fine-tuning.
- **Prediction (frozen).** (a) BACC of unweighted + threshold is within 1 pp of the weighted arm for each loss (CI of the difference above −1.5 pp). (b) ECE of P_maj falls by ≥ 40% relative, and the mean predicted θ on test is within ±0.03 of the observed mean vote fraction (the MICCAI-era runs gave 0.28–0.47 against 0.13–0.28). (c) |Spearman ρ(S, m)| falls (the weighting-induced S-tracks-m pattern disappears).
- **Refuted if** the weighted arm beats unweighted + threshold in BACC by > 1.5 pp with the CI excluding 0 on ≥ 2 datasets, or if ECE does not improve. Under fine-tuning (Round 2, reported only) a BACC benefit of weights would point to a feature effect, not a contradiction of (ii).

### H2. Remove the `+1` floor; softplus stays
- **Change.** α_c = softplus(f_c(x)) + ε with ε = 10⁻³, replacing softplus(·) + 1 + 10⁻⁶. The loss is computed in float32 on the GPU (moving it off the CPU), with gradient-norm clipping at 1.0 for all arms. **Fallback ladder, fixed in advance:** if, on any dataset, more than 1 of 5 seeds gives a non-finite loss, or the validation vote NLL rises by more than 10% after epoch 2: (F1) ε = 0.1; if that still fails, (F2) a mean–precision head, m = sigmoid(a), S = 0.5 + softplus(b), α₁ = mS, α₀ = (1 − m)S. F2 floors S only, still allows α < 1 at skewed m, and is an equivalent parametrisation of the same Beta (E.1 is respected).
- **Justification.** α ≥ 1 forces S ≥ 1/min(m, 1 − m): at m = 0.1, S ≥ 10. It excludes the U-shaped Betas that describe a panel that is usually unanimous but whose verdict is hard to predict. The data ask for them: the moment-matched marginal S is 2.3 / 1.0 / 1.5 (C1/C2/C3 train), below the floor wherever m < 0.5. Step 1 found 20–36% of C2/C3 predictions on the floor and a constant S beating the predicted S on held-out vote NLL in every run. Stability: lgamma(α) ≈ −log α as α → 0, so ∂L/∂α ≈ 1/α, which is at most ≈ 10³ at α = ε, and softplus′ ≤ 1. With clipping this is bounded. The `+1` comes from EDL's uniform-prior "evidence + 1", which has no role in a likelihood model.
- **Prediction.** On Endoscapes and CVS Challenge: (a) the share of predictions with min(α₀, α₁) within 0.05 of the floor falls from 20–36% to < 5% on C2/C3; (b) test vote NLL improves by ≥ 0.02 nats per frame per criterion on C2 and C3 (Step 1's gap between predicted and constant S was 0.14 nats on val C2); (c) the Brier score of P(split) improves. On CIFAR-10H the effect is smaller, because most images are near-unanimous, but the sign is the same. No seed needs the fallback.
- **Refuted if** test vote NLL does not improve (CI includes 0) on both CVS datasets, or if training is unstable even with F2.

### H3. Drop the KL; calibrate the spread on held-out data instead
- **Change.** λ = 0. After training, fit **one scalar c per criterion** on validation by maximising the validation vote likelihood of Beta(c·α₁, c·α₀). This keeps every m and rescales every S. One KL arm is kept for the comparison, with a corrected, reachable prior: π = mean vote fraction (not majority prevalence), ν = moment-matched marginal S of the training votes, λ = 0.05.
- **Justification.** The KL is applied per sample to the predicted distribution, so it is a penalty that does not fade with more data, not a prior over parameters. With the MICCAI settings the prior (α₁ᵖ = πν = 0.22–0.36) is unreachable under the floor, the KL caps S at ≈ 7–10 and pulls m toward π (01_foundations §3.2, A5–A6). Because S is identified only through annotator pairs, its overall level is the weakest-identified part of the model. Fitting that level on held-out data is the standard remedy for a weakly identified scale (it is the spread analogue of temperature scaling), and it is consistent, because c maximises a held-out proper score. The Global-S baseline gets the same one-parameter calibration, so the comparison is fair.
- **Prediction.** (a) Frozen encoder: fitted c lies in [0.7, 1.4] (the head does not over-fit S). (b) Test vote NLL of λ = 0 + c is not worse than the corrected-KL arm: CI of (no-KL − KL) has an upper bound below +0.005 nats on every dataset. (c) The KL arm shows m pulled toward π: a larger |m − m_noKL| at extreme m, and an unchanged BACC.
- **Refuted if** the corrected-KL arm beats λ = 0 + c on test vote NLL with the CI excluding 0 on ≥ 2 datasets.

### H4. A per-sample spread beats one global spread only where the data identify it (the core claim)
- **Change.** None to the recipe. The test: B²-r against Global-S (soft BCE + S* fitted on validation) and against soft BCE with a Binomial plug-in (S = ∞), on held-out vote-level scores. On CIFAR-10H the training votes are subsampled to n_train ∈ {1, 3, 10, all (~51)}. The validation votes are subsampled to the same n, since a practitioner has the same number of votes in validation; test is always scored against the full panel.
- **Justification.** Fisher information about log S per sample is 0 for n = 1, 0.014 for n = 3 and 0.35 for n = 50 at S = 10 (01_foundations §3.1). For n = 1 the Beta-Binomial pmf reduces to Bernoulli(m), so S receives no gradient and stays at initialisation; c and S* are then unidentified (the validation likelihood is flat in S), and both arms fall back to S = ∞. B²-r nests Global-S (a constant spread head), so its population optimum is never worse; in finite samples the extra variance can make it worse at small n.
- **Prediction.** (a) CIFAR-10H, n_train = all: B²-r beats Global-S on per-image full-panel test vote NLL by ≥ 0.02 nats, with the CI excluding 0, and reduces the MSE of P(split among 3) by ≥ 10%. (b) The B²-r − Global-S margin increases with n_train; it is 0 within CI at n_train = 1, and at n_train = 3 it is at most a third of the margin at n_train = all. (c) Endoscapes and CVS Challenge (n = 3, frozen): the margin is small (≤ 0.01 nats per frame per criterion) and may not exclude 0. Soft BCE with a Binomial plug-in loses to both on every dataset (it predicts too few unanimous panels).
- **Refuted if** (a) fails: then per-sample α carry no information beyond one global spread even with 51 raters and a frozen foundation encoder, and the paper becomes the "soft BCE + global spread is enough" story (§4.8, outcome C). (b) fails if the margin does not grow with n.

### H5. Encoder regime decides what S means
- **Change.** The same encoder (DINOv3 ViT-B/16, public weights; no majority-vote pre-adaptation) in three regimes: frozen (cached embeddings, MLP head), LoRA (rank 8 on the qkv projections of all blocks; the repo's `inject_lora_into_dinov3_qkv` already exists), and full fine-tuning. The median validation S is logged at every epoch. One additional arm, run unconditionally because it is cheap: **held-out spread head** for full fine-tuning. Train encoder + head with B²-r on 80% of the training videos (part A). Then freeze everything except the spread output (m fixed, S refit) and fit it on the remaining 20% (part B), with selection on validation.
- **Justification.** 01_foundations §3.4. With a frozen encoder, look-alike groups exist and S estimates the panel heterogeneity among embedding neighbours. With a trainable encoder, each training image can be isolated, its optimum is S → ∞, and S grows each epoch (Step 1: end-to-end validation S 5 → 16 in 4 epochs). The test-time S then encodes training duration and distance from training data, not panel heterogeneity. Post-hoc c can correct the overall level but not the per-sample ranking. Fitting the spread on data the encoder did not see restores the frozen-encoder meaning.
- **Prediction.** (a) Fitted c: frozen in [0.7, 1.4]; LoRA < 0.7; full fine-tuning < 0.5. (b) Median validation S rises monotonically over epochs under full fine-tuning in ≥ 4 of 5 seeds, and is flat (within ±20% after epoch 3) when frozen. (c) The H4 margin (Global-S NLL minus B²-r NLL) is largest when frozen, smaller with LoRA, and ≤ 0 with full fine-tuning; the held-out spread head recovers at least half of the frozen margin. (d) Discrimination (mAP) is higher with fine-tuning on the CVS datasets. This is context, not a claim.
- **Refuted if** the full-fine-tuning margin is at least as large as the frozen margin (CI excluding the frozen-larger ordering), or validation S does not grow under fine-tuning.

### H6. Report panel probabilities, not 2/S; use them for selective prediction
- **Change.** The model reports P(majority agrees with the decision), P(split) and P(unanimous) (and P(θ > 0.5) for a large panel). It never reports 2/S, which is not on any decision scale. Selective prediction uses the two pre-specified scores of U1.
- **Justification.** (i) For 0–1 loss against the panel majority, the optimal abstention rule ranks by max(P_maj, 1 − P_maj) (Chow's rule). (ii) Numerical check (§2): at n = 3, P_maj is nearly a function of m, so ranking gains over a mean-only score require S to vary strongly among samples with similar m; at n = 51 the spread dominates P(majority positive). (iii) 2/S has no direction: the same S means different panel behaviour at m = 0.05 and at m = 0.5.
- **Prediction.** (a) Endoscapes and CVS Challenge: AURC of B²-r's P(majority agrees) differs from the mean-only plug-in (computed from B²-r's own m) by < 0.005 in absolute value, with the CI including 0. It is within the same margin of soft-BCE confidence. P(split) AUROC for the observed split is at most 0.03 above the plug-in 3m(1−m). (b) CIFAR-10H, n_train = all, target = full-panel majority: the confidence based on P(θ > 0.5) lowers AURC by ≥ 10% relative to the mean-only confidence. (c) 2/S is the worst selective score of those tested on every dataset. (d) Not a hypothesis, reported: 5-seed ensembles (soft BCE and B²-r) improve AURC over single models, and the B²-r ensemble is best.
- **Refuted if** (b) fails: then the Beta does not help decisions even where the spread is identified. If (a) shows a clear gain on CVS (CI excluding 0, > 0.005), the n-dependence argument is incomplete, which is worth reporting.

---

## 4. Validation plan

### 4.1 Datasets

All three are used as **single images** with per-image vote counts available **for training** (E.1). Sources were checked on the web on 2026-10-08 (links in §7). Anything not confirmed is marked UNVERIFIED.

| | **Endoscapes2023 (CVS201)** | **SAGES CVS Challenge** | **CIFAR-10H** |
|---|---|---|---|
| Content | 201 LC videos; 11,090 frames with CVS labels, one every 5 s from the region of interest (58,813 frames at 1 fps in total) | 1,000 LC clips of 90 s from 54 institutions in 24 countries; 18 frames per clip (one every 5 s) | The 10,000 CIFAR-10 *test* images (32×32) |
| Votes per sample | 3 per frame and criterion, **fixed panel** of 3 experts | 3 per frame and criterion, **drawn from a pool of 20 qualified surgeons** (passed a competency exam). Identities vary across clips | 51 on average (range 47–63), from 2,571 Mechanical Turk participants; each did 200 images (20 per class). 511,400 labels in total; 14 participants excluded for failing attention checks |
| Label format | `ds` = mean of the 3 raters per criterion ∈ {0, ⅓, ⅔, 1} → k = 3·ds. Per-rater labels in `all_metadata.csv` ("CVS annotations from all annotators"); column names UNVERIFIED. The user confirms rater identities exist | Per frame: 3 binary criteria × 3 annotators, plus a self-reported annotator confidence on [0, 1] (per clip or per annotator: the challenge paper is inconsistent). File format and column names UNVERIFIED (only visible to registered participants) | `cifar10h-counts.npy` (10,000 × 10 vote counts), `cifar10h-probs.npy`, and `cifar10h-raw.csv` with `annotator_id`, `chosen_label`, `cifar10_test_set_idx`, reaction times, etc.; images from torchvision CIFAR-10 test |
| Official split | 120 / 41 / 40 videos (6,960 / 2,331 / 1,799 labelled frames) | 700 training / 300 test clips; **test labels withheld** (evaluated on Grand Challenge). No official validation split | None for CIFAR-10H |
| Split used here | Official | Within the 700 training clips: **490 / 105 / 105 clips** (70/15/15) by clip, stratified by country and by clip-level C1–C3 status; seed fixed and the split file committed. If the 300 test clips are released with labels, they replace our test split | Within the 10k: **6,000 / 2,000 / 2,000** images, stratified by CIFAR-10 class; fixed seed. The CIFAR-10 50k training set (one hard label each) is not used |
| Binary formulation | Native: 3 independent heads (C1–C3) | Native: 3 heads (C1–C3) | **10 one-vs-rest heads**: head c models "a rater chooses class c", k_c = votes for c, n = total votes on the image. The marginal of one category of a multinomial vote is Binomial given θ_c, so the fixed Beta-Binomial model holds exactly per head. This mirrors C1–C3 (independent heads, positive rate ≈ 10%) and uses all 10k images. Primary numbers are macro-averaged over the 10 heads |
| Access | Public, `s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip`; CC BY-NC-SA 4.0. Already installed | **Grand Challenge page: "currently not publicly available"; "any use of the dataset outside the official MICCAI challenge is strictly prohibited"; participants "agreed not to publish the data or any methods developed using the data" until the organisers' final manuscript.** The challenge paper (arXiv 2509.17100, v2 dated 28 Jan 2026) matches the rebuttal's "released 28/01/26". Whether a public release or a licence allows publishing results is **UNVERIFIED and a blocker**: confirm in writing with the organisers (info@cvschallenge.org) before Round 0 ends | Public, `github.com/jcpeterson/cifar-10h` (also TFDS `cifar10_h`). Licence: a LICENSE.txt exists; its type is UNVERIFIED |
| Dataloader work | Small: single-frame mode exists (`NoTemporal`, `b2_dinov3_notemporal.yaml`). Read k from the key frame; recompute k from the source annotations (the saved MICCAI result JSONs store rounded labels) | Medium (~1 day): extract the 18 annotated frames per clip at 5-s intervals (timestamp alignment UNVERIFIED), aggregate the 3 per-rater labels to k, build the split file. The repo has no CVS Challenge loader on `main` | Small (~½ day): counts → k_c, n; vote subsampling from `cifar10h-raw.csv` (n_train draws without replacement per training image, fixed per seed); exact hypergeometric "3 random raters" quantities for test |
| Known issues | Fixed panel with sequential and hierarchical marking by one rater (01b §B.4), which induces cross-criterion dependence; limitation paragraph plus one descriptive table (per-rater positive rate per criterion, conditional on the other criteria). Val/test prevalence shift (C1 0.163 vs 0.235; C3 0.167 vs 0.279): affects every method's val-chosen threshold equally; threshold-free metrics are reported too | ~7% robotic cases (keep; report). Confidence semantics ambiguous: **not used**. Reported annotator count 20 vs 19 in different sections of the challenge paper. Clip-level prevalence C1 41%, C2 60%, C3 40%; frame-level prevalence unknown until loaded | Low ambiguity overall: most images are near-unanimous, so each one-vs-rest head is very imbalanced and the disagreement sits in a minority of images (exact share computed in Round 0). Images are 32×32 and must be upsampled to 224. Whether DINOv3's pretraining data contain CIFAR images is UNVERIFIED; low concern because all arms share the encoder |

**Why CVS Challenge matters beyond "a second CVS dataset".** Its raters are drawn per clip from a pool of 20. That is much closer to the "randomly selected expert" reading of θ than Endoscapes' fixed panel. Endoscapes against CVS Challenge is therefore a natural, cheap check of whether the panel structure changes any conclusion (01b asks to judge assumption violations only by that).

**Optional: minimal synthetic check (01b §C).** One CPU script, about a day of the user's time: x ∈ ℝ², θ = sigmoid(g(x) + s(x)·h) with a hidden h ~ N(0, 1), S_true(x) known by quadrature, n ∈ {1, 3, 10, 50}, N ∈ {1k, 10k}, MLP heads, Bayes reference. It is the only place with a known *per-sample* S_true. Recommendation: **include only if the target is TMLR** or if the CIFAR-10H subsampling result (H4b) is ambiguous. It is never on the critical path.

### 4.2 Common protocol (identical for every arm)

- **Encoder.** DINOv3 ViT-B/16 from public weights; no majority-vote pre-adaptation anywhere (E.1). Input 384×384 for the surgical datasets (as in the repo) and 224×224 for CIFAR-10H.
- **Frozen regime.** Embeddings are computed once per dataset and cached, with no augmentation. Heads train for 50 epochs on the cache, which takes minutes per run. This makes Round 1 nearly free and removes encoder randomness from the loss comparison.
- **LoRA / full fine-tuning.** 10 epochs, AdamW, cosine schedule, the same light augmentation for all arms (small random-resized crop and colour jitter; no flips on surgical frames, because the anatomy has a side).
- **Hyperparameters.** Grid per loss family (hard BCE / soft BCE / B²): head lr ∈ {1e-4, 3e-4, 1e-3} × weight decay ∈ {1e-4, 1e-2}; for fine-tuning, encoder lr ∈ {1e-5, 3e-5} (full) or {1e-4, 3e-4} (LoRA). One seed, selected on **validation mAP** (threshold-free and common to all losses). Variant arms (weighted, +1 floor, KL) inherit their family's setting.
- **Model selection.** The epoch with the best validation mAP, the same rule for all arms. Validation S per epoch is logged for H5.
- **Post-hoc steps, all on validation.** Threshold per criterion (maximise validation BACC); S* for Global-S; c for B²-r; the legacy prior-shift readout for B²-M. All are fitted on the validation split only.
- **Test discipline.** Every run writes per-sample predictions (α or p, k, n, sample id, video/clip id) for validation and test. Test files are not opened before Round 3, after all training is frozen.
- **Code.** All work on a new branch `journal-extension`; never on `main`. λ, ε, the weighting mode and the loss are config fields; the loss moves to the GPU.

### 4.3 Baselines (01b §E.4)

| Arm | Training target | Spread | Class handling |
|---|---|---|---|
| Hard BCE | majority vote | none (Binomial plug-in of P_maj is reported for completeness) | unweighted + val threshold (weighted variant in H1 only) |
| Soft BCE | k/n (count-weighted by n_i/mean(n), which only matters if n varies) | none: Binomial plug-in, S = ∞ | same |
| **Global-S** | soft BCE, then one S* per criterion fitted on validation | constant | same |
| Deep ensembles | 5 seeds of soft BCE, and 5 seeds of B²-r (free: they are the seed runs) | ensemble mean (+ disagreement between members, for U1) | same |
| B²-r (proposed) | vote counts, Beta-Binomial NLL | per sample, ×c from validation | same |
| B²-M (legacy) | MICCAI recipe: +1 floor, weights, KL λ = 0.05, ν = 2 | per sample | MICCAI weights; prior-shift readout as secondary |

All arms share the encoder, head width, tuning budget, selection rule and threshold rule. The ensembles are reported next to single models, never in place of them.

### 4.4 Metrics (pre-specified)

*Discrimination, against the majority vote* (per criterion and macro): ACC, BACC, F1 at the validation-chosen threshold; mAP and AUROC (threshold-free).

*Panel prediction* (the paper's primary endpoints):
- **vote NLL**: mean of −log p(k_i | n_i, model) on test, per criterion. Arms without a spread use Binomial(n, m). On CIFAR-10H it is scored against the full panel, per image.
- ECE (15 equal-mass bins) and Brier of P_maj against the majority vote.
- P(split): Brier, plus AUROC/AP for the observed split (n = 3 datasets); MSE against the exact full-panel "3 random raters" value (CIFAR-10H).
- θ-MSE: (m − k/n)² against the full panel (CIFAR-10H).

*Use cases:* AURC; accuracy at 80% and 90% coverage; split rate among accepted frames at 80% coverage (U1); catch rate of split frames at the top 20% (U2); the vote-allocation simulation (U2, CIFAR-10H).

*Diagnostics (not endpoints):* fitted c and S*; share of α near the floor; Spearman ρ(S, m); median validation S per epoch.

### 4.5 Statistics (01b §E.6)

- **5 seeds** per configuration. Report mean ± sd across seeds.
- **Paired differences with a 95% CI from a two-level bootstrap** (2,000 resamples). Each resample draws test **videos/clips** with replacement (Endoscapes, CVS Challenge) or **images** stratified by class (CIFAR-10H), and *independently for each arm* draws 5 seeds with replacement. Seed pairing across arms is arbitrary, so seeds are not paired. The difference is computed on the same resampled test set. Clustering by video is a resampling unit only; there is no per-video evaluation.
- **Equivalence** (H0, H1a) is declared when the CI lies inside the stated margin; **superiority** when it excludes 0 in the predicted direction.
- **Primary comparisons, fixed now (about ten CIs per dataset):** P1 (H4) B²-r vs Global-S, vote NLL, frozen; P2 (H0) B²-r vs soft BCE, BACC and mAP, frozen; P3 (H1) unweighted vs weighted B²-r, BACC and ECE; P4 (H2) B²-r vs B²-r + floor, vote NLL; P5 (H3) B²-r vs corrected-KL arm, vote NLL; P6 (H6) AURC, P(majority agrees) vs mean-only plug-in; P7 (H5) P1's margin by regime. All other numbers are secondary and descriptive. No p-values. Because roughly one in twenty CIs excludes 0 by chance, **a claim needs the predicted sign on at least 2 of the 3 datasets**, except H4a and H6b, which concern CIFAR-10H by design.

### 4.6 Experiments: three compute rounds plus one analysis round

Budget assumptions: 3–4 A6000s; 10-epoch fine-tuning ≈ 1.5 h (Endoscapes), ≈ 2 h (CVS Challenge, about 25% more frames), ≈ 0.5 h (CIFAR-10H at 224²); frozen-head runs ≈ 3 min. The binding costs are the user's time and Claude usage, so each round is launched as one batch from a single config list and is not changed mid-round. A bug found in a round means fixing it and re-running the whole round, never patching individual cells.

#### Round 0: build and freeze (user ≈ 5 days; GPU ≈ 4 h; no results looked at)
- Branch `journal-extension`. Config fields: loss ∈ {hardBCE, softBCE, B2}, ε, λ, KL prior mode, weighting mode; loss on GPU; F2 mean–precision head behind a flag.
- Loaders: Endoscapes single-frame with exact k (and per-rater labels for the limitation table); CVS Challenge frames + split file (**only after the licence is confirmed**); CIFAR-10H counts, raw votes and subsampling.
- Embedding cache script (frozen regime). One post-hoc and metrics script: S*, c, thresholds, all §4.4 metrics, the two-level bootstrap, and per-sample prediction files.
- Smoke test: every arm, 1 seed, 2 epochs, **validation only**. Checks: finite losses at ε = 10⁻³ (otherwise trigger the H2 fallback ladder now, before Round 1), timings, file formats. Also compute descriptive label statistics (share of split votes per criterion, CIFAR-10H share of non-unanimous images).
- Optional: the synthetic script (§4.1), CPU only.
- **Exit criterion:** this document's §4.7 success criteria frozen (committed to the branch with a date) before Round 1 starts.

#### Round 1: frozen encoder, all losses (supports H0–H4, H6; Tables 1–2, Figs 2–3)
| Block | Datasets | Arms | Seeds | Runs |
|---|---|---|---|---|
| Tuning | 3 | 3 families × 6 grid points | 1 | 54 |
| Main grid | 3 | hard BCE, soft BCE, B²-r; hard BCE-w, soft BCE-w, B²-r-w (H1); B²-r + floor (H2); B²-r + corrected KL (H3); B²-M (legacy) = 9 | 5 | 135 |
| Vote subsampling (H4) | CIFAR-10H | soft BCE, B²-r at n_train ∈ {1, 3, 10} (n = all is in the main grid) | 5 | 30 |
| Post-hoc (no training) | 3 | Global-S, c-calibration, 5-seed ensembles, B²-M prior-shift readout | – | – |

≈ 220 head runs × 3 min ≈ 11 GPU-h plus < 1 h of embedding extraction: under 4 h wall-clock on 4 GPUs. User time ≈ ½ day.

#### Round 2: encoder regime (supports H5, and H0/H4 under fine-tuning; Fig 4, Table 1–2 extra rows)
| Block | Datasets | Arms | Seeds | Runs | GPU-h |
|---|---|---|---|---|---|
| Tuning (5 epochs) | 3 | hard BCE, soft BCE, B²-r × {LoRA, full} × 2 encoder lr | 1 | 36 | ≈ 24 |
| Main | 3 | hard BCE, soft BCE, B²-r × {LoRA, full} | 5 | 90 | ≈ 120 |
| Held-out spread head (H5) | Endoscapes, CVS Challenge | B²-r full fine-tuning on 80% of training videos, spread refit on the other 20% | 5 | 10 | ≈ 14 |

≈ 160 GPU-h ≈ 40 h wall-clock on 4 A6000s (about 2 days). User time ≈ 1 day of monitoring. Round 2 does not depend on Round 1's results: the hyperparameter grid is fixed, and only the head learning rate is inherited from Round 1 tuning, which is validation-only. The two rounds can therefore run back-to-back.

#### Round 3: lock and analyse (no training; supports H6 and U1–U3; Table 3, Figs 1, 5; supplement)
Open the test predictions once. Compute every §4.4 metric, the two-level bootstrap CIs for P1–P7, risk–coverage curves, the U2 catch rate and the CIFAR-10H vote-allocation simulation, the Endoscapes per-rater table, and Fig 1, which is theory-only (P_maj against S for n = 3/11/51, and Fisher information against n; from the existing `analysis/` scripts). User time ≈ 3–4 days.

**Totals:** ≈ 175 GPU-h; user time ≈ 2.5 weeks before writing (Round 0 is most of it). Round 0 plus Round 1 alone (≈ 1 week) already decide H0–H4, H6 and the paper's story (§4.8). Round 2 adds H5.

**Paper map.** Table 1: discrimination (all arms × datasets; frozen, plus fine-tuning rows). Table 2: panel-prediction metrics. Table 3: use cases (AURC, accuracy at coverage, split rate among accepted, catch rate). Fig 1: theory (n-dependence, Fisher information). Fig 2: CIFAR-10H margin against n_train. Fig 3: S against m for B²-M, + floor, + weights, + KL and B²-r (the three traps). Fig 4: validation S per epoch and fitted c by regime. Fig 5: risk–coverage on CVS and CIFAR-10H.

### 4.7 Success criteria (to be frozen before Round 1)

| ID | Endpoint (test, frozen unless stated) | Success | Datasets needed |
|---|---|---|---|
| S0 (H0) | Δ BACC, Δ mAP, B²-r − soft BCE | CI inside ±1.5 pp | all 3 |
| S1 (H1) | Δ BACC unweighted − weighted; ECE of P_maj | CI lower bound > −1.5 pp; ECE ≥ 40% lower | ≥ 2 of 3 |
| S2 (H2) | Δ vote NLL, B²-r − (B²-r + floor), C2/C3 | CI below 0; ≤ −0.02 nats | both CVS |
| S3 (H3) | Δ vote NLL, no-KL + c − corrected KL | CI upper bound < +0.005 | all 3 |
| S4 (H4) | Δ vote NLL, B²-r − Global-S | CIFAR-10H n = all: CI below 0 and ≤ −0.02 nats/image; margin increasing in n_train | CIFAR-10H (a, b); CVS reported (c) |
| S5 (H5) | H4 margin by regime; c by regime; S slope | frozen > LoRA > full; c ordering as predicted; S rises under full fine-tuning | ≥ 2 of 3 |
| S6 (H6) | Relative Δ AURC, P(θ > 0.5)-based vs mean-only | ≤ −10% (CIFAR-10H, full-panel target) | CIFAR-10H |

### 4.8 Outcome → story (decided after Round 3, not before)

| Outcome | Story | Venue (§6) |
|---|---|---|
| **A.** S4 met on CIFAR-10H **and** the H4 margin's CI excludes 0 on ≥ 1 CVS dataset (frozen) | "A per-sample panel spread is learnable from 3 raters under a frozen foundation encoder, and it helps CVS selective confirmation/triage." Full use-case section | MedIA |
| **B.** S4 met on CIFAR-10H only | "The spread needs many raters. For 3-rater CVS data, B²-r and Global-S are equivalent and both beat soft BCE on panel prediction. Here is the recipe, the identifiability rule of thumb, and the traps." Use cases shown with Global-S as the practical choice | MedIA (guidance framing) or TMLR |
| **C.** S4 fails | "B² = soft BCE + one global spread. Theory and three traps that make evidential heads look informative when they are not." Negative result, carefully argued | TMLR |

In every outcome H0–H3 are reportable, so the paper is never empty.

---

## 5. Non-image modality: not in the core

**Opinion: leave it out of the core plan.** The loss does not depend on the input type, and none of H0–H6 needs a second modality. CIFAR-10H already supplies the many-rater regime, and the CVS datasets supply the clinical one. A text or signal dataset would add a loader, an encoder and a tuning grid (≈ 1–2 days of the user's time) without testing a new hypothesis. Its only value is generality, which matters for TMLR or TPAMI and much less for MedIA.

**If it is wanted (outcome B/C with TMLR, or a reviewer request): ChaosNLI.** Verified facts: Nie, Zhou & Bansal, EMNLP 2020; **100 annotations per example**, 464,500 in total, on 4,645 examples: 3,113 from SNLI + MNLI and 1,532 from αNLI. Binary formulation: "entailment vs not" on the 3,113 SNLI/MNLI items, with one Beta-Binomial head and k = entailment votes out of 100. Use a frozen text encoder with cached embeddings and an MLP head (minutes per run), and a 60/20/20 split within the items. It pairs naturally with CIFAR-10H: the 2026 calibration-under-ambiguity work by Tao et al. uses exactly this pair, which makes it a recognised benchmark. Licence UNVERIFIED. Caveat: it is small (≈ 1,900 training items).

---

## 6. Candidate target journal

**Primary: Medical Image Analysis (MedIA).** It publishes methodological papers whose motivation and main evaluation are medical, and it tolerates a non-medical controlled experiment (CIFAR-10H) when that experiment serves the medical question. Its readers include the surgical data science community. Outcomes A and B (§4.8) both fit.

**Fallback: TMLR**, if the outcome is C or a strong B. TMLR reviews claims for correctness and evidence rather than novelty, so a careful "this is what the spread can and cannot do" paper with theory and negative results fits well. It is also fast and has open reviews. Add the synthetic check and possibly ChaosNLI in that case.

**Not recommended.**
- **IEEE TMI**: the contribution is a loss and an analysis of it, not imaging-specific; half the evidence would come from CIFAR-10H, and the CVS application is a frame-classification task. It is a weaker fit than MedIA (this is an opinion, not a stated TMI policy).
- **TPAMI**: the generic technique, a Dirichlet-/Beta-Multinomial likelihood on vote counts, is already used in other fields (§7), so the general-ML novelty bar is not met by this scope.

The plan is the same up to writing, so the venue is chosen after Round 3.

---

## 7. Novelty check (8 web searches, 2026-10-08; not a systematic review)

**What is already known**
1. **Beta-/Dirichlet-Multinomial likelihoods on vote counts with deep networks are not new.** Galaxy Zoo DECaLS (Walmsley et al., MNRAS 2022) trains CNNs on volunteer vote counts. Some galaxies have ≥ 30 votes and the rest ≈ 5. Its open `zoobot` library implements a Dirichlet-Multinomial negative log-likelihood over vote counts per question. The B² loss is the two-category case of this. **The journal paper must cite it and must not claim the loss as new.** (The MICCAI paper did claim it, implicitly.)
   [Walmsley et al. 2022, arXiv 2102.08414](https://arxiv.org/abs/2102.08414) · [zoobot losses](https://zoobot.readthedocs.io/en/latest/autodoc/pytorch/training/losses.html)
2. **Learning from soft or crowd labels.** Peterson et al. (2019) released CIFAR-10H (≈ 51 human labels per image) and showed that training on human label distributions improves generalisation and robustness. Uma et al. (JAIR 2021) survey learning from disagreement. Annotator-model approaches go back to Raykar et al. (JMLR 2010); identifiability of crowd-label models is studied, for example, by coupled cross-entropy minimisation (arXiv 2306.03288).
   [Peterson et al. 2019](https://arxiv.org/abs/1908.07086) · [CIFAR-10H repo](https://github.com/jcpeterson/cifar-10h) · [Uma et al. 2021](https://jair.org/index.php/jair/article/view/12752) · [Raykar et al. 2010](https://www.jmlr.org/beta/papers/v11/raykar10a.html) · [arXiv 2306.03288](https://arxiv.org/pdf/2306.03288)
3. **Predicting expert disagreement for referral.** Raghu et al. (ICML 2019, "Direct Uncertainty Prediction for Medical Second Opinions") predict which cases will cause high disagreement among doctors. They find that predicting this directly beats deriving it from a classifier's output. This is the closest prior art to U1/U2 and must be discussed. P(split) under a Beta-Binomial is a generative-model version of their target.
   [Raghu et al. 2019](https://proceedings.mlr.press/v97/raghu19a.html)
4. **Pitfalls of evidential / second-order losses.** Bengs, Hüllermeier & Waegeman (NeurIPS 2022) show that second-order (e.g. Dirichlet) predictors trained by loss minimisation on *single* labels do not give faithful epistemic uncertainty. This supports our "S is not epistemic" stance. The key difference: with **several votes per sample**, the second-order distribution becomes partly observable, and the Beta-Binomial NLL is a proper score for the vote counts.
   [Bengs et al. 2022](https://arxiv.org/abs/2203.06102)
5. **Higher-order calibration with k labels per example.** Ahdritz et al. (arXiv 2412.18808, 2024) define calibration for predictors that output distributions over label distributions, evaluated with "k-snapshots" (k labels per example). This is the closest theoretical framing for our identifiability analysis (information from annotator pairs, scaling with n), and it must be cited.
   [Ahdritz et al. 2024](https://arxiv.org/abs/2412.18808)
6. **Post-hoc calibration under ambiguous ground truth.** Tao et al. (arXiv 2603.22879, March 2026) show that temperature scaling fitted to majority labels under-estimates annotator uncertainty. They propose post-hoc calibrators that optimise proper scores against the full annotator distribution ("Dirichlet-Soft"), on CIFAR-10H, ChaosNLI and two dermatology sets. Our Global-S baseline and post-hoc c are close in spirit and should be positioned against it.
   [Tao et al. 2026](https://arxiv.org/abs/2603.22879)
7. **Medical multi-rater uncertainty.** This exists mainly for segmentation, e.g. label fusion for inter-rater uncertainty (arXiv 2202.07550). The searches found no CVS or surgical-video work that predicts the panel's vote distribution (not exhaustive).
   [arXiv 2202.07550](https://arxiv.org/pdf/2202.07550)

**What remains novel for the journal version** (to be re-checked by the critics)
- The **decomposition B² = soft BCE + an O(1/S) pairwise co-voting term**, and its consequences: majority decisions at n = 3 are almost independent of S, the dependence grows with panel size, and S is identified only through annotator pairs. No source in this search states it in this form for training losses.
- The **three implementation traps** of evidential heads trained on vote counts (the `+1` floor, label-dependent class weights, an unreachable per-sample KL), with their predicted and measured effects. Bengs et al. cover the epistemic claim, not these mechanics.
- A **real-data identifiability experiment** (CIFAR-10H vote subsampling against the full panel) that turns "how many raters does a per-sample spread need?" into a number.
- The **CVS application** with two CVS datasets that differ in panel structure (fixed panel against a rater pool), and use-case evaluation (selective confirmation, triage) against equally tuned and equally weighted baselines, including the direct competitor Global-S.

Not checked (search budget): Collins et al. 2022 (CIFAR-10S, soft labels from every annotator); "Soft-label training preserves epistemic uncertainty" (arXiv 2511.14117, seen in results, not read); Tanno et al. 2019; the Galaxy Zoo papers after 2022. The critics should check these.

---

## 8. Open items for the user (blocking marked ★)

1. ★ **CVS Challenge licence**: does the user's access allow publishing results outside the challenge? The Grand Challenge page says no, unless the data have been released since. Without written confirmation, drop the CVS Challenge from the plan (the plan still stands on Endoscapes + CIFAR-10H, with a weaker CVS story).
2. Endoscapes per-rater column names in `all_metadata.csv` (only needed for the limitation table).
3. Approve the defaults that cost user time: frozen-regime caching, the one-vs-rest CIFAR-10H formulation, the 490/105/105 CVS Challenge split, and leaving out the synthetic check and ChaosNLI unless the venue becomes TMLR.
