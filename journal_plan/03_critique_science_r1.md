# 03 — Science critique of 02_proposals_r1.md (round 1, final)

*Science critic, 2026-10-08. Judged only on what r1 says. New maths re-checked with scipy (scratch scripts). Pragmatism rule applied: only issues that could change a conclusion are kept.*

## Verdict

r1 resolves almost every round-0 issue. The ones that mattered most are fixed: H0 now scores m, the use case is rebuilt around the operating threshold, the S\*(m) and DUP baselines are in, n = 1 is an anchor, verdicts are three-way, H5's decisive arm moved to CIFAR-10H, and MHIST is the fallback third dataset.

Two errors remain that could change a conclusion:
1. **The equivalence margin rule δ = 1.5·h makes "inconclusive" likely even when the truth is exactly 0.** The per-test success probability is 0.67, and S0 alone needs four such tests.
2. **H4's per-sample test mixes mean and spread.** S\*(m) is built on soft-BCE's m, so B²-r can beat it through a better m alone.

Both have one-line fixes. With them, the plan is accept-worthy for TMLR, and for MedIA under outcomes A or B+. Marks: 22 accept, 7 revise, 3 cut (cuts agreed).

## Table

| Item | Mark | Reason (concise) | Minimal change to accept |
|---|---|---|---|
| POS | revise | Honest, well scoped. One false general statement in 1(b): "with an S that depends only on m, P_maj is strictly increasing in m". It holds for smooth S(m) (checked: S ∝ odds^b, b ∈ [−2, 3]) but **fails for a steep S\*(m)**: S = 1 at m ≤ 0.15 rising to 30 at m = 0.3 makes P_maj non-monotone. A 5-bin piecewise-linear S\*(m) can be steep | Say "for slowly varying S(m)", and check monotonicity of P_maj under the *fitted* S\*(m) in R3 (one line in the analysis script) |
| U1 | revise | The design is now right (FCR at matched sensitivity; each score paired with its risk). Problem: the threshold is set on validation for sensitivity 0.80, so the *test* sensitivity differs between scores. A score that lands at lower test sensitivity gets a lower FCR for free | Report FCR **at test recall 0.80** (a point on each score's PR curve; threshold-free, no test tuning) as primary, with the validation-threshold FCR and the realised test sensitivity as secondary |
| U2 | accept | Leaky simulation cut; DUP added; feasibility count is good. Minor: on CIFAR-10H/MHIST, DUP predicts the 3-rater split while the endpoint is k ∉ {0, n}. Score both on the 3-rater unbiased split target there | — |
| U3 | cut | Agreed | — |
| H0 | accept | Scored on m for all arms, so S cannot leak in. Three-way verdict, pilot margins, 3 pp cap with a pre-declared "no claim" branch. Refutable (fails if the CI is outside ±δ) | — (δ rule: see STAT) |
| H1 (→ HT-w) | accept | Justification corrected; ECE and mean-θ are diagnostics against validation; BACC equivalence is refutable | — |
| H2 (→ HT-floor) | accept | Gradient argument correct (matches my check). Stuck share now counts only k ≥ 1 samples. Single fallback is a reasonable cut for time: the "not testable" branch is pre-declared and the effect is predicted from data, not from stability | — |
| H3 (→ HT-KL) | accept | c on every B² arm; non-inferiority with a fail branch; ranges descriptive | — |
| H4 | revise | S\*(m), the n = 1 tie-break, n_train = 5, per-vote NLL and the ambiguous subset are all in. **New confound:** S\*(m) uses soft-BCE's m and B²-r uses its own, so "B²-r beats S\*(m)" can come from a better m (H0 equivalence on BACC/mAP does not imply equal NLL). That would wrongly confirm the per-sample claim. Minor: the H4 "fail" rule (point ≥ 0 with the CI covering 0) differs from the §4.5 definition | Add **S\*(m) fitted on B²-r's own m** (post-hoc, minutes) and make "B²-r vs S\*(m\_B²)" the per-sample test (S4a/b/c). Keep the soft-BCE S\*(m) as a secondary row. State that the H4-specific fail rule overrides §4.5 |
| H5 | accept | Decisive test (c) now on CIFAR-10H, where an effect is predicted; memorisation check plus a pre-set reading rule; augmentation confound removed. Minor: (a) needs a CI on log c, but c is fitted on validation and is not resampled, so with 3 seeds that CI cannot be computed. Make (a) descriptive. Apply the H4 m fix to the re-head margin too | — |
| H6 | revise | Linked to H4, so it can fail both ways. Inconsistency: (b) says t ≈ 0.1 on CIFAR-10H, while U1-a fixes the operating point by sensitivity 0.80 | One operating-point rule for all datasets (test recall 0.80, see U1); delete "t ≈ 0.1" |
| D-ENDO | accept | Disclosure, CIFAR-agreement rule and split-frame count added | — |
| D-CVSC | accept | Off the critical path with a dated decision rule. MHIST as the fallback is sound. Facts checked against the paper: 3,152 tiles, 7 pathologists, 2,175/977 split (1,545/630 and 617/360 HP/SSA), agreement 4/7–7/7, so k follows from label + agreement. With patient grouping unknown, a train/val leak would also bias c and S\*(m), not only model selection. Acceptable for a confirmatory dataset; say so | — |
| D-C10H | accept | Ambiguous subset fixed in advance; the "majority" definition is stated; rejecting the pair task is fine | — |
| D-SYNTH | cut | Agreed; real-data subsampling covers identifiability | — |
| D-CHAOSNLI | cut | Agreed | — |
| BL-HARD | accept | — | — |
| BL-SOFT | accept | `pos_weight` removed | — |
| BL-GLOBALS | accept | S\*(m) added (see H4 for the m-source fix) | — |
| BL-ENS | accept | Mixture predictive; test-only bootstrap stated | — |
| BL-B2M | accept | The table says both "× c" and "native output"; pick one (trivial) | — |
| MET | accept | Scored quantity fixed per arm; per-vote NLL; P(split) reliability with a jackknife noise floor | — |
| STAT | revise | Sound overall. **Margin rule error:** an equivalence test succeeds iff \|estimate\| < δ − h. With δ = 1.5h and a true difference of 0, P(success) = P(\|Z\| < 0.98) = **0.67** per test (scipy). S0 needs 4 tests (BACC and mAP on 2 datasets), so success is roughly 0.2–0.45 when the theory is exactly right. The anchor would read "inconclusive" by design | Set **δ = 2h** (P(success) = 0.95 per test at zero difference), keep the caps. The pilot's 5 seeds against the test's 10 make the test-time h smaller, so this stays conservative |
| SEL | accept | Two checkpoint rules, per-epoch spread fitting, enforced test silence | — |
| R0 | accept | Modular, pilot included, licence rule fixed | — |
| R1 | accept | Arms complete. The S\*(m\_B²) fix is post-hoc and adds no runs | — |
| R2 | accept | Scope cuts (LoRA, 3 seeds) leave H5c decidable | — |
| R3 | accept | — | — |
| SUCC | revise | Follows STAT (δ rule) and H4 (S4a–c on S\*(m\_B²)) | Apply those two changes to the table |
| OUTC | revise | Outcome A is triggered by "a CVS dataset". That lets the confirmatory CVS Challenge alone create the A story, contradicting §4.5 ("the third dataset neither creates nor removes a claim") | A requires the CVS gain on **Endoscapes** (CVS Challenge confirmatory) |
| JOUR | accept | The decision rule is sensible; B → TMLR is honest | — |
| NOV | accept | Corrected and complete enough; Tao et al. now positioned correctly | — |

## Notes on items still at revise (all one-line fixes; no re-review needed)

1. **STAT/SUCC (δ = 2h).** Success needs \|estimate\| + h < δ. With an estimate ~ N(0, (h/1.96)²) under a zero true difference: δ = 1.5h gives 0.67, δ = 2h gives 0.950, δ = 2.5h gives 0.997. Choose 2h. The substantive caps still prevent "equivalent within noise" claims when noise is large.
2. **H4/SUCC (S\*(m\_B²)).** The per-sample question is "given this model's m, does its S(x) beat the best function of m?". Fit S\*(m) on validation to B²-r's own m, and use it in S4a/b/c and in the H5 re-head margin. This removes the mean-quality confound at no compute cost.
3. **U1/H6 (test-recall FCR).** FCR at test recall 0.80 compares rankings at an identical sensitivity, which is the quantity per-sample S is predicted to change. The validation-threshold FCR remains as the deployment-realistic secondary. Use this rule for CVS, CIFAR-10H and MHIST alike.
4. **POS.** Replace "an S that depends only on m" with "a slowly varying S(m)", and check the fitted S\*(m) numerically. If P_maj is non-monotone under it, S\*(m) is an additional U1-a comparator (it is already listed).
5. **OUTC.** Restrict the A trigger to Endoscapes.
