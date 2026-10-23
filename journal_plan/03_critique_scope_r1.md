# 03 — Scope critique of `02_proposals_r1.md` (round 1, final)

*Scope critic, 2026-10-08. I judged only what r1 says, against brief 01b §E and the repo at `main@2f59ef9` (read-only).*

## 1. Verdict

r1 resolves almost everything I raised. It merges H1–H3 into one traps hypothesis (HT), cuts LoRA, the mean–precision head (F2), U3, the vote-allocation simulation, the synthetic check and ChaosNLI, writes the plan for outcome B, sizes the margins by a pilot, makes every criterion decidable on the two core datasets, and enforces test silence. The rounds are now few, large and pre-specified. Each remaining conditional (the ε fallback, the margins, the H5d head choice, the third dataset) is decided on validation or by a date, before R1.

The science critic's additions are all cheap (≤ ½ day each, or zero user time) and each makes a central claim refutable, so they stay: S\*(m), DUP, the margin pilot, the linear head, two checkpoints, 10 frozen seeds and the CIFAR-10H re-head.

What remains:
- **Cut the CVS Challenge.** Make MHIST the fixed third dataset. This removes a 2-day loader, an external blocker and a branch in the plan.
- **Fix four small specification gaps:** DUP is undefined for training panels of fewer than 3 votes; the checkpoint used inside H6 is not stated; the re-head trains on 5× fewer images than its frozen comparator; the R0 smoke test does not say it covers the fine-tuning path.
- **The effort estimate is still about 2–3 user-days low.**

## 2. Item-by-item marks

| Item | Mark | Reason | Required change (for revise) |
|---|---|---|---|
| POS | accept | 3 contributions, outcome B as the expected story, the delta from MICCAI and the frame-level paragraph are all added | — |
| U1 | accept | Merged into H6. All three endpoints are computed post hoc from saved predictions | — |
| U2 | accept | Simulation cut, DUP added, feasibility count of split frames in R0 | (DUP fix: see R1) |
| U3 | cut | Cut as asked; one sentence in the discussion | — |
| H0 | accept | Scored on m only, pilot margin with a cap, three-way verdict | — |
| H1 (→ HT-w) | accept | Merged; soft-BCE-w and the prior-shift readout cut; ECE is now a diagnostic | — |
| H2 (→ HT-floor) | accept | One fallback, decided in the smoke test; no new head | — |
| H3 (→ HT-KL) | accept | Merged; c applied to every B² arm; diagnostics are descriptive | — |
| H4 | accept | S\*(m) makes the per-sample claim refutable; the n = 1 tie-break is pre-set; the ambiguous subset is fixed now | — |
| H5 | revise | Full fine-tuning only, with no augmentation: good. The linear head is free. The re-head is cheap (about ½ day) and needed for the mechanism claim. **But** it trains on 20% of the training images (≈ 1,200) while the frozen comparator uses all 6,000, so the margin comparison mixes training size with regime | Add one frozen-encoder B²-r head trained on the **same 20% part B**, as the re-head's comparator. It is a config change, minutes per run |
| H6 | accept | Linked to H4, so it is refutable both ways | (Checkpoint: see SEL) |
| D-ENDO | accept | Test-split disclosure; split counts; per-rater table capped | — |
| D-CVSC | **cut** | The licence explicitly forbids publication, the loader costs about 2 days, and it adds no hypothesis that MHIST does not test better (n = 7 lets H4 vary the number of raters on medical data). Keeping it as a dated option still forces both loaders to be ready and both access requests to be tracked | Make MHIST the fixed third dataset. CVS Challenge goes to future work. Drop the 2026-11-06 rule |
| D-C10H | accept | Ambiguous subset; per-vote NLL; definition of "majority positive" | — |
| D-SYNTH | cut | Cut, as asked | — |
| D-CHAOSNLI | cut | Cut, as asked | — |
| BL-HARD | accept | — | — |
| BL-SOFT | accept | No `pos_weight` on soft targets | — |
| BL-GLOBALS | accept | Kept, with S\*(m) beside it | — |
| BL-ENS | accept | Soft-BCE ensemble only; cluster-only bootstrap stated | — |
| BL-B2M | accept | One legacy row; readout cut | — |
| MET | accept | Scored quantity fixed per arm. The primary list is long, but only 5 comparison families are primary, which is what matters | — |
| STAT | accept | 10 frozen seeds cost no user time; three-way verdicts; claim rule written for 2 datasets | — |
| SEL | revise | Two checkpoints (best val mAP; best val vote NLL after the concentration step) are cheap, given per-epoch silent predictions. But H6/U1-a compares P_maj against m **within B²-r** without saying which checkpoint. If m comes from checkpoint A and P_maj from checkpoint B, the comparison measures the checkpoint, not S | One line: every within-model comparison (H6, U1, U2) uses **checkpoint B for every score**; T1 uses A; T2 and T3 use B. Store test predictions only for the epochs that either rule selects, not for every epoch |
| R0 | revise | The modules are right, but the estimate absorbs this round's additions at no cost: S\*(m), the DUP soft target, two checkpoint rules, P(split) reliability with a jackknife, the verdict function, FCR at matched sensitivity, the ambiguous subset, the pilot, and caching from a fine-tuned checkpoint (for the re-head). The smoke test does not say it exercises the image (fine-tuning) path of the shared trainer | Re-estimate at **13–14 user-days** including MHIST. The smoke test must include 1 epoch of full fine-tuning per loss, and 1 re-head pass |
| R1 | revise | One batch, ≈ 26 GPU-h, fine. **DUP in the subsampling block is undefined at n_train ∈ {1}** (an unbiased 3-rater split estimate needs ≥ 3 votes per image). The third-dataset branch adds rows that must be laid out twice | Run DUP only at n_train ≥ 3. With MHIST fixed: one arm list, one config list |
| R2 | accept | Full fine-tuning only, 2 datasets, 3 seeds, ≈ 25 GPU-h, launchable straight after R1 | (+ the same-size frozen head from H5, which goes in R1) |
| R3 | revise | It now has 10 success rows, 6 figures, 3 tables with with-c/without-c and frozen/fine-tuned rows, and the MHIST subsampling curve. 4–5 days is optimistic even with scripts written in R0 | Budget **5–6 days**. Pre-declare that F6 (qualitative) and the without-c rows go to the supplement if R3 overruns, so there is no rescoping later |
| SUCC | accept | Every row decidable on Endoscapes + CIFAR-10H; δ committed at the end of R0 | — |
| OUTC | accept | With MHIST fixed, B+ is a reachable, pre-specified route to MedIA | — |
| JOUR | accept | Rule by outcome; same experiments for every venue | — |
| NOV | accept | Prior art handled correctly; Tao et al. correctly placed as first-order | — |

**New items introduced in r1 (not among the 32):** D-MHIST accept (fixed third dataset; send the access form on R0 day 1; if access has not arrived by the R1 launch, run on the two core datasets, which decide every criterion). BL-S\*(m), BL-DUP (n ≥ 3), the linear-head arm, the margin pilot and the CIFAR-10H re-head: accept (all cheap, all make a claim refutable).

**Counts (the 32 items):** 23 accept, 5 revise (H5, SEL, R0, R1, R3), 4 cut (U3, D-CVSC, D-SYNTH, D-CHAOSNLI).

## 3. Realistic user-days (before writing)

| Round | r1 claim | Realistic |
|---|---|---|
| R0 (modules a–g + MHIST loader) | 12–13 | **13–14** (c ≈ 2.5, e ≈ 2.5; MHIST ½–1 day because of the access form, 4-h download links and an unverified column header) |
| R1 | 0.5 | 0.5 |
| R2 | 0.5–1 | 1 (first large-scale use of the image path of the new trainer) |
| R3 | 4–5 | **5–6** |
| **Total** | 17–19.5 (MHIST) | **≈ 20–22**, with writing on top (≈ 15) |

GPU ≈ 55 h is realistic and not binding.

**Critical path:** module a (K heads) → c (shared trainer) → e (metrics and bootstrap) → f (smoke test, including fine-tuning) → g (pilot, freeze δ) → R1 and R2 back to back → R3. About 9 serial days. Module e is still the single riskiest item.

## 4. Notes on items still at revise or cut (minimal changes)

- **D-CVSC (cut).** Replace the dated rule with: "Third dataset = MHIST. The CVS Challenge is future work." Delete the CVS-Challenge column and the ≈ 2-day loader from §4.1 and §4.6.
- **H5.** Add "B²-r, frozen encoder, MLP head, trained on part B only" to the R1 arm list, as the comparator for the re-head margin in S5c's secondary reading.
- **SEL.** Add the line "within-model comparisons use checkpoint B for every score" to §4.2. Store test predictions only for the epochs the two rules select.
- **R0.** Change the estimate to 13–14 days. The smoke test covers 1 fine-tuning epoch per loss and one re-head pass.
- **R1.** DUP rows only at n_train ≥ 3. Single config list, with MHIST fixed.
- **R3.** Budget 5–6 days, with the pre-declared supplement fallback (F6 and the without-c rows).
