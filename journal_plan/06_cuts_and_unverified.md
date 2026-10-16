# 06 — What was cut, and what could not be verified

## A. Cut or demoted, and why

| Item | Where it came from | Decision | Reason |
|---|---|---|---|
| Per-video evaluation, precision = 1.0 video protocol | MICCAI paper | Cut | User instruction; adds an aggregation rule unrelated to the loss question |
| Spatiotemporal model (5-frame clips, gated pooling) | MICCAI | Cut | The journal paper compares losses on a common single-image model |
| Majority-vote encoder pre-adaptation (fine-tuned SwinV2) | MICCAI | Cut | User: wrong; it biases toward majority labels (reviewer R4's concern) |
| Reusing MICCAI numbers / SwinCVS comparison | MICCAI | Cut | Not reproducible from `main` (`FT_WEIGHTS` not implemented); not the question |
| One-sided Wilcoxon over 5 seeds | MICCAI rebuttal | Cut | User instruction. Replaced by a two-level bootstrap (test clusters × seeds) with three-way verdicts and margins sized on validation |
| `+1` floor, majority-vote weights, KL | MICCAI recipe | Removed from the proposed recipe; kept as ablation arms (HT) | 01_foundations §3, §5: each sets S instead of the data |
| LoRA regime | Brief (open item) | Cut | Not wired into the model; frozen vs full fine-tuning already answers what S means per regime |
| Held-out spread head on CVS | r0 H5 | Moved to CIFAR-10H only | No effect is predicted at n = 3, so it could not refute anything there |
| Mean–precision fallback head; cross-fitted S-head | r0 H2/H5, science critic | Cut | Cost; a single ε fallback (10⁻³ → 0.1) decided in the R0 smoke test is enough |
| Synthetic generator (user's design and the minimal version) | User | Cut (restore for TMLR if reviewers ask) | Not essential. The user's design mostly tests mean recovery (S_true = ∞ in its default regime) and its OOD test is ill-posed for this loss (01b §C). CIFAR-10H with ~51 votes gives near-known θ |
| ChaosNLI / non-image modality | r0 §5 | Cut | Tests no new hypothesis; restore only for generality at TMLR |
| CVS Challenge | User suggestion | Demoted to optional add-on; **the user decides** | The Grand Challenge page forbids use or publication outside the challenge; the loader costs ~2 user-days. It enters only with written permission before R1 |
| U3 (audit/teaching use case) | r0 | Folded into the metrics | No decision differs from U1/U2 |
| U2 vote-allocation simulation | r0 | Cut | Leaks the reference votes and has a straw-man comparator (uniform allocation) |
| Equivalence margins δ = 2h, caps, equivalence / non-inferiority verdicts | plan §4.3, §5.1 | Cut (2026-10-16, user) | δ followed the experiment's noise, not what matters; every comparison is now a difference with its bootstrap CI |
| S\*(m) and S\*(m_B²) (mean-dependent spread) | r1 H4 | Cut (2026-10-16, user) | Hard to explain; Global-S answers the "one constant" objection and the P_maj-vs-m ranking shows information beyond m |
| Recall-at-fixed-precision operating point | 2026-10-16 review | Not adopted | Adds a choice (the precision level) and breaks cross-dataset consistency; FCR at recall 0.90 kept, PR curves added |
| Label auditing / posterior-predictive check | 01b D.4 | At most one exploratory figure | The known Endoscapes artefact (sequential/hierarchical marking) is per rater and across criteria, so a per-criterion panel Beta cannot see it |
| Variable-n use case | 01b D.5 | Replaced by controlled vote subsampling | No core dataset has materially variable n |
| OOD detection via S | User's synthetic notes | Cut | Nothing in the likelihood makes S fall off-distribution; not a claim the paper can make |
| Rater-specific / annotator-confusion models | 01_foundations §6 | Cut | User: the dependence is procedural; pragmatism rule (flag only what could change a conclusion). One limitation paragraph plus a per-rater table |
| Data augmentation | — | None in any arm | Keeps arms comparable and the frozen-embedding cache valid; may cost fine-tuning accuracy, but accuracy is not the claim |

## B. Not verified

**Datasets**
- **CVS Challenge:** licence for publishing outside the challenge (page says no), file format and column names, frame-timestamp alignment, annotator count (20 vs 19 in different sections of the challenge paper).
- **MHIST:** exact column header for the per-image SSA vote count; patient grouping (a train/val leak would bias model selection, c and S\*(m)); access turnaround. MHIST facts (3,152 images, 7 pathologists, official 2,175/977 split) were checked by the proposer and the science critic, not by the orchestrator.
- **Endoscapes:** per-rater column names in `all_metadata.csv`. The user confirms rater identities exist and characterised the dependence; this was not inspected here.
- **CIFAR-10H:** licence type. Whether DINOv3's pretraining data contain CIFAR images (low concern: every arm shares the encoder).

**Literature** (found by the subagents' web searches; not read in full by the orchestrator)
- Tao et al. 2026 (arXiv 2603.22879), Kohli 2026 (arXiv 2605.29797), Singh et al. (arXiv 2511.14117). These post-date the orchestrator's knowledge; their existence and summaries rest on the agents' searches.
- Ahdritz et al. 2024 (2412.18808), Raghu et al. 2019, Walmsley et al. 2020/2022, Bengs et al. 2022: recalled and searched, but not read in full.
- The classical overdispersion references (Williams 1975, Crowder 1978, Prentice 1986): cited from memory; titles and venues to be checked.
- Collins et al. 2022, Baan et al. 2022, Malinin et al. 2019, Tanno et al. 2019: named, not checked.

**Estimates**
- User-days (≈ 20–22 before writing) and GPU-h (≈ 56): planning estimates by the proposer and the scope critic; nothing has been timed. Frozen-head runs are assumed to take ~3 min, and fine-tuning ~0.5–2 h per 10 epochs.
- The equivalence margins (δ = 2h) are unknown until the Round-0 validation pilot.

**From Step 1**
- How the MICCAI Fig. 3 was produced, given that saved result JSONs store rounded labels.
- The original reason for `softplus` (user does not remember). The likely reasons — positivity with smooth gradients, avoiding `exp` overflow and `ReLU`'s α = 0 — are inferred.
- All Step-1 numbers on trained models come from MICCAI-era single-seed runs (26/05 results). The Endoscapes **test** split was analysed in Step 1; this is disclosed in the plan and no Endoscapes-only claim is made.

**Repository state**
- No code was written and nothing was committed. `journal_plan/` is an untracked folder in the working tree on `main`. Per the user's rule, create branch `journal-extension` before Round-0 coding. Moving `journal_plan/` onto that branch can happen then; it was not done now, because git operations in the connected folder cannot delete lock files.
