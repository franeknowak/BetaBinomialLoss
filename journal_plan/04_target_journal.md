# 04 — Target journal recommendation

**Recommendation: write for TMLR by default; switch to Medical Image Analysis (MedIA) only if Round 3 gives outcome A or B+ (05_plan §5.2). Do not target TMI or TPAMI.**

The rule is fixed now, before any experiment, so the choice of venue cannot steer the analysis. The experiments are identical for both venues, so nothing is lost by deciding after Round 3.

## Why TMLR is the default
- **The most likely result is outcome B.** The per-sample spread helps where there are many raters (CIFAR-10H), and on 3-rater CVS data it is equivalent to a spread that depends only on the mean. The theory in 01_foundations predicts this, and both critics independently expected it. The defensible core is therefore an *analysis* paper:
  - the soft-BCE decomposition;
  - where the spread enters decisions (only at thresholds ≠ 0.5, through per-sample variation at fixed mean);
  - how many raters identify it;
  - the three implementation traps;
  - an honest comparison against matched baselines.
- **TMLR fits that paper.** It reviews for correctness and evidence, not novelty or significance, so conditional and partly negative results are publishable when they are rigorous.
- **The novelty of the loss is gone.** Galaxy Zoo uses Dirichlet-Multinomial vote likelihoods (Walmsley et al. 2020/2022). TMLR's criteria do not penalise this; MedIA's and TPAMI's would.
- **Half the evidence is non-medical** (CIFAR-10H). That is normal at TMLR.
- **Practical:** rolling submission, open review, no page limit for the theory and the traps ablation. A single student can turn it round faster.

## When MedIA is better (outcomes A or B+)
- MedIA is the natural home for an extended MICCAI paper, and its readers include the surgical data science community.
- It needs **positive medical evidence**: either
  - a CVS gain on Endoscapes (outcome A: S-aware confirmation lowers false confirmations, or the per-sample spread beats S\*(m)), or
  - the many-rater effect replicating on MHIST pathology (outcome B+).
- With one of these, the paper becomes "a method for predicting expert-panel responses in medical imaging, and when it works", which suits MedIA.
- Without them, MedIA reviewers would see a null medical result supported mainly by CIFAR-10H, and the risk of rejection is high.

## Why not the others
- **IEEE TMI:** the contribution is a training loss and its analysis, not an imaging method. The CVS task is frame classification, and much of the evidence is non-medical. A weaker fit than MedIA in every outcome (an editorial judgement, not a stated TMI policy).
- **TPAMI:** the general technique is prior art. The remaining contributions (traps, identifiability against n, decision analysis) are solid but too narrow for TPAMI's novelty and generality bar, and there is no new method family.

## Implication for the work now
None of Rounds 0–3 changes with the venue. Two small choices keep both doors open:
- write the methods and theory so they stand without the CVS framing (needed for TMLR);
- keep the CVS use-case figures (F5, F6) ready for MedIA.

The optional add-ons cut for scope (synthetic check, ChaosNLI) are the first things to restore if TMLR reviewers ask for more generality.
