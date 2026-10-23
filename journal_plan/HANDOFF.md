# HANDOFF — for the local Claude Code session that builds and runs the experiments

*Written 2026-10-09 by the planning session (cloud); updated 2026-10-16 after the user's plan review (cloud). Read this, then `STATE.md`, then `05_plan.md`. Do not redo the planning.*

**Terminology (2026-10-23).** S = α₀ + α₁ is the Beta **concentration** (precision): Var θ = m(1−m)/(S+1), P(split, n = 3) = 3m(1−m)·S/(S+1), between-rater correlation ρ = 1/(S+1). Larger S = θ closer to m and more split panels at a given m. The docs say "concentration" (formerly "spread"); code identifiers keep the old name (`mean_spread`, `spread_rule`, `fit_spread`, `diag.spread_ablation`, figure files `F3_spread_vs_m_*`, `F4_spread_over_training_*`).

**2026-10-16 review (read first).** The user reviewed the plan and agreed three changes, recorded as the 2026-10-16 amendments at the top of `05_plan.md` (§5 rewritten to match): (1) no equivalence margins δ — every comparison is a difference with its bootstrap 95% CI; (2) the operating point stays FCR at recall 0.90, and F5 gains P_maj-vs-m precision–recall curves; (3) S\*(m) and S\*(m_B²) are dropped — concentrations are compared as plug-in → Global-S (on soft BCE's m) → B²-r. The code was brought in line by module h (2026-10-16, `specs/h_review_2026-10-16.md`); the next step is the user questions in `STATE.md` → Exact next action.

## What exists
- `05_plan.md`: **the plan to execute**. The dated amendments at the top (2026-10-09, 2026-10-14, 2026-10-16) override anything below them; the latest wins where they differ.
- `01_foundations.md` + `01b_feedback_and_step2_brief.md`: the theory and the Step-1 findings (why the recipe changed).
- `04_target_journal.md`, `06_cuts_and_unverified.md`.
- History, read only if needed: `02_*`, `03_*`, `analysis/`.

## Your job
Round 0 of `05_plan.md` §4.4 (modules a → g in order, each with a short spec and a unit test, committed before the next starts). Then launch R1 and R2 on the cluster, R3 analysis, and R4 (CVS Challenge, last). Stop and report to the user at the end of each round, and whenever a pre-specified rule is triggered (ε fallback, access delay). Round D (validation-only method development, plan amendment 2026-10-14 #1) sits between R1 and R2.

## Hard rules
1. **Git:** work only on branch `journal-extension`. Never commit to or push `main`. Commit small; push after each module.
2. **Pre-registration:** the success criteria (`05_plan.md` §5, rewritten 2026-10-16; no δ values any more) are committed and dated at the end of R0, before any R1 job is submitted, and re-committed for B²-final at the end of round D.
3. **Test discipline (relaxed by the user 2026-10-23):** test results may be viewed and reviewed freely; they are never used to develop, tune or select anything (method, hyperparameters, thresholds, checkpoints, concentrations: validation only).
4. **Validation only** for tuning, thresholds, c, Global-S (S\*), checkpoint choice and the pilot.
5. **Do not change the design mid-round.** A bug means fix it and re-run the whole affected block. Anything else goes to the user.
6. **Keep `journal_plan/STATE.md` current** after every unit of work: current step, what is done, the exact next action, open questions. The user's Claude usage is limited and sessions may be cut off.
7. Fixed scientific choices (do not revisit):
   - DINOv3 ViT-B/16 only, from public weights; no majority-vote pre-adaptation;
   - softplus + ε head (no `+1`);
   - single images; no per-video evaluation;
   - 5 seeds everywhere; recall 0.90;
   - metrics include at least ACC, BACC, mAP, F1;
   - operating point FCR at test recall 0.90 on every dataset;
   - no equivalence margins; no S\*(m) (2026-10-16).

## Cluster access (from the user; treat as facts to work with, not a script)
- From the user's Mac: `ssh -o BatchMode=yes comic '<command>'`. `comic` is an `~/.ssh/config` alias for `fnowak@comic.cs.ucl.ac.uk`, reached via ProxyJump through `tails.cs.ucl.ac.uk`. Connections are multiplexed (ControlMaster, about 4 h).
- **Never ask for, store or type credentials. Always use `BatchMode=yes`.** On an authentication failure, stop and report the error; do not try alternatives.
- `comic` is a **login node**: file operations and scheduler commands only, no compute. Real work goes in submitted jobs.
- Project root: `/cluster/project7/FranekLapChole/` (folders: `BetaBinomialLoss`, `dinov3`, `MICCVS`, `SwinCVS`, `backup`; plus `download_cvschallenge*.qsub.sh`, `remove_files.qsub.sh`).
- The scheduler appears to be SGE. Confirm with `ssh -o BatchMode=yes comic 'which qsub sbatch'`, and use the existing `*.qsub.sh` files as templates.
- Suggested flow:
  - develop on the Mac;
  - push `journal-extension`;
  - on the cluster, `git fetch && git checkout journal-extension` inside `/cluster/project7/FranekLapChole/BetaBinomialLoss`;
  - submit one job array per round block;
  - poll `qstat` sparingly;
  - pull back only the small prediction and metric files.
- Find out on day 1 where the datasets live on the cluster (Endoscapes is installed; CVS Challenge download scripts exist), and the GPU queue and job time limits. Write them into `STATE.md`.

## First actions
1. If the branch `journal-extension` does not exist yet, create it from the current `main`, commit `CLAUDE.md` and `journal_plan/` (not `Claude outputs/`), and `git push -u origin journal-extension`. Otherwise `git fetch origin && git checkout journal-extension`.
2. Read this file, `STATE.md`, and `05_plan.md` §1–§5.
3. Check cluster access with one harmless command (`ssh -o BatchMode=yes comic 'hostname; which qsub sbatch'`). Report the result.
4. Remind the user to send the MHIST access request (R0 day 1).
5. Start module a.
