# BetaBinomialLoss — branch `journal-extension`

This branch carries the journal extension of the MICCAI 2026 B² paper.
Before doing anything, read `journal_plan/HANDOFF.md`, then `journal_plan/STATE.md`.

- Never commit to or push `main`.
- Update `journal_plan/STATE.md` after every unit of work: current step, done, exact next action, open questions.
- Cluster: `ssh -o BatchMode=yes comic '<cmd>'` only. Never handle credentials. Login node: no compute; submit jobs.
- Test results may be looked at and reviewed freely; never use them to develop, tune or select anything (all fitting and selection on validation).
