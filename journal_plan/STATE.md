# Journal plan — STATE

## Current step
Planning COMPLETE and amended with user decisions (2026-10-09). Handoff to a LOCAL Claude Code session for Round 0 (code) and cluster runs. See HANDOFF.md.

## Done
- Steps 1–4 (foundations, proposals, 2 critique rounds) + deliverables 04/05/06.
- 2026-10-09 amendments to 05_plan.md: +1 floor removed (not ablated); DINOv3 only; 5 seeds everywhere; recall 0.90; CVS Challenge added as 4th dataset, run last (R4); MHIST third. Pre-amendment copy in analysis/.
- HANDOFF.md written (cluster access rules, git rules, first actions). CLAUDE.md at repo root on the branch.
- NOT yet on GitHub: the cloud session's push to franeknowak/BetaBinomialLoss was refused (Claude GitHub App not installed for this repo). journal_plan/ and CLAUDE.md are untracked files in the local working tree on main.

## Exact next action (local Claude Code session)
1. Locally (repo root): `git checkout -b journal-extension && git add CLAUDE.md journal_plan && git commit -m "Add journal-extension plan and handoff" && git push -u origin journal-extension` (do NOT add 'Claude outputs/').
2. Read HANDOFF.md, then 05_plan.md §1–§5.
3. Check cluster access: `ssh -o BatchMode=yes comic 'hostname; which qsub sbatch'`.
4. Round 0, module a.

## Open questions (user)
- Send MHIST research-use request (R0 day 1).
- Keep/drop the legacy B²-M row.
