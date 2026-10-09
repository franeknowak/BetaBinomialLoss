# Journal plan — STATE

## Current step
Round 0, module a DONE (2026-10-09). Next: module b (loaders).

## Done
- Planning complete (steps 1–4, deliverables 04/05/06, 2026-10-09 amendments). See HANDOFF.md.
- 2026-10-09: branch `journal-extension` created from main, plan + CLAUDE.md committed and pushed to origin.
- 2026-10-09: module a — `b2j/{losses,heads,arms}.py`, spec `journal_plan/specs/a_heads_losses.md`, tests `tests/test_module_a.py` (25 pass, CUDA test skipped on Mac). B²-M reproduces MICCAI `total_bb_loss` exactly.

## Cluster
- 2026-10-09: `ssh -o BatchMode=yes comic ...` FAILED: "mux_client_request_session: read from master failed: Broken pipe", then no control socket (`~/.ssh/cm-fnowak@comic.cs.ucl.ac.uk:22` missing). The ControlMaster is dead; a fresh BatchMode connection cannot authenticate. User must re-open the master interactively from their own terminal. Not retried further (HANDOFF rule).
- Unknown yet: scheduler (SGE?), dataset paths, GPU queue, time limits.

## Exact next action
1. When the user confirms the SSH master is up: `ssh -o BatchMode=yes comic 'hostname; which qsub sbatch'`, then find dataset paths, GPU queue and limits; write them here.
2. Module b: loaders. Endoscapes exact k from source annotations (not the rounded MICCAI JSONs) + split-frame counts per criterion; CIFAR-10H counts, hypergeometric subsampling, ambiguous subset; MHIST (after access). Needs the raw Endoscapes annotation format: check on cluster or local `dataset/`.

## Open questions (user)
- MHIST research-use request: has it been sent? (R0 day 1)
- Keep/drop the legacy B²-M row (implemented; costs one arm).
- Module a choices: MLP head without LayerNorm (plan's 768→256→out); soft BCE not n-weighted on CIFAR-10H. See spec.
