# Module g — tuning and pilot margins (validation only)

> **Pending change (2026-10-16 review, 05_plan amendment #1):** the margins step is dropped (no δ). The `pilot` block's finished runs stay as a validation preview; `results/journal/pilot/margins.md` is historical. Module h removes the `margins` command.

Code: `b2j/pilot.py` (+ blocks `tune`, `pilot` in `b2j/blocks.py`). Tests: `tests/test_module_g.py`. Job: `jobs/pilot.qsub.sh` (cluster only).

## Tuning (block `tune`, 54 frozen runs, seed 0)
- Grid per loss family (plan §4.4 g): head lr ∈ {1e-4, 3e-4, 1e-3} × wd ∈ {1e-4, 1e-2}; families hard BCE, soft BCE, B²-r; datasets Endoscapes, CIFAR-10H, MHIST.
- Choice per dataset and family: best validation mAP at checkpoint A; **within 0.001 of the best, the lower validation vote NLL per vote (checkpoint B) wins**. Reason: CIFAR-10H mAP is at ceiling (≈ 0.996 in the smoke test), so mAP alone cannot separate the grid there.
- Inheritance (plan): hard BCE-w ← hard BCE; DUP ← soft BCE; B²-r-w, B²-r+KL, B²-r linear, B²-M ← B²-r.
- Budget check: a chosen setting whose A or B evaluation falls after 80% of the epoch budget is flagged in `tune/tune_report.md` (smoke test: MHIST often chose late evaluations at lr 3e-4). Flagged settings are reported to the user before R1; no automatic extension.
- Output: `results/journal/tune/chosen.json`, `tune_report.md`.

## Pilot (block `pilot`, 20 frozen runs)
Soft BCE and B²-r with the chosen settings, seeds 0–4, on Endoscapes and CIFAR-10H (the datasets on which every claim is decided; MHIST is confirmatory and its endpoints are superiority tests).

## Margins (`python -m b2j.pilot margins`)
On validation, two-level bootstrap (2,000 resamples; videos for Endoscapes, images within class for CIFAR-10H), h = CI half-width, **δ = 2h**, compared with the plan's caps (above the cap → "CI reported, no claim"):

| margin | contrast on validation | used by | cap |
|---|---|---|---|
| bacc | Δ BACC on m at A, B²-r − soft BCE | S0, ST-w | 3 pp |
| map | Δ mAP on m at A, B²-r − soft BCE | S0 | 3 pp |
| nll | Δ vote NLL at B, B²-r − S\*(m_B²) (same runs, paired) | S4c, ST-KL | 1% of soft-BCE plug-in val NLL |
| fcr | Δ FCR at recall 0.90, P_maj − m within B²-r (paired) | S6a | 5 pp |
| ap | Δ AP of P(split), B²-r − soft BCE's 3m(1−m) | H6a / U2 | 0.05 |

Vote NLL is per frame-criterion on Endoscapes and per vote on CIFAR-10H (plan §4.3). Output: `results/journal/pilot/margins.json`, `margins.md`. The δ values are then written into plan §5, committed and dated before any R1 run.
