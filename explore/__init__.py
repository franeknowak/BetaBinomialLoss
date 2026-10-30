"""Phase 2 beyond Endoscapes: the E1-E4 template on the other datasets, B²'s spread, fewer raters per image, and a
cluster bootstrap. Outputs under results/explore/ (see explore/run.py)."""
from pathlib import Path

from b2j.e1 import RESULTS

EXPLORE = Path(RESULTS) / "explore"
PHASE2_ROOT = EXPLORE / "root"           # final, final_hardbce -> ../../final*; phase2/<ds>/ written here


def sub_root(m):
    """Final runs with m raters per image on train and val (explore/subsample.py)."""
    return EXPLORE / "sub" / f"n{m}"
