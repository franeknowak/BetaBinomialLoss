"""Runs the phase-2 analyses outside Endoscapes.

  python -m explore.run phase2 <dataset> [e1,e2,e3,e4,spread]   E1-E4 (b2j code, unchanged) + spread check on the
                                                                main final runs -> results/explore/root/phase2/<ds>/
  python -m explore.run sub <dataset> <m>                       E1, E2, E3 + spread check on the runs trained with
                                                                m raters per image -> results/explore/sub/n<m>/phase2/
"""
import sys
import time

from b2j import e1, e2, e3, e4
from explore import PHASE2_ROOT, spread, sub_root

MODS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "spread": spread}


def _links():
    PHASE2_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("final", "final_hardbce"):
        link = PHASE2_ROOT / name
        if not link.exists():
            link.symlink_to(f"../../{name}")


def _run(ds, root, which):
    for name in which:
        t = time.time()
        print(f"######## {name} {ds} ({root})", flush=True)
        MODS[name].main(ds, root=root)
        print(f"######## {name} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    mode, ds = sys.argv[1], sys.argv[2]
    if mode == "phase2":
        _links()
        _run(ds, PHASE2_ROOT, sys.argv[3].split(",") if len(sys.argv) > 3 else list(MODS))
    elif mode == "sub":
        _run(ds, sub_root(int(sys.argv[3])), ["e1", "e2", "e3", "spread"])
    else:
        raise SystemExit(__doc__)
