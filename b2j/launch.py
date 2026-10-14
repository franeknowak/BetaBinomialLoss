"""Module f: run a block's specs. One SGE task runs every ntasks-th spec starting at its task id (1-based), so a
block can be one job or a job array: python -m b2j.launch smoke [--task $SGE_TASK_ID --ntasks N]."""
import argparse
import os
import time
import traceback

import torch

from b2j.blocks import BLOCKS
from b2j.runs import execute


def gpu_preflight():
    """Run every GPU op the losses need once; a broken card (e.g. job 7522356: 'CUDA driver error: invalid argument'
    in float64 lgamma) then fails the job at once instead of failing every B² run."""
    if not torch.cuda.is_available():
        return
    x = torch.linspace(0.5, 60, 64, dtype=torch.float64, device="cuda")
    vals = [torch.lgamma(x), torch.digamma(x), torch.nn.functional.softplus(x), torch.lgamma(x.float())]
    torch.cuda.synchronize()
    assert all(torch.isfinite(v).all() for v in vals)
    print(f"GPU preflight ok: {torch.cuda.get_device_name()} | torch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=sorted(BLOCKS))
    env_task = os.environ.get("SGE_TASK_ID", "")
    ap.add_argument("--task", type=int, default=int(env_task) if env_task.isdigit() else 1)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="print the specs and exit")
    a = ap.parse_args()
    specs = BLOCKS[a.block]()[a.task - 1::a.ntasks]
    if a.list:
        for s in specs:
            print(s.dataset, s.name)
        raise SystemExit
    gpu_preflight()
    failed = []
    for i, s in enumerate(specs):
        t0 = time.time()
        print(f"=== [{i + 1}/{len(specs)}] {s.dataset} {s.name}", flush=True)
        try:
            execute(s, verbose=False)
            print(f"    done in {time.time() - t0:.0f}s", flush=True)
        except Exception:                      # one failed run must not stop the block; reported at the end
            traceback.print_exc()
            failed.append(s.name)
    print(f"finished: {len(specs) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    raise SystemExit(1 if failed else 0)
