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
    """Forward + backward of every variant's loss on the GPU once, so a broken card fails the job at once.

    torch.lgamma/digamma on CUDA fail on some UCL nodes (GitHub issue #22; job 7522356): the losses keep those on
    the CPU (losses._on_cpu), and this check confirms the GPU path that remains works.
    """
    if not torch.cuda.is_available():
        return
    from b2j.variants import VARIANTS, build
    k = torch.tensor([[0.0, 1, 3], [2, 3, 1]] * 4).cuda()
    n = torch.full_like(k, 3.0)
    for name in VARIANTS:
        head, loss = build(name, 8, k.cpu(), n.cpu())
        head, loss = head.cuda(), loss.cuda()
        out = head(torch.randn(len(k), 8, device="cuda"))
        value = loss(out, k, n)
        value.backward()
        assert torch.isfinite(value), name
    torch.cuda.synchronize()
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
