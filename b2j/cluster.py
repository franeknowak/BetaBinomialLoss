"""SGE helpers. A job array splits a list of work items: task t (1-based, $SGE_TASK_ID) of N takes every N-th item
starting at t. The GPU preflight runs the B² loss forward + backward once so a broken card fails the job at once.

  python -m b2j.cluster preflight
"""
import os

import torch


def task_id() -> int:
    """$SGE_TASK_ID, or 1 outside an array job (SGE sets it to 'undefined' there)."""
    t = os.environ.get("SGE_TASK_ID", "")
    return int(t) if t.isdigit() else 1


def task_slice(items, task: int, ntasks: int):
    """The items of array task `task` (1-based) out of `ntasks`."""
    if not 1 <= task <= ntasks:
        raise ValueError(f"task {task} outside 1..{ntasks}")
    return list(items)[task - 1::ntasks]


def gpu_preflight():
    """B² loss forward + backward on the GPU. torch.lgamma/digamma on CUDA fail on some UCL nodes (GitHub issue #22;
    job 7522356): the loss keeps those on the CPU (losses._on_cpu), and this confirms the GPU path that remains works."""
    if not torch.cuda.is_available():
        print("GPU preflight skipped: no CUDA", flush=True)
        return
    from b2j.heads import MultiHead
    from b2j.losses import B2Loss
    k = torch.tensor([[0.0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 0, 3]], device="cuda")
    n = torch.full_like(k, 3.0)
    out = MultiHead(8, 3, 2).cuda()(torch.randn(len(k), 8, device="cuda"))
    value = B2Loss()(out, k, n)
    value.backward()
    assert torch.isfinite(value)
    torch.cuda.synchronize()
    print(f"GPU preflight ok: {torch.cuda.get_device_name()} | torch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["preflight"]:
        raise SystemExit("usage: python -m b2j.cluster preflight")
    gpu_preflight()
