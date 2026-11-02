#!/usr/bin/env python3
"""Long-lived GPU worker: reserves GPU memory once, then runs queued jobs (opt-in).

The memory budget is taken in the caching allocator at start and stays reserved between
jobs, so neighbours cannot take it in the gaps and no job restarts a CUDA context. `auto`
means the memory free right now minus a 1.5 GiB margin. Never kills other processes.
Submit with pqfl.train.worker.submit; stop with a file named STOP in the queue directory.
"""
import argparse
import os
import sys

from pqfl.train.worker import serve


def reserve(budget_gb: float | None, margin_gb: float = 1.5) -> float:
    import torch
    free, _ = torch.cuda.mem_get_info()
    target = free / 2 ** 30 - margin_gb if budget_gb is None else budget_gb
    while target >= 1.0:
        try:
            block = torch.empty(int(target * 2 ** 30), dtype=torch.uint8, device="cuda")
            del block          # freed to PyTorch's cache: the memory stays reserved by this process
            return target
        except torch.OutOfMemoryError:
            target -= 1.0
    raise SystemExit("Could not reserve at least 1 GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--reserve-gb", required=True, help="Number of GiB, or 'auto'")
    parser.add_argument("--poll-seconds", type=float, required=True)
    parser.add_argument("--idle-exit-hours", type=float, required=True,
                        help="Release the GPU after this long without jobs")
    args = parser.parse_args()
    budget = None if args.reserve_gb == "auto" else float(args.reserve_gb)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible")
    reserved = reserve(budget)
    print(f"pid {os.getpid()} reserved {reserved:.1f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB; queue {args.queue}", flush=True)
    reason = serve(args.queue, poll_seconds=args.poll_seconds, idle_exit_seconds=args.idle_exit_hours * 3600,
                   log=lambda m: print(m, flush=True))
    print(f"exiting: {reason}", flush=True)


if __name__ == "__main__":
    main()
