#!/usr/bin/env python3
"""
Launcher for VLMEvalKit that spawns independent processes per GPU.

Unlike torchrun, this does NOT create a torch.distributed process group,
so vLLM can initialise cleanly without port / shared-memory conflicts.

Each worker gets RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, and
CUDA_VISIBLE_DEVICES set in its environment.  The workers use a simple
file-based barrier (see vlmeval/smp/misc.py patching below) instead of
torch.distributed.barrier().

Usage:
    python launch_workers.py [--ngpu N] -- run.py --use-vllm --data ... --model ...
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ngpu", type=int, default=0,
                        help="Number of GPUs / workers. 0 = auto-detect.")
    parser.add_argument("--delay", type=float, default=15.0,
                        help="Seconds of stagger between launching workers "
                             "(gives vLLM time to finish init before the next starts).")
    # Everything after '--' is the command for each worker.
    args, remainder = parser.parse_known_args()
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("No worker command given.  Usage: launch_workers.py [opts] -- run.py [args]")
    return args, remainder


def detect_gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
        return len(out.strip().splitlines())
    except Exception:
        return 1


def main() -> None:
    args, worker_cmd = parse_args()
    ngpu = args.ngpu or detect_gpu_count()

    # Create a shared temp directory for file-based barriers.
    barrier_dir = tempfile.mkdtemp(prefix="vlmeval_barrier_")
    print(f"[launcher] ngpu={ngpu}, delay={args.delay}s, barrier_dir={barrier_dir}", flush=True)
    print(f"[launcher] worker command: {worker_cmd}", flush=True)

    procs: list[subprocess.Popen] = []

    for rank in range(ngpu):
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["WORLD_SIZE"] = str(ngpu)
        env["LOCAL_WORLD_SIZE"] = str(ngpu)
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["VLMEVAL_BARRIER_DIR"] = barrier_dir

        # Remove torchrun vars that might linger from a parent script.
        for k in ("MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
            env.pop(k, None)

        cmd = [sys.executable] + worker_cmd
        print(f"[launcher] starting rank {rank}/{ngpu}  CUDA_VISIBLE_DEVICES={rank}  cmd={cmd}", flush=True)
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)

        # Stagger launches so vLLM instances don't compete during init.
        if rank < ngpu - 1 and args.delay > 0:
            import time
            print(f"[launcher] waiting {args.delay}s before next worker ...", flush=True)
            time.sleep(args.delay)

    # Wait for all workers.  Propagate failure.
    def _kill_all(sig=None, frame=None):
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        sys.exit(1)

    signal.signal(signal.SIGINT, _kill_all)
    signal.signal(signal.SIGTERM, _kill_all)

    exit_codes = []
    for rank, proc in enumerate(procs):
        rc = proc.wait()
        exit_codes.append(rc)
        print(f"[launcher] rank {rank} exited with code {rc}", flush=True)

    # Cleanup barrier dir.
    import shutil
    shutil.rmtree(barrier_dir, ignore_errors=True)

    if any(rc != 0 for rc in exit_codes):
        failed = [r for r, rc in enumerate(exit_codes) if rc != 0]
        print(f"[launcher] ERROR: ranks {failed} failed", flush=True)
        sys.exit(1)

    print("[launcher] all workers completed successfully", flush=True)


if __name__ == "__main__":
    main()
