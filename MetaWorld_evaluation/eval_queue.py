#!/usr/bin/env python3
"""
eval_queue.py — Run MetaWorld evaluations for multiple checkpoints in parallel.

Checkpoints evaluated:
  - /home/tmprithvi/baseline/step_5000
  - /home/tmprithvi/baseline/step_10000
  - /home/tmprithvi/baseline/stage2/step_* (all subdirs)

Usage:
  python eval_queue.py --parallel 3
  python eval_queue.py --parallel 4 --base_port 9100 --out_root /tmp/eval_results
"""

import argparse
import os
import queue
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent.resolve()                        # MetaWorld_evaluation/
REPO_ROOT   = SCRIPT_DIR.parent.resolve()                            # Evo-1/
SERVER_SCRIPT = REPO_ROOT / "Evo_1" / "scripts" / "Evo1_server_eval.py"
CLIENT_SCRIPT = SCRIPT_DIR / "mt50_evo1_client_eval.py"

BASELINE_DIR  = Path("/home/tmprithvi/baseline")

# Defaults for norm stats keys (match what's in the checkpoint norm_stats.json)
DEFAULT_ARM_KEY     = "metaworld_sawyer"
DEFAULT_DATASET_KEY = "Evo1_MetaWorld"
DEFAULT_BASE_PORT   = 9000
DEFAULT_OUT_ROOT    = str(SCRIPT_DIR / "eval_outputs")

SERVER_READY_TIMEOUT = 180   # seconds to wait for server port to open
SERVER_BOOT_POLL     = 2     # seconds between port polls

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def get_checkpoints():
    """Return list of (name, abs_path) for all target checkpoints."""
    ckpts = []

    # baseline step_5000 and step_10000 only
    for step in ["step_5000", "step_10000"]:
        p = BASELINE_DIR / step
        if p.is_dir():
            ckpts.append((f"baseline_{step}", str(p)))
        else:
            print(f"[WARN] Not found, skipping: {p}")

    # all subdirs under stage2
    stage2 = BASELINE_DIR / "stage2"
    if stage2.is_dir():
        for d in sorted(stage2.iterdir()):
            if d.is_dir() and d.name.startswith("step"):
                ckpts.append((f"stage2_{d.name}", str(d)))
    else:
        print(f"[WARN] stage2 dir not found: {stage2}")

    return ckpts


# ---------------------------------------------------------------------------
# Port readiness check
# ---------------------------------------------------------------------------
def wait_for_port(port: int, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    """Return True once something is listening on port, False on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(SERVER_BOOT_POLL)
    return False


# ---------------------------------------------------------------------------
# Single-checkpoint runner (blocks until client exits)
# ---------------------------------------------------------------------------
def run_one(name: str, ckpt_dir: str, port: int, out_dir: str,
            arm_key: str, dataset_key: str) -> bool:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    server_log = out_path / "server.log"
    client_log = out_path / "client.log"

    tag = f"[{name}|port={port}]"
    print(f"{tag} Starting server  ckpt={ckpt_dir}")

    server_proc = subprocess.Popen(
        [
            sys.executable, str(SERVER_SCRIPT),
            "--ckpt_dir",    ckpt_dir,
            "--port",        str(port),
            "--arm_key",     arm_key,
            "--dataset_key", dataset_key,
        ],
        stdout=open(server_log, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT / "Evo_1" / "scripts"),
    )

    if not wait_for_port(port, timeout=SERVER_READY_TIMEOUT):
        print(f"{tag} ERROR: server never became ready on port {port}. Killing.")
        server_proc.terminate()
        server_proc.wait(timeout=10)
        return False

    print(f"{tag} Server ready. Starting client  out={out_dir}")

    client_proc = subprocess.Popen(
        [
            sys.executable, str(CLIENT_SCRIPT),
            "--port",    str(port),
            "--out_dir", str(out_path),
        ],
        stdout=open(client_log, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),   # client needs mt50_order.json / tasks.jsonl from here
    )

    ret = client_proc.wait()
    print(f"{tag} Client finished (exit={ret}). Stopping server.")
    server_proc.terminate()
    server_proc.wait(timeout=15)
    return ret == 0


# ---------------------------------------------------------------------------
# Queue + thread-pool orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parallel MetaWorld checkpoint evaluator")
    parser.add_argument(
        "--parallel", type=int, default=3,
        help="Number of simultaneous evaluations (default: 3)"
    )
    parser.add_argument(
        "--base_port", type=int, default=DEFAULT_BASE_PORT,
        help=f"First WebSocket port; slot i uses base_port+i (default: {DEFAULT_BASE_PORT})"
    )
    parser.add_argument(
        "--out_root", default=DEFAULT_OUT_ROOT,
        help=f"Root dir for per-checkpoint outputs (default: {DEFAULT_OUT_ROOT})"
    )
    parser.add_argument(
        "--arm_key", default=DEFAULT_ARM_KEY,
        help=f"Arm key in norm_stats.json (default: {DEFAULT_ARM_KEY})"
    )
    parser.add_argument(
        "--dataset_key", default=DEFAULT_DATASET_KEY,
        help=f"Dataset key in norm_stats.json (default: {DEFAULT_DATASET_KEY})"
    )
    args = parser.parse_args()

    checkpoints = get_checkpoints()
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoint(s). Running {args.parallel} in parallel.\n")
    for name, path in checkpoints:
        print(f"  {name}")
        print(f"    {path}")
    print()

    # Port pool: each concurrent slot gets its own port
    port_pool: queue.Queue = queue.Queue()
    for i in range(args.parallel):
        port_pool.put(args.base_port + i)

    results: dict = {}

    def worker(name: str, ckpt_dir: str) -> tuple:
        port = port_pool.get()           # blocks until a slot is free
        out_dir = os.path.join(args.out_root, name)
        try:
            ok = run_one(name, ckpt_dir, port, out_dir, args.arm_key, args.dataset_key)
            return name, ok
        finally:
            port_pool.put(port)          # release slot for the next queued job

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(worker, name, ckpt): name
            for name, ckpt in checkpoints
        }
        for fut in as_completed(futures):
            name, ok = fut.result()
            results[name] = ok
            status = "OK  " if ok else "FAIL"
            print(f"[{status}] {name}")

    # Final summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ok_count = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    print(f"\n{ok_count}/{len(results)} evaluations completed successfully.")
    print(f"Outputs in: {args.out_root}")


if __name__ == "__main__":
    main()
