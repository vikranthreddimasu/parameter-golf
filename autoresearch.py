#!/usr/bin/env python3
"""
Autoresearch loop for parameter-golf — no API keys needed.

Systematically tries promising techniques from the competition leaderboard,
keeps what improves val_bpb, discards what doesn't.

Usage:
    # On Colab (single GPU, 5 min per experiment)
    python autoresearch.py

    # Custom time budget and max experiments
    python autoresearch.py --time-budget 600 --max-experiments 10

    # Resume from previous run
    python autoresearch.py --resume
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = "train_gpt.py"
RESULTS_FILE = "autoresearch_results.tsv"
LOG_DIR = "autoresearch_logs"

# ---------------------------------------------------------------------------
# Experiment definitions
#
# Each experiment is a dict of environment variable overrides.
# They are tried in order. Successful experiments accumulate — later
# experiments run on top of all previously accepted changes.
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # --- Baseline (always first) ---
    {
        "name": "baseline",
        "description": "baseline — no modifications",
        "env": {},
    },

    # --- Depth recurrence (biggest architectural win) ---
    {
        "name": "recur_layers_3_4",
        "description": "depth recurrence layers 3,4 (always on)",
        "env": {"RECUR_LAYERS": "3,4", "RECUR_START_STEP": "0"},
    },
    {
        "name": "recur_layers_3_4_delayed",
        "description": "depth recurrence layers 3,4 (delayed start at step 2000)",
        "env": {"RECUR_LAYERS": "3,4", "RECUR_START_STEP": "2000"},
    },
    {
        "name": "recur_layers_3_4_5",
        "description": "depth recurrence layers 3,4,5 (delayed start at step 2000)",
        "env": {"RECUR_LAYERS": "3,4,5", "RECUR_START_STEP": "2000"},
    },

    # --- Model shape experiments ---
    {
        "name": "11_layers",
        "description": "11 layers instead of 9",
        "env": {"NUM_LAYERS": "11"},
    },
    {
        "name": "mlp_3x",
        "description": "MLP expansion 3x instead of 2x",
        "env": {"MLP_MULT": "3"},
    },
    {
        "name": "mlp_4x",
        "description": "MLP expansion 4x instead of 2x",
        "env": {"MLP_MULT": "4"},
    },

    # --- QK-Gain tuning ---
    {
        "name": "qk_gain_3",
        "description": "QK gain init 3.0 (up from 1.5)",
        "env": {"QK_GAIN_INIT": "3.0"},
    },
    {
        "name": "qk_gain_5",
        "description": "QK gain init 5.0 (top submissions use 4-5.25)",
        "env": {"QK_GAIN_INIT": "5.0"},
    },

    # --- Learning rate experiments ---
    {
        "name": "higher_matrix_lr",
        "description": "matrix LR 0.06 (up from 0.04)",
        "env": {"MATRIX_LR": "0.06"},
    },
    {
        "name": "lower_matrix_lr",
        "description": "matrix LR 0.03 (down from 0.04)",
        "env": {"MATRIX_LR": "0.03"},
    },
    {
        "name": "higher_embed_lr",
        "description": "tied embed LR 0.08 (up from 0.05)",
        "env": {"TIED_EMBED_LR": "0.08"},
    },

    # --- Warmdown tuning ---
    {
        "name": "warmdown_2000",
        "description": "warmdown 2000 iters (up from 1200)",
        "env": {"WARMDOWN_ITERS": "2000"},
    },
    {
        "name": "warmdown_3000",
        "description": "warmdown 3000 iters (much longer cooldown)",
        "env": {"WARMDOWN_ITERS": "3000"},
    },

    # --- Sequence length ---
    {
        "name": "seq_2048",
        "description": "sequence length 2048 (up from 1024)",
        "env": {"TRAIN_SEQ_LEN": "2048"},
    },

    # --- Optimizer tuning ---
    {
        "name": "muon_momentum_0.90",
        "description": "muon momentum 0.90 (down from 0.95)",
        "env": {"MUON_MOMENTUM": "0.90"},
    },
    {
        "name": "muon_steps_7",
        "description": "muon Newton-Schulz 7 steps (up from 5)",
        "env": {"MUON_BACKEND_STEPS": "7"},
    },

    # --- Batch size ---
    {
        "name": "larger_batch",
        "description": "batch size 1M tokens (up from 524K)",
        "env": {"TRAIN_BATCH_TOKENS": "1048576"},
    },

    # --- Logit softcap ---
    {
        "name": "softcap_50",
        "description": "logit softcap 50.0 (up from 30.0)",
        "env": {"LOGIT_SOFTCAP": "50.0"},
    },

    # --- Combinations of winners (added dynamically) ---
]


def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch loop for parameter-golf")
    parser.add_argument("--time-budget", type=int, default=300,
                        help="Training time budget per experiment in seconds (default: 300)")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Max experiments to run (0 = run all defined experiments)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results, skipping completed experiments")
    parser.add_argument("--nproc", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(time_budget: int, nproc: int, extra_env: dict) -> dict:
    """Run training and return parsed results."""
    log_file = Path(LOG_DIR) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    env = os.environ.copy()
    env["MAX_WALLCLOCK_SECONDS"] = str(time_budget)
    env["VAL_LOSS_EVERY"] = "0"
    env["TRAIN_LOG_EVERY"] = "100"
    env.update(extra_env)

    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", TRAIN_SCRIPT]

    env_str = " ".join(f"{k}={v}" for k, v in extra_env.items()) if extra_env else "(defaults)"
    print(f"  Command: {env_str} {' '.join(cmd)}")
    print(f"  Time budget: {time_budget}s")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=time_budget * 3,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        log_file.write_text(output)

        if result.returncode != 0:
            print(f"  CRASHED (exit code {result.returncode})")
            lines = output.strip().split("\n")
            for line in lines[-15:]:
                print(f"    {line}")
            return {"status": "crash", "val_bpb": 0.0, "elapsed": elapsed,
                    "artifact_bytes": 0, "log_file": str(log_file)}

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {time_budget * 3}s")
        return {"status": "crash", "val_bpb": 0.0, "elapsed": time.time() - start,
                "artifact_bytes": 0, "log_file": str(log_file)}

    return parse_training_output(output, elapsed, str(log_file))


def parse_training_output(output: str, elapsed: float, log_file: str) -> dict:
    """Parse training output to extract val_bpb and other metrics."""
    results = {"status": "ok", "elapsed": elapsed, "val_bpb": 0.0,
               "artifact_bytes": 0, "log_file": log_file}

    # Look for the final int8 roundtrip BPB (the real competition metric)
    match = re.search(r'final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)', output)
    if match:
        results["val_bpb"] = float(match.group(2))
    else:
        # Fallback: last val_bpb in log
        matches = re.findall(r'val_bpb:([\d.]+)', output)
        if matches:
            results["val_bpb"] = float(matches[-1])
        else:
            results["status"] = "crash"
            return results

    # Artifact size
    size_match = re.search(r'Total submission size int8\+zlib: (\d+) bytes', output)
    if size_match:
        results["artifact_bytes"] = int(size_match.group(1))

    # Peak memory
    mem_match = re.search(r'peak memory allocated: (\d+) MiB', output)
    if mem_match:
        results["peak_memory_mb"] = int(mem_match.group(1))

    # Steps completed
    step_match = re.findall(r'step:(\d+)/\d+', output)
    if step_match:
        results["steps"] = int(step_match[-1])

    return results


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results():
    """Initialize the results TSV file."""
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        results_path.write_text("experiment\tname\tval_bpb\tartifact_kb\tsteps\tstatus\tdescription\n")


def append_result(num: int, name: str, val_bpb: float, artifact_bytes: int,
                  steps: int, status: str, description: str):
    """Append a result to the TSV file."""
    artifact_kb = artifact_bytes / 1024 if artifact_bytes > 0 else 0
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{num}\t{name}\t{val_bpb:.6f}\t{artifact_kb:.1f}\t{steps}\t{status}\t{description}\n")


def get_completed_experiments() -> set[str]:
    """Get names of already-completed experiments."""
    completed = set()
    path = Path(RESULTS_FILE)
    if path.exists():
        for line in path.read_text().strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                completed.add(parts[1])
    return completed


def get_best_bpb() -> float:
    """Get the best val_bpb from results history."""
    path = Path(RESULTS_FILE)
    if not path.exists():
        return float("inf")
    best = float("inf")
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 6 and parts[5] in ("keep", "baseline"):
            try:
                bpb = float(parts[2])
                if bpb > 0:
                    best = min(best, bpb)
            except ValueError:
                pass
    return best


def get_kept_env() -> dict:
    """Build cumulative env overrides from all 'keep' experiments."""
    path = Path(RESULTS_FILE)
    if not path.exists():
        return {}

    kept_names = set()
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 6 and parts[5] == "keep":
            kept_names.add(parts[1])

    # Accumulate env vars from kept experiments
    env = {}
    for exp in EXPERIMENTS:
        if exp["name"] in kept_names:
            env.update(exp["env"])
    return env


def print_results_table():
    """Print a nice summary of all results."""
    path = Path(RESULTS_FILE)
    if not path.exists():
        return
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'#':<4} {'Name':<30} {'val_bpb':<12} {'Status':<10} {'Description'}")
    print("-" * 90)
    best = float("inf")
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 7:
            num, name, bpb_str, _, _, status, desc = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
            bpb = float(bpb_str) if bpb_str != "0.000000" else 0
            marker = ""
            if status in ("keep", "baseline") and bpb > 0:
                if bpb < best:
                    best = bpb
                    marker = " <-- BEST"
            print(f"{num:<4} {name:<30} {bpb_str:<12} {status:<10} {desc}{marker}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not Path(TRAIN_SCRIPT).exists():
        print(f"ERROR: {TRAIN_SCRIPT} not found. Run from the parameter-golf directory.")
        sys.exit(1)

    data_path = "./data/datasets/fineweb10B_sp1024"
    if not Path(data_path).exists():
        print(f"ERROR: Data not found at {data_path}")
        print("  Run: python data/cached_challenge_fineweb.py --variant sp1024")
        sys.exit(1)

    Path(LOG_DIR).mkdir(exist_ok=True)
    init_results()

    # Save backup
    backup_path = Path(f"{TRAIN_SCRIPT}.autoresearch_backup")
    if not backup_path.exists():
        shutil.copy2(TRAIN_SCRIPT, backup_path)

    completed = get_completed_experiments() if args.resume else set()

    experiments_to_run = EXPERIMENTS
    if args.max_experiments > 0:
        experiments_to_run = experiments_to_run[:args.max_experiments]

    total = len(experiments_to_run)
    print("=" * 70)
    print("AUTORESEARCH for Parameter Golf (no API keys needed)")
    print(f"  Experiments: {total}")
    print(f"  Time budget per experiment: {args.time_budget}s")
    print(f"  GPUs: {args.nproc}")
    print(f"  Estimated total time: ~{(total * (args.time_budget + 60)) // 60} min")
    print("=" * 70)

    for i, experiment in enumerate(experiments_to_run):
        name = experiment["name"]
        description = experiment["description"]
        extra_env = dict(experiment["env"])

        if name in completed:
            print(f"\n[{i}/{total}] Skipping '{name}' (already completed)")
            continue

        best_bpb = get_best_bpb()
        print(f"\n{'=' * 70}")
        print(f"[{i}/{total}] {name}")
        print(f"  Description: {description}")
        if best_bpb < float("inf"):
            print(f"  Best val_bpb so far: {best_bpb:.6f}")
        print("=" * 70)

        # For non-baseline experiments, include env vars from previously kept experiments
        if name != "baseline":
            kept_env = get_kept_env()
            # The experiment's own env overrides the kept env
            merged_env = {**kept_env, **extra_env}
            extra_env = merged_env

        result = run_training(args.time_budget, args.nproc, extra_env)
        steps = result.get("steps", 0)
        artifact_bytes = result.get("artifact_bytes", 0)

        if result["status"] == "crash" or result["val_bpb"] <= 0:
            print(f"  Result: CRASH")
            append_result(i, name, 0.0, 0, 0, "crash", description)

        elif name == "baseline":
            print(f"  Result: val_bpb = {result['val_bpb']:.6f} (baseline)")
            append_result(i, name, result["val_bpb"], artifact_bytes, steps, "baseline", description)

        elif result["val_bpb"] < best_bpb:
            improvement = best_bpb - result["val_bpb"]
            print(f"  Result: val_bpb = {result['val_bpb']:.6f} (IMPROVED by {improvement:.6f})")
            append_result(i, name, result["val_bpb"], artifact_bytes, steps, "keep", description)

        else:
            delta = result["val_bpb"] - best_bpb
            print(f"  Result: val_bpb = {result['val_bpb']:.6f} (worse by +{delta:.6f})")
            append_result(i, name, result["val_bpb"], artifact_bytes, steps, "discard", description)

        print(f"  Log: {result.get('log_file', 'n/a')}")

    print_results_table()

    best = get_best_bpb()
    kept = get_kept_env()
    if kept:
        print("\nTo reproduce the best result:")
        env_str = " ".join(f"{k}={v}" for k, v in kept.items())
        print(f"  {env_str} torchrun --standalone --nproc_per_node={args.nproc} {TRAIN_SCRIPT}")


if __name__ == "__main__":
    main()
