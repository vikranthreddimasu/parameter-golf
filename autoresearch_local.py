#!/usr/bin/env python3
"""
Local autoresearch for parameter-golf on MacBook (MLX).

Uses small batches and train_loss as a fast proxy metric to rank techniques
relative to each other. Winners get validated on Colab with full training.

Design:
  - Same model architecture as competition (techniques transfer)
  - Reduced batch size for fast steps (~1.8s instead of 14s)
  - Train loss at fixed step count as ranking metric (skip slow validation)
  - ~5 min per experiment → ~10 experiments/hour
  - Results inform what to run on Colab

Usage:
    python autoresearch_local.py                          # run all experiments
    python autoresearch_local.py --time-budget 300        # 5 min each (default)
    python autoresearch_local.py --max-experiments 5      # only first 5
    python autoresearch_local.py --resume                 # skip completed ones
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
# Config
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = "train_gpt_mlx.py"
RESULTS_FILE = "autoresearch_local_results.tsv"
LOG_DIR = "autoresearch_local_logs"

# Fast proxy config: small batch for fast steps, no validation overhead.
# Same model architecture so techniques transfer to full-scale runs.
FAST_PROXY_ENV = {
    "TRAIN_BATCH_TOKENS": "65536",      # 64K instead of 524K (8x smaller)
    "GRAD_ACCUM_STEPS": "1",            # no accumulation (8x faster steps)
    "WARMUP_STEPS": "3",                # minimal warmup (saves ~4.5 min)
    "VAL_LOSS_EVERY": "0",              # disable validation (saves ~4 min)
    "TRAIN_LOG_EVERY": "10",            # log more often for proxy metric
    "MLX_EAGER_EVAL": "1",              # keep memory safe on 24GB
}

# ---------------------------------------------------------------------------
# Experiments: each is {name, description, env overrides}
#
# env overrides are ADDED to FAST_PROXY_ENV + any previously kept winners.
# Order matters — promising, high-impact experiments first.
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "baseline config (fast proxy)",
        "env": {},
    },

    # --- Depth recurrence (our implementation) ---
    {
        "name": "recur_3_4_always",
        "description": "depth recurrence layers 3,4 always on",
        "env": {"RECUR_LAYERS": "3,4", "RECUR_START_STEP": "0"},
    },
    {
        "name": "recur_3_4_delayed",
        "description": "depth recurrence layers 3,4 delayed (step 50)",
        "env": {"RECUR_LAYERS": "3,4", "RECUR_START_STEP": "50"},
    },
    {
        "name": "recur_2_3_4",
        "description": "depth recurrence layers 2,3,4 always on",
        "env": {"RECUR_LAYERS": "2,3,4", "RECUR_START_STEP": "0"},
    },

    # --- QK-Gain (top submissions use 4-5.25) ---
    {
        "name": "qk_gain_3",
        "description": "QK gain 3.0",
        "env": {"QK_GAIN_INIT": "3.0"},
    },
    {
        "name": "qk_gain_5",
        "description": "QK gain 5.0 (SOTA range)",
        "env": {"QK_GAIN_INIT": "5.0"},
    },

    # --- Model shape ---
    {
        "name": "11_layers",
        "description": "11 layers (more depth)",
        "env": {"NUM_LAYERS": "11"},
    },
    {
        "name": "mlp_3x",
        "description": "MLP 3x expansion",
        "env": {"MLP_MULT": "3"},
    },
    {
        "name": "mlp_4x",
        "description": "MLP 4x expansion",
        "env": {"MLP_MULT": "4"},
    },
    {
        "name": "dim_640",
        "description": "model dim 640 (wider)",
        "env": {"MODEL_DIM": "640"},
    },

    # --- Learning rates ---
    {
        "name": "matrix_lr_0.06",
        "description": "matrix LR 0.06 (up from 0.04)",
        "env": {"MATRIX_LR": "0.06"},
    },
    {
        "name": "matrix_lr_0.03",
        "description": "matrix LR 0.03 (down from 0.04)",
        "env": {"MATRIX_LR": "0.03"},
    },
    {
        "name": "embed_lr_0.08",
        "description": "embed LR 0.08 (up from 0.05)",
        "env": {"TIED_EMBED_LR": "0.08"},
    },
    {
        "name": "scalar_lr_0.06",
        "description": "scalar LR 0.06 (up from 0.04)",
        "env": {"SCALAR_LR": "0.06"},
    },

    # --- Optimizer ---
    {
        "name": "muon_mom_0.90",
        "description": "muon momentum 0.90 (from 0.95)",
        "env": {"MUON_MOMENTUM": "0.90"},
    },
    {
        "name": "muon_steps_7",
        "description": "muon Newton-Schulz 7 steps",
        "env": {"MUON_BACKEND_STEPS": "7"},
    },

    # --- Sequence length ---
    {
        "name": "seq_2048",
        "description": "seq len 2048 (double context)",
        "env": {"TRAIN_SEQ_LEN": "2048"},
    },

    # --- Softcap ---
    {
        "name": "softcap_50",
        "description": "logit softcap 50 (from 30)",
        "env": {"LOGIT_SOFTCAP": "50.0"},
    },
    {
        "name": "softcap_20",
        "description": "logit softcap 20 (from 30)",
        "env": {"LOGIT_SOFTCAP": "20.0"},
    },

    # --- RoPE base ---
    {
        "name": "rope_base_500k",
        "description": "RoPE base 500000 (from 10000)",
        "env": {"ROPE_BASE": "500000.0"},
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Local autoresearch on Mac (MLX)")
    p.add_argument("--time-budget", type=int, default=300,
                   help="Seconds per experiment (default: 300)")
    p.add_argument("--max-experiments", type=int, default=0,
                   help="Max experiments (0 = all)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed experiments")
    p.add_argument("--compare-steps", type=int, default=0,
                   help="Compare train_loss at this step (0 = use last logged step)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(time_budget: int, extra_env: dict) -> dict:
    """Run MLX training and return results."""
    log_file = Path(LOG_DIR) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    env = os.environ.copy()
    env["MAX_WALLCLOCK_SECONDS"] = str(time_budget)
    env.update(FAST_PROXY_ENV)
    env.update(extra_env)

    # Build readable env string for logging
    custom_keys = set(extra_env.keys())
    env_str = " ".join(f"{k}={v}" for k, v in extra_env.items()) if extra_env else "(defaults)"

    cmd = [sys.executable, TRAIN_SCRIPT]
    print(f"  Config: {env_str}")
    print(f"  Budget: {time_budget}s | Batch: {env.get('TRAIN_BATCH_TOKENS', '?')} | Accum: {env.get('GRAD_ACCUM_STEPS', '?')}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=time_budget * 3,
        )
        elapsed = time.time() - start
        output = result.stdout + "\n" + result.stderr
        log_file.write_text(output)

        if result.returncode != 0:
            print(f"  CRASHED (exit {result.returncode})")
            lines = output.strip().split("\n")
            for line in lines[-10:]:
                print(f"    {line}")
            return {"status": "crash", "train_loss": 99.0, "steps": 0,
                    "elapsed": elapsed, "log_file": str(log_file)}

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({time_budget * 3}s)")
        return {"status": "crash", "train_loss": 99.0, "steps": 0,
                "elapsed": time.time() - start, "log_file": str(log_file)}

    return parse_output(output, elapsed, str(log_file))


def parse_output(output: str, elapsed: float, log_file: str) -> dict:
    """Parse MLX training output for proxy metrics."""
    res = {"status": "ok", "elapsed": elapsed, "log_file": log_file,
           "train_loss": 99.0, "steps": 0, "tok_s": 0, "step_ms": 0}

    # Extract all train_loss entries: step:N/M train_loss:X.XXXX
    losses = re.findall(r'step:(\d+)/\d+ train_loss:([\d.]+).*?step_avg:([\d.]+)ms.*?tok_s:(\d+)', output)
    if not losses:
        res["status"] = "crash"
        return res

    # Use the last logged train_loss as our proxy metric
    last = losses[-1]
    res["steps"] = int(last[0])
    res["train_loss"] = float(last[1])
    res["step_ms"] = float(last[2])
    res["tok_s"] = int(last[3])

    # Also collect the full loss curve for analysis
    res["loss_curve"] = [(int(s), float(l)) for s, l, _, _ in losses]

    # Check for val_bpb if validation ran
    bpb_match = re.findall(r'val_bpb:([\d.]+)', output)
    if bpb_match:
        res["val_bpb"] = float(bpb_match[-1])

    # Model params
    params_match = re.search(r'model_params:(\d+)', output)
    if params_match:
        res["params"] = int(params_match.group(1))

    return res


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results():
    path = Path(RESULTS_FILE)
    if not path.exists():
        path.write_text("num\tname\ttrain_loss\tsteps\tstep_ms\ttok_s\tstatus\tdescription\n")


def append_result(num, name, train_loss, steps, step_ms, tok_s, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{num}\t{name}\t{train_loss:.4f}\t{steps}\t{step_ms:.1f}\t{tok_s}\t{status}\t{description}\n")


def get_completed() -> set[str]:
    path = Path(RESULTS_FILE)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            done.add(parts[1])
    return done


def get_best_loss() -> tuple[float, str]:
    """Return (best_train_loss, experiment_name) from kept/baseline results."""
    path = Path(RESULTS_FILE)
    if not path.exists():
        return 99.0, "none"
    best_loss = 99.0
    best_name = "none"
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 7 and parts[6] in ("keep", "baseline"):
            try:
                loss = float(parts[2])
                if 0 < loss < best_loss:
                    best_loss = loss
                    best_name = parts[1]
            except ValueError:
                pass
    return best_loss, best_name


def get_kept_env() -> dict:
    """Accumulate env overrides from all kept experiments."""
    path = Path(RESULTS_FILE)
    if not path.exists():
        return {}
    kept_names = set()
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 7 and parts[6] == "keep":
            kept_names.add(parts[1])
    env = {}
    for exp in EXPERIMENTS:
        if exp["name"] in kept_names:
            env.update(exp["env"])
    return env


def print_summary():
    path = Path(RESULTS_FILE)
    if not path.exists():
        return
    lines = path.read_text().strip().split("\n")
    if len(lines) <= 1:
        return

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (lower train_loss = better)")
    print("=" * 100)
    print(f"{'#':<4} {'Name':<25} {'Loss':<10} {'Steps':<7} {'ms/step':<9} {'tok/s':<8} {'Status':<10} Description")
    print("-" * 100)

    best_loss = 99.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 8:
            num, name, loss_s, steps, ms, toks, status, desc = (
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
            )
            marker = ""
            loss_val = float(loss_s) if loss_s != "99.0000" else 99.0
            if status in ("keep", "baseline") and loss_val < best_loss:
                best_loss = loss_val
                marker = " <-- BEST"
            print(f"{num:<4} {name:<25} {loss_s:<10} {steps:<7} {ms:<9} {toks:<8} {status:<10} {desc}{marker}")

    print("=" * 100)

    # Print Colab command for winners
    kept = get_kept_env()
    if kept:
        print("\nRecommended Colab command (combining all winners):")
        env_str = " ".join(f"{k}={v}" for k, v in kept.items())
        print(f"  {env_str} torchrun --standalone --nproc_per_node=1 train_gpt.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not Path(TRAIN_SCRIPT).exists():
        print(f"ERROR: {TRAIN_SCRIPT} not found")
        sys.exit(1)

    data_path = "./data/datasets/fineweb10B_sp1024"
    if not Path(data_path).exists():
        print(f"ERROR: Data not found. Run: python data/cached_challenge_fineweb.py --variant sp1024")
        sys.exit(1)

    Path(LOG_DIR).mkdir(exist_ok=True)
    init_results()

    # Backup
    backup = Path(f"{TRAIN_SCRIPT}.autoresearch_backup")
    if not backup.exists():
        shutil.copy2(TRAIN_SCRIPT, backup)

    completed = get_completed() if args.resume else set()
    experiments = EXPERIMENTS
    if args.max_experiments > 0:
        experiments = experiments[:args.max_experiments]

    total = len(experiments)
    est_min = (total * (args.time_budget + 30)) // 60

    print("=" * 70)
    print("LOCAL AUTORESEARCH for Parameter Golf (MLX)")
    print("=" * 70)
    print(f"  Experiments:  {total}")
    print(f"  Time/exp:     {args.time_budget}s")
    print(f"  Est. total:   ~{est_min} min ({est_min / 60:.1f} hours)")
    print(f"  Proxy metric: train_loss (lower = better)")
    print(f"  Fast config:  batch=64K, accum=1, warmup=3, no validation")
    print("=" * 70)

    for i, experiment in enumerate(experiments):
        name = experiment["name"]
        desc = experiment["description"]
        exp_env = dict(experiment["env"])

        if name in completed:
            print(f"\n[{i}/{total}] Skipping '{name}' (done)")
            continue

        best_loss, best_name = get_best_loss()
        print(f"\n{'─' * 70}")
        print(f"[{i}/{total}] {name}")
        print(f"  {desc}")
        if best_loss < 99:
            print(f"  Current best: {best_loss:.4f} ({best_name})")
        print("─" * 70)

        # Accumulate kept experiment configs for non-baseline runs
        if name != "baseline":
            kept = get_kept_env()
            exp_env = {**kept, **exp_env}

        result = run_experiment(args.time_budget, exp_env)
        steps = result.get("steps", 0)
        step_ms = result.get("step_ms", 0)
        tok_s = result.get("tok_s", 0)
        train_loss = result.get("train_loss", 99.0)

        if result["status"] == "crash":
            print(f"  --> CRASH")
            append_result(i, name, 99.0, 0, 0, 0, "crash", desc)
        elif name == "baseline":
            print(f"  --> Baseline: loss={train_loss:.4f} steps={steps} ({step_ms:.0f}ms/step, {tok_s} tok/s)")
            append_result(i, name, train_loss, steps, step_ms, tok_s, "baseline", desc)
        elif train_loss < best_loss:
            delta = best_loss - train_loss
            print(f"  --> IMPROVED: loss={train_loss:.4f} (delta: -{delta:.4f})")
            append_result(i, name, train_loss, steps, step_ms, tok_s, "keep", desc)
        else:
            delta = train_loss - best_loss
            print(f"  --> No gain: loss={train_loss:.4f} (delta: +{delta:.4f})")
            append_result(i, name, train_loss, steps, step_ms, tok_s, "discard", desc)

        print(f"  Log: {result.get('log_file', 'n/a')}")

    print_summary()


if __name__ == "__main__":
    main()
