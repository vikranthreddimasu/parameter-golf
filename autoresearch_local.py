#!/usr/bin/env python3
"""
Local autoresearch for parameter-golf — Round 2: Research-backed experiments.

Implements and tests techniques from Claude Research, Perplexity Deep Research,
and AlphaXiv analysis. Each experiment toggles features via env vars.

Usage:
    python autoresearch_local.py                     # run all experiments
    python autoresearch_local.py --time-budget 600   # 10 min each (default)
    python autoresearch_local.py --resume            # skip completed
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

TRAIN_SCRIPT = "train_gpt_mlx.py"
RESULTS_FILE = "autoresearch_local_results.tsv"
LOG_DIR = "autoresearch_local_logs"

# Fast proxy config: small batch, no validation, no final eval.
FAST_PROXY_ENV = {
    "TRAIN_BATCH_TOKENS": "65536",
    "GRAD_ACCUM_STEPS": "1",
    "WARMUP_STEPS": "3",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "10",
    "MLX_EAGER_EVAL": "1",
    "SKIP_FINAL_EVAL": "1",
}

# ===========================================================================
# EXPERIMENTS — ordered by research-predicted impact
#
# Round 1 winners (baked into all experiments as baseline):
#   RECUR_LAYERS=4,5  RECUR_START_STEP=100  NUM_LAYERS=7  MLP_MULT=3
#   MATRIX_LR=0.08  TIED_EMBED_LR=0.03
# ===========================================================================

# Round 1 best config — applied to ALL round 2 experiments
ROUND1_WINNERS = {
    "RECUR_LAYERS": "4,5",
    "RECUR_START_STEP": "100",
    "NUM_LAYERS": "7",
    "MLP_MULT": "3",
    "MATRIX_LR": "0.08",
    "TIED_EMBED_LR": "0.03",
}

EXPERIMENTS = [
    # === BASELINE with round 1 winners ===
    {"name": "r2_baseline", "description": "round 1 winners as new baseline", "env": {}},

    # === TIER 1: HIGHEST IMPACT (research consensus) ===

    # Sigmoid Gated Attention — all 3 research sources recommend
    {"name": "sigmoid_gate", "description": "sigmoid gated attention (NeurIPS 2025)", "env": {"SIGMOID_GATE": "1"}},

    # Label smoothing — zero cost, prevents overconfidence
    {"name": "label_smooth_0.05", "description": "label smoothing eps=0.05", "env": {"LABEL_SMOOTHING": "0.05"}},
    {"name": "label_smooth_0.10", "description": "label smoothing eps=0.10 (Transformer paper)", "env": {"LABEL_SMOOTHING": "0.1"}},
    {"name": "label_smooth_0.15", "description": "label smoothing eps=0.15", "env": {"LABEL_SMOOTHING": "0.15"}},

    # Activation functions — SwiGLU and LeakyReLU² both beat baseline relu²
    {"name": "swiglu", "description": "SwiGLU activation (NanoGPT slowrun winner)", "env": {"SWIGLU": "1"}},
    {"name": "leaky_relu_sq", "description": "LeakyReLU(0.5)² (SOTA activation)", "env": {"LEAKY_RELU_SQ": "1"}},
    {"name": "leaky_relu_sq_0.3", "description": "LeakyReLU(0.3)² (less leakage)", "env": {"LEAKY_RELU_SQ": "1", "LEAKY_RELU_ALPHA": "0.3"}},

    # Meta tokens — Hymba architecture, 16K extra params for quality
    {"name": "meta_4", "description": "4 learnable meta tokens (Hymba-style)", "env": {"META_TOKENS": "4"}},
    {"name": "meta_8", "description": "8 learnable meta tokens", "env": {"META_TOKENS": "8"}},
    {"name": "meta_16", "description": "16 learnable meta tokens", "env": {"META_TOKENS": "16"}},
    {"name": "meta_32", "description": "32 learnable meta tokens", "env": {"META_TOKENS": "32"}},

    # === TIER 2: COMBINING WINNERS FROM TIER 1 ===

    # Sigmoid gate + label smoothing
    {"name": "gate+smooth", "description": "sigmoid gate + label smoothing 0.1", "env": {"SIGMOID_GATE": "1", "LABEL_SMOOTHING": "0.1"}},

    # Sigmoid gate + leaky relu
    {"name": "gate+leaky", "description": "sigmoid gate + LeakyReLU(0.5)²", "env": {"SIGMOID_GATE": "1", "LEAKY_RELU_SQ": "1"}},

    # Sigmoid gate + swiglu
    {"name": "gate+swiglu", "description": "sigmoid gate + SwiGLU", "env": {"SIGMOID_GATE": "1", "SWIGLU": "1"}},

    # Label smoothing + activations
    {"name": "smooth+leaky", "description": "label smoothing 0.1 + LeakyReLU(0.5)²", "env": {"LABEL_SMOOTHING": "0.1", "LEAKY_RELU_SQ": "1"}},
    {"name": "smooth+swiglu", "description": "label smoothing 0.1 + SwiGLU", "env": {"LABEL_SMOOTHING": "0.1", "SWIGLU": "1"}},

    # Meta tokens + sigmoid gate
    {"name": "meta8+gate", "description": "8 meta tokens + sigmoid gate", "env": {"META_TOKENS": "8", "SIGMOID_GATE": "1"}},

    # Triple combo
    {"name": "gate+smooth+leaky", "description": "sigmoid gate + smoothing 0.1 + LeakyReLU²", "env": {"SIGMOID_GATE": "1", "LABEL_SMOOTHING": "0.1", "LEAKY_RELU_SQ": "1"}},
    {"name": "gate+smooth+swiglu", "description": "sigmoid gate + smoothing 0.1 + SwiGLU", "env": {"SIGMOID_GATE": "1", "LABEL_SMOOTHING": "0.1", "SWIGLU": "1"}},

    # === TIER 3: DEEPER HYPERPARAMETER EXPLORATION ===

    # LR tuning on top of round 1 winners
    {"name": "matrix_lr_0.10", "description": "matrix LR 0.10 (push higher)", "env": {"MATRIX_LR": "0.10"}},
    {"name": "matrix_lr_0.12", "description": "matrix LR 0.12", "env": {"MATRIX_LR": "0.12"}},
    {"name": "embed_lr_0.02", "description": "embed LR 0.02 (push lower)", "env": {"TIED_EMBED_LR": "0.02"}},
    {"name": "embed_lr_0.01", "description": "embed LR 0.01", "env": {"TIED_EMBED_LR": "0.01"}},

    # Recurrence variants on 7-layer model
    {"name": "recur_2_3", "description": "recurrence layers 2,3 (early layers)", "env": {"RECUR_LAYERS": "2,3"}},
    {"name": "recur_3_4", "description": "recurrence layers 3,4 (mid)", "env": {"RECUR_LAYERS": "3,4"}},
    {"name": "recur_2_3_4", "description": "recurrence layers 2,3,4 (3-layer recurrence)", "env": {"RECUR_LAYERS": "2,3,4"}},
    {"name": "recur_del200", "description": "recurrence 4,5 delayed step 200", "env": {"RECUR_START_STEP": "200"}},
    {"name": "no_recur", "description": "no recurrence (check if it still helps)", "env": {"RECUR_LAYERS": "", "RECUR_START_STEP": "0"}},

    # Grad clipping (research says helps with gated attention)
    {"name": "grad_clip_1.0", "description": "gradient clipping 1.0", "env": {"GRAD_CLIP_NORM": "1.0"}},
    {"name": "grad_clip_0.5", "description": "gradient clipping 0.5", "env": {"GRAD_CLIP_NORM": "0.5"}},

    # Softcap with new activations
    {"name": "softcap_20", "description": "logit softcap 20", "env": {"LOGIT_SOFTCAP": "20.0"}},
    {"name": "softcap_50", "description": "logit softcap 50", "env": {"LOGIT_SOFTCAP": "50.0"}},

    # Model shape on 7-layer baseline
    {"name": "dim_640_7L", "description": "dim 640 with 7 layers", "env": {"MODEL_DIM": "640"}},
    {"name": "mlp_2x_7L", "description": "MLP 2x (smaller, faster steps)", "env": {"MLP_MULT": "2"}},

    # Muon tuning
    {"name": "muon_mom_0.92", "description": "muon momentum 0.92", "env": {"MUON_MOMENTUM": "0.92"}},
    {"name": "muon_mom_0.98", "description": "muon momentum 0.98", "env": {"MUON_MOMENTUM": "0.98"}},

    # === TIER 4: FULL COMBO SEARCH ===

    # Best of everything (will be filled based on tier 1-3 results)
    {"name": "full_combo_a", "description": "gate+smooth+leaky+meta8+clip1.0", "env": {
        "SIGMOID_GATE": "1", "LABEL_SMOOTHING": "0.1", "LEAKY_RELU_SQ": "1",
        "META_TOKENS": "8", "GRAD_CLIP_NORM": "1.0",
    }},
    {"name": "full_combo_b", "description": "gate+smooth+swiglu+meta16", "env": {
        "SIGMOID_GATE": "1", "LABEL_SMOOTHING": "0.1", "SWIGLU": "1",
        "META_TOKENS": "16",
    }},
    {"name": "full_combo_c", "description": "leaky+smooth+meta8+higher_lr", "env": {
        "LEAKY_RELU_SQ": "1", "LABEL_SMOOTHING": "0.1", "META_TOKENS": "8",
        "MATRIX_LR": "0.10",
    }},
]


def parse_args():
    p = argparse.ArgumentParser(description="Local autoresearch — Round 2")
    p.add_argument("--time-budget", type=int, default=600, help="Seconds per experiment (default: 600)")
    p.add_argument("--max-experiments", type=int, default=0, help="Max experiments (0 = all)")
    p.add_argument("--resume", action="store_true", help="Skip completed experiments")
    return p.parse_args()


def run_experiment(time_budget: int, extra_env: dict) -> dict:
    log_file = Path(LOG_DIR) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    env = os.environ.copy()
    env["MAX_WALLCLOCK_SECONDS"] = str(time_budget)
    env.update(FAST_PROXY_ENV)
    env.update(ROUND1_WINNERS)
    env.update(extra_env)

    display_env = {k: v for k, v in extra_env.items()} if extra_env else {"(round1 winners)": ""}
    env_str = " ".join(f"{k}={v}" for k, v in display_env.items())
    print(f"  Config: {env_str}")

    cmd = [sys.executable, TRAIN_SCRIPT]
    start = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=time_budget * 3)
        elapsed = time.time() - start
        output = result.stdout + "\n" + result.stderr
        log_file.write_text(output)
        if result.returncode != 0:
            print(f"  CRASHED (exit {result.returncode})")
            lines = output.strip().split("\n")
            for line in lines[-10:]:
                print(f"    {line}")
            return {"status": "crash", "train_loss": 99.0, "steps": 0, "elapsed": elapsed, "log_file": str(log_file)}
    except subprocess.TimeoutExpired:
        return {"status": "crash", "train_loss": 99.0, "steps": 0, "elapsed": time.time() - start, "log_file": "timeout"}

    return parse_output(output, elapsed, str(log_file))


def parse_output(output: str, elapsed: float, log_file: str) -> dict:
    res = {"status": "ok", "elapsed": elapsed, "log_file": log_file, "train_loss": 99.0, "steps": 0, "tok_s": 0, "step_ms": 0}
    losses = re.findall(r'step:(\d+)/\d+ train_loss:([\d.]+).*?step_avg:([\d.]+)ms.*?tok_s:(\d+)', output)
    if not losses:
        res["status"] = "crash"
        return res
    last = losses[-1]
    res["steps"] = int(last[0])
    res["train_loss"] = float(last[1])
    res["step_ms"] = float(last[2])
    res["tok_s"] = int(last[3])
    params_match = re.search(r'model_params:(\d+)', output)
    if params_match:
        res["params"] = int(params_match.group(1))
    return res


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
    return {line.split("\t")[1] for line in path.read_text().strip().split("\n")[1:] if len(line.split("\t")) >= 2}


def get_best_loss() -> tuple[float, str]:
    path = Path(RESULTS_FILE)
    if not path.exists():
        return 99.0, "none"
    best_loss, best_name = 99.0, "none"
    for line in path.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 7 and parts[6] in ("keep", "baseline"):
            try:
                loss = float(parts[2])
                if 0 < loss < best_loss:
                    best_loss, best_name = loss, parts[1]
            except ValueError:
                pass
    return best_loss, best_name


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
    print(f"{'#':<4} {'Name':<28} {'Loss':<10} {'Steps':<7} {'ms/step':<9} {'tok/s':<8} {'Status':<10} Description")
    print("-" * 100)
    best_loss = 99.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 8:
            num, name, loss_s, steps, ms, toks, status, desc = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
            marker = ""
            loss_val = float(loss_s) if loss_s != "99.0000" else 99.0
            if status in ("keep", "baseline") and loss_val < best_loss:
                best_loss = loss_val
                marker = " <-- BEST"
            print(f"{num:<4} {name:<28} {loss_s:<10} {steps:<7} {ms:<9} {toks:<8} {status:<10} {desc}{marker}")
    print("=" * 100)


def main():
    args = parse_args()
    if not Path(TRAIN_SCRIPT).exists():
        print(f"ERROR: {TRAIN_SCRIPT} not found")
        sys.exit(1)
    if not Path("./data/datasets/fineweb10B_sp1024").exists():
        print("ERROR: Data not found. Run: python data/cached_challenge_fineweb.py --variant sp1024")
        sys.exit(1)

    Path(LOG_DIR).mkdir(exist_ok=True)
    init_results()
    backup = Path(f"{TRAIN_SCRIPT}.autoresearch_backup")
    if not backup.exists():
        shutil.copy2(TRAIN_SCRIPT, backup)

    completed = get_completed() if args.resume else set()
    experiments = EXPERIMENTS[:args.max_experiments] if args.max_experiments > 0 else EXPERIMENTS
    total = len(experiments)
    est_hrs = (total * (args.time_budget + 30)) / 3600

    print("=" * 70)
    print("AUTORESEARCH Round 2 — Research-Backed Experiments")
    print("=" * 70)
    print(f"  Experiments:    {total}")
    print(f"  Time/exp:       {args.time_budget}s")
    print(f"  Est. total:     {est_hrs:.1f} hours")
    print(f"  Round 1 base:   7L MLP3x recur(4,5) matLR=0.08 embLR=0.03")
    print(f"  New features:   sigmoid gate, label smooth, SwiGLU, LeakyReLU²,")
    print(f"                  meta tokens, combinations")
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

        result = run_experiment(args.time_budget, exp_env)
        steps = result.get("steps", 0)
        step_ms = result.get("step_ms", 0)
        tok_s = result.get("tok_s", 0)
        train_loss = result.get("train_loss", 99.0)

        if result["status"] == "crash":
            print(f"  --> CRASH")
            append_result(i, name, 99.0, 0, 0, 0, "crash", desc)
        elif name == "r2_baseline":
            print(f"  --> Baseline: loss={train_loss:.4f} steps={steps} ({step_ms:.0f}ms/step)")
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
