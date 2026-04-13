# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameter Golf is an OpenAI-sponsored competition to train the best language model within strict constraints:
- **16MB artifact size limit** (code + compressed model)
- **10 minutes training** on 8×H100 GPUs
- **Metric**: Bits per byte (BPB) on FineWeb validation set (lower is better)
- Baseline: ~1.22 BPB; current SOTA: ~1.08 BPB

## Running Training

```bash
# 1. Download data (required first)
python3 data/cached_challenge_fineweb.py --variant sp1024

# 2a. Local training (Mac MLX)
python3 train_gpt_mlx.py

# 2b. Single GPU (CUDA)
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 2c. Full 8×H100 run
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With custom hyperparameters (all config via env vars)
RUN_ID=exp_1 ITERATIONS=20000 VOCAB_SIZE=8192 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

There are no unit tests. Evaluation is embedded in the training loop (validation loss + BPB computed during training).

## Architecture

Two parallel training scripts share the same architecture:
- **train_gpt.py** — PyTorch + torchrun distributed training (primary, used for submissions)
- **train_gpt_mlx.py** — Apple MLX variant for local Mac development

Key classes in both scripts:
- `Hyperparameters` — all config via environment variables (vocab size, model dim, layers, LRs, batch size, etc.)
- `Muon` — custom optimizer with orthogonalization (from modded-nanogpt)
- `GPT` — full model: transformer blocks with GQA, RoPE, RMSNorm, tied embeddings, logit softcap
- `TokenStream`/`DistributedTokenLoader` — data loading from preprocessed binary shards

## Key Metrics

- `val_loss`: token cross-entropy (nats)
- `val_bpb`: bits per byte — the competition metric, tokenizer-agnostic compression ratio

## Submission Structure

Submissions live in `records/track_10min_16mb/` (SOTA) or `records/track_non_record_16mb/`. Each contains:
- `train_gpt.py` — modified training script
- `submission.json` — metadata (author, val_bpb, artifact size)
- `README.md` — technical description
- Training logs from 3+ seed runs

SOTA submissions must improve by ≥0.005 nats with p<0.01 significance.

## Common Techniques in Top Submissions

Depth recurrence (looping layers), parallel residuals (GPT-J style), GPTQ/ternary quantization, test-time training (TTT), QK-gain, Brotli/LZMA compression, skip gates, partial RoPE.

## Data Pipeline

- `data/cached_challenge_fineweb.py` — downloads preprocessed FineWeb shards
- `data/download_hf_docs_and_tokenize.py` — creates SentencePiece tokenizers
- `data/tokenizer_specs.json` — tokenizer variant configs (SP1024, SP4096, SP8192)
- Validation set: fixed first-50k FineWeb documents
