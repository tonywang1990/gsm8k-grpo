# GSM8K GRPO

Standalone GRPO trainer for GSM8K math reasoning. No TRL, no HF Trainer — one file, one loop.

**Baseline result**: Qwen3-1.7B, 50 steps → **74.0% eval accuracy**
**Best ablation**: `num_groups=16` → **82.0%** (+8 points, same compute budget)

See [`ablation/README.md`](ablation/README.md) for full ablation findings and the recommended config.

---

## What problem this solves

Most open-source GRPO implementations delegate the hard parts to TRL's `GRPOTrainer`, which hides the training loop behind a callback-heavy abstraction. This makes it difficult to understand what's actually happening, experiment with the algorithm, or adapt it to a new setting. This repo strips all of that away.

Three specific problems that are rarely addressed cleanly elsewhere:

**1. Running vLLM and a trainable HF model in the same process.**
The standard approach is either vLLM-only (fast inference, not trainable) or HF-only (trainable, slow generation). The hard part is keeping both alive simultaneously without OOMing — vLLM pre-allocates GPU memory aggressively. This repo uses Unsloth's `unsloth_vllm_standby` mode, which holds vLLM in a suspended state during the backward pass and wakes it only for rollouts. After each optimizer step, updated LoRA weights are saved to disk and hot-reloaded into the vLLM engine, so rollouts always use the latest policy with no model duplication in memory.

**2. Correctly masking the loss to assistant tokens only.**
Most tutorials compute the loss over the full sequence including the prompt, or use a naive `[prompt_len:]` slice that accidentally includes role header tokens (`<|im_start|>assistant\n`). This repo builds an exact per-token binary mask by walking the token IDs and tracking `<|im_start|>` / `<|im_end|>` boundaries, so the loss only touches the actual generated content.

**3. Avoiding full-vocabulary logit materialization.**
At each training step, computing log-probabilities naively requires materializing a `(batch, seq_len, vocab_size)` tensor — for a 32K vocab and long sequences this alone can OOM. This repo passes `logits_to_keep=num_completion_tokens` to the forward pass so the model only produces logits for the completion suffix, then applies a fused gather+logsumexp to extract per-token log-probs without ever holding the full logit tensor.

---

## How it works

```
sample problems → vLLM rollouts → reward (format + correctness) →
group-normalize advantages → compute logprobs (HF model) →
PPO-clip loss → AdamW step → sync LoRA → vLLM → repeat
```

The whole loop lives in [`gsm8k_grpo.py`](gsm8k_grpo.py). The model is loaded once via Unsloth, which gives you a shared HF model for gradient computation and a vLLM engine for fast batched rollouts. After each optimizer step, LoRA weights are written to disk and hot-reloaded into vLLM.

---

## Setup

```bash
uv venv
uv pip install -e .
source setup.sh        # activates venv, installs CUDA compat if needed
```

Requirements: Python ≥ 3.10, CUDA GPU with ≥ 24GB VRAM.

---

## Usage

### Train

```bash
# Baseline (50 steps, ~30 min on A100)
python gsm8k_grpo.py --model_name Qwen/Qwen3-1.7B

# Recommended config from ablation study
python gsm8k_grpo.py \
    --model_name Qwen/Qwen3-1.7B \
    --learning_rate 2e-5 \
    --num_groups 16 \
    --group_size 16 \
    --mu 3 \
    --lora_rank 64 \
    --temperature 0.7 \
    --max_steps 500

# Smoke test (fast, ~2 min)
python gsm8k_grpo.py \
    --model_name Qwen/Qwen3-1.7B \
    --max_steps 5 \
    --num_groups 2 \
    --group_size 4 \
    --max_tokens 512
```

Outputs go to `runs/<slug>/` (or `--output_dir`). Each run writes:
- `step_stat_history.json` — per-step metrics (loss, accuracy, clip_frac, timing)
- `eval_history.json` — periodic test-set accuracy (greedy decode)
- `generations.log` — sample completions per group per step
- `accuracy_plot.txt` — ASCII training curve
- `summary.json` — final summary on completion
- `final_model/` — LoRA adapter weights + tokenizer

### Resume from checkpoint

```bash
python gsm8k_grpo.py \
    --model_name Qwen/Qwen3-1.7B \
    --resume_from runs/my-run/checkpoint-100 \
    --output_dir runs/my-run-continued
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-1.7B` | HuggingFace model ID |
| `--max_steps` | 50 | Training steps |
| `--num_groups` | 8 | Distinct problems per step |
| `--group_size` | 8 | Solutions per problem (G) |
| `--learning_rate` | 5e-6 | AdamW LR |
| `--lora_rank` | 32 | LoRA rank (minimum viable: 32) |
| `--mu` | 1 | Inner optimization steps per rollout |
| `--temperature` | 1.0 | Sampling temperature |
| `--epsilon` | 0.2 | PPO clip range |
| `--max_tokens` | 1024 | Max tokens per completion |
| `--eval_steps` | 10 | Eval frequency (0 = disabled) |
| `--eval_size` | 100 | Test problems per eval |
| `--save_steps` | 100 | Checkpoint frequency |
| `--early_stop_patience` | 0 | Stop after N steps without improvement (0 = disabled) |

---

## Ablation study

17 experiments sweeping 6 hyperparameter groups, each for 50 steps. Run all:

```bash
python ablation/ablation_study.py                          # runs all 17, then generates report
python ablation/ablation_study.py --only baseline lr_2e-5  # run specific experiments
python ablation/ablation_study.py --dry_run                # print commands without running
```

Results land in `runs/ablation/`. The pre-run results from our study are already there — see [`runs/ablation/report/`](runs/ablation/report/) for the figures and [`ablation/README.md`](ablation/README.md) for the analysis.

To regenerate the report from existing data:

```bash
python ablation/ablation_report.py --results_dir runs/ablation
```

---

## Reward function

```
1.0  — correct answer (extracted number matches ground truth)
0.1  — format present (#### found) but wrong answer
0.0  — no #### in output
```

The 0.1 format bonus ensures the model is rewarded for learning the output format even before it gets answers right.

---

## Files

```
gsm8k_grpo.py                  Training loop (load data → rollout → GRPO → save)
ablation/ablation_study.py     Runs N experiments sequentially, skips completed ones
ablation/ablation_report.py    Reads run JSONs, generates 5 figures + report.md
ablation/README.md             Ablation findings and recommendations
runs/ablation/                 Pre-run experiment data (metrics + logs + figures)
```
