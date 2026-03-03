# GSM8K GRPO

Standalone GRPO trainer for GSM8K math reasoning. No TRL, no HF Trainer — one file, one loop.

**Baseline result**: Qwen3-1.7B, 50 steps → **74.0% eval accuracy**
**Best ablation**: `num_groups=16` → **82.0%** (+8 points, same compute budget)

See [`summary.md`](summary.md) for full ablation findings and the recommended config.

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

Results land in `runs/ablation/`. The pre-run results from our study are already there — see [`runs/ablation/report/`](runs/ablation/report/) for the figures and [`summary.md`](summary.md) for the analysis.

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
summary.md                     Ablation findings and recommendations
runs/ablation/                 Pre-run experiment data (metrics + logs + figures)
```
