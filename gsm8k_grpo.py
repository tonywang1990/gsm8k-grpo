"""Standalone GRPO trainer for GSM8K math reasoning — no TRL, no HF Trainer.

Complete GRPO training loop in one file:
  load GSM8K problems -> generate solutions (vLLM) -> compute rewards ->
  group advantages -> compute logprobs (HF model) -> PPO-clip loss ->
  optimizer step -> sync LoRA weights back to vLLM -> repeat

Usage:
    python gsm8k_grpo.py \
        --model_name Qwen/Qwen3-1.7B \
        --max_steps 500 \
        --num_groups 8 \
        --group_size 8

Smoke test (fast):
    python gsm8k_grpo.py \
        --model_name Qwen/Qwen3-1.7B \
        --max_steps 5 \
        --num_groups 2 \
        --group_size 4 \
        --max_tokens 512
"""

import argparse
import json
import os
import random
import re
import sys
import time

# Prevent Cursor's debugger (pydevd) from attaching to vLLM subprocesses.
# The bundled debugger is incompatible with Python 3.12+ (missing `imp` module).
for _key in list(os.environ):
    if _key.startswith(("PYDEVD", "DEBUGPY")):
        del os.environ[_key]

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
from dataclasses import dataclass
from pathlib import Path

import torch
from vllm import SamplingParams

# Qwen3 chat template boundary tokens
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

ANSWER_PATTERN = re.compile(r"####\s*([\d,\.\-]+)")

GSM8K_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step, "
    "then state your final answer on the last line as: #### <number>"
)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SolutionData:
    prompt_ids: list[int]
    completion_ids: list[int]
    assistant_mask: list[int]
    reward: float
    info: dict  # question, ground_truth, extracted_answer, is_correct


# ============================================================================
# Data loading
# ============================================================================

def extract_ground_truth(answer_text: str) -> str:
    """Extract the numeric answer after '####' in a GSM8K answer string."""
    return answer_text.split("####")[1].strip().replace(",", "")


def load_gsm8k(split: str = "train") -> list[dict]:
    """Load GSM8K dataset and return list of {question, answer} dicts."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {
            "question": row["question"],
            "answer": extract_ground_truth(row["answer"]),
        }
        for row in ds
    ]


# ============================================================================
# Reward
# ============================================================================

def extract_model_answer(text: str) -> str | None:
    """Extract the number after '####' from model output, or None if missing."""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def compute_reward(completion: str, ground_truth: str) -> float:
    """Binary correctness reward + small format bonus.

    Returns:
        1.0  if extracted number matches ground truth
        0.1  if '####' present but answer is wrong
        0.0  if no '####' found
    """
    extracted = extract_model_answer(completion)
    if extracted is None:
        return 0.0
    # Small format bonus regardless of correctness
    reward = 0.1
    try:
        if float(extracted) == float(ground_truth):
            reward = 1.0
    except ValueError:
        if extracted == ground_truth:
            reward = 1.0
    return reward


# ============================================================================
# Model setup
# ============================================================================

def setup_model(model_name: str, max_seq_length: int, lora_rank: int, resume_from: str | None = None):
    """Load model with Unsloth + LoRA. Returns (model, tokenizer, vllm_engine).

    If resume_from is set, loads LoRA weights from that path after creating the
    PEFT model so training continues from a previous checkpoint.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
        unsloth_vllm_standby=True,
    )

    # Disable thinking in Qwen3 chat template:
    # 1. Default enable_thinking to false for add_generation_prompt
    # 2. Remove unconditional <think> wrapping on the last assistant message
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "enable_thinking is not defined or enable_thinking is false",
        )
        # The template adds <think> tags on loop.last even without reasoning_content.
        # Change "loop.last or (not loop.last and reasoning_content)" to just
        # "reasoning_content" so <think> tags only appear when there's actual reasoning.
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "loop.last or (not loop.last and reasoning_content)",
            "reasoning_content",
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing=False,
        random_state=3407,
    )

    # Resume from checkpoint: load saved LoRA weights into the PEFT model
    if resume_from:
        from peft import set_peft_model_state_dict
        import safetensors.torch
        resume_path = Path(resume_from)
        weights_file = resume_path / "adapter_model.safetensors"
        if not weights_file.exists():
            weights_file = resume_path / "adapter_model.bin"
        if weights_file.exists():
            if weights_file.suffix == ".safetensors":
                state_dict = safetensors.torch.load_file(str(weights_file))
            else:
                state_dict = torch.load(str(weights_file), map_location="cpu")
            set_peft_model_state_dict(model, state_dict)
            print(f"Resumed LoRA weights from: {resume_from}")
        else:
            raise FileNotFoundError(f"No adapter weights found in {resume_from}")

    vllm_engine = model.vllm_engine

    # Switch HF model from inference mode (in-place ops) to training mode
    FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

    return model, tokenizer, vllm_engine


# ============================================================================
# Rollout: generate math solutions with vLLM
# ============================================================================

def build_assistant_mask(
    token_ids: list[int], prompt_length: int,
    im_start_id: int, im_end_id: int, assistant_role_id: int,
) -> list[int]:
    """Build mask: 1 for assistant content tokens in the completion, 0 elsewhere."""
    completion_len = len(token_ids) - prompt_length
    mask = [0] * completion_len
    in_assistant = False
    skip_role_tokens = 0

    for i in range(prompt_length, len(token_ids)):
        idx = i - prompt_length
        tok = token_ids[i]

        if tok == im_end_id:
            in_assistant = False
            continue
        if tok == im_start_id:
            if i + 1 < len(token_ids) and token_ids[i + 1] == assistant_role_id:
                in_assistant = True
                skip_role_tokens = 2  # skip "assistant" + "\n"
            else:
                in_assistant = False
            continue
        if in_assistant:
            if skip_role_tokens > 0:
                skip_role_tokens -= 1
                continue
            mask[idx] = 1

    return mask


def rollout_solutions(
    vllm_engine, tokenizer, problems: list[dict],
    group_size: int,
    im_start_id: int, im_end_id: int, assistant_role_id: int,
    lora_request=None,
    temperature: float = 1.0, top_p: float = 0.9, top_k: int = 50,
    max_tokens: int = 1024,
) -> list[SolutionData]:
    """Generate group_size solutions per problem using a single batched vLLM call.

    Args:
        problems: list of {question, answer} dicts
        group_size: number of solutions to generate per problem
    Returns:
        list of SolutionData, length = len(problems) * group_size,
        ordered as [p0_s0, p0_s1, ..., p1_s0, p1_s1, ...]
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    system_msg = {"role": "system", "content": GSM8K_SYSTEM_PROMPT}

    # Build one prompt per (problem, solution_index) pair
    prompts = []
    prompt_meta = []  # (problem_dict,) for each prompt
    for problem in problems:
        user_msg = {"role": "user", "content": problem["question"]}
        prompt_str = tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )
        for _ in range(group_size):
            prompts.append(prompt_str)
            prompt_meta.append(problem)

    gen_kwargs = {}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    outputs = vllm_engine.generate(
        prompts, sampling_params=sampling_params,
        use_tqdm=False, **gen_kwargs,
    )

    solutions = []
    for problem, output in zip(prompt_meta, outputs):
        completion_text = output.outputs[0].text
        ground_truth = problem["answer"]

        reward = compute_reward(completion_text, ground_truth)
        extracted = extract_model_answer(completion_text)
        is_correct = reward == 1.0

        # Tokenize prompt + completion together for logprob computation
        user_msg = {"role": "user", "content": problem["question"]}
        assistant_msg = {"role": "assistant", "content": completion_text}

        full_ids = tokenizer.apply_chat_template(
            [system_msg, user_msg, assistant_msg],
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=True,
            add_generation_prompt=False,
        )
        completion_ids = full_ids[len(prompt_ids):]
        assistant_mask = build_assistant_mask(
            full_ids, len(prompt_ids), im_start_id, im_end_id, assistant_role_id,
        )

        solutions.append(SolutionData(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            assistant_mask=assistant_mask,
            reward=reward,
            info={
                "question": problem["question"],
                "ground_truth": ground_truth,
                "completion": completion_text,
                "extracted_answer": extracted,
                "is_correct": is_correct,
            },
        ))

    return solutions


# ============================================================================
# Logprob computation
# ============================================================================

def selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Memory-efficient log_softmax for selected tokens only.

    Args:
        logits: (batch, seq, vocab)
        target_ids: (batch, seq)
    Returns:
        (batch, seq) log probabilities for the target tokens
    """
    selected = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp = torch.logsumexp(logits, dim=-1)
    return selected - logsumexp


def compute_logprobs(model, input_ids, attention_mask, num_completion_tokens):
    """Forward pass -> per-token log probs for the completion portion.

    Uses logits_to_keep to avoid materializing logits for the full prompt.
    """
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=num_completion_tokens + 1,
    )
    # Shift: logits[i] predicts token[i+1]
    logits = output.logits[:, :-1, :]
    target_ids = input_ids[:, -num_completion_tokens:]
    return selective_log_softmax(logits, target_ids)


# ============================================================================
# GRPO loss
# ============================================================================

def grpo_loss(current_logps, old_logps, advantages, mask, epsilon=0.2):
    """GRPO / PPO-clip loss with assistant-only masking.

    Args:
        current_logps: (batch, seq) — differentiable
        old_logps: (batch, seq) — detached
        advantages: (batch,) — per-trajectory advantage
        mask: (batch, seq) — 1 for assistant tokens, 0 elsewhere
        epsilon: clip range
    Returns:
        scalar loss
    """
    log_ratio = current_logps - old_logps
    log_ratio = torch.clamp(log_ratio, -3.0, 3.0)  # prevent exp() overflow
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    adv = advantages.unsqueeze(1)
    per_token = -torch.min(ratio * adv, clipped * adv)
    return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()


# ============================================================================
# ASCII plot
# ============================================================================

def render_reward_plot(values, width=60, height=15, title="Accuracy"):
    """Render an ASCII plot of values. Returns the plot as a string."""
    n = len(values)
    if n == 0:
        return ""
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 0.1  # avoid division by zero

    # Bucket values into columns if more data points than width
    if n > width:
        bucket_size = n / width
        cols = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            cols.append(sum(values[start:end]) / (end - start))
    else:
        cols = values
        width = n

    # Build grid
    grid = [[" "] * width for _ in range(height)]
    for x, v in enumerate(cols):
        y = int((v - lo) / (hi - lo) * (height - 1))
        grid[y][x] = "█"

    # Render to string
    lines = []
    lines.append(f"\n{title:^{width + 10}}")
    for row in reversed(range(height)):
        if row == height - 1:
            label = f"{hi:>7.3f} │"
        elif row == 0:
            label = f"{lo:>7.3f} │"
        elif row == height // 2:
            mid = (lo + hi) / 2
            label = f"{mid:>7.3f} │"
        else:
            label = "        │"
        lines.append(label + "".join(grid[row]))
    lines.append("        └" + "─" * width)
    step_labels = f"{'1':<{width // 2}}{n}"
    lines.append(f"         {step_labels}")
    lines.append(f"{'Step':^{width + 10}}")
    return "\n".join(lines)


def print_reward_plot(values, width=60, height=15, title="Accuracy"):
    """Print an ASCII plot of values to the terminal."""
    print(render_reward_plot(values, width, height, title))


def save_reward_plot(values, path, width=60, height=15, title="Accuracy"):
    """Save an ASCII reward plot to a file."""
    plot = render_reward_plot(values, width, height, title)
    if plot:
        with open(path, "w") as f:
            f.write(plot + "\n")


# ============================================================================
# Logging
# ============================================================================

def log_solutions(log_path, step, solutions, group_size, step_stats=None):
    """Log per-group solution details to file."""
    for g in range(0, len(solutions), group_size):
        group = solutions[g:g + group_size]
        if not group:
            continue
        lines = [f"=== Step {step} | Group {g // group_size} ==="]
        if step_stats is not None:
            s = step_stats
            lines.append(
                "Training stats:\n"
                f"loss={s['loss']:.4f} grad={s['grad_norm']:.4f} "
                f"clip={s['clip_frac']:.3f} | "
                f"reward={s['reward_mean']:.3f} (std={s['reward_std']:.3f}) | "
                f"accuracy={s['accuracy']:.3f} | "
                f"t_roll={s['t_rollout']:.1f}s t_train={s['t_train']:.1f}s"
            )

        # Show the question (from first solution in group — same for all)
        first_info = group[0].info
        lines.append(f"Question: {first_info['question'][:200]}")
        lines.append(f"Ground truth: {first_info['ground_truth']}")
        lines.append("")

        for i, sol in enumerate(group):
            info = sol.info
            correct_marker = "✓" if info["is_correct"] else "✗"
            lines.append(
                f"Solution {i} [{correct_marker}]: "
                f"reward={sol.reward:.1f} | "
                f"extracted={info['extracted_answer']} "
                f"(gt={info['ground_truth']})"
            )
            # Show last 300 chars of completion (the answer region)
            completion_snippet = info["completion"][-300:].replace("\n", " ↵ ")
            lines.append(f"  ...{completion_snippet}")
            lines.append("")

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n\n")


# ============================================================================
# Evaluation
# ============================================================================

def eval_on_dataset(
    vllm_engine, tokenizer, problems: list[dict],
    lora_request=None,
    max_tokens: int = 1024,
) -> dict:
    """Evaluate model accuracy on a list of problems using greedy decoding.

    Returns a dict with keys: accuracy, format_rate, n_problems.
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    system_msg = {"role": "system", "content": GSM8K_SYSTEM_PROMPT}

    prompts = []
    for problem in problems:
        user_msg = {"role": "user", "content": problem["question"]}
        prompt_str = tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_str)

    gen_kwargs = {}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    outputs = vllm_engine.generate(
        prompts, sampling_params=sampling_params,
        use_tqdm=False, **gen_kwargs,
    )

    n_correct = 0
    n_formatted = 0
    for problem, output in zip(problems, outputs):
        completion = output.outputs[0].text
        extracted = extract_model_answer(completion)
        if extracted is not None:
            n_formatted += 1
            reward = compute_reward(completion, problem["answer"])
            if reward == 1.0:
                n_correct += 1

    n = len(problems)
    return {
        "accuracy": n_correct / n,
        "format_rate": n_formatted / n,
        "n_problems": n,
    }


# ============================================================================
# Training loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone GRPO trainer for GSM8K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--num_groups", type=int, default=8,
                        help="Number of distinct problems per training step")
    parser.add_argument("--group_size", type=int, default=8,
                        help="Number of solutions generated per problem (G)")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--mu", type=int, default=1,
                        help="Number of inner GRPO optimization steps per rollout batch")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens per solution generation")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="Run eval on test set every N steps (0=disabled)")
    parser.add_argument("--eval_size", type=int, default=100,
                        help="Number of test problems to sample per eval run")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to LoRA checkpoint dir to resume from")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop if accuracy doesn't improve for N steps (0=disabled)")
    parser.add_argument("--eval_seed", type=int, default=42,
                        help="Base seed for eval problem sampling; combined with step for reproducibility")
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        from coolname import generate_slug
        args.output_dir = f"./runs/{generate_slug(2)}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    gen_log_path = output_dir / "generations.log"
    gen_log_path.write_text("")

    print("================================================")
    print(f"Output: {output_dir}")
    print("================================================")
    print(f"Config: steps={args.max_steps}, groups={args.num_groups}, G={args.group_size}, "
          f"lr={args.learning_rate}, eps={args.epsilon}, mu={args.mu}, "
          f"max_tokens={args.max_tokens}, temp={args.temperature}")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")

    # ── 1. Setup ──────────────────────────────────────────────────────────
    model, tokenizer, vllm_engine = setup_model(
        args.model_name, args.max_seq_length, args.lora_rank,
        resume_from=args.resume_from,
    )

    im_start_id = tokenizer.convert_tokens_to_ids(IM_START_TOKEN)
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TOKEN)
    assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)
    assistant_role_id = assistant_tokens[0]

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=0.001)
    print(f"Trainable params: {sum(p.numel() for p in lora_params):,}")

    device = next(model.parameters()).device
    lora_request = None  # first rollout uses base model weights

    # If resuming, sync LoRA weights to vLLM for the first rollout
    if args.resume_from:
        lora_sync_path = str(output_dir / "live_lora")
        model.save_lora(lora_sync_path)
        lora_request = model.load_lora(lora_sync_path)
        print("Synced resumed LoRA weights to vLLM")

    # ── 2. Load dataset ──────────────────────────────────────────────────
    print("Loading GSM8K training data...")
    train_data = load_gsm8k("train")
    print(f"Loaded {len(train_data)} training problems")

    eval_data = []
    if args.eval_steps > 0:
        print("Loading GSM8K test data...")
        eval_data = load_gsm8k("test")
        print(f"Loaded {len(eval_data)} test problems (will sample {args.eval_size} per eval)")

    # ── 3. Training loop ──────────────────────────────────────────────────
    total_solutions = args.num_groups * args.group_size
    accuracy_history = []
    reward_history = []
    step_stat_history = []
    eval_history = []  # [{step, accuracy, format_rate}]
    best_accuracy = -float("inf")
    steps_since_improvement = 0

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # ── 3a. Sample problems for this step ────────────────────────
        problems_batch = random.sample(train_data, args.num_groups)

        # ── 3b. Wake vLLM, run rollouts, sleep vLLM ──────────────────
        if hasattr(vllm_engine, "wake_up"):
            if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                torch.cuda.empty_cache()
                vllm_engine.wake_up()

        solutions = rollout_solutions(
            vllm_engine, tokenizer, problems_batch,
            args.group_size,
            im_start_id, im_end_id, assistant_role_id,
            lora_request=lora_request,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )

        if hasattr(vllm_engine, "sleep"):
            if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                vllm_engine.sleep(int(os.environ.get("VLLM_SLEEP_MODE", 1)))

        t_rollout = time.time() - t0

        # ── 3c. Rewards -> advantages (GRPO group normalization) ──────
        rewards = torch.tensor(
            [s.reward for s in solutions], dtype=torch.float32, device=device,
        )
        rewards_grouped = rewards.view(args.num_groups, args.group_size)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = ((rewards_grouped - group_mean) / (group_std + 1e-4)).view(-1)

        # ── 3d. Compute old logprobs (π_θ_old, fixed for all μ steps) ─
        old_logps_list = []
        with torch.no_grad():
            for sol in solutions:
                ids = sol.prompt_ids + sol.completion_ids
                input_ids = torch.tensor([ids], device=device)
                attn_mask = torch.ones_like(input_ids)
                num_completion = len(sol.completion_ids)
                old_lp = compute_logprobs(model, input_ids, attn_mask, num_completion)
                old_logps_list.append(old_lp)

        # ── 3e. μ inner optimization steps ───────────────────────────
        total_loss = 0.0
        total_clip_frac = 0.0
        total_log_ratio_abs_mean = 0.0
        grad_norm = torch.tensor(0.0)

        for _mu_step in range(args.mu):
            optimizer.zero_grad()
            step_loss = 0.0
            step_clip_frac = 0.0
            step_log_ratio_sum = 0.0
            step_log_ratio_tokens = 0

            for i, sol in enumerate(solutions):
                ids = sol.prompt_ids + sol.completion_ids
                input_ids = torch.tensor([ids], device=device)
                attn_mask = torch.ones_like(input_ids)
                num_completion = len(sol.completion_ids)
                mask = torch.tensor(
                    [sol.assistant_mask], dtype=torch.float32, device=device,
                )
                adv_i = advantages[i : i + 1]

                current_logps = compute_logprobs(model, input_ids, attn_mask, num_completion)
                old_logps = old_logps_list[i]

                loss_i = grpo_loss(
                    current_logps, old_logps, adv_i, mask, args.epsilon,
                )
                (loss_i / total_solutions).backward()
                step_loss += loss_i.item()

                # Track clipping and log_ratio
                with torch.no_grad():
                    raw_log_ratio = current_logps - old_logps
                    ratio = torch.exp(raw_log_ratio)
                    is_clipped = ((ratio < 1 - args.epsilon) & (adv_i.unsqueeze(1) < 0)) | \
                                 ((ratio > 1 + args.epsilon) & (adv_i.unsqueeze(1) > 0))
                    clip_frac = (is_clipped.float() * mask).sum() / mask.sum().clamp(min=1)
                    step_clip_frac += clip_frac.item()
                    step_log_ratio_sum += (raw_log_ratio.abs() * mask).sum().item()
                    step_log_ratio_tokens += int(mask.sum().item())

            step_loss /= total_solutions
            step_clip_frac /= total_solutions
            step_log_ratio_abs_mean = step_log_ratio_sum / max(step_log_ratio_tokens, 1)

            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            if grad_norm.item() > 10.0:
                print(f"  [!] Skipping update: grad_norm={grad_norm.item():.1f}")
                optimizer.zero_grad()
            else:
                optimizer.step()

            total_loss = step_loss
            total_clip_frac = step_clip_frac
            total_log_ratio_abs_mean = step_log_ratio_abs_mean

        t_train = time.time() - t0 - t_rollout

        # ── 3f. Sync LoRA weights to vLLM ────────────────────────────
        lora_sync_path = str(output_dir / "live_lora")
        model.save_lora(lora_sync_path)
        lora_request = model.load_lora(lora_sync_path)

        # ── 3g. Eval on test set ─────────────────────────────────────
        eval_result = None
        if args.eval_steps > 0 and step % args.eval_steps == 0 and eval_data:
            if hasattr(vllm_engine, "wake_up"):
                if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                    torch.cuda.empty_cache()
                    vllm_engine.wake_up()

            eval_rng = random.Random(args.eval_seed + step)
            eval_problems = eval_rng.sample(eval_data, min(args.eval_size, len(eval_data)))
            eval_result = eval_on_dataset(
                vllm_engine, tokenizer, eval_problems,
                lora_request=lora_request,
                max_tokens=args.max_tokens,
            )
            eval_result["step"] = step
            eval_history.append(eval_result)

            if hasattr(vllm_engine, "sleep"):
                if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                    vllm_engine.sleep(int(os.environ.get("VLLM_SLEEP_MODE", 1)))

            with open(output_dir / "eval_history.json", "w") as f:
                json.dump(eval_history, f, indent=2)
            save_reward_plot(
                [e["accuracy"] for e in eval_history],
                output_dir / "eval_accuracy_plot.txt",
                title="Eval Accuracy",
            )

        # ── 3h. Metrics ───────────────────────────────────────────────
        mean_reward = rewards.mean().item()
        std_reward = rewards.std().item()
        accuracy = sum(1 for s in solutions if s.info["is_correct"]) / len(solutions)
        format_rate = sum(1 for s in solutions if s.info["extracted_answer"] is not None) / len(solutions)

        asst_tokens = sum(sum(s.assistant_mask) for s in solutions)
        total_tokens = sum(len(s.completion_ids) for s in solutions)

        accuracy_history.append(accuracy)
        reward_history.append(mean_reward)

        step_stats = {
            "step": step,
            "loss": round(total_loss, 6),
            "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
            "clip_frac": round(total_clip_frac, 4),
            "log_ratio_abs_mean": round(total_log_ratio_abs_mean, 4),
            "reward_mean": round(mean_reward, 4),
            "reward_std": round(std_reward, 4),
            "accuracy": round(accuracy, 4),
            "format_rate": round(format_rate, 4),
            "asst_token_ratio": round(asst_tokens / max(total_tokens, 1), 4),
            "seq_len_mean": round(total_tokens / max(len(solutions), 1), 1),
            "t_rollout": round(t_rollout, 2),
            "t_train": round(t_train, 2),
        }

        eval_str = ""
        if eval_result is not None:
            eval_str = f" | eval_acc={eval_result['accuracy']:.3f} eval_fmt={eval_result['format_rate']:.3f}"
        print(
            f"[Step {step}/{args.max_steps}] "
            f"loss={total_loss:.4f} grad={step_stats['grad_norm']:.4f} clip={total_clip_frac:.3f} "
            f"lra={total_log_ratio_abs_mean:.3f} | "
            f"reward={mean_reward:.3f} (std={std_reward:.3f}) | "
            f"accuracy={accuracy:.3f} fmt={format_rate:.3f} | "
            f"asst_ratio={asst_tokens / max(total_tokens, 1):.2f} "
            f"seq_len={total_tokens / max(len(solutions), 1):.0f} | "
            f"t_roll={t_rollout:.1f}s t_train={t_train:.1f}s"
            + eval_str
        )

        step_stat_history.append(step_stats)
        with open(output_dir / "step_stat_history.json", "w") as f:
            json.dump(step_stat_history, f, indent=2)
        log_solutions(gen_log_path, step, solutions, args.group_size, step_stats)
        save_reward_plot(accuracy_history, output_dir / "accuracy_plot.txt", title="Accuracy")

        # ── 3h. Save checkpoint ──────────────────────────────────────
        if step % args.save_steps == 0:
            ckpt_path = str(output_dir / f"checkpoint-{step}")
            model.save_lora(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # ── 3i. Early stopping ───────────────────────────────────────
        smoothed = sum(accuracy_history[-5:]) / min(len(accuracy_history), 5)
        if smoothed > best_accuracy + 1e-4:
            best_accuracy = smoothed
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if args.early_stop_patience > 0 and steps_since_improvement >= args.early_stop_patience:
            print(f"\n  Early stopping: no improvement for {args.early_stop_patience} steps")
            break

    # ── 4. Save final model ───────────────────────────────────────────────
    final_path = str(output_dir / "final_model")
    model.save_lora(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    # ── 5. Print accuracy plot ────────────────────────────────────────────
    if accuracy_history:
        print_reward_plot(accuracy_history, title="Accuracy")

    # ── 6. Write summary ──────────────────────────────────────────────────
    if step_stat_history:
        best_step = max(step_stat_history, key=lambda s: s["accuracy"])
        last5 = step_stat_history[-5:]
        summary = {
            "total_steps": len(step_stat_history),
            "early_stopped": steps_since_improvement >= args.early_stop_patience if args.early_stop_patience > 0 else False,
            "final_accuracy": round(sum(s["accuracy"] for s in last5) / len(last5), 4),
            "best_accuracy": round(best_step["accuracy"], 4),
            "best_step": best_step["step"],
            "final_reward": round(sum(s["reward_mean"] for s in last5) / len(last5), 4),
            "final_loss": round(sum(s["loss"] for s in last5) / len(last5), 6),
            "final_clip_frac": round(sum(s["clip_frac"] for s in last5) / len(last5), 4),
            "final_grad_norm": round(sum(s["grad_norm"] for s in last5) / len(last5), 4),
            "checkpoint_path": final_path,
        }
        if eval_history:
            best_eval = max(eval_history, key=lambda e: e["accuracy"])
            summary["best_eval_accuracy"] = round(best_eval["accuracy"], 4)
            summary["best_eval_step"] = best_eval["step"]
            summary["final_eval_accuracy"] = round(eval_history[-1]["accuracy"], 4)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        eval_summary_str = ""
        if eval_history:
            eval_summary_str = (
                f" eval_acc={summary['final_eval_accuracy']:.4f} "
                f"best_eval={summary['best_eval_accuracy']:.4f}@step{summary['best_eval_step']}"
            )
        print(
            f"Summary: accuracy={summary['final_accuracy']:.4f} "
            f"best={summary['best_accuracy']:.4f}@step{summary['best_step']} "
            f"reward={summary['final_reward']:.4f} "
            f"stopped_early={summary['early_stopped']}"
            + eval_summary_str
        )


if __name__ == "__main__":
    main()
