"""Standalone GRPO trainer for email search — no TRL, no HF Trainer.

The model learns to use tools (search_inbox, read_email) via multi-turn
agentic rollouts to answer questions about the Enron email dataset.

Complete GRPO training loop in one file:
  load scenarios -> multi-turn rollout (vLLM) -> execute tools -> compute rewards ->
  group advantages -> compute logprobs (HF model) -> PPO-clip loss ->
  optimizer step -> sync LoRA weights back to vLLM -> repeat

Usage:
    python email_search_grpo.py \
        --model_name Qwen/Qwen3-1.7B \
        --max_steps 100 \
        --num_groups 8 \
        --group_size 4 \
        --reward_model gpt-4.1-mini

Smoke test (fast, no GPU):
    python email_search_grpo.py \
        --model_name Qwen/Qwen3-1.7B \
        --max_steps 2 \
        --num_groups 2 \
        --group_size 2 \
        --max_turns 3 \
        --reward_model none
"""

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import time

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# Prevent Cursor's debugger (pydevd) from attaching to vLLM subprocesses.
# The bundled debugger is incompatible with Python 3.12+ (missing `imp` module).
for _key in list(os.environ):
    if _key.startswith(("PYDEVD", "DEBUGPY")):
        del os.environ[_key]

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from dataclasses import dataclass, field
from pathlib import Path

import torch
from vllm import SamplingParams

import email_search as _es

# Qwen3 chat template boundary tokens
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

# Tool call stop tokens — generation halts at these boundaries
TOOL_STOP_TOKENS = ["</tool_call>", "</final_answer>"]

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)

# Bonus added whenever the model produces a <final_answer> tag, regardless of correctness.
# This bootstraps learning when correctness rewards are sparse.
FORMAT_REWARD = 0.1


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SolutionData:
    prompt_ids: list[int]
    completion_ids: list[int]
    assistant_mask: list[int]
    reward: float
    info: dict  # question, ground_truth, final_answer, is_correct, turns, tool_calls


@dataclass
class TrajectoryState:
    """State for a single multi-turn rollout trajectory."""
    scenario: dict
    messages: list[dict]
    done: bool = False
    final_answer: str | None = None
    turns: int = 0
    tool_call_count: int = 0


# ============================================================================
# Dataset loading
# ============================================================================

def load_email_scenarios(split: str = "train") -> list[dict]:
    """Load email QA scenarios from corbt/enron_emails_sample_questions.

    Delegates to email_search.load_training_scenarios and converts to dicts.
    """
    scenarios = _es.load_training_scenarios(split=split)
    return [s.model_dump() for s in scenarios]


# ============================================================================
# Constants
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = (
    "You are an email search agent. You are given a user query and a list of tools "
    "you can use to search the user's email. Use the tools to search the user's emails "
    "and find the answer to the user's query. You may take up to {max_turns} turns to "
    "find the answer, so if your first search doesn't find the answer, try with different keywords.\n\n"
    "User's email address is {inbox_address}\n"
    "Today's date is {query_date}\n\n"
    "Before each action, briefly explain your reasoning, then either call a tool or "
    "provide your final answer inside <final_answer></final_answer> tags."
)

FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "When is the next all-hands meeting?",
    },
    {
        "role": "assistant",
        "content": (
            "I need to find an email about an all-hands meeting. I'll search for those keywords.\n"
            '<tool_call>\n{"name": "search_inbox", "arguments": {"keywords": ["all-hands", "meeting"]}}\n</tool_call>'
        ),
    },
    {
        "role": "user",
        "content": (
            "<tool_response>\n"
            "Found 1 email(s):\n"
            "- message_id=<ABC.1234@enron.com> | snippet=...the <b>all-hands</b> <b>meeting</b> is scheduled for Friday March 9th at 10am in the main conference room...\n"
            "</tool_response>"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The snippet looks relevant. Let me read the full email to confirm the details.\n"
            '<tool_call>\n{"name": "read_email", "arguments": {"message_id": "<ABC.1234@enron.com>"}}\n</tool_call>'
        ),
    },
    {
        "role": "user",
        "content": (
            "<tool_response>\n"
            "Date: 2000-03-07 09:15:00\n"
            "From: ceo@enron.com\n"
            "To: all-staff@enron.com\n"
            "Subject: All-Hands Meeting\n"
            "---\n"
            "Please join us for the all-hands meeting on Friday March 9th at 10am in the main conference room. Attendance is mandatory.\n"
            "</tool_response>"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The email confirms the all-hands meeting is on Friday March 9th at 10am. "
            "<final_answer>Friday March 9th at 10am</final_answer>"
        ),
    },
]

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_inbox",
            "description": "Search emails by keywords with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords":    {"type": "array", "items": {"type": "string"}, "description": "List of keywords to search (AND logic)"},
                    "from_addr":   {"type": "string", "description": "Filter by sender address (exact match)"},
                    "to_addr":     {"type": "string", "description": "Filter by recipient address (exact match)"},
                    "sent_after":  {"type": "string", "description": "Only emails after this date (YYYY-MM-DD)"},
                    "sent_before": {"type": "string", "description": "Only emails before this date (YYYY-MM-DD)"},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Retrieve the full body of an email by its message_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "The message_id from search results"},
                },
                "required": ["message_id"],
            },
        },
    },
]


# ============================================================================
# Database
# ============================================================================

def build_db(db_path: str | None = None) -> sqlite3.Connection:
    """Build or load the Enron email SQLite database via email_search."""
    if db_path:
        _es.DB_PATH = db_path
    return _es.get_db_connection()


# ============================================================================
# Prompt helpers
# ============================================================================

def _initial_messages(scenario: dict, max_turns: int) -> list[dict]:
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        max_turns=max_turns,
        inbox_address=scenario.get("inbox_address", "unknown"),
        query_date=scenario.get("query_date", "unknown"),
    )
    return [
        {"role": "system", "content": system_content},
        *FEW_SHOT_MESSAGES,
        {"role": "user", "content": scenario["question"]},
    ]


def _prompt_str(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS_SCHEMA,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _tokenize_messages(tokenizer, messages: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tools=TOOLS_SCHEMA,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )


# ============================================================================
# Tool execution
# ============================================================================

def execute_tool(tool_json_str: str, inbox_address: str = "", query_date: str = "", conn=None) -> str:
    """Parse and execute a tool call JSON string, returning the result as a string."""
    try:
        call = json.loads(tool_json_str.strip())
        name = call.get("name", "")
        args = call.get("arguments", call.get("args", {}))
    except json.JSONDecodeError as e:
        return f"Invalid tool call JSON: {e}"

    if name == "search_inbox":
        try:
            sent_before = args.get("sent_before") or query_date or None
            results = _es.search_emails(
                inbox=inbox_address,
                keywords=args.get("keywords", []),
                from_addr=args.get("from_addr"),
                to_addr=args.get("to_addr"),
                sent_after=args.get("sent_after"),
                sent_before=sent_before,
                conn=conn,
            )
            if not results:
                return "No emails found matching your search."
            lines = [f"Found {len(results)} email(s):"]
            for r in results:
                lines.append(f"- message_id={r.message_id} | snippet={r.snippet}")
            return "\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"

    elif name == "read_email":
        try:
            email = _es.read_email(args.get("message_id", ""), conn=conn)
            if email is None:
                return f"Email not found: message_id={args.get('message_id')}"
            body_snippet = (email.body or "")[:5000] or "(no body)"
            lines = [
                f"Date: {email.date}",
                f"From: {email.from_address}",
                f"To: {', '.join(email.to_addresses)}",
            ]
            if email.cc_addresses:
                lines.append(f"Cc: {', '.join(email.cc_addresses)}")
            if email.bcc_addresses:
                lines.append(f"Bcc: {', '.join(email.bcc_addresses)}")
            lines += [f"Subject: {email.subject}", "---", body_snippet]
            return "\n".join(lines)
        except Exception as e:
            return f"Read error: {e}"

    else:
        return f"Unknown tool: {name}. Available tools: search_inbox, read_email"


# ============================================================================
# Reward functions
# ============================================================================

def _tokenize_for_f1(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _compute_reward_word_overlap(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference. Returns score in [0, 1]."""
    if not prediction or not reference:
        return 0.0
    pred_tokens = _tokenize_for_f1(prediction)
    ref_tokens = _tokenize_for_f1(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    ref_counter: dict[str, int] = {}
    for t in ref_tokens:
        ref_counter[t] = ref_counter.get(t, 0) + 1
    common = sum(min(pred_counter.get(t, 0), ref_counter[t]) for t in ref_counter)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


async def _llm_judge_single(client, prediction: str, reference: str, question: str, model: str) -> float:
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model answer: {prediction}\n\n"
        "Is the model answer correct? Reply with exactly 'yes' or 'no'."
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().lower()
        return 1.0 if verdict.startswith("yes") else 0.0
    except Exception as e:
        print(f"  [LLM judge error] {e}")
        return 0.0


def _compute_rewards(
    trajectories: list[TrajectoryState],
    reward_model: str,
    openai_client,
) -> list[float]:
    rewards = []
    if reward_model == "none":
        for traj in trajectories:
            if traj.final_answer is None:
                rewards.append(0.0)
            else:
                correctness = _compute_reward_word_overlap(traj.final_answer, traj.scenario["answer"])
                rewards.append(min(FORMAT_REWARD + correctness, 1.0))
    else:
        async def _zero():
            return 0.0

        async def _batch():
            tasks = [
                _llm_judge_single(
                    openai_client,
                    traj.final_answer,
                    traj.scenario["answer"],
                    traj.scenario["question"],
                    reward_model,
                ) if traj.final_answer is not None else _zero()
                for traj in trajectories
            ]
            return await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        try:
            raw_rewards = loop.run_until_complete(_batch())
        finally:
            loop.close()

        for traj, raw_reward in zip(trajectories, raw_rewards):
            if traj.final_answer is None:
                reward = 0.0
            else:
                reward = min(FORMAT_REWARD + raw_reward, 1.0)
            rewards.append(reward)

    return rewards


# ============================================================================
# Rollout config
# ============================================================================

@dataclass
class RolloutConfig:
    tokenizer: object
    im_start_id: int
    im_end_id: int
    assistant_role_id: int
    conn: object = None  # Shared SQLite connection; if None, each call opens its own
    reward_model: str = "none"
    openai_client: object = None
    max_turns: int = 8
    max_tokens_per_turn: int = 512


# ============================================================================
# Multi-turn generation loop
# ============================================================================

def _run_turn_loop(
    vllm_engine,
    trajectories: list[TrajectoryState],
    sampling_params,
    config: RolloutConfig,
    lora_request=None,
) -> None:
    """Shared multi-turn loop. Mutates trajectories in-place."""
    gen_kwargs = {} if lora_request is None else {"lora_request": lora_request}

    for _ in range(config.max_turns):
        active_idxs = [i for i, t in enumerate(trajectories) if not t.done]
        if not active_idxs:
            break

        prompts = [_prompt_str(config.tokenizer, trajectories[i].messages) for i in active_idxs]
        outputs = vllm_engine.generate(prompts, sampling_params=sampling_params, use_tqdm=False, **gen_kwargs)

        for traj_i, output in zip(active_idxs, outputs):
            traj = trajectories[traj_i]
            text = output.outputs[0].text
            traj.turns += 1

            fa_match = FINAL_ANSWER_PATTERN.search(text)
            if fa_match:
                traj.final_answer = fa_match.group(1).strip()
                traj.messages.append({"role": "assistant", "content": text})
                traj.done = True
                continue

            tc_match = TOOL_CALL_PATTERN.search(text)
            if tc_match:
                traj.tool_call_count += 1
                tool_result = execute_tool(
                    tc_match.group(1).strip(),
                    inbox_address=traj.scenario.get("inbox_address", ""),
                    query_date=traj.scenario.get("query_date", ""),
                    conn=config.conn,
                )
                traj.messages.append({"role": "assistant", "content": text})
                traj.messages.append({"role": "user", "content": f"<tool_response>\n{tool_result}\n</tool_response>"})
            else:
                # No tool call and no <final_answer> tag — treat as done but with no answer
                traj.final_answer = None
                traj.messages.append({"role": "assistant", "content": text})
                traj.done = True

    for traj in trajectories:
        traj.done = True


def rollout(
    vllm_engine,
    scenarios: list[dict],
    group_size: int,
    config: RolloutConfig,
    lora_request=None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> list[SolutionData]:
    """Run multi-turn agentic rollouts for all (scenario × group_size) trajectories."""
    trajectories: list[TrajectoryState] = []
    for scenario in scenarios:
        for _ in range(group_size):
            msgs = _initial_messages(scenario, config.max_turns)
            trajectories.append(TrajectoryState(scenario=scenario, messages=list(msgs)))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=config.max_tokens_per_turn,
        stop=TOOL_STOP_TOKENS,
        include_stop_str_in_output=True,
    )

    _run_turn_loop(vllm_engine, trajectories, sampling_params, config, lora_request)
    rewards = _compute_rewards(trajectories, config.reward_model, config.openai_client)

    solutions = []
    for traj, reward in zip(trajectories, rewards):
        initial_msgs = _initial_messages(traj.scenario, config.max_turns)
        prompt_ids = _tokenize_messages(config.tokenizer, initial_msgs)
        full_ids = _tokenize_messages(config.tokenizer, traj.messages)
        completion_ids = full_ids[len(prompt_ids):]
        assistant_mask = build_assistant_mask(
            full_ids, len(prompt_ids),
            config.im_start_id, config.im_end_id, config.assistant_role_id,
        )
        is_correct = reward >= 0.5 if config.reward_model == "none" else reward == 1.0
        solutions.append(SolutionData(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            assistant_mask=assistant_mask,
            reward=reward,
            info={
                "question": traj.scenario["question"],
                "ground_truth": traj.scenario["answer"],
                "final_answer": traj.final_answer,
                "is_correct": is_correct,
                "turns": traj.turns,
                "tool_calls": traj.tool_call_count,
                "system_prompt": initial_msgs[0]["content"],
                "completion_messages": traj.messages[len(initial_msgs):],
            },
        ))

    return solutions


def evaluate(
    vllm_engine,
    scenarios: list[dict],
    config: RolloutConfig,
    lora_request=None,
) -> dict:
    """Evaluate model on scenarios using greedy decoding."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=config.max_tokens_per_turn,
        stop=TOOL_STOP_TOKENS,
        include_stop_str_in_output=True,
    )
    trajectories = [
        TrajectoryState(scenario=s, messages=list(_initial_messages(s, config.max_turns)))
        for s in scenarios
    ]
    _run_turn_loop(vllm_engine, trajectories, sampling_params, config, lora_request)
    rewards = _compute_rewards(trajectories, config.reward_model, config.openai_client)

    n = len(scenarios)
    is_llm = config.reward_model != "none"
    return {
        "accuracy": sum(1 for r in rewards if (r == 1.0 if is_llm else r >= 0.5)) / n,
        "final_answer_rate": sum(1 for t in trajectories if t.final_answer is not None) / n,
        "tool_call_rate": sum(1 for t in trajectories if t.tool_call_count > 0) / n,
        "turn_mean": round(sum(t.turns for t in trajectories) / max(n, 1), 2),
        "n_scenarios": n,
    }


# ============================================================================
# Model setup
# ============================================================================

def setup_model(model_name: str, max_seq_length: int, lora_rank: int, resume_from: str | None = None):
    """Load model with Unsloth + LoRA. Returns (model, tokenizer, vllm_engine)."""
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

    # Disable thinking in Qwen3 chat template
    if tokenizer.chat_template and "enable_thinking" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "enable_thinking is not defined or enable_thinking is false",
        )
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
    FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

    return model, tokenizer, vllm_engine


# ============================================================================
# Multi-turn rollout infrastructure (GRPO-specific, module-level)
# ============================================================================

def build_assistant_mask(
    token_ids: list[int], prompt_length: int,
    im_start_id: int, im_end_id: int, assistant_role_id: int,
) -> list[int]:
    """Build mask: 1 for assistant content tokens in the completion, 0 elsewhere.

    Handles multi-turn conversations by scanning all assistant turns.
    """
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


# ============================================================================
# Logprob computation
# ============================================================================

def selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Memory-efficient log_softmax for selected tokens only."""
    selected = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    logsumexp = torch.logsumexp(logits, dim=-1)
    return selected - logsumexp


def compute_logprobs(model, input_ids, attention_mask, num_completion_tokens):
    """Forward pass -> per-token log probs for the completion portion."""
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=num_completion_tokens + 1,
    )
    logits = output.logits[:, :-1, :]
    target_ids = input_ids[:, -num_completion_tokens:]
    return selective_log_softmax(logits, target_ids)


# ============================================================================
# GRPO loss
# ============================================================================

def grpo_loss(current_logps, old_logps, advantages, mask, epsilon=0.2):
    """GRPO / PPO-clip loss with assistant-only masking."""
    log_ratio = current_logps - old_logps
    log_ratio = torch.clamp(log_ratio, -3.0, 3.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    adv = advantages.unsqueeze(1)
    per_token = -torch.min(ratio * adv, clipped * adv)
    return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()


# ============================================================================
# ASCII plot
# ============================================================================

def render_reward_plot(values, width=60, height=15, title="Accuracy", eval_points=None):
    """Render ASCII bar chart. eval_points is list of (step, value) tuples plotted as 'X'."""
    n = len(values)
    if n == 0:
        return ""

    all_vals = list(values) + ([v for _, v in eval_points] if eval_points else [])
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        hi = lo + 0.1

    if n > width:
        bucket_size = n / width
        cols = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            cols.append(sum(values[start:end]) / (end - start))
        plot_width = width
    else:
        cols = values
        plot_width = n

    grid = [[" "] * plot_width for _ in range(height)]
    for x, v in enumerate(cols):
        y = int((v - lo) / (hi - lo) * (height - 1))
        grid[y][x] = "█"

    if eval_points:
        for step, v in eval_points:
            x = int((step - 1) / max(n - 1, 1) * (plot_width - 1)) if plot_width > 1 else 0
            x = min(x, plot_width - 1)
            y = int((v - lo) / (hi - lo) * (height - 1))
            grid[y][x] = "X"

    width = plot_width
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
    print(render_reward_plot(values, width, height, title))


def save_reward_plot(values, path, width=60, height=15, title="Accuracy", eval_points=None):
    plot = render_reward_plot(values, width, height, title, eval_points=eval_points)
    if plot:
        with open(path, "w") as f:
            f.write(plot + "\n")


# ============================================================================
# Logging
# ============================================================================

def log_trajectories(log_path, step, solutions, group_size, tokenizer, step_stats=None):
    """Log per-group trajectory details to file as raw decoded text.

    Trained tokens (assistant_mask=1) are wrapped in <<< >>> markers.
    Non-trained completion tokens (tool responses, etc.) appear as plain text.
    The prompt is shown as-is before the completion.
    """
    sep = "═" * 72

    for g in range(0, len(solutions), group_size):
        group = solutions[g:g + group_size]
        if not group:
            continue

        # Per-group stats
        g_rewards = [sol.reward for sol in group]
        g_mean = sum(g_rewards) / len(g_rewards)
        g_std = (sum((r - g_mean) ** 2 for r in g_rewards) / len(g_rewards)) ** 0.5
        g_accuracy = sum(1 for sol in group if sol.info["is_correct"]) / len(group)
        g_tools = sum(1 for sol in group if sol.info["tool_calls"] > 0) / len(group)
        g_turns = sum(sol.info["turns"] for sol in group) / len(group)

        lines = [sep, f"Step {step}  |  Group {g // group_size}"]

        # Step-level stats (loss/grad) shown once per group header alongside group reward
        if step_stats is not None:
            s = step_stats
            lines.append(
                f"loss={s['loss']:.3e}  grad={s['grad_norm']:.2e}  clip={s['clip_frac']:.3f}"
                f"  log_r={s['log_ratio_abs_mean']:.2e}"
                f"  adv={s['adv_abs_mean']:.2e}  zero_adv={s['zero_adv_frac']:.2f}"
                f"  |  reward={g_mean:.3f} ± {g_std:.3f}  accuracy={g_accuracy:.3f}"
                f"  |  turns={g_turns:.1f}  tools={g_tools:.2f}"
                f"  |  [step] reward={s['reward_mean']:.3f} ± {s['reward_std']:.3f}"
                f"  acc={s['accuracy']:.3f}"
            )
        else:
            lines.append(
                f"reward={g_mean:.3f} ± {g_std:.3f}  accuracy={g_accuracy:.3f}"
                f"  |  turns={g_turns:.1f}  tools={g_tools:.2f}"
            )
        lines.append(sep)

        first_info = group[0].info
        lines.append(f"Question:     {first_info['question']}")
        lines.append(f"Ground truth: {first_info['ground_truth']}")
        lines.append("")

        # Shared prompt — identical for all trajectories in this group
        if group[0].prompt_ids:
            lines.append(tokenizer.decode(group[0].prompt_ids, skip_special_tokens=False))
        lines.append("")

        for i, sol in enumerate(group):
            info = sol.info
            correct_marker = "✓" if info["is_correct"] else "✗"
            n_trained = sum(sol.assistant_mask)
            n_total = len(sol.completion_ids)

            lines.append(
                f"── Trajectory {i + 1}/{group_size} [{correct_marker}]"
                f"  reward={sol.reward:.3f}"
                f"  |  turns={info['turns']}  tool_calls={info['tool_calls']}"
                f"  |  trained_tokens={n_trained}/{n_total}"
            )
            lines.append("")

            # Completion only: split into runs by mask value, wrap trained spans in <<<...>>>
            if sol.completion_ids:
                run_ids: list[int] = []
                run_mask = sol.assistant_mask[0]
                chunks: list[tuple[bool, list[int]]] = []
                for tok, m in zip(sol.completion_ids, sol.assistant_mask):
                    if m == run_mask:
                        run_ids.append(tok)
                    else:
                        chunks.append((bool(run_mask), run_ids))
                        run_ids = [tok]
                        run_mask = m
                chunks.append((bool(run_mask), run_ids))

                parts = []
                for trained, ids in chunks:
                    text = tokenizer.decode(ids, skip_special_tokens=False)
                    if trained:
                        parts.append(f"==TRAIN_START=={text}==TRAIN_END==")
                    else:
                        parts.append(text)
                lines.append("".join(parts))
            else:
                lines.append("(empty completion)")

            lines.append("")
            lines.append(f"Final answer: {str(info['final_answer'])}")
            lines.append("")

        lines.append("")
        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")


# ============================================================================
# Training loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone GRPO trainer for email search")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_groups", type=int, default=4,
                        help="Number of distinct scenarios per training step")
    parser.add_argument("--group_size", type=int, default=16,
                        help="Number of trajectories generated per scenario (G)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--mu", type=int, default=1,
                        help="Number of inner GRPO optimization steps per rollout batch")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens_per_turn", type=int, default=512,
                        help="Max tokens per generation step in multi-turn rollout")
    parser.add_argument("--max_turns", type=int, default=8,
                        help="Max tool-use turns per trajectory")
    parser.add_argument("--reward_model", type=str, default="gpt-4.1-mini",
                        help="Reward model: 'none' for word-overlap F1, or OpenAI model name")
    parser.add_argument("--db_path", type=str, default=None,
                        help="Path to persist email SQLite DB (default: in-memory, rebuilt each run)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="Run eval on test set every N steps (0=disabled)")
    parser.add_argument("--eval_size", type=int, default=200,
                        help="Number of test scenarios to sample per eval run")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to LoRA checkpoint dir to resume from")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop if accuracy doesn't improve for N steps (0=disabled)")
    parser.add_argument("--eval_seed", type=int, default=42)
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        from coolname import generate_slug
        args.output_dir = f"./runs/{generate_slug(2)}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    gen_log_path = output_dir / "generations.log"
    gen_log_path.write_text("")

    print("================================================")
    print(f"Output: {output_dir}")
    print("================================================")
    print(
        f"Config: steps={args.max_steps}, groups={args.num_groups}, G={args.group_size}, "
        f"max_turns={args.max_turns}, reward_model={args.reward_model}, "
        f"lr={args.learning_rate}, eps={args.epsilon}, mu={args.mu}"
    )
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")

    # ── 1. Setup OpenAI client (if using LLM judge) ──────────────────────
    openai_client = None
    if args.reward_model != "none":
        try:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print(f"Using LLM judge: {args.reward_model}")
        except ImportError:
            print("Warning: openai package not installed, falling back to word-overlap reward")
            args.reward_model = "none"

    # ── 2. Setup email database ───────────────────────────────────────────
    db_conn = build_db(args.db_path)

    # ── 3. Load scenarios ─────────────────────────────────────────────────
    print("Loading email scenarios...")
    train_data = load_email_scenarios("train")
    print(f"Loaded {len(train_data)} training scenarios")

    eval_data = []
    if args.eval_steps > 0:
        try:
            eval_data = load_email_scenarios("test")
            print(f"Loaded {len(eval_data)} test scenarios")
        except Exception:
            eval_data = train_data[-50:]  # fallback: use tail of train
            print(f"Using {len(eval_data)} train scenarios as eval fallback")

    # ── 4. Setup model ────────────────────────────────────────────────────
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
    lora_request = None

    if args.resume_from:
        lora_sync_path = str(output_dir / "live_lora")
        model.save_lora(lora_sync_path)
        lora_request = model.load_lora(lora_sync_path)
        print("Synced resumed LoRA weights to vLLM")

    # ── 4b. Build rollout config ──────────────────────────────────────────
    config = RolloutConfig(
        tokenizer=tokenizer,
        im_start_id=im_start_id,
        im_end_id=im_end_id,
        assistant_role_id=assistant_role_id,
        conn=db_conn,
        reward_model=args.reward_model,
        openai_client=openai_client,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )

    # ── 5. Training loop ──────────────────────────────────────────────────
    total_solutions = args.num_groups * args.group_size
    accuracy_history = []
    reward_history = []
    step_stat_history = []
    eval_history = []
    best_accuracy = -float("inf")
    steps_since_improvement = 0

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # ── 5a. Sample scenarios for this step ───────────────────────
        scenarios_batch = random.sample(train_data, min(args.num_groups, len(train_data)))

        # ── 5b. Wake vLLM, run multi-turn rollouts, sleep vLLM ───────
        if hasattr(vllm_engine, "wake_up"):
            if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                torch.cuda.empty_cache()
                vllm_engine.wake_up()

        solutions = rollout(
            vllm_engine,
            scenarios_batch,
            args.group_size,
            config,
            lora_request=lora_request,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        if hasattr(vllm_engine, "sleep"):
            if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                vllm_engine.sleep(int(os.environ.get("VLLM_SLEEP_MODE", 1)))

        t_rollout = time.time() - t0

        # ── 5c. Rewards -> advantages (GRPO group normalization) ──────
        rewards = torch.tensor(
            [s.reward for s in solutions], dtype=torch.float32, device=device,
        )
        rewards_grouped = rewards.view(args.num_groups, args.group_size)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = ((rewards_grouped - group_mean) / (group_std + 1e-4)).view(-1)

        # ── 5d. Compute old logprobs ──────────────────────────────────
        old_logps_list = []
        with torch.no_grad():
            for sol in solutions:
                ids = sol.prompt_ids + sol.completion_ids
                input_ids = torch.tensor([ids], device=device)
                attn_mask = torch.ones_like(input_ids)
                num_completion = len(sol.completion_ids)
                if num_completion == 0:
                    old_logps_list.append(None)
                    continue
                old_lp = compute_logprobs(model, input_ids, attn_mask, num_completion)
                old_logps_list.append(old_lp)

        # ── 5e. μ inner optimization steps ───────────────────────────
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
            n_valid = 0

            for i, sol in enumerate(solutions):
                if not sol.completion_ids or old_logps_list[i] is None:
                    continue
                if sum(sol.assistant_mask) == 0:
                    continue

                ids = sol.prompt_ids + sol.completion_ids
                input_ids = torch.tensor([ids], device=device)
                attn_mask = torch.ones_like(input_ids)
                num_completion = len(sol.completion_ids)
                mask = torch.tensor(
                    [sol.assistant_mask], dtype=torch.float32, device=device,
                )
                adv_i = advantages[i:i + 1]

                current_logps = compute_logprobs(model, input_ids, attn_mask, num_completion)
                old_logps = old_logps_list[i]

                loss_i = grpo_loss(
                    current_logps, old_logps, adv_i, mask, args.epsilon,
                )
                (loss_i / total_solutions).backward()
                step_loss += loss_i.item()
                n_valid += 1

                with torch.no_grad():
                    raw_log_ratio = current_logps - old_logps
                    ratio = torch.exp(raw_log_ratio)
                    is_clipped = (
                        ((ratio < 1 - args.epsilon) & (adv_i.unsqueeze(1) < 0)) |
                        ((ratio > 1 + args.epsilon) & (adv_i.unsqueeze(1) > 0))
                    )
                    clip_frac = (is_clipped.float() * mask).sum() / mask.sum().clamp(min=1)
                    step_clip_frac += clip_frac.item()
                    step_log_ratio_sum += (raw_log_ratio.abs() * mask).sum().item()
                    step_log_ratio_tokens += int(mask.sum().item())

            if n_valid > 0:
                step_loss /= total_solutions
                step_clip_frac /= max(n_valid, 1)
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

        # ── 5f. Sync LoRA weights to vLLM ────────────────────────────
        lora_sync_path = str(output_dir / "live_lora")
        model.save_lora(lora_sync_path)
        lora_request = model.load_lora(lora_sync_path)

        # ── 5g. Eval on test set ──────────────────────────────────────
        eval_result = None
        if args.eval_steps > 0 and step % args.eval_steps == 0 and eval_data:
            if hasattr(vllm_engine, "wake_up"):
                if getattr(vllm_engine.llm_engine.vllm_config.model_config, "enable_sleep_mode", False):
                    torch.cuda.empty_cache()
                    vllm_engine.wake_up()

            eval_rng = random.Random(args.eval_seed + step)
            eval_scenarios = eval_rng.sample(eval_data, min(args.eval_size, len(eval_data)))
            eval_result = evaluate(vllm_engine, eval_scenarios, config, lora_request=lora_request)
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

        # ── 5h. Metrics ───────────────────────────────────────────────
        mean_reward = rewards.mean().item()
        std_reward = rewards.std().item()
        accuracy = sum(1 for s in solutions if s.info["is_correct"]) / len(solutions)
        final_answer_rate = sum(1 for s in solutions if s.info["final_answer"] is not None) / len(solutions)
        tool_call_rate = sum(1 for s in solutions if s.info["tool_calls"] > 0) / len(solutions)
        turn_mean = sum(s.info["turns"] for s in solutions) / len(solutions)

        asst_tokens = sum(sum(s.assistant_mask) for s in solutions)
        total_tokens = sum(len(s.completion_ids) for s in solutions)

        accuracy_history.append(accuracy)
        reward_history.append(mean_reward)

        adv_abs_mean = advantages.abs().mean().item()
        adv_std = advantages.std().item()
        n_zero_adv = (advantages.abs() < 1e-4).sum().item()

        step_stats = {
            "step": step,
            "loss": float(f"{total_loss:.6e}"),
            "grad_norm": float(f"{grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm:.4e}"),
            "clip_frac": round(total_clip_frac, 4),
            "log_ratio_abs_mean": float(f"{total_log_ratio_abs_mean:.4e}"),
            "adv_abs_mean": float(f"{adv_abs_mean:.4e}"),
            "adv_std": float(f"{adv_std:.4e}"),
            "zero_adv_frac": round(n_zero_adv / len(advantages), 4),
            "reward_mean": round(mean_reward, 4),
            "reward_std": round(std_reward, 4),
            "accuracy": round(accuracy, 4),
            "final_answer_rate": round(final_answer_rate, 4),
            "tool_call_rate": round(tool_call_rate, 4),
            "turn_mean": round(turn_mean, 2),
            "asst_token_ratio": round(asst_tokens / max(total_tokens, 1), 4),
            "seq_len_mean": round(total_tokens / max(len(solutions), 1), 1),
            "t_rollout": round(t_rollout, 2),
            "t_train": round(t_train, 2),
        }

        eval_str = ""
        if eval_result is not None:
            eval_str = (
                f" | eval_acc={eval_result['accuracy']:.3f} "
                f"eval_tool={eval_result['tool_call_rate']:.2f} "
                f"eval_turns={eval_result['turn_mean']:.1f}"
            )
        print(
            f"[Step {step}/{args.max_steps}] "
            f"loss={total_loss:.3e} grad={step_stats['grad_norm']:.2e} clip={total_clip_frac:.3f} "
            f"log_r={total_log_ratio_abs_mean:.2e} | "
            f"adv={adv_abs_mean:.2e} (std={adv_std:.2e}) zero={n_zero_adv}/{len(advantages)} | "
            f"reward={mean_reward:.3f} (std={std_reward:.3f}) acc={accuracy:.3f} "
            f"fa={final_answer_rate:.2f} tools={tool_call_rate:.2f} turns={turn_mean:.1f} | "
            f"t_roll={t_rollout:.1f}s t_train={t_train:.1f}s"
            + eval_str
        )

        step_stat_history.append(step_stats)
        with open(output_dir / "step_stat_history.json", "w") as f:
            json.dump(step_stat_history, f, indent=2)
        log_trajectories(gen_log_path, step, solutions, args.group_size, tokenizer, step_stats)
        eval_pts = [(e["step"], e["accuracy"]) for e in eval_history] if eval_history else None
        save_reward_plot(accuracy_history, output_dir / "accuracy_plot.txt", title="Accuracy", eval_points=eval_pts)

        # ── 5i. Save checkpoint ───────────────────────────────────────
        if step % args.save_steps == 0:
            ckpt_path = str(output_dir / f"checkpoint-{step}")
            model.save_lora(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # ── 5j. Early stopping ────────────────────────────────────────
        smoothed = sum(accuracy_history[-5:]) / min(len(accuracy_history), 5)
        if smoothed > best_accuracy + 1e-4:
            best_accuracy = smoothed
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if args.early_stop_patience > 0 and steps_since_improvement >= args.early_stop_patience:
            print(f"\n  Early stopping: no improvement for {args.early_stop_patience} steps")
            break

    # ── 6. Save final model ───────────────────────────────────────────────
    final_path = str(output_dir / "final_model")
    model.save_lora(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    # ── 7. Print accuracy plot ────────────────────────────────────────────
    if accuracy_history:
        print_reward_plot(accuracy_history, title="Accuracy")

    # ── 8. Write summary ──────────────────────────────────────────────────
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
            "final_tool_call_rate": round(sum(s["tool_call_rate"] for s in last5) / len(last5), 4),
            "final_turn_mean": round(sum(s["turn_mean"] for s in last5) / len(last5), 2),
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
            f"tools={summary['final_tool_call_rate']:.3f} "
            f"turns={summary['final_turn_mean']:.1f} "
            f"stopped_early={summary['early_stopped']}"
            + eval_summary_str
        )


if __name__ == "__main__":
    main()
