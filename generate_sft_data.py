"""
Synthetic SFT data generator for the email search agent.

Constructs gold trajectories from the scenarios dataset + Enron DB:
  1. search_inbox with keywords extracted from the question
  2. Tool response with real search results (target email injected if not found)
  3. read_email on the target message_id
  4. Tool response with full email content
  5. <final_answer>correct answer</final_answer>

Output: JSONL where each line is {"messages": [...]} in chat format.
"""

import argparse
import json
import os
import re
import sys

import email_search as _es

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())


# ── Stopwords for keyword extraction ─────────────────────────────────────────

STOPWORDS = {
    "what", "when", "where", "who", "which", "how", "why", "is", "are", "was",
    "were", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "did", "do", "does", "has", "have", "had", "will",
    "would", "could", "should", "can", "may", "might", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "my", "your", "their", "our",
    "his", "her", "me", "him", "them", "us", "about", "any", "some", "also",
    "than", "then", "just", "more", "most", "very", "much", "many", "each",
    "there", "here", "they", "said", "says", "tell", "told", "know", "knew",
    "want", "need", "like", "make", "made", "give", "given", "take", "taken",
    "come", "came", "going", "into", "over", "after", "before", "between",
    "email", "emails", "inbox", "sent", "received", "message", "messages",
}


def extract_keywords(question: str, max_keywords: int = 4) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
    seen = set()
    result = []
    for w in words:
        if w not in STOPWORDS and w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= max_keywords:
            break
    return result


# ── Tool response formatters (match execute_tool in email_search_grpo.py) ────

def _fmt_search_results(results: list[_es.SearchResult]) -> str:
    if not results:
        return "No emails found matching your search."
    lines = [f"Found {len(results)} email(s):"]
    for r in results:
        lines.append(f"- message_id={r.message_id} | snippet={r.snippet}")
    return "\n".join(lines)


def _fmt_email(email: _es.Email) -> str:
    body_snippet = (email.body or "")[:5000] or "(no body)"
    lines = [
        f"Date: {email.date}",
        f"From: {email.from_address}",
        f"To: {', '.join(email.to_addresses)}",
    ]
    if email.cc_addresses:
        lines.append(f"Cc: {', '.join(email.cc_addresses)}")
    lines += [f"Subject: {email.subject}", "---", body_snippet]
    return "\n".join(lines)


def _tool_response(content: str) -> dict:
    return {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}


# ── Trajectory construction ───────────────────────────────────────────────────

def build_trajectory(
    scenario: _es.Scenario,
    system_prompt: str,
    conn=None,
) -> list[dict] | None:
    """
    Build a gold trajectory for the scenario. Returns None if the target email
    cannot be found (email missing from DB).
    """
    target_id = scenario.message_ids[0]

    # Look up the target email
    target_email = _es.read_email(target_id, conn=conn)
    if target_email is None:
        return None

    # Try progressively fewer keywords until the search returns ≥1 result
    keywords = extract_keywords(scenario.question)
    search_results = None
    used_keywords = keywords

    for n in range(len(keywords), 0, -1):
        kw = keywords[:n]
        try:
            results = _es.search_emails(
                inbox=scenario.inbox_address,
                keywords=kw,
                sent_before=scenario.query_date,
                conn=conn,
            )
            if results:
                search_results = results
                used_keywords = kw
                break
        except Exception:
            continue

    # If no search results, fall back to a single-keyword subject search
    if not search_results:
        subject_words = extract_keywords(target_email.subject or "", max_keywords=2)
        if subject_words:
            try:
                results = _es.search_emails(
                    inbox=scenario.inbox_address,
                    keywords=subject_words,
                    sent_before=scenario.query_date,
                    conn=conn,
                )
                if results:
                    search_results = results
                    used_keywords = subject_words
            except Exception:
                pass

    # Ensure the target email appears in the search results
    if search_results:
        ids_in_results = {r.message_id for r in search_results}
        if target_id not in ids_in_results:
            # Inject target at the top with a plain snippet
            snippet = (target_email.body or "")[:120].replace("\n", " ")
            injected = _es.SearchResult(message_id=target_id, snippet=snippet)
            search_results = [injected] + list(search_results[:4])
    else:
        # No results at all — build a minimal result with just the target
        snippet = (target_email.body or "")[:120].replace("\n", " ")
        search_results = [_es.SearchResult(message_id=target_id, snippet=snippet)]
        used_keywords = extract_keywords(scenario.question, max_keywords=2) or ["information"]

    # Build the message list with <think> blocks before each action
    kw_str = ", ".join(f'"{k}"' for k in used_keywords)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
        {
            "role": "assistant",
            "content": (
                f"I need to find emails relevant to this question. "
                f"I'll search with keywords: {kw_str}.\n"
                f'<tool_call>\n'
                f'{{"name": "search_inbox", "arguments": {{"keywords": {json.dumps(used_keywords)}}}}}\n'
                f'</tool_call>'
            ),
        },
        _tool_response(_fmt_search_results(search_results)),
        {
            "role": "assistant",
            "content": (
                f"I can see the relevant email in the results. Let me read it in full.\n"
                f'<tool_call>\n'
                f'{{"name": "read_email", "arguments": {{"message_id": "{target_id}"}}}}\n'
                f'</tool_call>'
            ),
        },
        _tool_response(_fmt_email(target_email)),
        {
            "role": "assistant",
            "content": (
                f"I have the email content. The answer to the question is: {scenario.answer} "
                f"<final_answer>{scenario.answer}</final_answer>"
            ),
        },
    ]
    return messages


# ── System prompt (kept in sync with email_search_grpo.py) ───────────────────

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


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT data for email search agent")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Max scenarios to process")
    parser.add_argument("--max_messages", type=int, default=1, help="Max relevant emails per scenario")
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--db_path", type=str, default="./enron.db")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (default: sft_{split}.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"sft_{args.split}.jsonl"

    # Load DB — open once and reuse for all queries
    if args.db_path:
        _es.DB_PATH = args.db_path
    conn = _es.get_db_connection()
    print(f"DB loaded: {args.db_path}")

    # Load scenarios
    scenarios = _es.load_training_scenarios(
        split=args.split,
        limit=args.limit,
        max_messages=args.max_messages,
        shuffle=True,
        seed=args.seed,
    )

    # Generate trajectories
    written = 0
    skipped = 0

    with open(args.output, "w") as out:
        for i, scenario in enumerate(scenarios):
            if i % 500 == 0:
                print(f"  [{i}/{len(scenarios)}] written={written} skipped={skipped}")

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                max_turns=args.max_turns,
                inbox_address=scenario.inbox_address,
                query_date=scenario.query_date,
            )

            messages = build_trajectory(scenario, system_prompt, conn=conn)
            if messages is None:
                skipped += 1
                continue

            out.write(json.dumps({"messages": messages}) + "\n")
            written += 1

    print(f"\nDone. Written: {written}  Skipped: {skipped}  Output: {args.output}")


if __name__ == "__main__":
    main()
