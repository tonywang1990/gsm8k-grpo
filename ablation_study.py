"""Ablation study runner for GSM8K GRPO training.

Runs experiments (one baseline + ablations), each for 50 steps,
then generates a report via ablation_report.py.

Usage:
    python ablation_study.py --dry_run
    python ablation_study.py --only baseline
    python ablation_study.py              # run all
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

BASELINE = {
    "learning_rate": 5e-6,
    "num_groups": 8,
    "group_size": 8,
    "epsilon": 0.2,
    "temperature": 1.0,
    "mu": 1,
    "lora_rank": 32,
}

EXPERIMENTS = [
    {"name": "baseline",         "overrides": {}},
    {"name": "lr_1e-6",          "overrides": {"learning_rate": 1e-6}},
    {"name": "lr_2e-5",          "overrides": {"learning_rate": 2e-5}},
    {"name": "group_size_4",     "overrides": {"group_size": 4}},
    {"name": "group_size_16",    "overrides": {"group_size": 16}},
    {"name": "num_groups_4",     "overrides": {"num_groups": 4}},
    {"name": "num_groups_16",    "overrides": {"num_groups": 16}},
    {"name": "epsilon_0.1",      "overrides": {"epsilon": 0.1}},
    {"name": "epsilon_0.4",      "overrides": {"epsilon": 0.4}},
    {"name": "temperature_0.7",  "overrides": {"temperature": 0.7}},
    {"name": "temperature_1.4",  "overrides": {"temperature": 1.4}},
    {"name": "mu_2",             "overrides": {"mu": 2}},
    {"name": "mu_3",             "overrides": {"mu": 3}},
    {"name": "lora_rank_16",     "overrides": {"lora_rank": 16}},
    {"name": "lora_rank_8",      "overrides": {"lora_rank": 8}},
    {"name": "lora_rank_64",     "overrides": {"lora_rank": 64}},
    {"name": "lora_rank_128",    "overrides": {"lora_rank": 128}},
]

FIXED_ARGS = {
    "max_steps": 50,
    "eval_steps": 10,
    "eval_size": 100,
    "model_name": "Qwen/Qwen3-1.7B",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def build_cmd(exp: dict, base_dir: Path) -> list[str]:
    """Build the subprocess argv list for a given experiment."""
    config = {**BASELINE, **exp["overrides"]}
    run_dir = base_dir / exp["name"]
    cmd = [sys.executable, "gsm8k_grpo.py"]
    cmd += ["--output_dir", str(run_dir)]
    for k, v in {**FIXED_ARGS, **config}.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def is_complete(run_dir: Path) -> bool:
    """Return True if this experiment has already finished successfully."""
    return (run_dir / "summary.json").exists()


def run_experiment(exp: dict, base_dir: Path) -> dict:
    """Run a single experiment, skipping if already complete.

    Returns a manifest entry dict.
    """
    run_dir = base_dir / exp["name"]
    name = exp["name"]

    if is_complete(run_dir):
        print(f"[ablation] Skipping {name!r} — already complete.")
        return {
            "name": name,
            "status": "skipped",
            "elapsed_s": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    cmd = build_cmd(exp, base_dir)
    print(f"\n[ablation] Starting {name!r}")
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = round(time.time() - t0, 1)
    status = "success" if result.returncode == 0 else f"failed(rc={result.returncode})"
    print(f"[ablation] {name!r} finished in {elapsed}s — {status}")
    return {
        "name": name,
        "status": status,
        "elapsed_s": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_manifest(manifest_path: Path) -> list[dict]:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return []


def save_manifest(manifest_path: Path, entries: list[dict]) -> None:
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K GRPO ablation study")
    parser.add_argument(
        "--base_dir", default="runs/ablation",
        help="Root directory for all ablation run outputs",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Run only these experiment names",
    )
    parser.add_argument(
        "--skip_report", action="store_true",
        help="Don't auto-generate report after runs complete",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_dir / "manifest.json"
    # Touch the log file so external `tee` processes can open it immediately
    (base_dir / "ablation_study.log").touch(exist_ok=True)

    # Filter experiments
    exps = EXPERIMENTS
    if args.only:
        exps = [e for e in EXPERIMENTS if e["name"] in args.only]
        if not exps:
            known = [e["name"] for e in EXPERIMENTS]
            print(f"[ablation] No experiments matched {args.only}.")
            print(f"  Known names: {known}")
            sys.exit(1)

    if args.dry_run:
        print("[ablation] DRY RUN — commands that would be executed:\n")
        for exp in exps:
            cmd = build_cmd(exp, base_dir)
            print(f"  [{exp['name']}]")
            print(f"  {' '.join(cmd)}\n")
        return

    manifest = load_manifest(manifest_path)

    for exp in exps:
        entry = run_experiment(exp, base_dir)
        # Replace any existing entry for this name
        manifest = [m for m in manifest if m["name"] != entry["name"]]
        manifest.append(entry)
        save_manifest(manifest_path, manifest)

    print(f"\n[ablation] All experiments done. Manifest saved to {manifest_path}")

    if not args.skip_report:
        report_cmd = [
            sys.executable, "ablation_report.py",
            "--results_dir", str(base_dir),
        ]
        print(f"\n[ablation] Generating report...")
        print(f"  cmd: {' '.join(report_cmd)}")
        subprocess.run(report_cmd, check=False)


if __name__ == "__main__":
    main()
