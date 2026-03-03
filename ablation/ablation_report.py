"""Generate ablation study report from completed GSM8K GRPO runs.

Reads all */args.json, */step_stat_history.json, */eval_history.json,
*/summary.json from the ablation directory and produces:
  - fig1_eval_accuracy_curves.png  (4×2 grid, per-group eval accuracy)
  - fig2_train_accuracy_curves.png (4×2 grid, per-group train accuracy, EMA-smoothed)
  - fig3_final_performance.png     (horizontal bar chart, all configs)
  - fig4_training_dynamics.png     (loss / clip_frac / grad_norm for key runs)
  - fig5_speed_quality.png         (scatter: training time vs final accuracy)
  - report.md                      (summary table + per-group analysis)

Usage:
    python ablation_report.py --results_dir runs/ablation
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not found. Please install it: pip install matplotlib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Experiment metadata (must stay in sync with ablation_study.py)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"name": "baseline",        "group": "baseline",     "label": "baseline"},
    {"name": "lr_1e-6",         "group": "lr",           "label": "lr=1e-6"},
    {"name": "lr_2e-5",         "group": "lr",           "label": "lr=2e-5"},
    {"name": "group_size_4",    "group": "group_size",   "label": "group_size=4"},
    {"name": "group_size_16",   "group": "group_size",   "label": "group_size=16"},
    {"name": "num_groups_4",    "group": "num_groups",   "label": "num_groups=4"},
    {"name": "num_groups_16",   "group": "num_groups",   "label": "num_groups=16"},
    {"name": "epsilon_0.1",     "group": "epsilon",      "label": "ε=0.1"},
    {"name": "epsilon_0.4",     "group": "epsilon",      "label": "ε=0.4"},
    {"name": "temperature_0.7", "group": "temperature",  "label": "temp=0.7"},
    {"name": "temperature_1.4", "group": "temperature",  "label": "temp=1.4"},
    {"name": "mu_2",            "group": "mu",           "label": "μ=2"},
    {"name": "mu_3",            "group": "mu",           "label": "μ=3"},
    {"name": "lora_rank_16",    "group": "lora_rank",    "label": "lora_rank=16"},
    {"name": "lora_rank_8",     "group": "lora_rank",    "label": "lora_rank=8"},
    {"name": "lora_rank_64",    "group": "lora_rank",    "label": "lora_rank=64"},
    {"name": "lora_rank_128",   "group": "lora_rank",    "label": "lora_rank=128"},
]

ABLATION_GROUPS = [
    {"key": "lr",          "title": "Learning Rate",    "members": ["lr_1e-6", "lr_2e-5"]},
    {"key": "group_size",  "title": "Group Size",       "members": ["group_size_4", "group_size_16"]},
    {"key": "num_groups",  "title": "Num Groups",       "members": ["num_groups_4", "num_groups_16"]},
    {"key": "epsilon",     "title": "Epsilon (clip)",   "members": ["epsilon_0.1", "epsilon_0.4"]},
    {"key": "temperature", "title": "Temperature",      "members": ["temperature_0.7", "temperature_1.4"]},
    {"key": "mu",          "title": "Mu (inner steps)", "members": ["mu_2", "mu_3"]},
    {"key": "lora_rank",   "title": "LoRA Rank",        "members": ["lora_rank_8", "lora_rank_16", "lora_rank_64", "lora_rank_128"]},
]

GROUP_COLORS = {
    "baseline":    "gray",
    "lr":          "tab:blue",
    "group_size":  "tab:orange",
    "num_groups":  "tab:green",
    "epsilon":     "tab:red",
    "temperature": "tab:purple",
    "mu":          "tab:brown",
    "lora_rank":   "tab:pink",
}

EXP_META = {e["name"]: e for e in EXPERIMENTS}

# Colors for ablation variants within each subplot
VARIANT_COLORS = ["tab:blue", "tab:orange"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class RunData:
    """Loaded data for one experiment run."""

    def __init__(self, name: str, run_dir: Path):
        self.name = name
        self.run_dir = run_dir
        self.args = self._load_json("args.json")
        self.step_stats: list[dict] = self._load_json("step_stat_history.json") or []
        self.eval_history: list[dict] = self._load_json("eval_history.json") or []
        self.summary = self._load_json("summary.json")

    def _load_json(self, filename: str):
        path = self.run_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    @property
    def is_complete(self) -> bool:
        return self.summary is not None

    @property
    def has_eval(self) -> bool:
        return len(self.eval_history) > 0

    @property
    def has_steps(self) -> bool:
        return len(self.step_stats) > 0

    def final_eval_accuracy(self):
        if self.eval_history:
            return self.eval_history[-1]["accuracy"]
        return None

    def final_train_accuracy(self):
        if self.step_stats:
            return self.step_stats[-1]["accuracy"]
        return None

    def best_eval_accuracy(self):
        if self.eval_history:
            return max(e["accuracy"] for e in self.eval_history)
        return None

    def total_train_time_minutes(self):
        if not self.step_stats:
            return None
        total_s = sum(
            s.get("t_rollout", 0) + s.get("t_train", 0)
            for s in self.step_stats
        )
        return total_s / 60.0

    def final_accuracy_best_source(self):
        """Return (accuracy, source_label) preferring eval over train."""
        ea = self.final_eval_accuracy()
        if ea is not None:
            return ea, "[eval]"
        ta = self.final_train_accuracy()
        if ta is not None:
            return ta, "[train]"
        return None, "[no data]"


def load_all_runs(results_dir: Path) -> dict:
    runs = {}
    for exp in EXPERIMENTS:
        name = exp["name"]
        run_dir = results_dir / name
        runs[name] = RunData(name, run_dir)
    return runs


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------

def ema_smooth(values: list, alpha: float = 0.1) -> list:
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


# ---------------------------------------------------------------------------
# Figure 1: Eval accuracy curves (4×2 grid)
# ---------------------------------------------------------------------------

def fig1_eval_accuracy_curves(runs: dict, report_dir: Path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle("Eval Accuracy vs Training Step (per hyperparameter group)", fontsize=14)

    baseline_run = runs["baseline"]
    bl_steps = [e["step"] for e in baseline_run.eval_history]
    bl_accs  = [e["accuracy"] for e in baseline_run.eval_history]

    for idx, grp in enumerate(ABLATION_GROUPS):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        ax.set_title(grp["title"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Eval Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 55)
        ax.grid(True, alpha=0.3)

        # Baseline (gray dashed)
        if bl_steps:
            ax.plot(bl_steps, bl_accs,
                    color="gray", linestyle="--", linewidth=1.5, label="baseline", zorder=3)
        else:
            ax.text(0.5, 0.5, "(no baseline data)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)

        # Ablation variants
        for i, member_name in enumerate(grp["members"]):
            run = runs[member_name]
            meta = EXP_META[member_name]
            color = VARIANT_COLORS[i % len(VARIANT_COLORS)]
            if run.has_eval:
                steps = [e["step"] for e in run.eval_history]
                accs  = [e["accuracy"] for e in run.eval_history]
                ax.plot(steps, accs, color=color, linewidth=2, label=meta["label"])
            else:
                ax.text(0.5, 0.4 - i * 0.12,
                        f"{meta['label']}: (no data)",
                        transform=ax.transAxes, ha="center",
                        color=color, fontsize=8)

        ax.legend(fontsize=8, loc="lower right")

    # Hide unused 8th subplot
    axes[3][1].set_visible(False)

    plt.tight_layout()
    out_path = report_dir / "fig1_eval_accuracy_curves.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Train accuracy curves (EMA-smoothed, 4×2 grid)
# ---------------------------------------------------------------------------

def fig2_train_accuracy_curves(runs: dict, report_dir: Path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle("Training Accuracy vs Step (EMA smoothed α=0.1)", fontsize=14)

    baseline_run = runs["baseline"]
    bl_steps = [s["step"] for s in baseline_run.step_stats]
    bl_accs  = [s["accuracy"] for s in baseline_run.step_stats]
    bl_ema   = ema_smooth(bl_accs)

    for idx, grp in enumerate(ABLATION_GROUPS):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        ax.set_title(grp["title"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Baseline: faint raw + bold EMA
        if bl_steps:
            ax.plot(bl_steps, bl_accs,
                    color="gray", alpha=0.2, linewidth=1)
            ax.plot(bl_steps, bl_ema,
                    color="gray", linestyle="--", linewidth=1.5,
                    label="baseline", zorder=3)
        else:
            ax.text(0.5, 0.5, "(no baseline data)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)

        # Ablation variants
        for i, member_name in enumerate(grp["members"]):
            run = runs[member_name]
            meta = EXP_META[member_name]
            color = VARIANT_COLORS[i % len(VARIANT_COLORS)]
            if run.has_steps:
                steps = [s["step"] for s in run.step_stats]
                accs  = [s["accuracy"] for s in run.step_stats]
                accs_ema = ema_smooth(accs)
                ax.plot(steps, accs,     color=color, alpha=0.2, linewidth=1)
                ax.plot(steps, accs_ema, color=color, linewidth=2, label=meta["label"])
            else:
                ax.text(0.5, 0.4 - i * 0.12,
                        f"{meta['label']}: (no data)",
                        transform=ax.transAxes, ha="center",
                        color=color, fontsize=8)

        ax.legend(fontsize=8, loc="lower right")

    axes[3][1].set_visible(False)

    plt.tight_layout()
    out_path = report_dir / "fig2_train_accuracy_curves.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Final performance horizontal bar chart
# ---------------------------------------------------------------------------

def fig3_final_performance(runs: dict, report_dir: Path):
    names   = []
    values  = []
    colors  = []
    sources = []

    baseline_acc, _ = runs["baseline"].final_accuracy_best_source()

    for exp in EXPERIMENTS:
        name = exp["name"]
        run  = runs[name]
        acc, src = run.final_accuracy_best_source()
        names.append(name)
        values.append(acc if acc is not None else 0.0)
        sources.append(src)
        colors.append(GROUP_COLORS.get(exp["group"], "gray"))

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = list(range(len(names)))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    # Hatch the baseline bar
    baseline_idx = names.index("baseline")
    bars[baseline_idx].set_hatch("//")
    bars[baseline_idx].set_edgecolor("black")

    # Annotate each bar with value
    for bar, val, src in zip(bars, values, sources):
        label = f"{val:.3f} {src}" if val > 0 else "(no data)"
        ax.text(
            max(val + 0.01, 0.02),
            bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8,
        )

    # Vertical dashed line at baseline accuracy
    if baseline_acc is not None:
        ax.axvline(
            baseline_acc, color="gray", linestyle="--", linewidth=1.2,
            label=f"baseline ({baseline_acc:.3f})",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Accuracy")
    ax.set_title("Final Performance — All Configs")

    # Group color legend
    patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = report_dir / "fig3_final_performance.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Training dynamics (loss / clip_frac / grad_norm)
# ---------------------------------------------------------------------------

def fig4_training_dynamics(runs: dict, report_dir: Path, highlight_names: list):
    fig, (ax_loss, ax_clip, ax_grad) = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("Training Dynamics — Key Runs", fontsize=14)

    palette = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]

    for i, run_name in enumerate(highlight_names):
        if run_name not in runs:
            continue
        run = runs[run_name]
        if not run.has_steps:
            continue

        color = palette[i % len(palette)]
        ls    = "--" if run_name == "baseline" else "-"
        label = EXP_META.get(run_name, {}).get("label", run_name)

        steps  = [s["step"]      for s in run.step_stats]
        losses = [s["loss"]      for s in run.step_stats]
        clips  = [s["clip_frac"] for s in run.step_stats]
        grads  = [s["grad_norm"] for s in run.step_stats]

        losses_ema = ema_smooth(losses)
        clips_ema  = ema_smooth(clips)
        grads_ema  = ema_smooth(grads)

        ax_loss.plot(steps, losses,     color=color, alpha=0.2, linewidth=1)
        ax_loss.plot(steps, losses_ema, color=color, linestyle=ls, linewidth=2, label=label)

        ax_clip.plot(steps, clips,     color=color, alpha=0.2, linewidth=1)
        ax_clip.plot(steps, clips_ema, color=color, linestyle=ls, linewidth=2, label=label)

        ax_grad.plot(steps, grads,     color=color, alpha=0.2, linewidth=1)
        ax_grad.plot(steps, grads_ema, color=color, linestyle=ls, linewidth=2, label=label)

    for ax, title, ylabel in [
        (ax_loss, "Loss",          "loss"),
        (ax_clip, "Clip Fraction", "clip_frac"),
        (ax_grad, "Gradient Norm", "grad_norm"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = report_dir / "fig4_training_dynamics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Speed-quality scatter + Pareto frontier
# ---------------------------------------------------------------------------

def fig5_speed_quality(runs: dict, report_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Speed vs Quality Tradeoff")
    ax.set_xlabel("Total Training Time (minutes)")
    ax.set_ylabel("Final Accuracy")

    points = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        run  = runs[name]
        t    = run.total_train_time_minutes()
        acc, _ = run.final_accuracy_best_source()
        if t is not None and acc is not None:
            points.append((t, acc, name, exp["group"]))

    if not points:
        ax.text(0.5, 0.5, "No complete runs yet",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        plt.tight_layout()
        out_path = report_dir / "fig5_speed_quality.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")
        return

    for t, acc, name, grp in points:
        color = GROUP_COLORS.get(grp, "gray")
        ax.scatter(t, acc, color=color, s=80, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(name, (t, acc),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Pareto frontier (min time, max accuracy)
    sorted_pts = sorted(points, key=lambda p: p[0])
    pareto = []
    best_acc = -1.0
    for t, acc, name, grp in sorted_pts:
        if acc > best_acc:
            best_acc = acc
            pareto.append((t, acc))

    if len(pareto) >= 2:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.step(px, py, where="post",
                color="black", linestyle=":", linewidth=1.5,
                label="Pareto frontier", zorder=2)
        ax.legend(fontsize=8)

    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = report_dir / "fig5_speed_quality.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report_md(runs: dict, report_dir: Path):
    lines = ["# GSM8K GRPO Ablation Study Report", ""]

    # --- Summary table ---
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Name | Group | Final Eval Acc | Best Eval Acc | Final Train Acc | Source |")
    lines.append("|------|-------|:--------------:|:-------------:|:---------------:|--------|")

    for exp in EXPERIMENTS:
        name = exp["name"]
        run  = runs[name]
        ea   = run.final_eval_accuracy()
        be   = run.best_eval_accuracy()
        ta   = run.final_train_accuracy()
        ea_s = f"{ea:.3f}" if ea is not None else "(no data)"
        be_s = f"{be:.3f}" if be is not None else "(no data)"
        ta_s = f"{ta:.3f}" if ta is not None else "(no data)"
        src  = "[eval]" if ea is not None else "[train]" if ta is not None else "[none]"
        lines.append(f"| {name} | {exp['group']} | {ea_s} | {be_s} | {ta_s} | {src} |")

    lines.append("")

    # --- Per-group analysis ---
    lines.append("## Per-Group Analysis")
    lines.append("")

    baseline_acc, _ = runs["baseline"].final_accuracy_best_source()
    bl_str = f"{baseline_acc:.3f}" if baseline_acc is not None else "N/A"
    lines.append(f"**Baseline accuracy**: {bl_str}")
    lines.append("")

    for grp in ABLATION_GROUPS:
        lines.append(f"### {grp['title']}")
        lines.append("")

        member_results = []
        for member_name in grp["members"]:
            run  = runs[member_name]
            acc, src = run.final_accuracy_best_source()
            meta = EXP_META[member_name]
            member_results.append((meta["label"], acc, src))

        for label, acc, src in member_results:
            if acc is not None and baseline_acc is not None:
                delta = acc - baseline_acc
                sign  = "+" if delta >= 0 else ""
                lines.append(f"- **{label}**: {acc:.3f} {src} ({sign}{delta:.3f} vs baseline)")
            elif acc is not None:
                lines.append(f"- **{label}**: {acc:.3f} {src}")
            else:
                lines.append(f"- **{label}**: (no data)")

        # Auto-generated observation
        valid = [(lbl, acc) for lbl, acc, _ in member_results if acc is not None]
        if valid and baseline_acc is not None:
            best_label, best_acc_val = max(valid, key=lambda x: x[1])
            delta = best_acc_val - baseline_acc
            if abs(delta) < 0.01:
                obs = (
                    f"Little difference from baseline; "
                    f"**{best_label}** is marginally best."
                )
            elif delta > 0:
                obs = f"**{best_label}** outperforms baseline by {delta:+.3f}."
            else:
                obs = (
                    f"All variants underperform baseline; "
                    f"**{best_label}** is least harmful ({delta:+.3f})."
                )
            lines.append("")
            lines.append(f"> **Observation**: {obs}")
        else:
            lines.append("")
            lines.append("> **Observation**: Insufficient data for analysis.")

        lines.append("")

    lines.append("---")
    lines.append("*Generated by `ablation_report.py`*")
    lines.append("")

    report_path = report_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GSM8K GRPO ablation study report"
    )
    parser.add_argument(
        "--results_dir", default="runs/ablation",
        help="Directory containing ablation run subdirectories",
    )
    parser.add_argument(
        "--highlight_runs", nargs="+",
        default=["baseline", "lr_1e-6", "lr_2e-5", "mu_3"],
        help="Runs to highlight in fig4 training dynamics plot",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report_dir  = results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from: {results_dir}")
    runs = load_all_runs(results_dir)

    n_complete  = sum(1 for r in runs.values() if r.is_complete)
    n_with_eval = sum(1 for r in runs.values() if r.has_eval)
    print(f"  {n_complete}/{len(EXPERIMENTS)} complete, {n_with_eval} with eval data")

    print("Generating figures...")
    fig1_eval_accuracy_curves(runs, report_dir)
    fig2_train_accuracy_curves(runs, report_dir)
    fig3_final_performance(runs, report_dir)
    fig4_training_dynamics(runs, report_dir, highlight_names=args.highlight_runs)
    fig5_speed_quality(runs, report_dir)

    print("Generating report.md...")
    generate_report_md(runs, report_dir)

    print(f"\nReport saved to: {report_dir}")


if __name__ == "__main__":
    main()
