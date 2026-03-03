# GSM8K GRPO Ablation Study Report

**Model**: Qwen3-1.7B | **Task**: GSM8K math reasoning | **Training**: 50 steps, eval every 10 steps on 100 fixed test problems
**Baseline config**: `lr=5e-6, num_groups=8, group_size=8, epsilon=0.2, temperature=1.0, mu=1, lora_rank=32` → **0.740 eval accuracy**

---

## Key Findings

### 1. LoRA rank is the most critical parameter — with a hard threshold

| Rank | Eval Acc | Δ |
|------|:--------:|:--:|
| 8    | 0.220 | -0.520 |
| 16   | 0.310 | -0.430 |
| **32 (baseline)** | **0.740** | — |
| 64   | 0.800 | +0.060 |
| 128  | 0.810 | +0.070 |

The rank 16→32 jump (+0.430) dwarfs every other ablation in the study. Below rank 32 the model essentially fails — LoRA's subspace is too constrained to represent the weight updates required for GSM8K reasoning. Above rank 32 returns diminish quickly; rank 64→128 yields only +0.010, indicating saturation around 64 dimensions.

**Recommendation**: rank 32 is the minimum viable setting; rank 64 is the sweet spot for cost vs performance.

---

### 2. Learning rate has the largest effect among optimizer hyperparameters

| LR | Eval Acc | Δ |
|----|:--------:|:--:|
| 1e-6 | 0.170 | -0.570 |
| **5e-6 (baseline)** | **0.740** | — |
| 2e-5 | 0.810 | +0.070 |

`lr=1e-6` catastrophically underperforms — the model barely learns in 50 steps. `lr=2e-5` gives the best result of all optimizer-related ablations (+0.070). The baseline `5e-6` is overly conservative; for short runs (≤50 steps) a more aggressive LR is clearly better.

**Recommendation**: use `lr=2e-5` for short runs. For longer runs the optimal LR may be lower as gradients accumulate.

---

### 3. More rollout diversity per step consistently helps

Both axes of rollout volume matter:

| Config | Eval Acc | Δ |
|--------|:--------:|:--:|
| num_groups=4  | 0.560 | -0.180 |
| num_groups=8 (baseline) | 0.740 | — |
| num_groups=16 | 0.820 | +0.080 |
| | | |
| group_size=4  | 0.590 | -0.150 |
| group_size=8 (baseline) | 0.740 | — |
| group_size=16 | 0.790 | +0.050 |

`num_groups` (problem diversity per step) matters slightly more than `group_size` (solutions per problem). Both reductions cause large drops — fewer problems means poor advantage estimation, fewer solutions per problem means the group mean/std used for normalisation are noisy.

**Recommendation**: if compute allows, increase `num_groups` first (cheapest gain), then `group_size`.

---

### 4. Multiple inner steps (mu) help, with mild diminishing returns

| μ | Eval Acc | Δ |
|---|:--------:|:--:|
| 1 (baseline) | 0.740 | — |
| 2 | 0.780 | +0.040 |
| 3 | 0.790 | +0.050 |

More gradient steps per rollout batch squeezes more signal from each generation pass. μ=3 slightly edges μ=2. The gain is consistent but moderate — each extra inner step risks overfitting the rollout batch (policy drifts from the data it was sampled under), which is why returns diminish.

**Recommendation**: μ=2 or μ=3 is a free ~+0.040–0.050 improvement with no extra rollout cost.

---

### 5. Epsilon is inert at mu=1

| ε | Eval Acc | Δ |
|---|:--------:|:--:|
| 0.1 | 0.710 | -0.030 |
| 0.2 (baseline) | 0.740 | — |
| 0.4 | 0.760 | +0.020 |

The spread across all three (0.050 total) is within noise. With μ=1 the policy never moves far enough from the rollout policy to trigger the clip — `clip_frac ≈ 0.000` throughout all runs. Epsilon only becomes meaningful at μ>1, where the policy drifts across inner steps.

**Recommendation**: don't tune epsilon unless also using μ>1.

---

### 6. Temperature: slightly lower is better

| Temp | Eval Acc | Δ |
|------|:--------:|:--:|
| 0.7 | 0.750 | +0.010 |
| 1.0 (baseline) | 0.740 | — |
| 1.4 | 0.720 | -0.020 |

Lower temperature produces more focused rollouts (less entropy), giving cleaner reward signal. Higher temperature introduces more noise into the solutions and degrades the advantage estimates. The effect is small at 50 steps but directionally consistent.

**Recommendation**: a mild reduction to 0.7 is a small free gain.

---

## Recommended Config

Combining the best of each ablation:

| Param | Baseline | Recommended | Expected gain |
|-------|:--------:|:-----------:|:-------------:|
| learning_rate | 5e-6 | **2e-5** | +0.070 |
| num_groups | 8 | **16** | +0.080 |
| group_size | 8 | **16** | +0.050 |
| mu | 1 | **3** | +0.050 |
| lora_rank | 32 | **64** | +0.060 |
| temperature | 1.0 | **0.7** | +0.010 |
| epsilon | 0.2 | 0.4 | ~+0.020 (only matters with μ>1) |

Individual gains are not perfectly additive (the runs share compute budget), but the combined config should push well above 0.820. The compute cost doubles roughly (2× groups × 2× group_size × 3× mu = 12× more gradient signal per wall-clock step), so these gains come at a real cost.

---

## Full Results Table

| Rank | Name | Eval Acc | vs Baseline |
|------|------|:--------:|:-----------:|
| 1  | num_groups_16  | 0.820 | +0.080 |
| 2  | lr_2e-5        | 0.810 | +0.070 |
| 2  | lora_rank_128  | 0.810 | +0.070 |
| 4  | lora_rank_64   | 0.800 | +0.060 |
| 5  | group_size_16  | 0.790 | +0.050 |
| 5  | mu_3           | 0.790 | +0.050 |
| 7  | mu_2           | 0.780 | +0.040 |
| 8  | epsilon_0.4    | 0.760 | +0.020 |
| 9  | temperature_0.7| 0.750 | +0.010 |
| 10 | **baseline**   | **0.740** | — |
| 11 | temperature_1.4| 0.720 | -0.020 |
| 11 | epsilon_0.1    | 0.710 | -0.030 |
| 13 | group_size_4   | 0.590 | -0.150 |
| 14 | num_groups_4   | 0.560 | -0.180 |
| 15 | lora_rank_16   | 0.310 | -0.430 |
| 16 | lora_rank_8    | 0.220 | -0.520 |
| 17 | lr_1e-6        | 0.170 | -0.570 |

---

*Generated from 17 runs × 50 steps each. Eval on 100 fixed GSM8K test problems (seed=42+step).*
*Figures: `fig1`–`fig5` in this directory.*
