# Optimized Smartwatch Stress Detection Threshold Formula

## Purpose

Canonical stress index and parameter justification for **consumer smartwatches** using only PPG + 3-axis accelerometer. All numbers below are derived from the latest LOSO run (WESAD, smartwatch-only, use_temperature=False).

## Sensors

1. **PPG** — heart rate and HRV
2. **3-axis Accelerometer** — motion, fidgeting, activity context

## Canonical Formula (double-sigmoid)

Let each feature be z-scored on a rolling or per-session baseline: $z_i = (x_i - \mu_i) / \sigma_i$. Apply sign flips for features where *lower = more stress*.

Weighted score: $u = \sum_i w_i z_i$.

Clamp: $u = \min(\max(u, -3), 3)$ to avoid extreme tails.

Double-sigmoid mapping (sharp near ±2σ):

\[
p = \begin{cases}
\sigma(k\,(u - b)), & u \ge 0 \\
1 - \sigma(k\,(-u - b)), & u < 0
\end{cases}
\]

with $\sigma(x) = 1 / (1 + e^{-x})$, slope $k = 1.8$, offset $b = 0.5$. This yields $p \approx 0.1$ at $u \approx -2$ and $p \approx 0.9$ at $u \approx +2$.

Suggested decision threshold: $p \in [0.50, 0.60]$ for stress; keep continuous $p$ for graded feedback.

## Empirical Weights (LOSO RF, PPG+ACC)

Direct importances normalized to sum 1.0 (16-feature set used in LOSO pipeline):

- hr_std: 0.1863
- sdnn: 0.1301
- acc_z_energy: 0.1096
- rmssd: 0.0942
- pnn50: 0.0803
- acc_dominant_frequency: 0.0643
- acc_magnitude_std: 0.0587
- acc_magnitude_mean: 0.0464
- acc_y_energy: 0.0390
- acc_x_energy: 0.0358
- acc_entropy: 0.0348
- hr_mean: 0.0266
- hr_max: 0.0259
- acc_activity_level: 0.0245
- hr_min: 0.0245
- lf_hf_ratio: 0.0192

## Deployment Weight Mapping (8 features used in on-device detector)

The on-device implementation aggregates related energies to match the eight available features. The aggregated weights are renormalized to sum 1.0:

- hr_mean: 0.038
- hr_std: 0.266
- hrv_rmssd: 0.134
- hrv_pnn50: 0.114
- acc_mean: 0.066
- acc_std: 0.084
- acc_energy (x+y+z combined): 0.263
- activity_level: 0.035

These eight weights come from summing the axis-level energies and renormalizing the subset of LOSO importances to the deployed feature set. If you later include TEMP or more axis-level terms, recompute weights from the pipeline and update accordingly.

## Baseline Statistics (population priors)

Used only as fallbacks; live systems should maintain a short rolling baseline per user.

```python
baseline_stats = {
   'hr_mean': {'mean': 72.0, 'std': 12.0},
   'hr_std': {'mean': 8.0, 'std': 4.0},
   'hrv_rmssd': {'mean': 35.0, 'std': 15.0},
   'hrv_pnn50': {'mean': 18.0, 'std': 12.0},
   'acc_mean': {'mean': 0.5, 'std': 0.3},
   'acc_std': {'mean': 0.2, 'std': 0.15},
   'acc_energy': {'mean': 0.3, 'std': 0.4},
   'activity_level': {'mean': 0.3, 'std': 0.3}
}
```

## Reference Implementation (pseudo-code)

```python
def stress_index(features, baselines, weights, k=1.8, b=0.5):
   z = []
   for name, value in features.items():
      mu, sd = baselines[name]['mean'], baselines[name]['std']
      z_i = 0.0 if sd == 0 else (value - mu) / sd
      if name in {'hrv_rmssd', 'hrv_pnn50'}:
         z_i = -z_i  # lower HRV = more stress
      z.append(z_i)

   u = sum(w * z_i for w, z_i in zip(weights, z))
   u = max(-3.0, min(3.0, u))

   if u >= 0:
      p = 1 / (1 + exp(-k * (u - b)))
   else:
      p = 1 - 1 / (1 + exp(-k * (-u - b)))

   return max(0.0, min(1.0, p))
```

## Stress Level Bands (optional)

- 0.00–0.25: relaxed (monitor for hypoarousal if sustained)
- 0.25–0.45: mild (elevated but not crisis)
- 0.45–0.65: moderate (heightened, consider intervention)
- 0.65–0.85: high (intervene)
- 0.85–1.00: severe (crisis intervention recommended)

## Rationale for Parameters

- **Weights**: direct normalization of LOSO RF importances (PPG+ACC, no TEMP). Axis-level energies are summed for deployment convenience.
- **Slope/offset (k=1.8, b=0.5)**: targets $p \approx 0.1$ at -2σ and $p \approx 0.9$ at +2σ without saturating early.
- **Clamp [-3, 3]**: prevents rare outliers from dominating.

## If Temperature Becomes Available

Re-run the LOSO pipeline with `use_temperature=True`, export the new feature importances, and re-normalize weights for the deployed feature list. The double-sigmoid mapping remains unchanged.