# Quick Reference: Stress Index Formula & Thresholds

**One-page cheat sheet for your advisor. Print this.**

---

## Core Formula

### Step 1: Z-Score Normalization
For each feature $x_i$:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

where $\mu_i$ and $\sigma_i$ are per-subject/session mean and std.

**HRV inversion** (lower HRV = stress):
$$z_{\text{rmssd}} = -z_{\text{rmssd}}, \quad z_{\text{pnn50}} = -z_{\text{pnn50}}$$

### Step 2: Weighted Aggregation
$$u = \sum_{i=1}^{8} w_i \cdot z_i$$

where $w = [0.038, 0.266, 0.134, 0.114, 0.066, 0.084, 0.263, 0.035]$ for:
- hr_mean, hr_std, hrv_rmssd, hrv_pnn50, acc_mean, acc_std, acc_energy, activity_level

### Step 3: Clamping
$$u' = \max(-3, \min(3, u))$$

### Step 4: Stress Index (Double-Sigmoid)
$$p = \begin{cases}
\sigma(1.8 \cdot (u' - 0.5)) & \text{if } u' \geq 0 \\
1 - \sigma(1.8 \cdot (-u' - 0.5)) & \text{if } u' < 0
\end{cases}$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$.

**Result:** $p \in [0, 1]$; interpret as stress probability.

---

## Threshold Bands

| Score | Label | Interpretation | Action |
|-------|-------|-----------------|--------|
| 0.00–0.25 | Relaxed | Monitor for hypoarousal if sustained |  OK |
| 0.25–0.45 | Mild | Elevated but not crisis |  Monitor |
| 0.45–0.65 | Moderate | Consider intervention |  Intervene |
| 0.65–0.85 | High | Significant stress |  Alert |
| 0.85–1.00 | Severe | Crisis-level |  Urgent |

---

## Feature Weights (Why These Numbers?)

| Feature | Weight | % | Reason |
|---------|--------|---|--------|
| hr_std | 0.266 | 38% | HR variability = sympathetic activation |
| acc_energy | 0.263 | 37% | Motion/fidgeting = behavioral stress marker |
| hrv_rmssd | 0.134 | 19% | Parasympathetic tone (lower = stress) |
| hrv_pnn50 | 0.114 | 16% | HRV variability |
| acc_std | 0.084 | 12% | Movement variability |
| acc_mean | 0.066 | 9% | Baseline activity |
| hr_mean | 0.038 | 5% | Absolute HR (weak alone) |
| activity_level | 0.035 | 5% | Context (sedentary vs. active) |

**Derived from:** LOSO Random Forest feature importances (WESAD, 15 subjects, 600 windows).

---

## Performance Metrics (LOSO CV, PPG+ACC)

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 87.6% ± 9.6% | 9 out of 10 windows classified correctly |
| **AUC** | 0.952 ± 0.079 | Excellent ranking; low false-positive rate |
| **Sensitivity** | ~88% | Catches most stress events |
| **Specificity** | ~87% | Minimal false alarms |
| **Model** | Random Forest | Best among RF, XGB, SVM |
| **Features** | 16 (PPG + ACC) | Smartwatch-only; no TEMP |

**Context:** Dropping temperature (unavailable on smartwatch) costs ~2.6% accuracy vs. research-grade.

---

## Parameter Justification

### Why Double-Sigmoid?
- **Single sigmoid saturates too fast** near ±2σ
- **Double-sigmoid stays linear** longer; better clinical sensitivity
- **k=1.8, b=0.5** chosen so $p \approx 0.1$ at $-2\sigma$ and $p \approx 0.9$ at $+2\sigma$

### Why Clamp ±3σ?
- Prevents rare outliers from dominating
- ~99.7% of normal data falls within ±3σ
- Beyond ±3σ, signal becomes unreliable (sensor artifact, edge case)

### Why Invert HRV?
- **Physiology:** Lower HRV = parasympathetic withdrawal = stress
- **Implementation:** Flip sign so stress formula is consistent (high z-score → high stress)

---

## Example: Relaxed → Stressed Transition

### Baseline (Relaxed State)
```
hr_mean = 70 BPM      → z = (70-72)/12 = -0.17
hr_std = 6 BPM        → z = (6-8)/4 = -0.50
hrv_rmssd = 40 ms     → z = -(40-35)/15 = -0.33  [inverted]
acc_energy = 0.2 m²/s² → z = (0.2-0.3)/0.4 = -0.25
... [other features near 0]

u = 0.266*(-0.5) + 0.263*(-0.25) + ... ≈ -0.3
p = 1 - σ(1.8 * (0.3 - 0.5)) = 1 - σ(-0.36) ≈ 0.41 → "Mild"
```

### Stressed State
```
hr_mean = 90 BPM      → z = (90-72)/12 = 1.50
hr_std = 15 BPM       → z = (15-8)/4 = 1.75
hrv_rmssd = 15 ms     → z = -(15-35)/15 = 1.33  [inverted]
acc_energy = 0.8 m²/s² → z = (0.8-0.3)/0.4 = 1.25
... [other features elevated]

u = 0.266*(1.75) + 0.263*(1.25) + ... ≈ 1.4
p = σ(1.8 * (1.4 - 0.5)) = σ(1.62) ≈ 0.83 → "High"
```

---

## Code Reference

| Use Case | Script | Command |
|----------|--------|---------|
| Train & validate | `pipelines/smartwatch_loso_validation.py` | `python smartwatch_loso_validation.py` |
| Compute stress index | `pipelines/smartwatch_optimized_stress_detector.py` | Import; call `detector.calculate_stress_index(sensor_data)` |
| Export for mobile | `pipelines/convert_model_to_tflite.py` | `python convert_model_to_tflite.py` |
| Visualize results | `analysis/thesis_visualization_suite.py` | `python thesis_visualization_suite.py` |

---

## Key Limitations & Future Work

1. **Subject variability:** ±9.6% in accuracy suggests personalization (rolling baselines) needed
2. **Accelerometer confounding:** Motion weight (37%) can cause false positives in active users
3. **No TEMP:** ~2.6% accuracy cost vs. full-sensor research setup
4. **Population stats:** Baseline means/stds are population averages; live deployment should learn per-user

---

## Questions Your Advisor Might Ask

**Q: Why not just use raw z-scores?**  
A: Raw z-scores are unbounded; double-sigmoid maps to [0,1] and provides gentle sensitivity around clinical thresholds.

**Q: Why these specific threshold bands (0.25, 0.45, 0.65, 0.85)?**  
A: Chosen to align with clinical practice (relaxed, mild, moderate, high, severe) and validate against stress literature.

**Q: Why drop temperature?**  
A: Temperature is unreliable on consumer smartwatches (no active cooling sensor); PPG+ACC are standard. 2.6% accuracy cost is acceptable.

**Q: Why 87.6% accuracy, not higher?**  
A: Feature ceiling with smartwatch sensors; adding more data doesn't help. Personalization & contextual signals (app, calendar) could push higher.

**Q: How do you handle subject variability?**  
A: Per-subject z-scoring normalizes individual physiological differences. LOSO CV measures generalization across new subjects. Variance (±9.6%) is expected and documentable.

---

**For deeper understanding, see:**
- `docs/METHODOLOGY.md` — Data, features, algorithms
- `docs/RESULTS_SUMMARY.md` — Full metrics & tables
- `notebooks/THESIS_WALKTHROUGH.ipynb` — Interactive demo
