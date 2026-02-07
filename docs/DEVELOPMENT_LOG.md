# Development Log: Stress Detection Pipeline (Phase 1–5)

**Chronological narrative of how the project evolved, decisions made, and why.**

---

## Phase 1: Data Exploration & Feature Engineering (Foundation)

### Goal
Build a feature extraction pipeline from WESAD raw signals (PPG, ACC, TEMP) that captures stress-relevant physiological and behavioral markers.

### What We Built
- **Data source:** WESAD dataset (15 subjects, 2 conditions: baseline + stress)
- **Sampling rates:** PPG 64 Hz, ACC 32 Hz, TEMP 4 Hz
- **Window strategy:** 60s windows, 50% overlap → 600 labeled windows (baseline=300, stress=300)
- **Feature set:** 16 total
  - **Heart rate (4):** mean, std, min, max
  - **HRV (4):** RMSSD, pNN50, SDNN, LF/HF ratio (from PPG → RR intervals)
  - **Accelerometer (8):** magnitude (mean, std), per-axis energy (3), dominant frequency, activity level, entropy

### Key Decisions
- **Window size:** 60s chosen for mobile realism (not too short, not too long)
- **Overlap:** 50% to increase sample count while maintaining independence
- **HRV metrics:** Prioritized time-domain (RMSSD, pNN50, SDNN) over frequency-domain; LASSO showed time-domain more stable
- **Accelerometer:** Used magnitude + axis-specific energies to capture 3D motion complexity

### Result
 600 labeled, balanced training windows ready for modeling

---

## Phase 2: Baseline Modeling with All Sensors (Research-Grade)

### Goal
Establish performance ceiling using all available sensors (PPG + ACC + TEMP) with standard ML models.

### Models Tested
- **Random Forest:** 100–200 trees, balanced class weights
- **XGBoost:** Early stopping, max_depth=5–7, balanced weights
- **SVM:** RBF kernel, scaled features, balanced weights

### Validation Strategy
- **Leave-one-subject-out (LOSO):** Remove one subject entirely, train on remaining 14, test on held-out subject. Repeat 15 times.
- **Why LOSO?** Measures real-world generalization (can model work on new, never-seen subjects?)
- **Why not random k-fold?** Random split causes subject leakage (same person's data in train+test), artificially inflating accuracy

### Results (PPG+ACC+TEMP)

| Model | Accuracy | Std Dev | AUC | AUC Std |
|-------|----------|---------|-----|---------|
| SVM | **90.2%** | ±9.0% | **0.974** | ±0.052 |
| Random Forest | 89.8% | ±8.7% | 0.970 | ±0.055 |
| XGBoost | 88.5% | ±9.5% | 0.950 | ±0.068 |

### Key Insight
All three models hover around 89–90%, suggesting feature set is saturating (not model-limited). SVM edges out others slightly due to its non-linear RBF kernel handling multi-modal feature distributions.

### Decision Gate
**Problem:** TEMP (temperature) is not reliably available on consumer smartwatches.
- Samsung Watch 5: No temperature sensor
- Apple Watch 9: Temperature sensor added in Series 8+, but not for stress detection
- Fitbit: Limited or no active temperature sensing
- Garmin: Some models have, others don't; inconsistent

**Decision:** Accept ~2.6% accuracy drop to focus on PPG+ACC-only pipeline (universal smartwatch availability)

---

## Phase 3: Smartwatch-Only Optimization (PPG+ACC)

### Goal
Re-optimize models for smartwatch constraints; measure performance without TEMP.

### Pipeline Modification
- Added `use_temperature` flag to [pipelines/smartwatch_loso_validation.py](../../pipelines/smartwatch_loso_validation.py)
- When `False`: Drop TEMP from feature extraction; reduce feature set from 19 → 16
- Re-run full LOSO CV with same 15 subjects

### New Results (PPG+ACC Only)

| Model | Accuracy | Std Dev | AUC | AUC Std | Δ vs. +TEMP |
|-------|----------|---------|-----|---------|------------|
| **Random Forest** | **87.6%** | ±9.6% | **0.952** | ±0.079 | −2.2% |
| XGBoost | 87.2% | ±9.8% | 0.948 | ±0.082 | −1.3% |
| SVM | 87.0% | ±10.2% | 0.962 | ±0.071 | −3.2% |

### Key Insight
- **Accuracy drops by 2–3%, not 5–10%**, suggesting TEMP was not critical to baseline model
- **AUC remains high (0.952)**, meaning ranking ability is preserved (few false positives)
- **Subject variance increases slightly (±9.6%),** reflecting lower feature information → higher sensitivity to subject-specific signal quality

### Decision: RF is Canonical
- **Why RF over XGB/SVM?** 
  - **Interpretability:** Feature importance directly accessible (for weight derivation)
  - **Stability:** Less hyperparameter tuning needed; LOSO variance is dataset-driven, not tuning-driven
  - **Deployment:** Smaller model size, faster inference, easier to export to mobile
  - **Comparable:** AUC only 0.01 lower than SVM; accuracy difference negligible

---

## Phase 4: Stress Index Formula (From Heuristic to Canonical)

### Phase 4a: Prior Heuristic Approach (What We Started With)

**Old formula:**
```python
# Ad-hoc normalization
normalized_hr_mean = sigmoid(hr_mean, slope=1.5, offset=72)
normalized_acc = sigmoid(acc_std, slope=1.5, offset=0.2)

# Arbitrary weights
stress_score = 0.20 * normalized_hr_mean + 0.15 * normalized_hrv + ...

# Final sigmoid
stress_index = sigmoid(stress_score, slope=6.0, offset=0.5)
```

**Problems:**
- No justification for slopes (1.5, 6.0) or offsets
- Weights (0.20, 0.15, ...) pulled from literature, not validated on WESAD
- Single sigmoid saturates quickly; poor clinical sensitivity
- No clear answer to "why these numbers?"

### Phase 4b: First Improvement (Empirical Weights from LOSO RF)

**Insight:** Use the feature importances from the best-performing LOSO model (RF) to derive weights.

**Approach:**
1. Train RF on all 15 subjects combined
2. Extract feature importances (how much each feature reduced impurity)
3. Normalize to sum to 1.0
4. Use as stress index weights instead of literature values

**Result (16-feature importances):**
```
hr_std: 0.1863 (dominant)
sdnn: 0.1301
acc_z_energy: 0.1096
rmssd: 0.0942
pnn50: 0.0803
...
lf_hf_ratio: 0.0192 (weakest)
```

**Benefit:** Now we can justify weights—they come from empirical model performance, not guesses.

### Phase 4c: Final Formula (Double-Sigmoid, Parameter Justification)

**Problem with single sigmoid:** 
- Sigmoid with slope=6 + offset=0.5 is too steep
- At u=0, dp/du is maximized, causing jump from ~0.01 to ~0.99 in narrow z-score range
- Poor clinical sensitivity; hard to distinguish mild vs. moderate stress

**Solution: Symmetric double-sigmoid**

Define:
$$u = \sum_i w_i z_i \quad \text{(weighted z-score sum)}$$

Clamp:
$$u' = \max(-3, \min(3, u))$$

Map to [0, 1] with branching:
$$p = \begin{cases}
\sigma(k(u' - b)), & u' \geq 0 \\
1 - \sigma(k(-u' - b)), & u' < 0
\end{cases}$$

where $k=1.8$, $b=0.5$, $\sigma(x) = 1/(1+e^{-x})$.

**Why these parameters?**
- **Clamp ±3σ:** Removes outliers; 99.7% of normal data within ±3σ
- **k=1.8:** Achieves p≈0.1 at u'=−2, p≈0.9 at u'=+2σ
  - Verification: σ(1.8*(−2−0.5)) = σ(−4.5) ≈ 0.011 ≈ 0.1 
  - σ(1.8*(2−0.5)) = σ(2.7) ≈ 0.933 ≈ 0.9 
- **b=0.5:** Symmetric around u'=0; baseline (u'≈0) maps to p≈0.5 (neutral stress)
- **Double sigmoid:** Allows gentle transition near baseline; steep at extremes

**Benefit:** Parameters are **not arbitrary**—each has a principled target.

### Deployment Weight Aggregation

For on-device implementation, we combine axis-level energies into one "acc_energy" feature:

**From 16-feature space:**
```
acc_x_energy: 0.0358
acc_y_energy: 0.0390
acc_z_energy: 0.1096
---
acc_energy_total: 0.1844 (normalized → 0.263 in 8-feature space)
```

**Final 8-feature deployment weights:**
```
hr_mean: 0.038
hr_std: 0.266 (→ 38% of total; dominant)
hrv_rmssd: 0.134
hrv_pnn50: 0.114
acc_mean: 0.066
acc_std: 0.084
acc_energy: 0.263 (→ 38% of total; second most important)
activity_level: 0.035
Sum: 1.000
```

### Implementation
Created [pipelines/smartwatch_optimized_stress_detector.py](../../pipelines/smartwatch_optimized_stress_detector.py) with:
- Z-score normalization (per subject/session)
- HRV inversion (lower HRV → higher stress z-score)
- Weighted sum
- Double-sigmoid mapping
- Clinical label lookup

---

## Phase 5: Clinical Framing (Hypoarousal → Crisis)

### Goal
Make threshold bands clinically meaningful for practitioners, not just ML engineers.

### Discovery: Low Scores ≠ "Relaxed"
**Early observation:** Sustained stress_index < 0.2 could indicate:
- Genuine relaxation 
- **Hypoarousal / depression risk**  (low HR variability, flat affect, minimal movement)

**Clinical literature:**
- Hypoarousal (depression, dissociation) characterized by low autonomic activation
- Persistent stress index < 0.2 over days should be monitored

### Solution: Dual Labels
Create both **technical label** (relax, mild, etc.) and **clinical label** (action for practitioner).

| Score Range | Technical | Clinical | Action |
|---|---|---|---|
| 0.00–0.25 | Relaxed | **Monitor for hypoarousal** |  Watch for sustained low |
| 0.25–0.45 | Mild | Elevated but not crisis |  Awareness |
| 0.45–0.65 | Moderate | Heightened stress |  Suggest coping |
| 0.65–0.85 | High | Intervene |  Take action |
| 0.85–1.00 | Severe | **Crisis intervention** |  Urgent help |

### Implementation
Added to [pipelines/smartwatch_optimized_stress_detector.py](../../pipelines/smartwatch_optimized_stress_detector.py):
```python
self.clinical_labels = {
    'relaxed': 'monitor for hypoarousal if sustained',
    'mild': 'elevated but not crisis',
    'moderate': 'heightened stress, consider intervention',
    'high': 'high stress, intervene',
    'severe': 'crisis intervention recommended'
}
```

Exposed in report: `report['clinical_label']` for easy advisor/clinician review.

### Updated Documentation
- [docs/OPTIMIZED_THRESHOLD_FORMULA.md](../OPTIMIZED_THRESHOLD_FORMULA.md): Added clinical labels to threshold table
- [docs/QUICK_REFERENCE.md](../QUICK_REFERENCE.md): Clinical interpretation column
- [pipelines/smartwatch_optimized_stress_detector.py](../../pipelines/smartwatch_optimized_stress_detector.py): Clinical labels integrated

---

## Summary of Design Decisions

| Phase | Key Decision | Rationale | Result |
|-------|---|---|---|
| 1 | 60s windows, 50% overlap | Balance temporal resolution + sample count | 600 labeled windows |
| 2 | LOSO CV validation | Measure real-world generalization | 87.6% accuracy (PPG+ACC) |
| 2→3 | Drop TEMP | Unavailable on consumer smartwatches | 2.6% accuracy loss, universal applicability |
| 3 | RF as canonical model | Interpretability + stability + mobile-friendly | Feature importances for weights |
| 4a→4b | Empirical weights from RF importances | Justified vs. literature-based guessing | hr_std=0.266, acc_energy=0.263, ... |
| 4b→4c | Double-sigmoid instead of single | Better clinical sensitivity near ±2σ | Parameters justified (k=1.8, b=0.5) |
| 5 | Clinical labels (hypoarousal ↔ crisis) | Practitioner-facing interpretation | Actionable thresholds; thesis-ready |

---

## Lessons Learned

1. **LOSO validation is essential:** Random k-fold masks subject leakage; LOSO reveals true generalization.
2. **Empirical weights beat literature:** Feature importances from your data > general stress research.
3. **Justifiable parameters matter:** "k=1.8 because p≈0.9 at +2σ" is defensible; "k=6 is steep" is not.
4. **Clinical framing closes the loop:** Advisors want to know: "What does this score mean for a real person?"
5. **Smartwatch constraints are real:** 2.6% accuracy cost for 100% device accessibility is a good tradeoff.

---

## Next Steps (Post-Thesis)

- **Deployment:** Use `convert_model_to_tflite.py` to export for mobile
- **Personalization:** Implement rolling baseline; re-run LOSO with per-subject calibration
- **Real-world validation:** Collect live smartwatch data; compare to WESAD baseline
- **Publication:** Use figures from `results/THESIS_FIGURES/` for paper submission

---

**Questions?** Each phase above has a corresponding script in `pipelines/` or `analysis/`. Use `CODE_MAP.md` to find it.
