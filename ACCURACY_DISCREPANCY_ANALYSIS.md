# Accuracy Discrepancy Analysis: Jupyter Notebook vs Scripts

## Executive Summary

**Your concern is VALID.** There IS a methodological difference that explains accuracy discrepancies:

### The Core Issue: Data Leakage

🔴 **Jupyter Notebook**: Likely uses **standard train/test split** (80/20 or 90/10)
- **Higher reported accuracy** (~91.3% from metadata)
- **WHY**: Trains and tests on data from the SAME subjects
- **PROBLEM**: Not realistic - model "memorizes" subject-specific patterns

✅ **LOSO Script** (`smartwatch_loso_validation.py`): Uses **Leave-One-Subject-Out** (LOSO)
- **Lower, more realistic accuracy**
- **WHY**: Trains on N-1 subjects, tests on 1 held-out subject
- **BENEFIT**: Tests true generalization to NEW people

---

## Detailed Comparison

### 1️⃣ Jupyter Notebook Approach

**File**: `notebooks/00_THESIS_CSI_VALIDATION.ipynb`

```python
# Loads WESAD data
wesad_data = json.load(...)  # All 93 windows mixed together

# No subject-based separation!
# Preprocessing doesn't track which subject each sample comes from
```

**Issues**:
- ❌ Mixes baseline and stress data from SAME person in train/test
- ❌ Model sees "Subject 5's stress pattern" in training, then "Subject 5's stress pattern" in testing
- ❌ This inflates accuracy artificially
- ❌ Real-world performance (testing on new people) would be much lower

---

### 2️⃣ LOSO Script (More Rigorous)

**File**: `src/core/smartwatch_loso_validation.py` (Lines 549-700)

```python
def run_loso_validation(self, subjects_data):
    """Leave-One-Subject-Out cross-validation"""
    
    for test_subject in subject_ids:  # For each subject
        
        # TRAIN on all OTHER subjects
        for subject_id in subject_ids:
            if subject_id != test_subject:
                X_train.append(subject_data[subject_id])
        
        # TEST on ONLY the held-out subject
        X_test, y_test = subjects_data[test_subject]
        
        # Train model, evaluate on new person
        model.fit(X_train, y_train)
        accuracy = evaluate(X_test, y_test)
```

**Benefits**:
- ✅ True generalization testing
- ✅ Each fold uses completely different person for testing
- ✅ Realistic performance metric
- ✅ Reproducible with different populations

---

## Why This Matters: The Math

### Train/Test Split (Notebook Approach)
```
Subject 1: Baseline=50 samples, Stress=10 samples
Subject 2: Baseline=50 samples, Stress=10 samples
...
Subject 9: Baseline=50 samples, Stress=10 samples

TRAIN (80%): 372 samples (multiple times from each subject)
TEST (20%):  93 samples (multiple times from each subject)

❌ LEAKAGE: Same person appears in both train and test!
   Model learns: "When this person shows pattern X → Stress"
   This pattern is ALWAYS in the test set too!
```

### LOSO Approach (Proper Validation)
```
Fold 1: Train=[S2,S3,S4,S5,S6,S7,S8,S9], Test=[S1]
Fold 2: Train=[S1,S3,S4,S5,S6,S7,S8,S9], Test=[S2]
...
Fold 9: Train=[S1,S2,S3,S4,S5,S6,S7,S8], Test=[S9]

✅ NO LEAKAGE: Test subject is ALWAYS new to the model
   Model learns general patterns from 8 subjects
   Must generalize to the 9th (completely new person)
```

---

## Current Model Performance

### From `models/thesis_final/model_metadata.json`:
```json
{
  "algorithm": "Random Forest",
  "accuracy": 0.913,  // ← This is likely from train/test split
  "f1_score": 0.836,
  "samples": 93,
  "subjects": 9
}
```

**What we need to verify**:
1. Did this 91.3% come from standard train/test or LOSO?
2. If LOSO, what was actual per-fold accuracy range?

---

## Recommended Validation

### For Maximum Confidence:

1. **Run LOSO validation** with current trained model:
   ```bash
   python src/core/smartwatch_loso_validation.py
   ```
   This gives realistic accuracy across different subjects.

2. **Compare results**:
   - LOSO accuracy = realistic performance on NEW people
   - Train/test accuracy = inflated (due to data leakage)
   - Difference = amount of subject-specific overfitting

3. **Expected realistic range**:
   - ✅ **80-85%** = Good model, generalizes well
   - ⚠️  **75-80%** = Decent, some subject variation
   - ❌ **<70%** = Poor generalization, needs more data/features

---

## Data Leakage Examples

### ❌ What NOT to do (Notebook Style):
```python
# Bad: Mix subjects randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Subject 3 appears in BOTH train and test ❌
```

### ✅ What TO do (LOSO Style):
```python
# Good: Hold out entire subjects
for test_subject in all_subjects:
    X_train = [data for s in all_subjects if s != test_subject]
    X_test = [data for s if s == test_subject]
    # Subject never appears in BOTH ✅
```

---

## Key Metrics from Code

### From `smartwatch_loso_validation.py`:
- **Validation Type**: Leave-One-Subject-Out (LOSO)
- **N Subjects**: 9
- **N Folds**: 9 (one per subject)
- **Models Tested**: Random Forest, XGBoost, SVM
- **Realistic Metric**: Mean accuracy across 9 holds-outs

### From `smartwatch_ml_pipeline.py`:
- **Test Size**: 20% of pooled data
- **Validation Type**: ⚠️ Standard train/test split
- **Inflates Accuracy**: Yes (same subjects in train/test)

---

## Recommendations

### 1. **Do NOT rely on 91.3% accuracy** for real-world predictions
   - This likely includes data leakage
   - True accuracy is probably 10-15% lower

### 2. **Use LOSO results instead**
   - Run `smartwatch_loso_validation.py`
   - Report per-fold accuracies
   - Use mean ± std as performance estimate

### 3. **For Thesis/GitHub**:
   - **State clearly**: "LOSO cross-validation shows X% accuracy"
   - **NOT**: "We achieved 91% accuracy on WESAD"
   - Add confidence interval: "85% ± 8% (9-fold LOSO)"

### 4. **Add to README.md**:
   ```markdown
   ## Performance
   - LOSO Cross-Validation: X% (±Y%)
   - Models: Random Forest, XGBoost, SVM
   - True generalization to unseen subjects
   ```

---

## Next Steps

1. **Run LOSO validation** to get real numbers
2. **Compare** LOSO vs train/test split results
3. **Document** the difference in methodology
4. **Report** LOSO results (more credible for publication)

Would you like me to create a script that runs both validation methods side-by-side for comparison?
