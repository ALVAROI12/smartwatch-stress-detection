# Arousal-controlled evaluation kit (novelty plan step 2, 2026-09-23)

Script: `scripts/arousal_kit.py` (test: `tests/test_arousal_kit.py`). Tables: `outputs/tables/jbhi_v2/contribution_probes/arousal_kit/`. The script reuses the committed external and within-dataset scores (`contribution_probes/arousal/scores.csv`), so no model is retrained; run time is about 80 s. It reproduces the committed matched-arousal values exactly (rest 0.584, UBFC control 0.577, Lego 0.355, anger 0.390, fear 0.633, hyperventilation 0.523).

`arousal_controlled_auroc(scores, index, labels, subjects)` is the reusable function: the AUROC of any score inside quantile bins of any arousal index, weighted by bin size, with a subject-bootstrap CI.

## Results

**1. Index baseline (`index_baseline.csv`).** The untrained index (mean of per-subject z-scored `hr_mean` and `eda_tonic_mean`) against the model, stress versus all non-stress, AUROC:

| Dataset | Index | External model | External − index [95% CI] | Within − index [95% CI] |
|---|---|---|---|---|
| WESAD | 0.952 | 0.968 | 0.016 [−0.02, 0.07] | 0.033 [0.00, 0.08] |
| PhysioNet | 0.804 | 0.807 | 0.003 [−0.06, 0.06] | 0.037 [−0.00, 0.08] |
| Stress-Predict | 0.739 | 0.708 | −0.031 [−0.06, 0.00] | 0.034 [−0.00, 0.07] |
| UBFC-Phys | 0.840 | 0.906 | 0.066 [0.00, 0.15] | 0.118 [0.04, 0.22] |
| Campanella | 0.948 | 0.918 | −0.030 [−0.08, 0.01] | −0.010 [−0.05, 0.02] |

The external model is no better than the index on any dataset. Correction: report 05 of 2026-09-23 gave PhysioNet index 0.662. That value came from a join with 294 duplicate keys; the aligned value is 0.804.

**2. Equivalence and power (`equivalence_mde.csv`).** Only stress versus hyperventilation has a 90% CI inside 0.40–0.60. A synthetic positive control (a logit shift added to every stress window of the pair) gives the smallest within-bin AUROC detectable at 95%: 0.60–0.64 for the four non-separable pairs (hyperventilation 0.63, UBFC control 0.64, Lego 0.61, anger 0.60). The nulls rule out a moderate evaluative signature, not a small one.

**3. Probe state versus rest at matched arousal (`probe_vs_rest.csv`).** The external model scores these states above baseline and rest at equal arousal: Lego 0.79 [0.70, 0.88], anger 0.71 [0.62, 0.80], UBFC control 0.70 [0.54, 0.93] and hyperventilation 0.62 [0.51, 0.71]. It does not score exercise (0.49) or fear clips (0.54) above rest. What the model adds beyond arousal looks like task engagement, which evaluated and non-evaluated tasks share.

**4. Iso-HR test of Kwon et al. (2026) (`iso_hr.csv`).** Within-subject 5-bpm HR bins, majority class subsampled, 7 seeds, external score. Stress versus non-stress survives HR matching on every dataset (WESAD 0.957, PhysioNet 0.731, Stress-Predict 0.674, UBFC-Phys 0.984, Campanella 0.993), which reproduces Kwon. Against probe states: exercise 0.685, hyperventilation 0.434, Lego 0.690. The UBFC control task and the EPM-E4 clips have no within-subject overlap with stress, so they cannot be tested this way. For Lego, HR-only matching gives 0.69, but HR plus EDA matching gives 0.36: matching on HR alone leaves EDA free to separate the classes.

**5. Arousal plus residual (`residual.csv`).** Regressing the model's logit on the index within each dataset (label-free) gives R² of 0.51–0.68. The residual still ranks stress above chance (AUROC 0.59–0.73).

## What this changes in the paper

- It strengthens the central claim. The trained model adds nothing measurable over a two-feature arousal index when it is transferred to a new dataset.
- It refines what the detector's non-arousal part is. It is shared by all active tasks (Lego, anger, non-evaluated speech, hyperventilation), not specific to evaluation.
- It bounds the nulls: effects below a within-bin AUROC of about 0.60–0.64 cannot be excluded.

These results are added to `paper/main.tex` (Table V, the "Arousal-controlled checks" and "Power" paragraphs, the abstract, the Discussion and the Limitations).
