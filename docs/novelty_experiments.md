# Novelty-search experiments, 2026-09-21

These experiments test three questions raised by `docs/literature/novelty-search.md`. The chronological few-shot rerun was done separately, in commit `bae4e5e`.

All runs use `scripts/leave_one_dataset_out.py` with 20 subject splits per target and XGBoost on a single thread. The input is `harmonized_windows_v2.csv` (after the label fixes) unless marked "before fixes". Tables are in `outputs/tables/jbhi_v2/novelty_experiments/`, and `scripts/novelty_experiments_summary.py` rebuilds the comparisons. Paired tests are Nadeau–Bengio corrected t-tests, Holm-corrected within each table.

**Reproduction check.** The default run (`--tag _session_check`) reproduces the committed `leave_one_dataset_out_summary.csv` exactly.

All numbers below are balanced accuracy (BA) with HR/HRV/EDA features, unless stated otherwise.

## 1. Is the small transfer cost an artefact of whole-session z-scoring?

Whole-session z-scoring uses the test subject's unlabelled windows, so it is transductive. It was compared with three alternatives:
- **raw**: no per-subject scaling;
- **baseline**: statistics from the subject's Baseline windows only;
- **causal**: statistics from strictly earlier, non-overlapping windows of the same recording. The first 4 windows (2.5 min) share the statistics of that opening period.

| Target | Within / external BA: session | raw | baseline | causal | Cost (within − external): session | raw | baseline | causal |
|---|---|---|---|---|---|---|---|---|
| Campanella | 0.882 / 0.867 | 0.681 / 0.691 | 0.988 / 0.952 | 0.874 / 0.765 | 0.015 | −0.010 | 0.036 | 0.109 |
| PhysioNet | 0.776 / 0.746 | 0.696 / 0.705 | 0.713 / 0.701 | 0.774 / 0.744 | 0.030 | −0.009 | 0.012 | 0.030 |
| Stress-Predict | 0.697 / 0.646 | 0.662 / 0.616 | 0.708 / 0.612 | 0.688 / 0.665 | 0.051 | 0.046 | 0.096 | 0.023 |
| UBFC-Phys | 0.877 / 0.837 | 0.737 / 0.724 | 0.896 / 0.764 | 0.880 / 0.742 | 0.040 | 0.013 | 0.132 | 0.138 |
| WESAD | 0.916 / 0.884 | 0.834 / 0.807 | 0.791 / 0.778 | 0.845 / 0.841 | 0.032 | 0.027 | 0.013 | 0.004 |

**Reading.**
- The transfer cost stays small without any per-subject scaling: raw features give −0.01 to 0.05. So the small cost is **not** produced by transductive normalisation.
- Whole-session z-scoring mainly raises the absolute level on both sides of the comparison.
- The deployable causal variant matches whole-session z-scoring on PhysioNet and Stress-Predict, and comes within 0.04 on WESAD. It loses about 0.10 externally on Campanella and UBFC-Phys, whose recordings are short (about 7 and 11 windows per subject), so causal statistics rest on very few windows.
- No external difference against whole-session z-scoring survives Holm correction across the 60 tests (`normalisation_vs_session.csv`). Report the table, not "equivalence".
- The baseline variant is unstable. It gives 0.95–0.99 on Campanella but a cost of 0.10–0.13 on Stress-Predict and UBFC-Phys, consistent with the order confound (baseline is recorded first).

**For the paper.** Keep whole-session z-scoring as the main result and state that it is transductive. Report raw and causal as sensitivity analyses. Say that the small transfer cost holds without per-subject scaling. Say that a deployable causal scaling keeps it on long recordings.

## 2. Exercise windows as non-stress negatives (PhysioNet)

`--exercise-negatives` keeps PhysioNet's Aerobic and Anaerobic sessions as non-stress. They are scaled with the subject's stress-protocol statistics.

| Trained on | BA with / without exercise | AUROC with / without | Exercise windows called stress |
|---|---|---|---|
| Other subjects of PhysioNet | 0.805 / 0.776 | 0.911 / 0.854 | 4.3% |
| Other datasets only | **0.583 / 0.746** | **0.625 / 0.811** | **69.5%** |
| All datasets | 0.822 / 0.783 | 0.910 / 0.854 | 6.4% |

The external drop is significant after Holm correction: BA −0.163, p_holm = 0.011; AUROC −0.186, p_holm = 0.031 (`exercise_negatives_physionet.csv`). With all physiology features, external BA falls from 0.735 to 0.565 and 70.3% of exercise windows are called stress.

**Reading.** A model trained on datasets without exercise mistakes exercise for stress about 70% of the time. This matches the 82.6% that Aydoğan & Villagra Povina (2026) report on PhysioNet. Once exercise is among the training negatives, the error nearly vanishes (4–6%).

**For the paper.** Partly anticipated: Aydoğan & Villagra Povina (2026, read in full 2026-09-22) also report 71.0% from a WESAD-trained model and 6.2–7.0% once exercise is a training class, within one pairwise transfer and one dataset. What is ours is the pooled five-dataset design with per-subject normalisation and corrected tests (see `docs/literature/novelty-search.md`). It limits the headline: the small transfer cost holds for seated or lab non-stress states. Exercise must be among the training negatives, or the model needs an activity gate. It also bears on the EPM-E4 / exercise scope decision.

## 3. The Kwon et al. (2026) setting: WESAD + Stress-Predict only

Run with `--datasets WESAD Stress-Predict`, on the tables before and after the label fixes, with whole-session z-scoring or raw features.

| Target | Trained on | AUROC after / before fixes (session) | AUROC after / before (raw) | BA after / before (session) |
|---|---|---|---|---|
| WESAD | Stress-Predict only | 0.914 / 0.897 | 0.834 / 0.836 | 0.817 / 0.802 |
| WESAD | within | 0.977 / 0.977 | 0.923 / 0.923 | 0.916 / 0.916 |
| Stress-Predict | WESAD only | 0.669 / 0.671 | 0.659 / 0.646 | 0.631 / 0.627 |
| Stress-Predict | within | 0.768 / 0.771 | 0.715 / 0.721 | 0.697 / 0.709 |

These are HR/HRV/EDA features. Kwon et al. report LODO AUROC of 0.890 (WESAD target) and 0.564 (Stress-Predict target) with logistic regression.

**Reading.**
- In this two-dataset setting the label fixes change transfer by at most 0.02. The WESAD paired test is not significant after Holm (p_holm ≥ 0.60; `kwon_setting_wesad_label_fix_test.csv`). Stress-Predict lost S01 in the fixes, so only its means can be compared.
- Our pipeline beats Kwon's numbers on Stress-Predict (external AUROC 0.67 vs 0.56) with or without the fixes. The difference therefore comes from features, windows and task definition, not from the label audit.
- Stress-Predict stays the weak dataset for us too: within-dataset AUROC 0.77.
- The committed five-dataset test (`label_fix_sensitivity/before_after_significance.csv`) agrees: after Holm, only WESAD external AUROC improves significantly (+0.0075).

**For the paper.** Contribution 2 cannot rest on "fixing labels cut the transfer cost from 0.04–0.09 to 0.015–0.05" as a causal effect. Most of that change is within split noise. Present the label audit as a data-quality contribution: seven documented errors, and every external score rose, but the gains are mostly not significant. The contrast with Kwon should rest on the pipeline (per-subject z-scoring helps WESAD as target: 0.834 → 0.914 AUROC), not on the label fixes.

## 4. Mild-stress pilot (2026-09-22): self-rated intensity does not predict missed detections

`scripts/mild_stress_probe.py`; tables in `outputs/tables/jbhi_v2/mild_stress/`. Same task, HR/HRV/EDA features and per-subject z-scoring as leave-one-dataset-out. Each graded window gets an out-of-sample probability from leave-one-subject-out inside its dataset (within) and from one model trained on the other five datasets (external).

**Graded labels.** Only PhysioNet (per-task stress 1–10) and WESAD (PANAS "Stressed" 1–5 for the TSST) rate each task for the same person. Stress-Predict has only STAI before and after the session. UBFC-Phys ctrl/test is between-subject, and Campanella has no ratings. The grade is the rise over the person's own Baseline rating. One unit is one rated task of one person: 54 in PhysioNet (35 subjects; Stroop and TMCT, since opinion tasks are too short for a 60 s window) and 15 in WESAD.

| Dataset | Trained on | Spearman ρ, rise vs recall [95% CI, subject bootstrap] | Recall (mild / moderate / strong units) | Non-stress false-stress |
|---|---|---|---|---|
| PhysioNet | within | −0.08 [−0.42, 0.21] | 0.69 / 0.69 / 0.63 (29 / 19 / 6) | 0.11 |
| PhysioNet | external | 0.02 [−0.30, 0.31] | 0.76 / 0.82 / 0.75 | 0.28 |
| WESAD | within | −0.49 [−0.87, 0.17] | 0.99 / 0.96 / 0.50 (7 / 6 / 2) | 0.02 |
| WESAD | external | −0.60 [−0.84, −0.08] | 0.99 / 0.91 / 0.62 | 0.13 |

Bands: rise ≤ 1 is mild; PhysioNet moderate is 1–3 and strong > 3; WESAD moderate is 2 and strong > 2.

- The task people rated milder is not the one that is missed. In PhysioNet, Stroop (mean rise 0.67) is detected at least as well as TMCT (mean rise 2.49). Paired over the 18 subjects with both tasks, TMCT − Stroop recall is −0.18 within (Wilcoxon p = 0.064) and −0.02 external (p = 0.89).
- WESAD's negative ρ rests on 2 subjects with the largest rise and should not be read as an effect.

**Reading.** On public E4 data, "mild stress" defined by self-report has no measurable detection gap. The misses (PhysioNet TMCT within recall 0.64) are not the low-rated tasks. The pilot is small (69 units, 8 strong), so it cannot exclude a modest effect. It does not support building the method contribution on self-rated intensity with these datasets. Two things would change that: a dataset with graded stressor doses within a subject, or a different definition of "mild", such as a small physiological response. The second is circular unless it is defined on data kept apart from evaluation.

## Not done

- A literal rerun of Kwon's public code (github.com/RURUGURU/isohr-wearable-stress). It needs the Nurse dataset, and section 3 already answers the question that matters.
- Domain adaptation in the five-dataset setting. It is still WESAD↔PhysioNet only, so state that scope.
- EDA quality rules (Kleckner et al., 2018) as a sensitivity analysis.
