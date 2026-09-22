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

## Not done

- A literal rerun of Kwon's public code (github.com/RURUGURU/isohr-wearable-stress). It needs the Nurse dataset, and section 3 already answers the question that matters.
- Domain adaptation in the five-dataset setting. It is still WESAD↔PhysioNet only, so state that scope.
- EDA quality rules (Kleckner et al., 2018) as a sensitivity analysis.
