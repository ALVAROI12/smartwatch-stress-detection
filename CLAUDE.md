# Smartwatch stress detection: JBHI revision

Alvaro Ibarra's wrist-wearable stress detection research (UTSA, ECE). This repo is the working copy for the journal revision (IEEE JBHI) of the six-class paper. Start every chat about the stress project here, so all sessions share this file.

## Repo layout and branches

- Remote: `github.com/ALVAROI12/smartwatch-stress-detection`.
- `jbhi-revision` (this worktree): active paper work. `jbhi-revision-with-memo` adds the advisor report PDF.
- `~/Projects/smartwatch-stress-detection` is a second worktree of the same repo on `thesis-final-figures` (thesis report, poster, chapter figures). The raw datasets live there: `WESAD/`, `EPM-E4/`, `Stress-Predict/`, `UBFC-Phys/`, `Campanella2024/`, `wearable-device-dataset/` (PhysioNet).
- `scripts/`: feature extraction, relabeling, evaluation, tuning, domain adaptation, leave-one-dataset-out. `tests/`: pytest for the audit, split and relabel utilities.
- `docs/advisor_correction_sheet.md`: the current state of the paper. Read it first.
- `docs/literature/`: reading list (abstract-level), its full-text evidence report, and `dataset-and-claims-verification.md` (full-text check of all five dataset descriptors plus nine cited papers, 2026-09-21). Section 1 of that file lists 7 label fixes; they were applied in commit `6070b6b` (2026-09-21): Stress-Predict S01 dropped and untagged task boundaries dropped (180 s tag snapping), Campanella subtraction span ended per subject, PhysioNet f13 excluded. Second-half rests and v2 baseline length are sensitivity switches in `extract_features.py`, with results in `outputs/tables/jbhi_v2/label_fix_sensitivity/`. The pre-fix feature table is kept as `data/processed/combined/harmonized_windows_v2_before_label_fixes.csv` in the thesis worktree.
- `sources/` (git-ignored): local full texts of verified papers, the own IEEE draft (`ibarra-ieee-draft-six-class.txt`), Kaggle metadata for candidate E4 datasets, and scratch results from the 2026-09-20 JBHI chat (`jbhi-chat-scratch/`, including two probe CSVs not saved elsewhere).

**The README headline numbers (94.53% accuracy, 96 subjects) are from the old leaky pipeline and are wrong.** See defect A in the correction sheet.

## Device

Empatica E4 on the wrist: BVP 64 Hz, EDA 4 Hz, skin temperature 4 Hz, accelerometer 32 Hz. All datasets used are E4-based.

## Where the paper stands (from `docs/advisor_correction_sheet.md`, 2026-09-20)

Four defects in the original pipeline were found and fixed: a window-level train/test split (subject leak), misaligned WESAD labels, unfiltered BVP peak detection, and whole-session "stress" labels in PhysioNet. The corrected, subject-held-out result is 63.9%, not 93.8%.

Rebuilt and validated:
- Wrist HR/HRV with NeuroKit2 and artefact rejection, checked against WESAD chest ECG: HR error 2.4 bpm, r = 0.94. Wrist RMSSD is weak (r = 0.53, +34 ms bias). Only 37% of stress windows keep usable PPG, because speech moves the wrist.
- Evaluation: 20 subject-grouped splits plus leave-one-subject-out, nested subject-grouped tuning, Nadeau–Bengio corrected t-tests with Holm correction.
- Baseline versus stress is confounded with time in session. The unconfounded task is stress versus all non-stress states, with per-subject z-scored physiology.
- With WESAD and PhysioNet only, cross-dataset transfer cost 0.05–0.06 balanced accuracy on that task. PhysioNet is harder because its seated arithmetic stressors barely move EDA.
- Leave-one-dataset-out over five datasets (`outputs/tables/jbhi_v2/leave_one_dataset_out_summary.csv`, physiology features) widens the gap to 0.01–0.30: PhysioNet −0.01, WESAD −0.07, Campanella −0.10, Stress-Predict −0.10, UBFC-Phys −0.30. UBFC-Phys keeps AUROC 0.928 while balanced accuracy drops to 0.579, so ranking survives but the decision threshold does not transfer. Cause confirmed by `scripts/threshold_transfer_probe.py` (2026-09-21): UBFC-Phys has no temperature in 100% of windows, so XGBoost's missing-value branch pushes its mean stress score to 0.168 (others 0.31–0.43). Ranking is intact (oracle-threshold balanced accuracy 0.909). Dropping temperature (HR/HRV/EDA features) lifts it to 0.790.
- With HR/HRV/EDA features, external transfer cost 0.04–0.09 balanced accuracy on all five datasets before the label fixes, and **0.015–0.05 after them** (commit `6070b6b`; WESAD −0.03, PhysioNet −0.03, Stress-Predict −0.05, UBFC-Phys −0.04, Campanella −0.015). External accuracy rose on every dataset, including UBFC-Phys (0.790 to 0.837) whose own data did not change. But `scripts/label_fix_significance.py` (paired, WESAD and UBFC-Phys only) finds all 16 differences favour the fixes while only 1 survives Holm correction, and Campanella's smaller gap is partly a lower within-dataset score (0.908 to 0.882). Treat "label noise hurt transfer" as suggestive, not proven. Stress-Predict remains the weakest (0.697 within).
- PhysioNet sensitivity: v2 baseline of 120 or 240 s instead of 180 s moves results by at most 0.006. Using only second-half rests (as the dataset authors did) raises external PhysioNet from 0.746 to 0.797, so post-stressor recovery labelled as rest explains much of PhysioNet's difficulty; report both.
- `run_jbhi_experiments.py` must be run on the three-dataset table (WESAD, PhysioNet, EPM-E4: `harmonized_windows_v2_three_datasets_fixed.csv`), matching its committed results; on all six datasets its tasks change and it takes hours. `harmonization_table.csv` covers all six datasets, so regenerate it from the full table.
- The remaining external loss is mostly ranking: after the label fixes a best-possible threshold recovers only 0.02–0.05 (HR/HRV/EDA features). A label-free cut matching the source stress rate is not a general fix (it hurts Campanella, 0.821 to 0.733). HR/HRV/EDA is the defensible main feature set.
- The current advisor report is `~/Desktop/JBHI_revision_report_for_Dr_Pan_2026-09-22.pdf` (the 09-21 and undated copies are superseded), built by `docs/make_advisor_report.py` (source committed only on the local `jbhi-revision-with-memo` branch; excluded here via `.git/info/exclude`). The older `JBHI_revision_report_for_Dr_Pan.pdf` and `docs/advisor_correction_sheet.md` predate the five-dataset results.
- Tuning (`tune_baselines.py`, complete 96-configuration run on fixed labels, 2026-09-21): on the stress task, all modalities vs physiology only is 0.82–0.83 vs 0.78–0.80 for XGBoost, with no difference surviving Holm correction; 1 of 72 model comparisons is significant (nine-class task only). The earlier tuning run had been incomplete (accelerometer-only configurations missing).
- The model is not just detecting movement: physiology alone reaches 0.80 versus 0.83 with all modalities.
- CORAL, MMD and DANN do not beat source-only training. Unsupervised DA in the hard direction: 0.54–0.66. Few-shot personalisation (`domain_adaptation.py --support chronological --gap 30`, 2026-09-21): only PhysioNet→WESAD is evaluable, because PhysioNet subjects have a median of 5 Baseline windows, all consumed by k = 5, leaving 3 Baseline query windows in total; the committed WESAD→PhysioNet few-shot numbers are therefore not meaningful. On WESAD, with each class's first 5 windows as support and a 30 s gap, few-shot reaches 0.82–0.95 balanced accuracy (+0.16 to +0.26 over source-only); random support inflated this by up to 0.08. Per-subject z-scoring there still uses the whole recording.
- Leave-one-dataset-out now covers five datasets (commit `53eaf0e`).

Open decisions waiting on the advisor: scope (drop or keep EPM-E4 and the exercise classes), the method contribution (candidate problem: detecting mild stress), a thesis correction note, labelling assumptions, more data, and compute.

## Literature (`docs/literature/`)

The evidence report checked 20 reading-list claims against full text: 8 supported, 7 overstated, 3 misattributed, 2 contradicted. Points that matter for this paper:
- Cite Vos et al. (2023) for small datasets, weak validation practice and binary labels, and Schmidt et al. (2019) for the feature inventory, the activity confound and the leave-one-subject-out recommendation.
- Prajod et al. (2024, ICMI) found stressor type drives cross-dataset transfer, but their only failing pair was chest ECG and differed in stress intensity. Our five-dataset result is a counterpoint worth making in the paper: once labels and features are fixed, the gap is small for most datasets, and the largest one (UBFC-Phys) is a threshold shift with ranking intact.
- The Nurse dataset's labels are model-generated and nurse-confirmed, with no validated non-stress class. Do not use it as a clean test set.
- LifeSnaps has no depression measure and only Fitbit aggregates.
- Khan et al. (2025) dementia-agitation E4 dataset: cite in related work only (1-minute windows won, trees beat deep models, window-level folds likely leak). Not a training dataset.
- The evidence report's Section 4 plan (four-class taxonomy, WESAD-to-VerBIO step) was written before this repo was reviewed. Where it conflicts with the correction sheet, the correction sheet wins.

Candidate extra E4 datasets (metadata in `sources/kaggle-candidates/`, not downloaded or checked): BIOSTRESS (Data in Brief 2023; psychologist-led stress session plus exams) and WorkStress3D (Mendeley; E4 downsampled to 4 Hz, so BVP-derived HRV may be unavailable).

## Working rules

- Split by subject, never by window. Report balanced accuracy with corrected tests.
- Every claim about a paper needs a full-text check with a verbatim quote. The reading list is abstract-level and was wrong or overstated in 12 of 20 checked claims.
- Label wrist-device conclusions drawn from chest or clinical ECG studies as extrapolations.
- The `academic-research-skills` plugin is installed (`deep-research`, `academic-paper`, `academic-paper-reviewer`, `academic-pipeline`).

## Novelty experiments (merged 2026-09-22, `29afa47`)

Write-up in `docs/novelty_experiments.md`; tables in `outputs/tables/jbhi_v2/novelty_experiments/`; new opt-in flags in `leave_one_dataset_out.py` (default output unchanged, verified).
- Non-transductive normalisation keeps the small transfer cost: raw features −0.01 to 0.05. Causal (earlier windows only) scaling keeps it on WESAD, PhysioNet and Stress-Predict but loses about 0.10 externally on UBFC-Phys and Campanella (short recordings). Keep whole-session z-scoring as main, state it is transductive, report raw and causal as sensitivity.
- Exercise: models trained on the other datasets call 69.5% of PhysioNet exercise windows "stress" (external PhysioNet 0.746 to 0.583, p_holm 0.011); with exercise in training only 4–6%. The small transfer cost holds for seated/lab non-stress only.
- Kwon et al. (2026) WESAD + Stress-Predict setting: label fixes change transfer by at most 0.02 (n.s.); our pipeline beats their Stress-Predict AUROC (0.67 vs 0.56). The label audit is a correctness contribution, not the cause of the small transfer cost.
- Still unread: AutoStress benchmark (Schreiber & Maleshkova, IEEE CAI 2026) and Dahal (2026); get them via the UTSA library.
