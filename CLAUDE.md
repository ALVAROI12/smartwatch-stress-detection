# Smartwatch stress detection: JBHI revision

Alvaro Ibarra's wrist-wearable stress detection research (UTSA, ECE). This repo is the working copy for the journal revision (IEEE JBHI) of the six-class paper. Start every chat about the stress project here, so all sessions share this file.

## Repo layout and branches

- Remote: `github.com/ALVAROI12/smartwatch-stress-detection`.
- `jbhi-revision` (this worktree): active paper work. `jbhi-revision-with-memo` adds the advisor report PDF.
- `~/Projects/smartwatch-stress-detection` is a second worktree of the same repo on `thesis-final-figures` (thesis report, poster, chapter figures). The raw datasets live there: `WESAD/`, `EPM-E4/`, `Stress-Predict/`, `UBFC-Phys/`, `Campanella2024/`, `wearable-device-dataset/` (PhysioNet).
- `scripts/`: feature extraction, relabeling, evaluation, tuning, domain adaptation, leave-one-dataset-out. `tests/`: pytest for the audit, split and relabel utilities.
- `docs/advisor_correction_sheet.md`: the current state of the paper. Read it first.
- `docs/literature/`: reading list (abstract-level) and its full-text evidence report.
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
- Cross-dataset transfer costs only 0.05–0.06 balanced accuracy on that task. PhysioNet is harder because its seated arithmetic stressors barely move EDA.
- The model is not just detecting movement: physiology alone reaches 0.80 versus 0.83 with all modalities.
- CORAL, MMD and DANN do not beat source-only training. Few-shot personalisation reaches 0.85–0.95.
- Leave-one-dataset-out now covers five datasets (commit `53eaf0e`).

Open decisions waiting on the advisor: scope (drop or keep EPM-E4 and the exercise classes), the method contribution (candidate problem: detecting mild stress), a thesis correction note, labelling assumptions, more data, and compute.

## Literature (`docs/literature/`)

The evidence report checked 20 reading-list claims against full text: 8 supported, 7 overstated, 3 misattributed, 2 contradicted. Points that matter for this paper:
- Cite Vos et al. (2023) for small datasets, weak validation practice and binary labels, and Schmidt et al. (2019) for the feature inventory, the activity confound and the leave-one-subject-out recommendation.
- Prajod et al. (2024, ICMI) found stressor type drives cross-dataset transfer, but their only failing pair was chest ECG and differed in stress intensity. Our own result (transfer costs 0.05–0.06 once labels and features are fixed; PhysioNet is simply harder) is a direct counterpoint worth making in the paper.
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
