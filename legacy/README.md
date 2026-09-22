# Legacy: old thesis pipeline (not valid results)

This folder keeps the notebooks, trained models, figures, tables and API from the original thesis pipeline. They are kept because the thesis and poster (branch `thesis-final-figures`) cite them.

**Do not use these numbers.** That pipeline had four defects, all fixed in the current pipeline (`scripts/`):
- a window-level train/test split that put the same subjects in training and test;
- misaligned WESAD labels;
- unfiltered BVP peak detection;
- whole-session "stress" labels in PhysioNet.

The thesis figures were affected in different ways:
- **94.53% (5-fold cross-validation):** window-level; the same subjects were in training and test.
- **93.8% (hold-out):** also window-level. The thesis text describes a subject hold-out, but notebook 12 used `train_test_split` on windows.
- **72.0% (leave-one-subject-out):** genuinely subject-level, but computed on the defective labels and features.

There is no like-for-like corrected six-class figure. The corrected labels split the classes further (emotion into four classes; rest and meditation separate). On that 11-class version, leave-one-subject-out gives 51% accuracy (balanced accuracy 0.35; `outputs/tables/jbhi_v2/loso_summary.csv`). A random window split on the same corrected data still gives 92% (`leakage_check.csv`).

The notebooks and `api.py` expect the old folder layout and will not run from here without path changes.
