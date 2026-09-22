# Legacy: old thesis pipeline (not valid results)

This folder keeps the notebooks, trained models, figures, tables and API from the original thesis pipeline. They are kept because the thesis and poster (branch `thesis-final-figures`) cite them.

**Do not use these numbers.** That pipeline had four defects, all fixed in the current pipeline (`scripts/`):
- a window-level train/test split that put the same subjects in training and test;
- misaligned WESAD labels;
- unfiltered BVP peak detection;
- whole-session "stress" labels in PhysioNet.

Its 94.53% accuracy falls to 49% (balanced accuracy 0.36) when subjects are held out. See `outputs/tables/jbhi_v2/leakage_check.csv` and the main README.

The notebooks and `api.py` expect the old folder layout and will not run from here without path changes.
