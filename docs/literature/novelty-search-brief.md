# Novelty search brief (for a separate chat)

Written 2026-09-21. Goal: before rewriting the JBHI draft, find out whether our main claims are already published, and what the closest prior work reports. Read `CLAUDE.md` and `docs/literature/dataset-and-claims-verification.md` first.

## Our candidate contributions (what must be checked for prior art)

1. **Five-dataset leave-one-dataset-out for wrist Empatica E4 stress detection.** WESAD, PhysioNet (Hongn et al. 2025), Stress-Predict, UBFC-Phys, Campanella 2024. Task: stress vs all non-stress states, per-subject z-scored HR/HRV + EDA, XGBoost, subject-grouped splits. External transfer costs only 0.015–0.05 balanced accuracy, and the remaining loss is mostly ranking, not threshold.
2. **Label quality drives apparent transfer failure.** Checking each dataset's descriptor paper found seven label problems; fixing them cut the transfer cost from 0.04–0.09 to 0.015–0.05. Prior papers report near-chance cross-dataset transfer (e.g. Prajod & André 2022, ICMLA; Prajod et al. 2024, ICMI, chest ECG).
3. **Pitfalls with controlled evidence:** window-level splitting (92% vs 49% on unseen subjects); baseline-order confound (baseline always recorded first, so baseline-referenced features call 51–85% of post-stressor rest "stress"); a missing sensor channel (UBFC-Phys has no temperature) masquerading as a domain gap.
4. **Candidate method:** few-shot personalisation (0.85–0.95 balanced accuracy with 5 labelled windows per class per new user), possibly across unseen datasets.
5. **Wrist HR/HRV validated against chest ECG** within the stress pipeline (HR r = 0.94, MAE 2.4 bpm; RMSSD r = 0.53).

## Questions to answer

1. Has anyone published cross-dataset or leave-one-dataset-out stress detection on **three or more** wrist wearable datasets? Which datasets, task, validation, and reported transfer cost?
2. Has anyone shown that **label errors or label harmonization** explain cross-dataset failure in wearable stress detection?
3. Is the **baseline-order confound** described anywhere (baseline recorded first, confounding baseline-vs-stress with time in session)?
4. Is **few-shot / label-efficient personalisation** for wearable stress already shown to close cross-dataset or cross-subject gaps? With what label budgets and results?
5. What are the strongest 2023–2026 results on WESAD and on multi-dataset E4 benchmarks under subject-independent validation, for a fair comparison table?

## Search scope and rules

- Years: 2020–2026, weighted to 2023–2026. Sources: PubMed, arXiv, IEEE Xplore, ACM DL (IMWUT, ICMI, CHI), Scientific Data, Sensors, JBHI, IEEE TAFFC.
- Search terms (combine): cross-dataset, leave-one-dataset-out, generalization, domain shift, transfer, WESAD, Empatica E4, wrist, stress detection, personalization, few-shot, label noise, harmonization.
- Use the `academic-research-skills:deep-research` skill. Same rules as the earlier verification: full text for every paper that looks close to our claims, verbatim quotes with section, exact figures, "Unverifiable" if full text cannot be obtained. Save full texts to `sources/` (git-ignored).
- Already verified, do not redo: Prajod & André 2022; Prajod et al. 2024; Sahu et al. 2025; Islam & Washington 2023; Vos et al. 2023; Schmidt et al. 2019 (see the two files in this folder).

## Output

Write `docs/literature/novelty-search.md`:
- A verdict per contribution above: **novel / partly anticipated / already published**, with the closest papers and what they did differently.
- A comparison table of the closest works (datasets, device, task, validation, transfer cost).
- Recommended framing and any claims to drop or soften.
- APA 7 references.

Do not change code or result tables; this is a literature task only.
