# Novelty-collision check: "What Do Wrist-Worn Stress Detectors Detect?"

Date: 2026-09-23. This extends `docs/literature/novelty-search.md` (2026-09-21/22) and `docs/literature/contribution-gap-scan.md` (2026-09-22) and does not repeat them. Papers already read there (Kwon 2026, Aydoğan 2026, Richer 2024, Mishra 2020, Prajod 2024, Calza-Metre 2026, Xiao 2025, AutoStress, Dahal, Sosa 2026, Sahu 2025, Liu & Ning 2026, Tognotti 2026, FEEL, EmoWork, and others) are cited only where a new reading changes the verdict.

Sources searched: WebSearch; arXiv listing pages (queries "stress wearable arousal", "stress detection cross-dataset", "WESAD", sorted newest first); PubMed (two queries, 2023 onward, 74 hits screened by title and abstract); PMC full text for Kwon et al. 2026. The arXiv API and Semantic Scholar could not be reached from the sandbox.

Evidence grades: **FT** = full text read, quote verbatim; **FT-partial** = HTML full text read through a summarising fetch, so quotes are verbatim sentences the fetch returned but were not checked line by line; **AO** = abstract only.

## Most important new findings (read these first)

1. **Zhou, Soleymani & Matarić (2023, IEEE BIBM), "Investigating the Generalizability of Physiological Characteristics of Anxiety"** (arXiv:2402.15513). This paper was not in the project's earlier searches and already makes the headline claim in general form. Abstract (AO, verbatim): "it is unclear whether ML models are learning physiological features specific to stress … We found that models trained on an arousal dataset perform relatively well on a previously unseen stress dataset, and vice versa. Our experimental results suggest that the evaluated models may be identifying emotional arousal instead of stress. This work is the first cross-corpus evaluation across stress and arousal from ECG and EDA signals". FT-partial: "the stress detection models we tested may be learning features general across emotional arousal instead of specific to stress". Datasets: APD, WESAD and CASE, analysed mainly with **chest ECG** (RespiBAN, Zephyr) and within-corpus 5-fold CV. They did not match arousal, use a non-stress arousal task panel, or evaluate on wrist E4 only. **The paper must cite it; we can no longer frame "arousal, not stress" as a new idea.**
2. **Kwon et al. (2026) already ran an arousal-matched test, and its conclusion points the other way from ours.** Their §2.5 "Iso-HR Matching and the HR-Leakage Index" (FT via PMC): "To separate stress-specific physiology from non-specific arousal, we balance HR before evaluation. Within each subject, windows are binned into 5 bpm HR bins, and the majority class is randomly subsampled so that the two classes have identical HR distributions per bin." Result, §3.6: "In WESAD … the purely non-cardiac (EDA+temperature) signal remains strong after matching; this argues against an HR confound as the explanation for WESAD's signal." The project's earlier notes mention iso-HR only in passing. A reviewer who knows Kwon will ask why their matched test finds stress-specific signal and ours does not. The answer is defensible but has to be written: (i) they match **stress against rest** on HR only, so EDA, which is itself an arousal channel, is left free to separate the classes; (ii) we match on an HR + tonic-EDA index and compare stress against **other arousing tasks**, not rest. The two results agree: EDA separates stress from rest at equal HR, and nothing separates stress from arousing non-stress tasks at equal HR and EDA.
3. **Bosch et al. (2026, *Int J Psychophysiol*; arXiv:2605.15756)** and **Kaya et al. (2026, arXiv:2604.12671)** are new stress-versus-exertion specificity studies. Both are single-cohort, chest or cortisol based, and small. They support claim (f) and partly anticipate its exercise arm; they do not collide with the non-exercise arms.

## (a) Wearable stress models detect arousal or activity, not stress

| Paper | Venue, year | What they showed | Grade | How ours differs |
|---|---|---|---|---|
| Zhou, Soleymani & Matarić | IEEE BIBM 2023, pp. 4848–4855; https://arxiv.org/abs/2402.15513 | Cross-corpus stress (APD, WESAD) ↔ arousal (CASE) transfer. WESAD→CASE "accuracies up to 69.6% and AUC scores up to 0.617". Concludes models "may be identifying emotional arousal instead of stress". | AO + FT-partial | Chest ECG and wrist GSR; transfer is to arousal *labels* from video ratings, not to non-stress tasks; no matching; within-corpus 5-fold CV, likely window-level. We use wrist E4 only, subject-held-out LODO over five datasets, a fixed detector, matched arousal and six named non-stress states. |
| Kwon, Yoon, Hur & Kang | *Healthcare* 2026, 14(15):2434; https://doi.org/10.3390/healthcare14152434 | Introduction: "A wearable classifier may therefore achieve high apparent performance by separating high-arousal movement from low-arousal rest, rather than detecting stress-specific physiology." Iso-HR test: WESAD signal survives HR matching. | FT | They frame arousal as HR/movement leakage and conclude WESAD is *not* HR-confounded. We make the stronger construct claim with a different test (see (b)). |
| Aydoğan & Villagra Povina | *Med Eng Phys* 2026; https://doi.org/10.1088/1873-4030/aea41e | Title "Stress or arousal? Exercise confounding in wearable stress detection". Abstract: rest-vs-stress validation "does not establish that their outputs are specific to stress rather than broader physiological activation". | FT (project copy) | Exercise only; PhysioNet plus WESAD. |
| Kaya, Athanassopoulou, Malliaras & Alban-Paccha | arXiv:2604.12671, Apr 2026; https://arxiv.org/abs/2604.12671 | n = 6, Latin-square rest / cycling / TSST days; chest ECG + E4 EDA + cortisol. "Autonomic signals primarily index general arousal … Consequently, wearable-based stress detection models tend to conflate physiologically distinct states that share similar arousal levels." "Psychological stress was misclassified as rest in 50% of samples". | FT-partial | Tiny single cohort; no cross-dataset test; no matching. |
| Sosa et al. | *Sensors* 2026, 26(5):1584 | Review: "physiological signals primarily index arousal and regulatory processes rather than affective valence". | FT (gap scan) | Review; no experiment. |
| Abdel-Ghaffar et al. (Google/Fitbit "Body Response") | arXiv:2504.21242, 2025 | Industry reframes the product target as "autonomic arousal events" rather than stress. | FT (gap scan) | Proprietary Fitbit data; states the construct but does not test it. |
| Schreiber et al. (VitaStress) | arXiv:2508.10468, Aug 2025; https://arxiv.org/abs/2508.10468 | Treadmill gave the highest HR while participants self-rated "relaxed". | FT-partial | Single Corsano dataset; descriptive only. |

**Verdict (a): PARTLY SCOOPED.** The general statement "stress models may detect arousal" is published, as a hypothesis supported by cross-corpus transfer (Zhou 2023), in paper titles (Aydoğan 2026) and in reviews (Sosa 2026). What remains new is the *explanatory* use: arousal as the reason wrist detectors **transfer** across five E4 datasets. This rests on the combination of small LODO cost, matched-arousal non-separability and exercise as the only separable class. No paper links cross-dataset transfer success to arousal detection.

## (b) Arousal-matched or confound-controlled evaluation of stress classifiers

| Paper | What is matched | Comparison | Result | Grade |
|---|---|---|---|---|
| Kwon et al. 2026 | Within-subject HR (5-bpm bins, class-balanced subsampling, 7 seeds) | Stress vs non-stress (baseline/rest) | HR-leakage index ≈ 0 everywhere; WESAD EDA+TEMP signal survives; Stress-Predict and Nurse at floor | FT |
| Aydoğan & Villagra Povina 2026 | "matched within-dataset stress-specificity experiment" (participants with matched aerobic and anaerobic sessions) | Stress vs exercise windows | 82.6% of exercise windows called stress (XGBoost) | FT (project copy) |
| Bosch et al. 2026, *Int J Psychophysiol*; arXiv:2605.15756 | 2 × 3 factorial: stress (audio 3-back + social evaluation) × activity (sitting, walking, cycling); randomised order | Effect sizes per signal, no classifier transfer | "Tonic electrodermal activity showed a robust, additive response to both cognitive stress and physical exertion, with no interaction"; HR stress effect r = 0.11 vs activity r = 0.72 | FT-partial |
| Kriklenko & Kovaleva 2026; Richer et al. 2024 | Design-level controls (friendly TSST) | — | Qualitative | FT (earlier) |

**Verdict (b): PARTLY SCOOPED.** Matching on HR (Kwon) and a design-level stress × exertion factorial (Bosch) both exist. Not found anywhere: (i) matching on a *label-free arousal index that includes EDA*; (ii) within-arousal-bin AUROC of a *fixed, externally trained* detector; (iii) comparisons against **non-stress, non-exercise arousing tasks** (manual task, non-evaluated speech, hyperventilation, anger/fear clips). The paper must state the Kwon contrast explicitly (see finding 2); otherwise a reviewer will read the two results as contradictory.

## (c) Time-in-session or protocol-order confound in WESAD-style datasets

| Paper | Statement | Grade |
|---|---|---|
| Richer et al. 2024, *Sci Rep* | TSST vs pre-stress baseline "can potentially induce a classification bias … sequence effects" | FT (earlier) |
| Li et al. 2021, *IEEE TPAMI* 43(1):316–333, "The Perils and Pitfalls of Block Design for EEG Classification Experiments"; https://doi.org/10.1109/TPAMI.2020.2973153 | In EEG, block designs let classifiers learn temporal correlation instead of the condition. This is the closest methodological precedent in another modality. Not previously in the project's notes. | AO (search snippet) |
| EmpathicSchool, *Sci Data* 2025 | "We could not mitigate [the order effect] by testing different task orders" | FT (gap scan) |
| Chatzaki & Tsiknakis 2025, *Sensors* (review of open stress datasets), PMC12694032 | Generic: "varying the order of affective blocks can reduce potential bias from participant familiarization or expectancy effects"; names no dataset-specific confound | FT-partial |
| Rah & Chen 2026, *Sensors*, PMC13211236 | Temperature/device-warming confound of EDA ("device-induced warming rather than psychological stress"); proposes thermal compensation. Not about time since donning or protocol order. | FT-partial |
| Tognotti et al. 2026; Fecke & Rehof 2026 | Normalisation leakage, not position | FT (earlier) |

Searches for "time since start/elapsed time as a feature beats physiology", "warm-up after donning E4" and "baseline-first confound quantified" returned nothing on wearable stress.

**Verdict (c): NOVEL (quantification); concept anticipated.** Cite Richer 2024 and Li et al. 2021 (EEG block design) for the concept. Four results are new: minutes-since-start alone beats physiology (PhysioNet 0.83 vs 0.78; Stress-Predict 0.95 vs 0.70); the share of baseline windows in the first 10 minutes on public wrist datasets; the E4 warm-up overlap; and a position-matched evaluation rule.

## (d) Multi-dataset E4 benchmarks with leave-one-dataset-out

| Paper | Datasets | Design | Grade |
|---|---|---|---|
| Kwon et al. 2026 | WESAD, Stress-Predict, Nurse (3 E4) | Pooled LODO; AUROC ≈ 0.90 (WESAD target), 0.52–0.57 (others) | FT |
| Liu & Ning 2026, arXiv:2609.22622 | Neuro, WESAD, MAUS, AffectiveROAD, EmpaticaE4Stress (EDA only) | LOSO + LODO; WESAD MCC 0.739 → 0.197 cross-domain; code "upon acceptance" | FT (gap scan) |
| Xiao et al. 2025 (HHISS), arXiv:2506.02256 | 7 datasets (4 private) | LODO; ~75% mean OOD accuracy | FT-partial |
| Zhou et al. 2023 BIBM | APD, WESAD, CASE | Leave-one-corpus-out; chest ECG | AO |
| VIAStress, *IEEE TBME* 2026, doi 10.1109/TBME.2026.3653495 | WESAD, UBFC-Phys, VerBIO, CAN-STRESS | "within-dataset and cross-dataset subject-independent settings"; method paper (domain generalisation) | AO (PubMed) |
| Shahriar 2025/2026, *Physiol Meas*, doi 10.1088/1361-6579/ae520c | 3 E4 datasets incl. Hongn | Per-dataset LOSO; "establishes a unified benchmark" but no cross-dataset test | AO |
| AutoStress (Schreiber & Maleshkova 2026, IEEE CAI) | WESAD, PhysioNet, VitaStress | Pooled LOSO; LODO named as future work | FT (earlier) |
| Akkaya 2026, *BMC MIDM* | WESAD, Nurse, Hongn | LOSO, calibration | AO |

No 2024–2026 benchmark suite (NeurIPS D&B, IMWUT, ACII, ICMI) with harmonised E4 stress labels and frozen splits was found. FEEL (Singh et al.) is arousal/valence, not stress.

**Verdict (d): PARTLY SCOOPED.** LODO on E4 corpora is published (Kwon: 3 corpora; Liu & Ning: 5 EDA datasets; Xiao: 7). New: five **public** E4 datasets with **audited** labels, every dataset as target, per-subject normalisation, corrected tests, and the finding that the cost is small (0.015–0.05). VIAStress (TBME 2026) is new to the project's notes; add it to related work as a cross-dataset method paper.

## (e) Label errors in public stress datasets

| Paper | What | Grade |
|---|---|---|
| Aydoğan & Villagra Povina 2026 | PhysioNet STRESS sessions "contain both rest and stress-induction blocks"; second-half rests; f07/f13 excluded | FT |
| Kwon et al. 2026 | Global ±64 s label shift on Stress-Predict does not rescue AUROC; no per-subject fixes | FT |
| Iqbal et al. 2022 (Stress-Predict descriptor) | Admits "labelling was performed without considering these delayed changes" | FT |
| Singh et al. (FEEL) | "Labeling strategy emerged as a major factor" (correlational; arousal/valence) | FT (earlier) |
| Chatzaki & Tsiknakis 2025 | Generic ground-truth variability; no dataset-specific errors | FT-partial |
| Generic label-noise literature (Saeed et al. 2025, *Sci Rep*) | Synthetic random noise | earlier |

Searches for WESAD, Stress-Predict, UBFC-Phys and Campanella label audits or errata found nothing beyond the above.

**Verdict (e): NOVEL** as a descriptor-level audit of five public E4 datasets with seven concrete fixes and a measured effect. One arm (PhysioNet whole-session labels, second-half rests, f13) was found independently by Aydoğan 2026; cite it. The measured effect on transfer is modest and mostly not significant after Holm, so claim a correctness contribution, not a performance one.

## (f) Specificity panels (exercise, speech, cognitive load, emotions vs stress)

| Paper | Non-stress states tested | Setting | Grade |
|---|---|---|---|
| Aydoğan & Villagra Povina 2026 | Aerobic, anaerobic exercise | Within PhysioNet + WESAD→PhysioNet | FT |
| Bosch et al. 2026, *IJP* | Walking, cycling (× stress) | n = 19, chest ECG, Shimmer EDA; effect sizes | FT-partial |
| Kaya et al. 2026 | Cycling, recovery | n = 6, LOPO, five classes | FT-partial |
| Zhou et al. 2023 | Video-elicited high-arousal emotions (CASE) | Cross-corpus | AO |
| Sahu et al. 2025 (arXiv:2504.03695) | Different anxiety activities | Cross-activity, AUROC 0.82 → 0.59–0.62 | FT (gap scan) |
| Liu et al. 2026 (arXiv:2601.17890) | Cognitive load × time pressure | Within one study | FT (gap scan) |
| Richer et al. 2024 | Friendly TSST | Body motion | FT (earlier) |
| Darwish et al. 2025, *MethodsX* | WESAD amusement only | "three-stage validation"; no arousal panel | FT-partial |
| Krishna, Sankar & Ghiasi 2026 (arXiv:2603.15880) | Exercise vs rest from EDA (no stress) | Single dataset | AO |

**Verdict (f): PARTLY SCOOPED.** The exercise arm is anticipated (Aydoğan 2026; also Bosch 2026 and Kaya 2026). Individual states have been tested one at a time. Not found: one fixed, subject-independent wrist detector trained on other datasets and scored against a panel of **manual task (Lego), same tasks without evaluation (UBFC control), hyperventilation, and anger/fear/happiness/sadness clips, plus exercise**, with per-state false-stress rates and the "add as training negative" test. The finding that adding a state to training fixes exercise but not the others is also unmatched.

## Verdict summary

| Claim | Verdict | Must cite |
|---|---|---|
| (a) Detectors detect arousal, not stress | **PARTLY SCOOPED** | Zhou 2023 BIBM (new), Aydoğan 2026, Kwon 2026, Kaya 2026 (new), Sosa 2026 |
| (b) Arousal-matched evaluation | **PARTLY SCOOPED** (HR-only iso-HR exists; the HR+EDA index against non-stress tasks with an external model is new) | Kwon 2026 iso-HR (explain the contrast), Bosch 2026 (new) |
| (c) Time-in-session / order confound | **NOVEL** as quantification; concept anticipated | Richer 2024, Li et al. 2021 TPAMI (new) |
| (d) Multi-dataset E4 LODO | **PARTLY SCOOPED** | Kwon 2026, Liu & Ning 2026, Xiao 2025, VIAStress TBME 2026 (new) |
| (e) Label errors in public datasets | **NOVEL** (one arm independently found) | Aydoğan 2026, Kwon 2026 (label-shift test) |
| (f) Specificity panel | **PARTLY SCOOPED** (exercise arm scooped; the multi-state panel is new) | Aydoğan 2026, Bosch 2026 (new), Kaya 2026 (new) |

No single paper scoops the combination. The closest threats are Zhou 2023 for the headline idea and Kwon 2026 for the matched test.

## Sharpest defensible novelty statement

"It has been suggested that wearable stress models detect arousal rather than stress [Zhou 2023; Aydoğan 2026], and heart-rate-matched tests have found stress-versus-rest signal that survives in EDA [Kwon 2026]. We give the first test of this on a single subject-independent wrist detector across five public Empatica E4 datasets with audited labels. We show that the same property explains both of its behaviours: it transfers across datasets at a cost of only 0.015–0.05 balanced accuracy, and, once HR and tonic EDA are matched, it cannot separate evaluative stress from a manual task, non-evaluated speech, hyperventilation or emotional clips (within-bin AUROC 0.36–0.58). Only exercise remains separable (0.89). Adding a state as a training negative fixes exercise but none of the others, and time in session alone matches or beats physiology within two of the datasets."

Wording rules for the manuscript:
- Drop "no study has asked whether stress remains separable at equal arousal" from Related Work. Kwon's iso-HR asks a version of it. Replace with "no study has matched on a combined cardiac and electrodermal arousal index, or compared stress against non-stress arousing tasks rather than rest".
- Add Zhou 2023 as the first sentence of the arousal subsection and present ours as a wrist, multi-dataset, matched test of their hypothesis.
- Keep "first descriptor-level label audit" and "first quantification of time-in-session and warm-up confounds on public wrist stress datasets".

## References added by this check

- Zhou, E., Soleymani, M., & Matarić, M. J. (2023). Investigating the generalizability of physiological characteristics of anxiety. *IEEE BIBM 2023*, 4848–4855. https://arxiv.org/abs/2402.15513 [AO + FT-partial]
- Bosch, E., et al. (2026). Separating acute psychological stress from physical exertion in biometric signals. *International Journal of Psychophysiology*. https://arxiv.org/abs/2605.15756 [FT-partial]
- Kaya, O., Athanassopoulou, N., Malliaras, G. G., & Alban-Paccha, M. V. (2026). Differentiating physical and psychological stress using wearable physiological signals and salivary cortisol. arXiv:2604.12671. [FT-partial]
- Li, R., Johansen, J. S., Ahmed, H., et al. (2021). The perils and pitfalls of block design for EEG classification experiments. *IEEE TPAMI*, 43(1), 316–333. https://doi.org/10.1109/TPAMI.2020.2973153 [AO]
- VIAStress: Variational instance-adaptive personalized stress recognition based on wearable sensor signals. (2026). *IEEE Transactions on Biomedical Engineering*. https://doi.org/10.1109/TBME.2026.3653495 [AO]
- Xiao, Y., Sharma, H., Kaur, S., Bergen-Cico, D., & Salekin, A. (2025). Human heterogeneity invariant stress sensing. arXiv:2506.02256. [FT-partial]
- Schreiber, P., Cinar, B., Mackert, L., & Maleshkova, M. (2025). Stress detection from multimodal wearable sensor data (VitaStress). arXiv:2508.10468. [FT-partial]
- Chatzaki, C., & Tsiknakis, M. (2025). An overview of stress analysis based on physiological signals: Systematic review of open datasets and current trends. *Sensors*. PMC12694032. [FT-partial]
- Rah, A., & Chen, Y. (2026). Electrodermal temperature-adjusted EDA for stress detection in virtual reality. *Sensors*. PMC13211236. [FT-partial]
- Darwish, et al. (2025). From lab to real-life: A three-stage validation of wearable technology for stress monitoring. *MethodsX*, 103205. https://doi.org/10.1016/j.mex.2025.103205 [FT-partial]
- Krishna, R. M., Sankar, R., & Ghiasi, S. (2026). Electrodermal activity as a unimodal signal for aerobic exercise detection in wearable sensors. arXiv:2603.15880. [AO]
- Kwon et al. (2026) full text via PubMed Central: PMC13465799, https://doi.org/10.3390/healthcare14152434 (Iso-HR section §2.5, §3.6).

Not retrieved: Enhanced PPG-based stress detection: a multivariate cross-dataset analysis across devices and tasks (*BSPC* 2025, S1746809425006603; HTTP 403); the 2021–2025 systematic review of validation protocols (ResearchGate, HTTP 403).
