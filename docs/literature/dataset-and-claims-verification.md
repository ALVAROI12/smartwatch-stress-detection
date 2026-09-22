# Dataset and claim verification (full text), 2026-09-21

Seven verification agents read the full text of 16 sources: the descriptors of all five datasets used, EPM-E4, two candidate datasets, and nine papers the JBHI draft would cite. Every finding below was backed by a verbatim quote in the agent's report. Full-text copies are in `sources/` (git-ignored). Findings marked "inference" are the verifier's, not the authors'.

## 1. Label fixes

**Status (2026-09-21): fixes 1–5 applied in commit `6070b6b`; 6 and 7 are sensitivity switches in `extract_features.py`.** Fix 1 was resolved by tag order: S02, S03, S09, S14 and S19 boundaries snap to the correct tag once the snapping window is 180 s; S06 (Stroop, Interview), S17 (Interview), S18 and S30 (Stroop) task stages are dropped with the stage sharing the unresolved boundary. Effects, HR/HRV/EDA features, leave-one-dataset-out: external transfer cost fell from 0.04–0.09 to 0.015–0.05 balanced accuracy, with external accuracy up on all five datasets. That improvement is consistent but mostly not significant: in a paired test (WESAD and UBFC-Phys, whose test subjects are unchanged; `scripts/label_fix_significance.py`), all 16 differences favour the fixes and 1 survives Holm correction. Part of Campanella's smaller gap is a lower within-dataset score. PhysioNet v2 baseline length (120/240 s) changes results by at most 0.006; second-half rests raise external PhysioNet from 0.746 to 0.797. Results on WESAD, PhysioNet and EPM-E4 alone (evaluation suite, transfer gap, normalisation probe) changed by at most about 0.03 and every earlier conclusion holds. Tuning and domain adaptation were not rerun.

Ordered by likely effect on results.

| # | Dataset | Issue | Evidence | Proposed fix |
|---|---|---|---|---|
| 1 | Stress-Predict | 10 stage boundaries lie more than 90 s from any E4 tag, so they keep minute-resolution log times: S06 Stroop start (442 s off), S18 Stroop start (362), S17 Interview end (348), S30 Stroop start (330), S06 Interview start (256), S19 Interview start (163), S09 Interview start (109), S14 Stroop start (97), S03 Stroop start (94), S02 Stroop end (92). Median log-to-tag distance over all 210 task boundaries: 20.5 s. | Verifier's computation on local `Time_logs.xlsx` and `tags_*.csv` | Inspect these by hand (EDA/ACC around the boundary); exclude the affected stages if not resolvable. Stress-Predict is our weakest dataset (external AUROC 0.704), so label noise may explain part of it. |
| 2 | Stress-Predict | S01 was excluded by the authors: "The data collection protocol was not followed properly for one (1) participant". Its log has Baseline after Stroop. | Iqbal et al. (2022), Section 4 | Drop S01 (N = 34). |
| 3 | Campanella 2024 | Backward subtraction has "no time constraint". Our Stress span (1650–1800 s) assumes it lasts at least 180 s. From recording length, the nominal schedule implies 134–174 s for subjects 03, 05, 22, 23, 24, so their last 6–46 s of "Stress" fall in the following rest (roughly 5–10 of 116 stress windows). | Campanella et al. (2024), Section 4.3 and Fig. 1; verifier's inference from file lengths | Estimate each subject's subtraction end (recording length minus the fixed 300 s tail, or ACC), or shorten the span to 1650–1740. |
| 4 | Campanella 2024 | Margins are thin: the Lego-with-countdown span ends exactly at its nominal boundary (0 s margin); baseline tolerates 15 s of lag. The paper describes no transition time; our own ACC check found breaks starting about a minute late. | Campanella et al. (2024), Section 4.3 | Add margins, e.g. 1380–1470. |
| 5 | PhysioNet | f13 was discarded by the authors for "bad fit of the wristband"; our code keeps it. (f07 and S02 are already handled.) | Hongn et al. (2025, Sci Data), Machine Learning algorithms | Exclude f13, or keep and state it. |
| 6 | PhysioNet | The authors used only "the second half of both the First Rest and Second Rest blocks to minimize the influence of any residual stress effects". We use the full rest spans, so our Rest includes post-stressor recovery. | Hongn et al. (2025, Sci Data), Data preprocessing | Sensitivity run with second-half rests. May raise PhysioNet's 0.74 ceiling. |
| 7 | PhysioNet | V2 has no baseline start mark; "3-minute baseline" is documented for v1 only. Our rule (last 180 s before the first tag) is defensible and available in every session (shortest gap 229 s). | Hongn et al. (2025), Protocols; authors' notebook diagram `sources/hongn-2025-protocol-v2.png` | Keep. Reword the code comment and paper as our choice; optional sensitivity at 120 s and 240 s. |

## 2. Deviations from the authors' labels to state in the paper

These are defensible choices, but each differs from the dataset descriptor and must be stated as ours.

- **Campanella 2024**: the authors label every task as stress ("0 for rest and 1 for stress"); we count only the seated subtraction. Our schedule approximates the authors' per-subject segmentation, whose scripts were never published. The Lego-with-countdown task includes counting backwards from 180, so our "Manual task" class contains some cognitive load (used for scaling only).
- **UBFC-Phys**: the authors treat T2/T3 as stress for all participants; we label the control group's T2/T3 "Control task". Do not call the control tasks "non-evaluative": the control group still spoke to an experimenter and counted down, and the authors write that the tasks "imply a high stress level in both scenarios". Test vs control differed only in somatic anxiety (p = 0.03); physiology did not separate the groups. Our 19 subjects: 11 test, 8 control.
- **Stress-Predict**: the authors count Hyperventilation as a stressor (released labels mark it 1); we keep it as its own non-stress class. Their "TSST" was a friendly interview ("the interviewers were friendly and kind"), sometimes with another participant in the room.
- **PhysioNet**: the authors label Baseline as rest (0); we keep Baseline as its own class.

## 3. Dataset facts to correct in the draft

- **UBFC-Phys**: cite as IEEE Trans. Affective Computing 14(1), 622–636 (2023). 56 subjects recorded. The E4 records temperature, but only BVP and EDA were released; the paper gives no reason, so ours must not either. T1 is the middle 3 min of a 10-min rest; T2 and T3 are the first 3 min of 6- and 4-min tasks.
- **PhysioNet (Hongn et al.)**: 36 participants recorded; the authors' own analysis used 34 (f07 and f13 excluded). Real stage durations differ from nominal (V2 First Rest 711–1086 s vs 10 min nominal), so describe stages from tags. The 13th V1 tag row is undocumented and unused.
- **Stress-Predict**: 35 recorded, 34 analysed by the authors. E4 on the non-dominant wrist. No classifier accuracy is reported (linear mixed models only: HR +1.40 bpm under stress, 95% CI 1.10–1.71).
- **Campanella 2024**: 29 subjects (21 male, 8 female, aged 20–60), all released; our local copy is byte-identical to Mendeley v2. No self-report. The authors note rests may be too short for full relaxation. File quirks: EDA, BVP and HR never have header rows; subject_01 has partial ACC and missing TEMP headers; 23 of 29 subjects have EDA values like "1.177.810"; subjects 25 and 28 appear truncated at 2,500 s.
- **EPM-E4**: 53 recorded (Zenodo), 47 released, 33 with clip-slice timing. The draft's 49 is wrong. No peer-reviewed descriptor exists; cite the Zenodo record (its 2020 date predates the 2021 recordings). No stress condition. The anger clip is about 79 s in the data, not the documented 106 s. Happiness arousal is 6.55 for n = 33 (6.34 for n = 47). The UTC+2 slice-time offset is our inference, consistent with every session.

## 4. Candidate datasets

- **EmpathicSchool** (Hosseini et al., 2025, Scientific Data): leaning yes. E4 raw BVP and EDA, 30 subjects (22–28 usable), separate task folders for stress (presentation, Stroop/IQ) and non-stress (reading, rest, music, breathing). Access requires a data-use agreement (the Zenodo record is restricted, not open as the paper states). The authors caution that the tasks measure "arousal rather than stress". Fixed task order.
- **BIOSTRESS** (Çöpürkaya et al., 2023): leaning no. Raw E4 is released, but there is no separate baseline or scripted stressor, labels are 30-s audio annotations inside one session, and the labelled CSVs carry no subject IDs.
- **WorkStress3D**: skip (E4 downsampled to 4 Hz, no HRV).

## 5. Claim papers: how to cite

| Paper | What it actually is | Use in the paper |
|---|---|---|
| Prajod & André (2022), ICMLA, arXiv 2210.06225 | Chest ECG, WESAD ↔ SWELL-KW; per-user min-max from 5 min of baseline. Cross-dataset accuracy fell near chance (e.g., SVM 0.871 → 0.538); HRV features transferred better than end-to-end deep models. | Main contrast for our five-dataset result (0.04–0.09 loss). |
| Sahu et al. (2025), arXiv 2504.03695 (preprint) | Person-level self-reported anxiety, Shimmer chest ECG and finger EDA, no normalisation, AUROC 0.6–0.7. | Related work only; not comparable. |
| Islam & Washington (2023), Applied Sciences 13(21), 12035 (published form of arXiv 2308.02731) | Per-subject SSL regression of STAI answers from WESAD chest EDA; about 10 labelled 10-s windows per subject; no population model. | Supports "personalisation from about 10 labels is feasible"; not a benchmark. |
| Nagaraj et al. (2023), npj Digit Med | 365 nurses, Oura Ring at night, causal graphs; over 80% changed under stress, no clusters, no edge shared by 10% of people. | Motivation for per-subject normalisation and personalisation. Not a wrist or E4 study. |
| Moser et al. (2024), Sensors (title: "An Explainable Deep Learning Approach…") | Own E4 dataset, 28 people, air-horn stressor, EDA and skin temperature only, subject-level split; recall 0.76, precision 0.36; 0.98 accuracy is dominated by non-events. | Explainability point only (phasic EDA rise and recovery). Not comparable. |
| Campanella et al. (2023), Sensors (title: "A Method for Stress Detection Using Empatica E4 Bracelet and Machine-Learning Techniques") | Same dataset as Campanella 2024; 76.5% RF with random 10-fold over 1-min windows (subjects on both sides); all-stress guess scores 62%. | Example of window-level leakage; do not compare. |
| Vest et al. (2017), J Electrocardiol | Software benchmark of two HRV toolboxes on clean ECG; preprocessing was held equal, not tested. | Only for "remove artefactual intervals without interpolation". Our 230 ms vs 49 ms comparison carries the inflation claim. |
| Jerath et al. (2023), Sensors | Narrative review, no accuracy figures, no E4, technical errors. | Drop. Replace with a quantitative wrist-PPG vs ECG validation study (none verified yet). |
| Posada-Quintero & Chon (2020), Sensors | Systematic review of EDA processing; no sampling-rate guidance, no E4. EDA content mostly 0.045–0.15 Hz. Decomposition results vary with method and parameters; motion artefacts remain a concern. | Cite for SCL/SCR decomposition. Report our decomposition settings; consider EDA artefact handling (we have none, unlike PPG). |

## 6. Methods points reviewers will likely raise

- Per-subject z-scoring over the whole session (including stress windows) makes each test subject's scaling depend on how much stress their session contains. Prajod & André used baseline-only statistics. Our choice is deliberate (baseline-referenced scaling produced an order confound; see the advisor correction sheet) and must be justified in the paper.
- EDA: state the decomposition method and parameters; no EDA artefact rejection is applied.
- Wrist-HRV validity: our own ECG validation (HR r = 0.94; RMSSD r = 0.53) needs a peer-reviewed wrist validation study beside it.

## References

Campanella, S., Altaleb, A., Belli, A., Pierleoni, P., & Palma, L. (2023). A method for stress detection using Empatica E4 bracelet and machine-learning techniques. *Sensors, 23*(7), Article 3565. https://doi.org/10.3390/s23073565

Campanella, S., Altaleb, A., Belli, A., Pierleoni, P., & Palma, L. (2024). PPG and EDA dataset collected with Empatica E4 for stress assessment. *Data in Brief, 53*, Article 110102. https://doi.org/10.1016/j.dib.2024.110102

Çöpürkaya, Ç., Meriç, E., Erik, E. B., Kocaçınar, B., Akbulut, F. P., & Catal, C. (2023). Investigating the effects of stress on achievement: BIOSTRESS dataset. *Data in Brief, 49*, Article 109297. https://doi.org/10.1016/j.dib.2023.109297

Garcia-Moreno, F. M., & Badenes-Sastre, M. (2020). *EmoPulse Moments E4 Dataset (EPM-E4): An exhaustive collection of emotion-related data from Empatica E4 wearable* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8431638

Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable device dataset from induced stress and structured exercise sessions* (version 1.0.1) [Data set]. PhysioNet. https://doi.org/10.13026/he0v-tf17

Hongn, A., Bosch, F., Prado, L. E., Ferrández, J. M., & Bonomini, M. P. (2025). Wearable physiological signals under acute stress and exercise conditions. *Scientific Data, 12*, Article 520. https://doi.org/10.1038/s41597-025-04845-9

Hosseini, M., Sohrab, F., Gottumukkala, R., Bhupatiraju, R. T., Katragadda, S., Raitoharju, J., Iosifidis, A., & Gabbouj, M. (2025). A multimodal stress detection dataset with facial expressions and physiological signals. *Scientific Data, 12*(1), Article 1844. https://doi.org/10.1038/s41597-025-05812-0

Iqbal, T., Simpkin, A. J., Roshan, D., Glynn, N., Killilea, J., Walsh, J., Molloy, G., Ganly, S., Ryman, H., Coen, E., Elahi, A., Wijns, W., & Shahzad, A. (2022). Stress monitoring using wearable sensors: A pilot study and Stress-Predict dataset. *Sensors, 22*(21), Article 8135. https://doi.org/10.3390/s22218135

Islam, T., & Washington, P. (2023). Individualized stress mobile sensing using self-supervised pre-training. *Applied Sciences, 13*(21), Article 12035. https://doi.org/10.3390/app132112035

Jerath, R., Syam, M., & Ahmed, S. (2023). The future of stress management: Integration of smartwatches and HRV technology. *Sensors, 23*(17), Article 7314. https://doi.org/10.3390/s23177314

Meziati Sabour, R., Benezeth, Y., De Oliveira, P., Chappé, J., & Yang, F. (2023). UBFC-Phys: A multimodal database for psychophysiological studies of social stress. *IEEE Transactions on Affective Computing, 14*(1), 622–636. https://doi.org/10.1109/TAFFC.2021.3056960

Moser, M. K., Ehrhart, M., & Resch, B. (2024). An explainable deep learning approach for stress detection in wearable sensor measurements. *Sensors, 24*(16), Article 5085. https://doi.org/10.3390/s24165085

Nagaraj, S., Goodday, S., Hartvigsen, T., Boch, A., Garg, K., Gowda, S., Foschini, L., Ghassemi, M., Friend, S., & Goldenberg, A. (2023). Dissecting the heterogeneity of "in the wild" stress from multimodal sensor data. *npj Digital Medicine, 6*(1), Article 237. https://doi.org/10.1038/s41746-023-00975-9

Posada-Quintero, H. F., & Chon, K. H. (2020). Innovations in electrodermal activity data collection and signal processing: A systematic review. *Sensors, 20*(2), Article 479. https://doi.org/10.3390/s20020479

Prajod, P., & André, E. (2022). On the generalizability of ECG-based stress detection models. In *2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)* (pp. 549–554). IEEE. https://doi.org/10.1109/ICMLA55696.2022.00090

Sahu, N. K., Gupta, S., & Lone, H. R. (2025). *Are anxiety detection models generalizable? A cross-activity and cross-population study using wearables* (arXiv:2504.03695) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2504.03695

Vest, A. N., Li, Q., Liu, C., Nemati, S., Shah, A., & Clifford, G. D. (2017). Benchmarking heart rate variability toolboxes. *Journal of Electrocardiology, 50*(6), 744–747. https://doi.org/10.1016/j.jelectrocard.2017.08.006

AI disclosure: full texts were retrieved and quoted by AI agents (Claude); no human has yet checked the quotations against the sources.
