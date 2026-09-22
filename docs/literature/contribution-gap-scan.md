# Literature scout: eight candidate directions for the JBHI revision

Date: 2026-09-22. Scope: wrist-wearable (Empatica E4) stress detection, 2019–2026. Papers already read by the project (Kwon 2026, Prajod, Mishra, Xiao, Vos, Calza-Metre, Tervonen, Stewart, Han, Richer 2024, AutoStress, Dahal, Aydoğan, Fecke & Rehof, Zhu, O'Brien, Singh FEEL, Saeb, Bahameish) are only referenced, not re-reviewed.

Evidence grades used in the tables: **FT** = full text read (quote is verbatim from the text); **AO** = abstract/landing page only; **UV** = unverifiable (source blocked or claim not locatable). Four full texts (Böttcher 2022; Iqbal 2022 Stress-Predict; EmpathicSchool 2025; EmoWork 2025) were retrieved from PubMed Central via PubMed; DOIs are given in the reference list. Springer/BMC, MDPI and OSF pages blocked automated access; those items are marked AO.

Three findings that matter before the tables:

1. **The Q2 premise does not hold.** The only "PULSE" stress paper by Zhao et al. (arXiv:2510.24058, Oct 2025) uses **WESAD only**, not the Hongn PhysioNet dataset, and reports AUROC 0.989 (binary) / 0.994 (3-class) under LOSO, not 0.965. No arXiv or PubMed paper applying a foundation model or self-supervised encoder to the Hongn dataset was found. The one 2026 paper that does evaluate Hongn ("Exercise-Stress") with a deep model is Akkaya (2026, BMC MIDM), and it reports **AUROC ~0.59–0.60** there, with gradient boosting beating the CNN.
2. **Stress-Predict labels its 2-min hyperventilation block as a "stress-inducing task."** Iqbal et al. (2022): "All the readings have been tagged as the duration of the stress-inducing task and baseline/rest period", and the hyperventilation test exists only "to estimate the respiratory rate from the PPG signal … to obtain a reference reading." If the project's harmonised table inherits that tag, Stress-Predict's stress class contains a pure respiratory-arousal block. That is both a label-audit item and a ready-made Q1 probe. Also from the same paper: "the interviewees were friendly and kind to the participant … which might have resulted in less induced stress", i.e. Stress-Predict's TSST is closer to a friendly-TSST.
3. **An E4 dataset with a non-evaluative speech control already exists** (EmoWork, Sci Data 2025, controlled access on Zenodo): baseline B2 is "speaking without emotional workload" (reading a neutral script aloud), with E4 BVP/EDA/TEMP/ACC plus Polar H10 ECG.

---

## Q1. Specificity against non-stress arousal ("arousal confounder panel")

| Paper | Venue/year | What they did | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Sosa, Fontecchio, Chrysikou & Atchison | *Sensors* 2026, 26(5):1584 | Systematic review of multimodal SNS-arousal classifiers for stress; names the arousal-vs-stress specificity problem; reviewed studies mostly did not test classifiers on non-stress arousal | — | "physiological signals primarily index arousal and regulatory processes rather than affective valence. Elevated sympathetic activation… can therefore arise not only during stress, but also during non-stressful states such as excitement, engagement, or novelty detection." | FT |
| Sahu, Gupta & Lone (speech) | arXiv:2501.01471, 2025 | 51 participants, baseline / anticipation / speech / recovery; physiology only, no classifier | HR up, HRV down during speech vs baseline | "we analyzed data from 51 participants and found that HRV decreased and HR increased during the anticipation and speech activity phases compared to baseline" | FT |
| Sahu, Gupta & Lone (anxiety cross-activity) | arXiv:2504.03695, 2025 | 111 participants, three anxiety-provoking activities (speech, group discussion, …), ECG+EDA; train on one activity, test on another; also WESAD and APD | Within-activity AUROC 0.82 → cross-activity 0.59–0.62 | "In the within-activity analysis, we used 5-fold cross-validation and achieved the best AUROC of 0.82 … In the cross-activity analysis, AUROC values ranged from 0.59 to 0.62" | FT |
| Liu, Chiossi, Henninger et al. | arXiv:2601.17890, 2026 | 2×2 within-subjects (question difficulty × time pressure), N=29, physiological + behavioural features, 10-fold and LOSO; 4-class "stress × difficulty" | 4-class LOSO accuracy 0.27 | "we conducted a 2×2 within-subjects study (N = 29)"; "[LOSO] cross-validation achieved an average accuracy of only 0.2706 (±0.0764)" (4-class; context from the results table) | FT |
| Shahriar | arXiv:2512.06099, 2025 | Three E4 datasets (Hongn stress/exercise, exam, CogWear); separate per-dataset tasks, not a cross-state test | 0.96–0.99 AUC within each dataset | "Labels differ depending on the dataset: (i) stress/aerobic/anaerobic activity, (ii) baseline vs. cognitive load, and (iii) high vs. low performance in an examination task." | FT |
| Brandebusemeyer, Georgi & Arnrich | *Sensors* 2026, 26(14):4376 | Device reliability across baseline, n-back 1/2/3, acoustic startle, jogging in place, 4-7-8 breathing (E4, EmbracePlus, Shimmer, Pixel Watch 2); no classifier; data not public | n=20 + 18 | "To protect privacy, all data were stored locally and not transmitted to any external database or service." | FT |
| Iqbal et al. (Stress-Predict) | *Sensors* 2022, 22(21):8135 | Hyperventilation block tagged as stress | 35 subjects | "35 healthy volunteers performed three different stress-inducing tasks (i.e., Stroop colour word test, Trier Social Stress Test and Hyperventilation Provocation Test session) with a baseline/relax period in-between each task." | FT |

Also relevant, already read: Kwon et al. 2026 (iso-HR matching and a movement gate, which is the closest existing "specificity control"); Aydoğan 2026 (exercise called stress).

**Verdict: partly explored.** Pieces exist one at a time (exercise: Aydoğan, Kwon; amusement: every WESAD 3-class paper; speech physiology: Sahu 2025; cognitive load: Liu 2026 within one study; hyperventilation: present in Stress-Predict but labelled stress). Nobody has scored one fixed, subject-independent stress detector across a panel of non-stress arousal blocks and reported per-block false-positive rates. The project can assemble such a panel from data it already holds: Hongn aerobic/anaerobic, WESAD amusement, Stress-Predict hyperventilation (relabelled), EmoWork B2 speech (if access is granted), Brandebusemeyer-style n-back is not public.

---

## Q2. Raw-signal representation learning headroom

| Paper | Venue/year | Labels / windows / validation | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Zhao, Pendiyala, Mortazavi & Yan (PULSE) | arXiv:2510.24058, 2025 | **WESAD only.** Protocol-block labels (baseline/stress/amusement), transient windows dropped; 60 s windows, 0.25 s stride; 15 LOSO folds; SSL pretraining pool described as all subjects, fold hygiene for pretraining not stated | Binary AUROC 0.989±0.017, AUPRC 0.977±0.033, acc 93.97±5.77%; 3-class AUROC 0.994±0.011 | "We use the publicly available WESAD benchmark … 15 subjects … 35-min protocol (baseline → social-stress → comedy)"; "cut into 60-second windows with a 0.25-second stride (96% overlap)"; "Label 0 (transient) samples are discarded; only windows whose entire 60s lie in a single class {1=baseline, 2=stress, 3=amusement} are kept"; "We build fifteen LOSO folds: in fold k, subject S_k is the test set and the remaining 14 subjects form the training set"; "This segmentation yields on the order of 10^5 windows across all subjects, sufficient for self-supervised pretraining"; "We train on WESAD windows using a 90/10 train/validation split." | FT |
| Saha, Xu, Mao, Neupane, Rehg & Kumar (Pulse-PPG) | arXiv:2502.01108, 2025 | Pretrained on MOODS field PPG (120 participants, 100 days, subject-wise 84/18/18 split). WESAD binary = stress vs (baseline+amusement), non-overlapping 1-min windows, ~10 vs 26 windows per subject; downstream split protocol for WESAD not stated in text I could read | Table 7, WESAD Stress(2): Pulse-PPG P 0.869 / R 0.885 vs PaPaGei P 0.686 / R 0.694 (column identity linear-probe vs fine-tune not verified) | "we combined baseline and amusement sessions to form the non-stress class, while the stress class was the original stress session"; "each session was divided into non-overlapping 1-minute windows. Hence, the stress vs. non-stress class had approximately 10 vs. 26 windows" | FT (partial) |
| Pillai et al. (PaPaGei) | ICLR 2025, arXiv:2410.20542 | WESAD used only for SAM valence/arousal (T17/T18), binarised at 5; subject-level splits; WESAD not in pretraining | — (no stress-vs-baseline task) | "we binarized these values by categorizing valence and arousal as low (1) when less than 5 and high (0)"; "The splitting is performed at the subject level ensuring no overlap." | FT |
| Luo et al. (NormWear) | arXiv:2412.09758, 2024 | WESAD stress (neutral/stress/amusement); single 80/20 split stratified by subject ID, linear probe; downstream datasets disjoint from pretraining | AUROC 76.06 vs statistical features 66.21, Chronos 71.49 | "The split is stratified on the anonymized subject ID if this information is provided by the dataset" | FT |
| Alchieri et al. (UME, EDA foundation model) | arXiv:2603.16878, 2026 | EDAMAME (24 datasets, 634 users; 17 for pretraining, 7 held out for evaluation); LOPO; WESAD arousal/valence, HHISS stress/calm, Nurses; Stress-Predict listed but not in main results | HHISS BA 0.60±0.02; UME ≈ handcrafted EDA features | "models trained from the UME features have similar balanced accuracy to models trained from either the EDA-specific handcrafted features" | FT |
| Simon & Chetouani (RAP) | arXiv:2606.24985, 2026 | Frozen MOMENT / Chronos-2 / HuBERT; WESAD 3-class, 60 s windows, 3 s stride; repeated LOSO ×5 seeds; retrieval from target user's own history (query excluded) | 83.99% acc / 77.07 F1 → 87.91 / 81.83 | "top-K elements from the same user are retrieved… excluding the corresponding embedding" | FT |
| Mao et al. (ProtoMM) | arXiv:2510.09764, 2025 | PPG+ACC prototypes pretrained on MOODS; WESAD binary and 4-class | F1 0.910 vs 0.848 (binary); 0.622 vs 0.500 (4-class) | "improves the F1-score for 4-class stress detection by a significant 24.6% (0.622 vs. 0.500) and for binary stress detection by over 7% (0.910 vs. 0.848)" | FT (split protocol not verified) |
| Kataria et al. | arXiv:2510.14254, 2025 | MOMENT vs PPG-GPT; WESAD used for HR estimation, not stress | — | — | FT |
| Akkaya | *BMC MIDM* 2026 | GBM vs 1D-CNN on WESAD, Nurse, Exercise-Stress (Hongn), LOSO | WESAD AUROC ~0.99; Exercise-Stress ~0.59–0.60; CNN < GBM | "Discrimination was saturated on WESAD but modest on other datasets" (abstract) | AO |

Apple, Google (LSM-2) and SiamQuality foundation models: none evaluate WESAD or Hongn under subject-independent validation as far as any indexed text shows (LSM-2 is Fitbit-only; OpenMHC has no stress task). UV.

**Verdict: done on WESAD, unexplored on the hard datasets.** WESAD LOSO is already at ~0.99 AUROC for trees (Akkaya) and PULSE's 0.989 adds nothing over that; PULSE's SSL pretraining fold hygiene is unstated. No foundation model has been scored on Hongn, Stress-Predict, UBFC-Phys or Campanella under LOSO/LODO. The literature therefore cannot say whether raw-signal representation learning lifts the 0.70–0.78 within-dataset ceilings; that is an experiment, not a citation.

---

## Q3. Selective classification / abstention / conformal prediction under shift

| Paper | Venue/year | What they did | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Farahani, Cao & Rahmani (ICCM) | arXiv:2608.18397, Aug 2026 | Conformal-style rank score on inter-signal coupling (EDA–BVP–TEMP), three-zone gate (predict / defer / abstain); RF, LOSO on WESAD and Stress-Predict; calibration on the test subject's own labelled non-stress windows; not gated by PPG quality | Coverage 96.8% (WESAD) / 94.4% (Stress-Predict); FP 29→27 and 94→92, p=.157/.317; S14 F1=0 not repaired | "an interpretable signal of unsupported physiology and individual failure, rather than a stand-alone safety guarantee"; "overlapping calibration windows prevent formal coverage guarantees" | FT |
| Akkaya | *BMC MIDM* 2026 | Calibration (slopes, ECE) and decision-curve net benefit across three E4 datasets; few-shot per-subject recalibration | ECE 0.22→0.07 retrospectively; did not persist prospectively | "few-shot per-subject recalibration reduced the Nurse calibration error from 0.22 to 0.07" (abstract) | AO |
| Zhou & Wang | arXiv:2606.15153, 2026 | Audits selective-classification risk bounds under distribution shift (audio, images, some ECG references) | bounds loosen under shift | (generic; no stress data) | AO |
| Singh, Georgara, Deshmukh, Nguyen & Ramchurn | arXiv:2605.14878, 2026 | Uncertainty-weighted decision-level fusion on WESAD; no abstention | fusion ≥ feature-level "84 percent of the time" | — | AO |

No paper gates abstention by a PPG signal-quality index and reports a coverage–accuracy curve, and none does it under leave-one-dataset-out. The project's own 37%-usable-PPG figure is exactly the quantity such a curve would use.

**Verdict: partly explored.** One within-dataset LOSO paper (ICCM) with non-significant gains; nothing SQI-gated, nothing cross-dataset.

---

## Q4. Sensor warm-up after donning (TEMP and EDA drift) as a baseline-first confound

| Paper | Venue/year | What they did | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Parry & Briganti (Wearanize+) | medRxiv 2026 | E4 vs PSG in sleep; identifies passive hydration as an EDA confound (different mechanism, but the same physics as post-donning moisture build-up) | — | "sweat to accumulate under the wristband without evaporating, increasing apparent skin conductance through a passive hydration mechanism rather than active sympathetic sweat gland stimulation"; no warm-up statement | FT |
| Böttcher et al. | *Sci Rep* 2022 | E4 data-quality tools (completeness, on-body, per-modality quality) in epilepsy monitoring | — | Nothing on settling; only "the sampling rate has a certain drift over time, such that timestamps … can be inaccurate by up to one second per hour" | FT |
| Tognotti, Otesteanu, Anceschi & Menon | *Front Digit Health* 2026 | Baseline-normalisation choices; test-inclusive baseline inflates BA | 3–13 pp inflation; 2-min test-exclusive baseline recommended | "Test-inclusive baseline normalization inflated balanced accuracy by 3–13 percentage points" | FT |
| van Lier et al. 2020; Milstein & Gordon 2020; Brandebusemeyer 2026; ambulatory-EDA review (PMC8653913, 2021) | various | E4 validation papers | — | None mentions a warm-up/settling period; Brandebusemeyer uses a 5-min nature-video acclimatisation but does not analyse it | FT |
| Vendor/blog claims ("at least 15 minutes" for dry wrist electrodes; "wear the device 10–15 minutes … to reach a stable equilibrium") | web | — | — | Could not be located in any retrievable primary text (Empatica support pages returned 403; the sentence is not in arXiv 2104.02410, 2105.06637 or 2207.03405) | UV |

**Verdict: unexplored.** No wearable-stress ML paper treats time-since-donning as a covariate or checks whether the first baseline block is contaminated by TEMP rise / EDA hydration. Cheap test on existing data: regress E4 TEMP and tonic EDA on minutes-since-recording-start within the first baseline block of all five datasets, then re-run the stress task with the first N minutes dropped or with a time-since-donning feature. This also bears on the project's "baseline vs stress is confounded with time" finding.

---

## Q5. Subject-level detectability as a trait

| Paper | Venue/year | What they did | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Yildiz & Subasi (PAMG-AT) | bioRxiv 2026 | Graph model, WESAD LOSO; identifies low responders from raw signals | S2 86.99%, S3 85.99%, S9 80.93% vs ≥98% for others | "attenuated heart rate increases, minimal electrodermal activity, and reduced respiratory changes during the stress condition relative to baseline"; "This limitation reflects a characteristic of the monitored population rather than a deficiency of any specific algorithm." | FT |
| Sah & Ghasemzadeh | arXiv:2107.05666, 2021 | WESAD LOSO per subject, then personalisation | S14 55.9%, S11 66.1% | "On the test data for left out subjects S2, S3, S7, S11, S14, and S17 the trained models performed poorly"; explanations are anecdotal ("subject S3 was looking forward to stress conditions and was cheerful during data collection") | FT |
| Farahani et al. (ICCM) | arXiv:2608.18397, 2026 | S14 as high-leverage failure; coupling-based explanation | S14 F1=0 despite 93% cohort accuracy; r=−0.607 → 0.185 without S14 | "EDA–BVP coupling weakens near stress onset" | FT |
| Carroll, Ginty, Whittaker, Lovallo & de Rooij | *Neurosci Biobehav Rev* 2017 | Review of blunted cardiovascular/cortisol reactivity | no prevalence figure given | "blunted reactors also exhibited reduced activation" (neural) | FT (review) |
| Veit, Brody & Rau | *J Behav Med* 1997 | 4-year stability of HR reactivity to mental arithmetic (n=75) | r=.76 for HR change | — | AO |
| Howard et al. | *Psychophysiology* 2023 | Life-event stress → blunted responses to both salient and non-salient tasks | — | — | AO |

Across the three WESAD papers the "hard" subjects only partly overlap (S2/S3 recur; S14 appears in two of three), and nobody has asked whether the same people are missed across stressors (WESAD TSST vs Hongn arithmetic vs UBFC-Phys) or across models, nor tied ML misses to a reactivity index computed from chest ECG or self-report. The psychophysiology literature says HR reactivity is trait-like over years (r≈.76), which makes the question answerable.

**Verdict: partly explored** (single-dataset anecdotes); the cross-stressor / cross-model trait question and the link to reactivity magnitude are open. The project's mild-stress probe (self-rated intensity does not predict misses) is the closest existing evidence and points the other way from the "blunting" hypothesis; a physiological reactivity index (ΔHR from chest ECG in WESAD) is the missing test.

---

## Q6. Time-aware / recovery-aware labels; MIL, PU learning, temporal smoothing

| Paper | Venue/year | What they did | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Abdel-Ghaffar, Galatzer-Levy, Heneghan, Liu et al. (Fitbit Body Response) | arXiv:2504.21242 (IEEE TAC), 2025 | Replace block labels with autonomic-arousal event-onset labels; Fitbit Sense 2; data not public | — | "Traditionally, time during the baseline and recovery periods are labeled as not-stress and time during the anticipation and stressor periods are labeled as stress"; arousal events occur "at varying offsets during both the anticipation and stress periods, and well into the recovery period. Because the primary aim of this study was to build an algorithm that can identify the onset of autonomic arousal events, and not to conduct a traditional…" | FT |
| Skat-Rørdam, Das, Rasmussen, Lønfeldt & Clemmensen | arXiv:2509.03240, 2025 | Window-based F1 with temporal tolerance for event-style stress labels; E4 datasets ADARP, Wrist Angel (temporal within-subject split) and ROAD (subject-wise split); evaluation only, no time-aware training | — | "We introduce a window-based F1 metric (F1w) that incorporates temporal tolerance"; "For ROAD, a subject-wise split was performed" | FT |
| Kyprakis et al. | arXiv:2604.06990, 2026 | Attention-MIL over visual encodings of Garmin/Polar windows; bags = patient × 3-month horizon; questionnaire labels; patient-level LOSO; oncology | — | "we adopt a leave-one-subject-out (LOSO) split at the patient level to prevent information leakage across horizons" | FT |
| Iqbal et al. (Stress-Predict) | *Sensors* 2022 | Acknowledges the transition problem in their own labels | — | "at the start and end of the stress task, the HR and RR gradually changed, but there is no accurate way to determine this gradual change. Thus, labelling was performed without considering these delayed changes." | FT |
| Romeo et al. | *IEEE TAC* 2022 | MIL for continuous emotion (DEAP + own dataset); not stress, not wearable-stress | — | — | AO |
| "Hybrid Probability Fusion with Temporal Smoothing for HRV-Based Recognition of Rest, Fatigue, and Stress in Athletes" | Springer 2026 chapter | HMM/Viterbi smoothing over HRV windows; leave-one-file-out | acc 0.94 | — | AO (blocked) |
| Yu et al. | *IMWUT* 2023 | Semi-supervised in-the-wild stress with unlabeled data | +7–10% | — | AO |

No positive-unlabelled formulation, no time-since-stressor label decay, and no HMM smoothing evaluated under LOSO across multiple protocol datasets was found.

**Verdict: unexplored for protocol datasets under subject-independent validation.** The project's own sensitivity switch (second-half rests only, PhysioNet 0.746→0.797) is already the strongest published-style evidence that recovery contamination matters; a time-decayed soft label or a PU treatment of "rest" windows is a natural next step and has no direct prior.

---

## Q7. Harmonised multi-dataset wrist-stress benchmark (2025–2026)

| Paper | Venue/year | What they did | Key number | Verbatim quote | Grade |
|---|---|---|---|---|---|
| Liu & Ning (EDA survey + benchmark) | arXiv:2609.22622, Sept 2026 | Catalogue of 26 EDA datasets; benchmark on Neuro, WESAD, MAUS, AffectiveROAD, EmpaticaE4Stress (95 subjects, 9,751 segments); LOSO + leave-one-dataset-out; no label harmonisation; code not yet released | WESAD LOSO MCC 0.739 → cross-domain 0.197; RF 0.437 MCC > end-to-end DL | "The code will be made publicly available upon acceptance."; "Reliable EDA-based stress detection depends more on consistent processing and evaluation protocols, effective signal representation, and domain compatibility than on model complexity" | FT |
| Akkaya | *BMC MIDM* 2026 | WESAD + Nurse + Exercise-Stress (Hongn), leakage-controlled LOSO, calibration and net benefit; whether splits/code are released could not be checked | — | "Using three public Empatica E4 datasets (WESAD, Nurse, and Exercise-Stress)" (abstract) | AO |
| Shahriar | arXiv:2512.06099, 2025 | Three E4 datasets, common pipeline, LOSO; claims "a unified benchmark"; no split files verified | — | "establish a unified benchmark to guide the development of more robust wearable health-monitoring systems" | FT |
| Almadhor et al. | *Front Bioeng Biotechnol* 2025 | WESAD + ScientISST-MOVE + DREAMER; no harmonisation, no released splits | WESAD 98% | "further inquiries can be directed to the corresponding authors" (data statement) | FT |
| Tognotti et al. | *Front Digit Health* 2026 | Not a benchmark, but the methodological rule a benchmark would need (test-exclusive baseline windows) | 3–13 pp | see Q4 | FT |
| "PhysioBench" (arXiv:2609.20836), "StressBench" (IEEE 2022), OpenMHC (arXiv:2607.16235) | — | Physiological-signal QA benchmark; network I/O benchmark; Fitbit survey-outcome benchmark. None is a wrist-stress benchmark | — | OpenMHC tracks: "32 self-reported survey variables …", no stress detection task | FT/AO |

**Verdict: unexplored.** As of 2026-09-22 there is no released benchmark with (a) harmonised stress/non-stress labels across E4 datasets, (b) frozen subject splits, (c) protocol-stage anchors. Liu & Ning is the nearest (EDA-only, LODO, code pending). The project's five-dataset table with audited labels and 20 frozen subject-grouped splits would be the first, provided it ships as files plus a loader.

---

## Q8. Counterbalanced / crossover lab protocols separating speech from evaluation threat, with E4 data

| Paper / dataset | Venue/year | What they did | Public? | Verbatim quote | Grade |
|---|---|---|---|---|---|
| EmoWork | *Sci Data* 2025 (KAIST) | 31 call-centre agents; E4 + Polar H10 + Muse S; B1 rest, **B2 reading a neutral script aloud**, B3 typing; then C1 neutral call, C2 shouting, C3 swearing; C2/C3 order randomised; continuous retrospective stress/arousal/valence/suppression ratings | Controlled access (Zenodo + DUA) | "Baseline 2 (B2) was a condition in which participants read a neutral script in a natural speaking tone, unrelated to customer service work. B2 (speaking without emotional workload) …"; "The order of the conditions was randomized as C1-C2-C3 or C1-C3-C2 to avoid order effects." | FT |
| EmpkinS (PEPbench) | *Psychophysiology* 2025; OSF sh3xn | 15 participants, TSST vs f-TSST on separate occasions; ECG + ICG only | Public (OSF) | "synchronized electrocardiogram (ECG) and impedance cardiography (ICG) recordings … from 15 participants undergoing an acute stress test and a stress-free control condition" | FT (OSF API) |
| Ringgold … Rohleder | ISPNE 2025 abstract | Within-subject TSST vs f-TSST on consecutive days; HR + facial AUs | No release stated | "Participants underwent the TSST and the f-TSST on subsequent days."; "The TSST elicited greater cortisol and heart rate increases, as well as higher SSSQ-G scores than the f-TSST." | AO |
| Toner et al. | arXiv:2304.01293, 2023 | 46 socially anxious students; E4; dyadic and group conversations, explicitly vs implicitly evaluative, randomised order; RF LOPO | Not stated | "The order of events (y-axis) was randomized"; "one was designated as being explicitly socially evaluative … The other two conversations did not involve this instruction"; RF "predictive accuracy of 58%" for evaluative vs not | FT (availability UV) |
| VerBIO | 2021 (CU Boulder) | 55 participants, 344 speeches, real vs VR audience, E4 + ECG; no non-evaluative control | On request | — | AO |
| EmpathicSchool | *Sci Data* 2025 (Zenodo) | 30 students, E4; silent reading, presentation prep, presentation to camera, Stroop/IQ, music, amusing video, breathing; not counterbalanced | Public | "We could not mitigate [the order effect] by testing different task orders" | FT |
| Richer et al. 2024 (already read) | f-TSST+ | posture/movement, cortisol | — | — | — |

**Verdict: partly explored.** Public TSST-vs-f-TSST data exist without E4 (EmpkinS); a public-ish E4 dataset with a speech-without-evaluation baseline exists (EmoWork, not TSST); Toner et al. manipulate evaluative threat with E4 but availability is unclear. No public, counterbalanced E4 dataset contrasts TSST with friendly/placebo-TSST. Filling it needs new data collection (or a request to FAU's MaD Lab / KAIST).

---

## Ranked openness (most open first)

1. **Q4 warm-up/donning drift** — no prior work at all; testable on existing baselines in a day; directly relevant to the paper's "baseline confounded with time" claim.
2. **Q7 harmonised benchmark** — nothing released; the project already has the artefacts (audited labels, frozen splits, LODO harness). Liu & Ning (code pending) is the only near-competitor and is EDA-only.
3. **Q6 recovery-aware / PU labels under LOSO+LODO** — only event-onset labelling on proprietary Fitbit data and an evaluation metric paper; no training-side method on protocol datasets.
4. **Q1 arousal confounder panel** — pieces exist but no panel; the project can build one now (Hongn exercise, WESAD amusement, Stress-Predict hyperventilation relabelled, EmoWork speech). The Stress-Predict hyperventilation tag is also a label-audit correction to make regardless.
5. **Q5 detectability as a trait** — three WESAD anecdotes, no cross-stressor test, no reactivity-index link; needs chest-ECG ΔHR in WESAD plus the other datasets' baseline-to-stress deltas.
6. **Q3 SQI-gated abstention under dataset shift** — ICCM (Aug 2026) occupies the within-dataset LOSO slot with null gains; the coverage–accuracy curve gated by PPG usability, across LODO, is untaken.
7. **Q8 speech vs evaluation crossover with E4** — clear gap, but requires new data; cite EmoWork/EmpkinS as partial.
8. **Q2 foundation-model headroom** — saturated on WESAD; the premise paper (PULSE) is WESAD-only with unstated pretraining hygiene; the meaningful version (FM on Hongn/Stress-Predict/UBFC-Phys under LODO) is an experiment the literature has not run, not a citation gap.

---

## References (APA 7)

Abdel-Ghaffar, S., Galatzer-Levy, I., Heneghan, C., Liu, X., et al. (2025). *Passive measurement of autonomic arousal in real-world settings*. arXiv. https://arxiv.org/abs/2504.21242

Akkaya, A. (2026). Calibration, not architecture, limits cross-subject wearable stress detection: A multi-dataset, decision-focused evaluation with label-efficient recalibration. *BMC Medical Informatics and Decision Making*. https://doi.org/10.1186/s12911-026-03842-1

Alchieri, L., Garzon, M., Alecci, L., Bombassei De Bona, F., Gjoreski, M., De Felice, G., & Santini, S. (2026). *A foundation model for electrodermal activity data*. arXiv. https://arxiv.org/abs/2603.16878

Almadhor, A., Ojo, S., Nathaniel, T. I., Ukpong, K., Alsubai, S., & Al Hejaili, A. (2025). A cross-domain framework for emotion and stress detection using WESAD, SCIENTISST-MOVE, and DREAMER datasets. *Frontiers in Bioengineering and Biotechnology*. https://doi.org/10.3389/fbioe.2025.1659002

Böttcher, S., et al. (2022). Data quality evaluation in wearable monitoring. *Scientific Reports*, 12. https://doi.org/10.1038/s41598-022-25949-x (full text via PubMed Central)

Brandebusemeyer, C., Georgi, F., & Arnrich, B. (2026). Reliability assessment of wearable technologies for physiological measurements: An evaluation of Shimmer3 GSR+, Empatica E4, EmbracePlus, and Pixel Watch 2 across cognitive, affective and physical activity tasks. *Sensors*, 26(14), 4376. https://doi.org/10.3390/s26144376

Carroll, D., Ginty, A. T., Whittaker, A. C., Lovallo, W. R., & de Rooij, S. R. (2017). The behavioural, cognitive, and neural corollaries of blunted cardiovascular and cortisol reactions to acute psychological stress. *Neuroscience & Biobehavioral Reviews*, 77, 74–86. https://doi.org/10.1016/j.neubiorev.2017.02.025

EmoWork authors (KAIST; author list not captured). (2025). A multimodal dataset for assessing emotion, stress, and emotional workload in interpersonal work scenario. *Scientific Data*. https://doi.org/10.1038/s41597-025-06531-2 (full text via PubMed Central)

EmpathicSchool authors (author list not captured). (2025). A multimodal stress detection dataset with facial expressions and physiological signals. *Scientific Data*. https://doi.org/10.1038/s41597-025-05812-0 (full text via PubMed Central)

Farahani, S. A., Cao, H., & Rahmani, A. M. (2026). *When clean signals are not enough: Detecting structural ambiguity for safe wearable stress classification*. arXiv. https://arxiv.org/abs/2608.18397

Howard, S., et al. (2023). Life event stress is associated with blunted cardiovascular responding to both personally salient and personally non-salient laboratory tasks. *Psychophysiology*, 60(3), e14199. https://doi.org/10.1111/psyp.14199

Iqbal, T., et al. (2022). Stress monitoring using wearable sensors: A pilot study and Stress-Predict dataset. *Sensors*, 22(21), 8135. https://doi.org/10.3390/s22218135 (full text via PubMed Central)

Kataria, S., et al. (2025). *Generalist vs specialist time series foundation models: Investigating potential emergent behaviors in assessing human health using PPG signals*. arXiv. https://arxiv.org/abs/2510.14254

Kyprakis, I., Skaramagkas, V., Karanasiou, G., et al. (2026). *Stress estimation in elderly oncology patients using visual wearable representations and multi-instance learning*. arXiv. https://arxiv.org/abs/2604.06990

Liu, A., Chiossi, F., Henninger, F., Andersen, L. B., Wistuba, T., Greven, S., Kreuter, F., & Draxler, F. (2026). *Physiological and behavioral modeling of stress and cognitive load in web-based question answering*. arXiv. https://arxiv.org/abs/2601.17890

Liu, G., & Ning, Z. (2026). *Electrodermal activity (EDA) for stress detection: A comprehensive survey and benchmark*. arXiv. https://arxiv.org/abs/2609.22622

Luo, Y., Chen, Y., Salekin, A., & Rahman, T. (2024). *Toward foundation model for multivariate wearable sensing of physiological signals* (NormWear). arXiv. https://arxiv.org/abs/2412.09758

Mao, W., et al. (2025). *Leveraging shared prototypes for a multimodal pulse motion foundation model* (ProtoMM). arXiv. https://arxiv.org/abs/2510.09764

Parry, ?, & Briganti, ?. (2026). Validity and limitations of the Empatica E4 wristband for autonomic and thermoregulatory sleep monitoring against concurrent polysomnography: A Wearanize+ dataset study. *medRxiv*. https://doi.org/10.64898/2026.06.10.26355348

Pillai, A., Spathis, D., Kawsar, F., & Malekzadeh, M. (2025). PaPaGei: Open foundation models for optical physiological signals. *ICLR 2025*. https://arxiv.org/abs/2410.20542

Richer, R., et al. (2025). PEPbench: Open, reproducible, and systematic benchmarking of automated pre-ejection period extraction algorithms. *Psychophysiology*. https://doi.org/10.1111/psyp.70176 ; EmpkinS dataset: https://osf.io/sh3xn/

Ringgold, Burkhardt, Abel, Kurz, Müller, Richer, Eskofier, Shields, & Rohleder. (2025). *Multimodal stress responses during the TSST and friendly-TSST* [Conference abstract]. ISPNE 2025. https://cris.fau.de/publications/352488646/

Romeo, L., et al. (2022). Multiple instance learning for emotion recognition using physiological signals. *IEEE Transactions on Affective Computing*. (abstract only)

Sah, R. K., & Ghasemzadeh, H. (2021). *Stress classification and personalization: Getting the most out of the least*. arXiv. https://arxiv.org/abs/2107.05666

Saha, M., Xu, M. A., Mao, W., Neupane, S., Rehg, J. M., & Kumar, S. (2025). *Pulse-PPG: An open-source field-trained PPG foundation model for wearable applications across lab and field settings*. arXiv. https://arxiv.org/abs/2502.01108

Sahu, N. K., Gupta, S., & Lone, H. R. (2025a). *Exploring heart rate variability and heart rate dynamics using wearables before, during, and after speech activity*. arXiv. https://arxiv.org/abs/2501.01471

Sahu, N. K., Gupta, S., & Lone, H. R. (2025b). *Are anxiety detection models generalizable? A cross-activity and cross-population study using wearables*. arXiv. https://arxiv.org/abs/2504.03695

Shahriar, K. A. (2025). *Why nonlinear models matter: Unified analysis of cognitive load, stress, and exercise using wearable physiological signals*. arXiv. https://arxiv.org/abs/2512.06099

Simon, L., & Chetouani, M. (2026). *Retrieval-augmented personalization with foundation models for wearable stress detection*. arXiv. https://arxiv.org/abs/2606.24985

Singh, L., Georgara, A., Deshmukh, J., Nguyen, T. V. T., & Ramchurn, S. D. (2026). *Decision-level fusion for robust wearable affect recognition*. arXiv. https://arxiv.org/abs/2605.14878

Skat-Rørdam, H. V., Das, S., Rasmussen, K. S., Lønfeldt, N. N., & Clemmensen, L. (2025). *Evaluation of stress detection as time series events: A novel window-based F1-metric*. arXiv. https://arxiv.org/abs/2509.03240

Sosa, S., Fontecchio, A. K., Chrysikou, E. G., & Atchison, J. S. (2026). Beyond EDA: A systematic review of multimodal sympathetic nervous system arousal classification for stress detection. *Sensors*, 26(5), 1584. https://doi.org/10.3390/s26051584

Tognotti, A., Otesteanu, C. F., Anceschi, L., & Menon, C. (2026). Baseline normalization choices inflate classification performance in wearable health monitoring: Quantification and mitigation strategies. *Frontiers in Digital Health*. https://doi.org/10.3389/fdgth.2026.1827279

Toner, E. R., Rucker, M., Wang, Z., Larrazabal, M. A., Cai, L., Datta, D., Thompson, E., Lone, H., Boukhechba, M., Teachman, B. A., & Barnes, L. E. (2023). *Wearable sensor-based multimodal physiological responses of socially anxious individuals across social contexts*. arXiv. https://arxiv.org/abs/2304.01293

Veit, R., Brody, S., & Rau, H. (1997). Four-year stability of cardiovascular reactivity to psychological stress. *Journal of Behavioral Medicine*, 20, 447–460. https://doi.org/10.1023/a:1025599415918

Yildiz, O., & Subasi, A. (2026). PAMG-AT: A physiological attention multi-graph model with adaptive topology for stress detection using wearable devices. *bioRxiv*. https://doi.org/10.64898/2026.03.02.709179

Yu, H., et al. (2023). Semi-supervised learning for wearable-based momentary stress detection in the wild. *Proceedings of the ACM on IMWUT*. (abstract only)

Zhao, Z., Pendiyala, K., Mortazavi, M., & Yan, N. (2025). *PULSE: Privileged knowledge transfer from electrodermal activity to low-cost sensors for stress monitoring*. arXiv. https://arxiv.org/abs/2510.24058

Zhou, J., & Wang, M. (2026). *False sense of safety in selective signal classification: Auditing bound tightness and exchangeability for risk control*. arXiv. https://arxiv.org/abs/2606.15153
