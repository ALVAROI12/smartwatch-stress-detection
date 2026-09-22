# Wearable Biomarkers of Stress and Mental Health — Reading List

Compiled 2026-09-20. Sources: PubMed (via PubMed search; entries with a PMC ID have free full text in PubMed Central), arXiv, and public dataset pages.

Selection was made from abstracts and metadata; full texts were not read. arXiv and dataset links come from web search results and were not individually opened.

Free full text URL pattern: `https://pmc.ncbi.nlm.nih.gov/articles/<PMCID>/`

---

## 1. Physiology: what the signals mean

| Paper | Why it matters | Links |
|---|---|---|
| Kim 2018 — Stress and Heart Rate Variability: A Meta-Analysis and Review of the Literature (Psychiatry Investig) | Stress lowers parasympathetic activity: high-frequency power drops, low-frequency power rises. | [DOI](https://doi.org/10.30773/pi.2017.08.17) · PMC5900369 |
| Immanuel 2023 — HRV for Evaluating Psychological Stress Changes in Healthy Adults: Scoping Review (Neuropsychobiology) | RMSSD is the HRV metric most often linked to stress; compares induced vs real-life stress. | [DOI](https://doi.org/10.1159/000530376) · PMC10614455 |
| Shaffer & Ginsberg 2017 — An Overview of Heart Rate Variability Metrics and Norms (Front Public Health) | Reference for defining HRV features. | [DOI](https://doi.org/10.3389/fpubh.2017.00258) · PMC5624990 |
| Posada-Quintero & Chon 2020 — Innovations in Electrodermal Activity Data Collection and Signal Processing (Sensors) | Tonic and phasic EDA features. | [DOI](https://doi.org/10.3390/s20020479) · PMC7014446 |
| Järvelin-Pasanen 2018 — HRV and occupational stress: systematic review (Ind Health) | Chronic work stress goes with lower RMSSD and HF power. | [DOI](https://doi.org/10.2486/indhealth.2017-0190) · PMC6258751 |
| Sundas 2025 — Heart rate variability over the decades: scoping review (PeerJ) | Measurement techniques, applications, limitations. | [DOI](https://doi.org/10.7717/peerj.19347) · PMC12047215 |
| Vest 2017 — Benchmarking heart rate variability toolboxes (J Electrocardiol) | Preprocessing choices change HRV values. | [DOI](https://doi.org/10.1016/j.jelectrocard.2017.08.006) · PMC5696039 |
| Xu 2026 — Accuracy of PPG-derived pulse rate variability vs ECG-derived HRV: systematic review and meta-analysis (Sensors) | How well wrist optical pulse agrees with ECG. | [DOI](https://doi.org/10.3390/s26165192) · PMC13517957 |

## 2. Psychiatric signatures (for separating conditions)

| Paper | Why it matters | Links |
|---|---|---|
| Ramesh 2023 — Heart Rate Variability in Psychiatric Disorders: Systematic Review (Neuropsychiatr Dis Treat) | Overview across depression, anxiety, schizophrenia, substance use. | [DOI](https://doi.org/10.2147/NDT.S429592) · PMC10596135 |
| Schneider 2020 — Autonomic dysfunction in PTSD indexed by HRV: meta-analysis (Psychol Med) | PTSD: lower RMSSD and HF power, higher heart rate, at rest and under stress. | [DOI](https://doi.org/10.1017/S003329172000207X) · PMC7525781 |
| Čukić 2023 — When Heart Beats Differently in Depression: Nonlinear HRV Measures (JMIR Ment Health) | Entropy measures give larger effect sizes than time/frequency features. | [DOI](https://doi.org/10.2196/40342) · PMC9890355 |
| Pinter 2019 — Cardiac dysautonomia in depression (Neuropsychiatr Dis Treat) | Pathophysiology of autonomic shift in depression. | [DOI](https://doi.org/10.2147/NDT.S200360) · PMC6529729 |
| Guichard 2024 — HRV wrist-wearable biomarkers identify adverse posttraumatic neuropsychiatric sequelae (Psychiatry Res) | Wrist HRV after trauma exposure, large cohort. | [DOI](https://doi.org/10.1016/j.psychres.2024.116260) · PMC11617258 |
| Magal 2026 — Circadian instability following mass trauma predicts PTSD risk (Transl Psychiatry) | Circadian variability, not mean level, discriminates. | [DOI](https://doi.org/10.1038/s41398-026-04068-5) · PMC13338036 |
| Tsai 2022 — Panic Attack Prediction Using Wearable Devices and Machine Learning (JMIR Med Inform) | Panic attack prediction model. | [DOI](https://doi.org/10.2196/33063) · PMC8889475 |
| McGinnis 2023 — Discovering Digital Biomarkers of Panic Attack Risk in Consumer Wearables Data (medRxiv preprint) | Not peer reviewed. | [DOI](https://doi.org/10.1101/2023.03.01.23286647) · PMC10002787 |
| Llamocca 2022 — Bipolar Depression Forecasting Based on Wearable Data Collection (Front Physiol) | Bipolar vs unipolar depression. | [DOI](https://doi.org/10.3389/fphys.2021.777137) · PMC8821957 |
| Gershon 2015 — Daily Actigraphy Profiles Distinguish Depressive and Interepisode States in Bipolar Disorder (Clin Psychol Sci) | Activity patterns per mood state. | [DOI](https://doi.org/10.1177/2167702615604613) · PMC5022043 |
| Liu 2024 — Digital phenotyping from wearables using AI characterizes psychiatric disorders (Cell) | Large adolescent cohort (ABCD), wearable plus genetic data. | [DOI](https://doi.org/10.1016/j.cell.2024.11.012) · PMC12278733 |

No PMC ID — check institutional access:
- Cheng 2022 — HRV in patients with anxiety disorders: systematic review and meta-analysis (Psychiatry Clin Neurosci). Low resting HRV, but no difference in HRV reactivity. [DOI](https://doi.org/10.1111/pcn.13356)
- Koch 2019 — HRV in depression meta-analysis (Psychol Med). PMID 31239003.

## 3. Wearable stress detection: reviews and wrist ML studies

| Paper | Why it matters | Links |
|---|---|---|
| Hickey 2021 — Smart Devices and Wearable Technologies to Detect and Monitor Mental Health Conditions and Stress: Systematic Review (Sensors) | Devices and the physiological processes behind detection. | [DOI](https://doi.org/10.3390/s21103461) · PMC8156923 |
| Vos 2023 — Generalizable machine learning for stress monitoring from wearable devices: systematic review (Int J Med Inform) | Generalization across subjects and datasets. | [arXiv 2209.15137](https://arxiv.org/abs/2209.15137) · [DOI](https://doi.org/10.1016/j.ijmedinf.2023.105026) |
| Schmidt 2019 — Wearable-Based Affect Recognition: A Review (Sensors) | Full pipeline: sensors, labels, features, models. | [DOI](https://doi.org/10.3390/s19194079) · PMC6806301 |
| Jerath 2023 — Integration of Smartwatches and HRV Technology (Sensors) | Consumer smartwatch HRV accuracy. | [DOI](https://doi.org/10.3390/s23177314) · PMC10490434 |
| Klimek 2023 — Wearables measuring EDA to assess perceived stress in care: scoping review (Acta Neuropsychiatr) | EDA wearables in practice. | [DOI](https://doi.org/10.1017/neu.2023.19) · PMC13130278 |
| Barac 2024 — Wearable Technologies for Detecting Burnout and Well-Being in Health Care Professionals: Scoping Review (JMIR) | Chronic occupational stress markers. | [DOI](https://doi.org/10.2196/50253) · PMC11234055 |
| Chen 2021 — Pain and Stress Detection Using Wearable Sensors: Review (Sensors) | Pain can look like stress in these signals. | [DOI](https://doi.org/10.3390/s21041030) · PMC7913347 |
| Campanella 2023 — Stress Detection Using Empatica E4 and Machine Learning (Sensors) | PPG + EDA, 27 features, Random Forest 76.5%. | [DOI](https://doi.org/10.3390/s23073565) · PMC10098696 |
| Kim 2023 — Machine learning classification of knowledge worker mental stress (Front Public Health) | Galaxy Watch3; needs windows longer than 2–3 minutes. | [DOI](https://doi.org/10.3389/fpubh.2023.1302794) · PMC10661277 |
| Moser 2024 — Explainable Deep Learning for Stress Detection in Wearable Sensor Measurements (Sensors) | LSTM on E4; model features overlap with EDA literature. | [DOI](https://doi.org/10.3390/s24165085) · PMC11359526 |
| Almadhor 2023 — Wrist-Based EDA Monitoring for Stress Detection Using Federated Learning (Sensors) | Privacy-preserving training on WESAD. | [DOI](https://doi.org/10.3390/s23083984) · PMC10146352 |
| Xu 2024 — A physicochemical-sensing electronic skin for stress response monitoring (Nat Electron) | Vital signs plus sweat biomarkers; separates 3 stressor types at 98%. | [DOI](https://doi.org/10.1038/s41928-023-01116-6) · PMC10906959 |
| Torrente-Rodríguez 2020 — Wearable sweat cortisol sensing (Matter) | Endocrine stress marker on a wearable. | [DOI](https://doi.org/10.1016/j.matt.2020.01.021) · PMC7138219 |

## 4. Real-life data and labeling

| Paper | Why it matters | Links |
|---|---|---|
| Smets 2018 — Large-scale wearable data reveal digital phenotypes for daily-life stress detection (NPJ Digit Med) | Large daily-life cohort. | [DOI](https://doi.org/10.1038/s41746-018-0074-9) · PMC6550211 |
| Nagaraj 2023 — Dissecting the heterogeneity of "in the wild" stress from multimodal sensor data (NPJ Digit Med) | Individual differences in stress response. | [DOI](https://doi.org/10.1038/s41746-023-00975-9) · PMC10733336 |
| Sano 2018 — Physiological Markers for Self-Reported Stress and Mental Health Using Wearables and Phones, SNAPSHOT study (JMIR) | Wearable + phone features vs self-report. | [DOI](https://doi.org/10.2196/jmir.9410) · PMC6015266 |
| Başaran 2024 — Relieving the burden of intensive labeling for stress monitoring in the wild with semi-supervised learning (Front Psychol) | Fewer labels needed. | [DOI](https://doi.org/10.3389/fpsyg.2023.1293513) · PMC10797089 |
| King 2019 — micro-Stress EMA: Passive Sensing Framework for In-the-Wild Stress in Pregnant Mothers (IMWUT) | Self-reported stress does not match physiology. | [DOI](https://doi.org/10.1145/3351249) · PMC7236910 |
| Bari 2020 — Automated Detection of Stressful Conversations Using Wearable Sensors (IMWUT) | Labels the stress event itself. | [DOI](https://doi.org/10.1145/3432210) · PMC8180313 |
| Awada 2023 — Predicting Office Workers' Productivity (Sensors) | Predicts eustress and distress as separate states. | [DOI](https://doi.org/10.3390/s23218694) · PMC10647707 |
| Egger 2020 — Real-Time Assessment of Stress Using Digital Phenotyping: Study Protocol (Front Digit Health) | Protocol design reference. | [DOI](https://doi.org/10.3389/fdgth.2020.544418) · PMC8521792 |
| Kim 2019 — Integrating ecological momentary assessment into mHealth systems (Biopsychosoc Med) | EMA as ground truth. | [DOI](https://doi.org/10.1186/s13030-019-0160-5) · PMC6688314 |

arXiv preprints:
- [Stressor Type Matters! Cross-dataset generalizability of physiological stress detection](https://arxiv.org/pdf/2405.09563)
- [Personalization of Stress Mobile Sensing using Self-Supervised Learning](https://arxiv.org/pdf/2308.02731)
- [Personalized Prediction of Recurrent Stress Events Using Self-Supervised Learning](https://arxiv.org/pdf/2307.03337)
- [Are Anxiety Detection Models Generalizable? Cross-activity and cross-population study](https://arxiv.org/pdf/2504.03695)
- [On the Generalizability of ECG-based Stress Detection Models](https://arxiv.org/pdf/2210.06225)

## 5. Depression and anxiety from wearables

| Paper | Links |
|---|---|
| Abd-Alrazaq 2023 — Wearable AI for Detecting Anxiety: Systematic Review and Meta-Analysis (JMIR) | [DOI](https://doi.org/10.2196/48754) · PMC10666012 |
| Abd-Alrazaq 2023 — Wearable AI for Anxiety and Depression: Scoping Review (JMIR) | [DOI](https://doi.org/10.2196/42672) · PMC9896355 |
| Ancillon 2022 — Machine Learning for Anxiety Detection Using Biosignals: Review (Diagnostics) | [DOI](https://doi.org/10.3390/diagnostics12081794) · PMC9332282 |
| Pedrelli 2020 — Monitoring Changes in Depression Severity Using Wearable and Mobile Sensors (Front Psychiatry) | [DOI](https://doi.org/10.3389/fpsyt.2020.584711) · PMC7775362 |
| Shah 2021 — Personalized machine learning of depressed mood using wearables (Transl Psychiatry) | [DOI](https://doi.org/10.1038/s41398-021-01445-0) · PMC8187630 |
| Lekkas 2023 — Depression deconstructed: wearables and passive digital phenotyping for individual symptoms (Behav Res Ther) | [DOI](https://doi.org/10.1016/j.brat.2023.104382) · PMC10529827 |
| Tazawa 2020 — Evaluating depression with multimodal wristband device (Heliyon) | [DOI](https://doi.org/10.1016/j.heliyon.2020.e03274) · PMC7005437 |
| Moshe 2021 — Predicting Symptoms of Depression and Anxiety Using Smartphone and Wearable Data (Front Psychiatry) | [DOI](https://doi.org/10.3389/fpsyt.2021.625247) · PMC7876288 |
| Zhuparris 2023 — Smartphone- and wearable-based biomarker for unipolar depression severity (Sci Rep) | [DOI](https://doi.org/10.1038/s41598-023-46075-2) · PMC10620211 |
| Wang 2018 — Tracking Depression Dynamics in College Students Using Mobile Phone and Wearable Sensing (IMWUT) | [DOI](https://doi.org/10.1145/3191775) · PMC11501090 |
| Mullick 2022 — Predicting Depression in Adolescents Using Mobile and Wearable Sensors (JMIR Form Res) | [DOI](https://doi.org/10.2196/35807) · PMC9270714 |
| Shen 2025 — Passive Sensing for Mental Health Monitoring Using ML: Scoping Review (JMIR) | [DOI](https://doi.org/10.2196/77066) · PMC12395114 |
| Choi 2024 — Digital Phenotyping for Stress, Anxiety, and Mild Depression: Systematic Review (JMIR Mhealth Uhealth) | [DOI](https://doi.org/10.2196/40689) · PMC11157179 |
| Bufano 2023 — Digital Phenotyping for Monitoring Mental Disorders: Systematic Review (JMIR) | [DOI](https://doi.org/10.2196/46778) · PMC10753422 |
| Sheikh 2021 — Wearable, Environmental, and Smartphone-Based Passive Sensing for Mental Health Monitoring (Front Digit Health) | [DOI](https://doi.org/10.3389/fdgth.2021.662811) · PMC8521964 |
| Maatoug 2022 — Digital phenotype of mood disorders: conceptual and critical review (Front Psychiatry) | [DOI](https://doi.org/10.3389/fpsyt.2022.895860) · PMC9360315 |
| Lui 2022 — The Apple Watch for Monitoring Mental Health-Related Physiological Symptoms (JMIR Ment Health) | [DOI](https://doi.org/10.2196/37354) · PMC9494213 |
| Arnold 2025 — Wearable Sensors for Detecting Cognitive Rumination: Scoping Review (Sensors) | [DOI](https://doi.org/10.3390/s25030654) · PMC11820721 |

## 6. Open datasets

| Dataset | Content | Links |
|---|---|---|
| WESAD | Wrist (Empatica E4) and chest recordings, 15 subjects: baseline, stress (Trier Social Stress Test), amusement | [paper PDF](https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf) · [DOI](https://doi.org/10.1145/3242969.3242985) |
| Nurse stress dataset | Empatica E4, 15 nurses, about 1,250 hours of hospital shifts | [Scientific Data paper](https://www.nature.com/articles/s41597-022-01361-y) · [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f) |
| Stress-Predict | Wrist PPG stress-monitoring pilot dataset | [DOI](https://doi.org/10.3390/s22218135) · PMC9654418 |
| LifeSnaps | 4 months of Fitbit data with stress and mood surveys | [DOI](https://doi.org/10.1038/s41597-022-01764-x) · PMC9622868 |
| Multimodal stress dataset | Facial expressions plus physiological signals | [PMC12635167](https://pmc.ncbi.nlm.nih.gov/articles/PMC12635167/) |

---

## What the abstracts imply for the model

- **Acute stress vs chronic conditions.** Acute stress shows as changes in HRV and EDA reactivity. Anxiety disorders and PTSD show as low resting HRV with normal reactivity (Cheng 2022, Schneider 2020). The model needs resting baselines and circadian features, not only short windows.
- **Depression.** Shows mainly in sleep, activity rhythm and nonlinear HRV, more than in EDA bursts. Needs windows of days.
- **Stressor type.** Limits how well models transfer across datasets (arXiv 2405.09563). Multimodal sensing separates stressor types better than HRV alone (Xu 2024).
- **Labels.** Self-report labels are noisy (King 2019). Semi-supervised and personalized models help (Başaran 2024, Shah 2021).
- **Physical activity.** Use the accelerometer to separate physical load from mental stress.
- **Wrist pulse vs ECG.** Check agreement before using frequency-domain features (Xu 2026).

## Recommended next steps

1. Fix the label taxonomy: acute stress, chronic stress / anxiety-like, depression-like, physical load. Add panic or PTSD only with clinical labels.
2. Build a two-timescale feature set: 2–5 minute windows for acute stress; daily aggregates and circadian features for chronic states.
3. Baseline on public data (WESAD, Nurse dataset, LifeSnaps) with leave-one-subject-out validation.
4. Run a cross-dataset test (train WESAD, test Nurse dataset) to quantify the generalization gap.
5. Move to own data with per-subject baseline normalization and semi-supervised labels.

Suggested reading order: Vos 2023, Schmidt 2019, Cheng 2022, Čukić 2023.
