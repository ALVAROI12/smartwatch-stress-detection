# Full-Text Verification of the Wearable Stress Biomarkers Reading List: An Evidence Report

Prepared 2026-09-21. Companion to `wearable-stress-biomarkers-reading-list.md` (compiled 2026-09-20).

## Abstract

The reading list on wearable biomarkers of stress and mental health was compiled from abstracts and search snippets; no full text had been read and the arXiv and dataset links had not been opened. This report checks the list's load-bearing claims against the full text of 13 sources: ten papers and three dataset descriptors. Twenty discrete claims were assessed. Eight were supported, seven were partially supported or overstated, three were not supported by the source they were attributed to, and two were contradicted. The most consequential findings are these. First, the four-class label taxonomy and the two-timescale feature design are the reading-list author's own synthesis; the two reviews said to "set" them (Vos et al., 2023; Schmidt et al., 2019) cover binary acute stress and emotion recognition only. Second, the proposed cross-dataset experiment (train on WESAD, test on the nurse dataset) is attributed to a paper that never used the nurse dataset (Prajod et al., 2024), and the nurse dataset's labels are model-generated, which makes it unsuitable as a clean external test set (Hosseini et al., 2022). Third, LifeSnaps contains no depression measure and at most nine weeks of data per person (Yfantidou et al., 2022), so it cannot support a "depression-like" class. Fourth, much of the physiological evidence comes from chest or clinical ECG, not wrist photoplethysmography. Three of the five recommended next steps need revision; revised steps are given in the Discussion.

## 1. Introduction

The reading list ends with five recommended next steps: fix a four-class label taxonomy, build a two-timescale feature set, baseline on public data, run a cross-dataset test, and then move to own data. Each step rests on specific claims about what named papers found. Because the list states that "full texts were not read," the purpose of this report is to establish, claim by claim, whether the full text supports, partially supports, or contradicts the list's wording, and then to say whether each recommended step still holds.

## 2. Method

**Mode.** Fact-check (source verification), reported in APA 7.0 style.

**Scope.** Nine priority claim groups covering ten papers, plus the three public datasets named in step 3. The remaining sources on the reading list (roughly 55 entries) were not checked and their descriptions remain abstract-level.

**Procedure.** Six verification agents ran in parallel, each assigned two or three sources. Each agent was required to (a) obtain the full text, (b) confirm bibliographic metadata from the fetched page, (c) support every finding with a verbatim quote and section name, (d) copy figures exactly, and (e) return "Unverifiable" when full text could not be obtained, never filling gaps from memory or from the abstract.

**Access routes.** PubMed Central full text through the PubMed full-text tool for eight sources; arXiv HTML (v1) for Prajod et al. (2024); arXiv v3 PDF for Vos et al. (2023); the author PDF for WESAD; and an Internet Archive snapshot of the free-access Wiley page for Cheng et al. (2022), because the live Wiley site blocked automated access.

**Verdict scale.** Supported; Partially supported (true in part, or overstated in the reading list's wording); Not supported by this source (the claim may be true but the cited paper does not contain it); Contradicted.

## 3. Findings

### 3.1 Summary of verdicts

| # | Claim in the reading list | Source | Verdict |
|---|---|---|---|
| 1 | Study used a Galaxy Watch3 | Kim et al. (2023) | Supported |
| 2 | Windows under 2 min fail; needs longer than 2–3 min | Kim et al. (2023) | Partially supported (overstated) |
| 3 | Anxiety disorders show low resting HRV | Cheng et al. (2022) | Supported |
| 4 | Anxiety disorders and PTSD show normal HRV reactivity | Cheng et al. (2022) | Partially supported (overstated) |
| 5 | Nonlinear HRV measures give larger effect sizes in depression than linear measures | Čukić et al. (2023) | Partially supported |
| 6 | PTSD: lower RMSSD and HF, higher heart rate at rest | Schneider & Schwerdtfeger (2020) | Supported (RMSSD fragile) |
| 7 | The same holds under stress | Schneider & Schwerdtfeger (2020) | Partially supported (overstated) |
| 8 | Stressor type limits cross-dataset transfer | Prajod et al. (2024) | Supported, with a design confound |
| 9 | "Train WESAD, test Nurse dataset. Expect big drop" | Prajod et al. (2024) | Contradicted as an attribution |
| 10 | Systematic review and meta-analysis of PPG versus ECG agreement | Xu et al. (2026) | Supported, narrow scope |
| 11 | Check agreement before using wrist frequency-domain features | Xu et al. (2026) | Not supported by this source |
| 12 | Self-reported stress does not match physiology; labels are noisy | King et al. (2019) | Partially supported |
| 13 | Semi-supervised learning means fewer labels are needed | Başaran et al. (2024) | Partially supported (overstated) |
| 14 | Vos 2023 and Schmidt 2019 set the four-class taxonomy | Vos et al. (2023); Schmidt et al. (2019) | Not supported by these sources |
| 15 | Schmidt 2019 covers the full pipeline and feature list | Schmidt et al. (2019) | Supported |
| 16 | Two-timescale design (2–5 min windows; daily and circadian features) | Vos et al. (2023); Schmidt et al. (2019) | Not supported by these sources |
| 17 | WESAD description | Schmidt et al. (2018) | Supported, incomplete |
| 18 | Nurse dataset description (E4, 15 nurses, about 1,250 hours) | Hosseini et al. (2022) | Supported on stated facts; critical label facts omitted |
| 19 | LifeSnaps: "4 months of Fitbit data" | Yfantidou et al. (2022) | Partially supported (misleading) |
| 20 | LifeSnaps for long-window mood or depression-like modelling | Yfantidou et al. (2022) | Contradicted |

Totals: 8 supported, 7 partially supported, 3 not supported by the cited source, 2 contradicted.

### 3.2 Window length (claims 1, 2, 16)

Kim et al. (2023) used a Samsung Galaxy Watch3 and only two signals, heart rate and peak-to-peak interval; there was no electrodermal, accelerometer, or temperature input. They tested windows from 30 s to 300 s in 30-s steps with logistic regression under leave-one-subject-out validation (n = 67). Performance rose gradually: accuracy 0.762 and F1 0.747 at 30 s, 0.801 and 0.787 at 120 s, a peak F1 of 0.800 at 150 s (accuracy 0.816), and the highest accuracy, 0.826, at 300 s. The authors conclude that "measuring physiological data for at least 2–3 min is necessary to accurately distinguish between stressed and non-stressed states" (Section 4.2). No statistical test supports their word "significantly." A five-point gap is not failure, and the authors wrote "at least," not "longer than." Two further caveats: per-participant min–max normalisation means the model needs data from each new user, and feature ranking was computed on the whole dataset, a mild selection leak.

Neither review supplies the 2–5 minute figure. Vos et al. (2023) report that "summary windows of 30 and 60 seconds are most often utilized" and that two studies found 10 to 17.5 minutes better (Section 3.3.2); they recommend no length. Schmidt et al. (2019) cite a meta-analysis finding that physiological features "are commonly aggregated over fixed window lengths of 30 to 60 s" (Section 5.1). The WESAD baseline itself used 60-s windows (Schmidt et al., 2018). Neither review mentions daily aggregates or circadian features; the word "circadian" appears in neither.

### 3.3 Resting HRV and reactivity in anxiety and PTSD (claims 3, 4, 6, 7)

Cheng et al. (2022) pooled 99 studies and found lower resting HRV in anxiety disorders (Hedges' g = −0.3897, 95% CI [−0.4976, −0.2819], I² = 81.7%). PTSD is included in that pool and is its largest subgroup (38 studies; g = −0.6098, 95% CI [−0.8578, −0.3618] per Table 1; the Results text prints a different interval). The authors caution that resting HRV "is not so useful for differentiating subjects with AD and with these psychiatric disorders."

The "normal reactivity" half of the claim is overstated for four reasons. First, the paper defines reactivity as a level, not a change: "HRV reactivity refers to the HRV values during specific situations." Second, the pooled null (26 studies; g = −0.1121, 95% CI [−0.2873, 0.0632], P = 0.210) hides a significant effect in generalised anxiety disorder (5 studies; g = −0.3391, P = 0.005) and a borderline one for psychological challenges (P = 0.055). Third, the PTSD subgroup rests on 6 studies and 132 patients. Fourth, the authors attribute the null to heterogeneity of the challenge tasks.

Schneider and Schwerdtfeger (2020) confirm lower resting RMSSD (16 studies; g = −0.38, 95% CI [−0.62, −0.13]), lower HF-HRV (28 studies; g = −0.23, 95% CI [−0.34, −0.11]), and higher heart rate (18 studies; g = 0.78, 95% CI [0.46, 1.11]) in PTSD. The RMSSD effect becomes non-significant after trim-and-fill correction (g = −0.16, 95% CI [−0.40, 0.08]). Under stress, only heart rate (4 studies; g = 0.41) and HF-HRV (8 studies; g = −0.24, p = .089, "did not reach significance") were analysed; RMSSD under stress was not analysed at all. The paper states that "data on stress reactivity (relative to baseline), as well as recovery, could not be included." Neither meta-analysis measured reactivity as change from baseline, so "normal reactivity" in PTSD is not established by either source.

### 3.4 Nonlinear HRV in depression (claim 5)

Čukić et al. (2023) is a random-effects meta-analysis of 26 studies (1,537 patients, 1,041 controls). Entropy measures gave a pooled effect size of 1.05 (95% CI [0.572, 1.52], 15 studies) and other nonlinear measures 0.702 (95% CI [0.422, 0.982]). Detrended fluctuation analysis gave 0.364 (95% CI [0.237, 0.491]), which is below the 0.46 ceiling the authors quote for conventional measures, so the claim does not hold for all nonlinear measures. The comparison with linear measures is not head-to-head: linear effect sizes are quoted from earlier meta-analyses, and the authors state that "direct quantitative comparisons could not have been made." The overall estimate is inflated by design, since for each study "we presented the largest effect size," and it includes outliers as large as 7.7. Eleven of 26 studies were not significant. All recordings were ECG. The words PPG, wrist, and wearable do not appear in the body text, and the authors warn that aggressive preprocessing "can contribute to misleading results due to the loss of the exact order of samples," which is what PPG beat-correction pipelines do.

### 3.5 Cross-dataset generalisation (claims 8, 9)

The arXiv preprint is now a peer-reviewed conference paper (Prajod et al., 2024) and should be cited as such. It used four datasets: WESAD, SWELL-KW, ForDigitStress, and VerBIO. It did not use the nurse dataset; the word "nurse" does not occur in the text. The reading list therefore credits the paper with a prediction it never tested.

The paper's finding is more specific than "expect big drop." Transfer between datasets with the same stressor type (social) on blood volume pulse worked well: models trained on WESAD lost only "4% to 6%" on ForDigitStress and VerBIO. Transfer across stressor types failed: WESAD to SWELL-KW fell from F1 0.844 within-dataset to 0.468 (multilayer perceptron). That failing pair is confounded, because it is also the only chest-ECG pair and the two datasets differ sharply in stress intensity; the authors concede that low intensity in SWELL-KW "might have contributed." Cross-stressor transfer on wrist blood volume pulse was never tested.

### 3.6 PPG versus ECG (claims 10, 11)

Xu et al. (2026) is a PRISMA-compliant, PROSPERO-registered meta-analysis restricted to healthy participants and to two metrics, RMSSD and SDNN. Forty-three studies entered the qualitative synthesis and 10 the pooled analysis. Absolute standardised error was 0.188 for RMSSD (95% CI [0.066, 0.309]) and 0.134 for SDNN (95% CI [0.014, 0.255]), with PPG reading higher than ECG for SDNN. The pooled correlation for RMSSD was r = 0.980 with I² = 97.33%.

The paper contains no analysis of frequency-domain features: studies reporting only frequency-domain outcomes were excluded, and there is no subgroup analysis by sensor site. The reading list's caution about wrist frequency-domain features may be correct, but it needs a different citation. What the paper does support is broader caution: "the pooled estimates should not be generalized to sleep, exercise, stress, or free-living settings without further validation."

### 3.7 Labels (claims 12, 13)

King et al. (2019) report only fair agreement between single self-report questions and clustered ECG physiology in the laboratory (best Cohen's κ = 0.34; two questions at 0.00 and 0.07). The sentence the reading list relies on is a framing premise in the abstract and is never tested as a hypothesis. The phrase "label noise" does not occur, and low κ could equally reflect a weak physiological proxy. The study used a chest ECG patch, 18 non-pregnant women in the laboratory, and 18 pregnant women for roughly one day in the field.

Başaran et al. (2024) show label propagation plus a random forest reaching 77% accuracy with about 17–20% of labels. Fully supervised models reached 90.38–91.35%, supervised models restricted to the same 17% reached 72.73–74.89%, and K-Means with no labels reached 73%. The gain from semi-supervision is therefore 2 to 4 points, with no significance tests, no repeated runs, and no subject-independent validation (a 30% sample-level hold-out of 1-s samples). The test data were about 75% non-stress, so 77% accuracy sits close to a majority-class baseline; this last point is the verifier's inference, which the paper does not resolve.

### 3.8 Taxonomy and features (claims 14, 15)

Vos et al. (2023) reviewed 33 papers (the arXiv listing abstract still says 24). The review is explicitly binary: "stress is considered as a binary condition for prediction." Papers focused on psychiatry were excluded, participants were predominantly healthy, and the words depression, anxiety, and circadian do not appear. Its real contribution concerns generalisation: all public datasets have 25 subjects or fewer except one, WESAD (15 subjects) achieves 45% statistical power, "at least 34 test subjects would be required to achieve 80% power," 8 of 23 machine-learning studies used leave-one-subject-out validation, and only 2 tested on an unseen dataset.

Schmidt et al. (2019) is a narrative review of 46 studies. It is a sound source for the pipeline, for the feature inventory in its Table 8 (heart rate and HRV time, frequency, and nonlinear features; tonic and phasic electrodermal features; accelerometer statistics; temperature mean and slope), for physical activity as a confound ("it is sufficient to estimate the intensity level of an activity"), and for the recommendation of leave-one-subject-out validation. Depression appears only as a questionnaire and a future application. The review states that "distinguishing between eustress and distress is a largely unsolved problem," which argues for caution about fine physiological class boundaries.

Neither review proposes, endorses, or contradicts a split into acute stress, chronic or anxiety-like, depression-like, and physical load. The statement "Literature supports this split, not finer" cannot be attributed to them.

### 3.9 Datasets (claims 17–20)

**WESAD** (Schmidt et al., 2018). Seventeen subjects recorded, 15 usable (12 male; mean age 27.5). Conditions were baseline, stress (Trier Social Stress Test), and amusement, plus two meditation periods the reading list omits. About 36 minutes per subject were used, of which 30% was stress. Wrist signals: blood volume pulse 64 Hz, electrodermal activity 4 Hz, temperature 4 Hz, accelerometer 32 Hz, worn on the non-dominant hand. Wrist-only baselines under leave-one-subject-out validation were 76% (three-class) and 88% (binary). The authors warn that classifiers "might have partially learned to distinguish between speaking (stress condition) and non-speaking episodes."

**Nurse dataset** (Hosseini et al., 2022). The three stated facts are correct. The omitted facts determine its usability. Candidate stress events were generated by a random forest trained on the AffectiveRoad driving dataset using electrodermal activity, skin temperature, and heart rate; nurses confirmed or rejected them at the end of the shift. Only 83 of about 1,250 hours carry validated labels. There is no validated non-stress class: "Unlabelled data does not necessarily imply a lack of stress." No nurse used the self-report button. The device was worn on the dominant arm, all participants were women aged 30 to 55 in an emergency department during COVID-19, and sampling rates in the paper are internally inconsistent. Testing a physiological classifier on events pre-selected by another physiological classifier is partly circular.

**LifeSnaps** (Yfantidou et al., 2022). Seventy-one participants wore a Fitbit Sense. "4 months" is the study total across two rounds; each participant has at most nine weeks, with mean engagement of 41.37 days. Surveys were weekly PANAS and S-STAI (the latter mistakenly administered on a 5-point scale) and a thrice-daily categorical mood item with 43% compliance. There is no PHQ-9, BDI, CES-D, or any other depression instrument. There is no raw PPG and no beat-to-beat interval series, only Fitbit aggregates, and electrodermal data exist only for 143 user-initiated sessions.

## 4. Discussion: do the recommended next steps still hold?

**Step 1, fix the label taxonomy. Holds as a proposal, not as a literature finding.** The acute-stress class and the handling of physical load are supported (Schmidt et al., 2019; Vos et al., 2023). The chronic or anxiety-like class has some support from Cheng et al. (2022), but in clinical samples and with the authors' own warning that resting HRV does not differentiate between disorders. The depression-like class rests on ECG-based entropy evidence with no wrist validation (Čukić et al., 2023). Present the taxonomy as the project's working hypothesis, and find sources that test separability of chronic stress from depression in wearable data; none of the 13 sources checked here does.

**Step 2, two-timescale features. Revise.** Drop the statement that windows under 2 minutes fail. Treat short-window length as a tuned hyperparameter and report results across at least 30 s, 60 s, 120 s, 180 s, and 300 s, since the literature's common practice is 30–60 s (Schmidt et al., 2019; Vos et al., 2023), one wrist study favours 150–180 s (Kim et al., 2023), and two studies favour 10 minutes or more (Vos et al., 2023). Note that very-low-frequency HRV features cannot be computed on short windows. The daily and circadian tier has no support among the checked sources and needs its own citations; Magal (2026) and the actigraphy papers on the reading list are candidates but were not verified here.

**Step 3, baseline on public data. Holds for WESAD only.** Use WESAD with leave-one-subject-out validation and compare against the published wrist-only figures of 76% and 88%. Report the speaking confound. Use the nurse dataset only for exploratory or weakly supervised evaluation. Do not use LifeSnaps for a depression-like class; if used at all, label the target as negative affect, state anxiety, or mood, and accept that only device aggregates are available. A dataset with a validated depression measure is still needed. With 15 subjects, WESAD results are underpowered (Vos et al., 2023).

**Step 4, cross-dataset test. Replace.** WESAD to nurse is not a clean external validation and was never run by the paper cited for it. A stronger design follows Prajod et al. (2024): train on WESAD wrist blood volume pulse, then test on a same-stressor dataset (VerBIO or ForDigitStress) and on a different-stressor dataset. The observation that cross-stressor transfer has never been tested on wrist blood volume pulse is a genuine gap and a better motivation for the paper than an expected "big drop."

**Step 5, own data. Holds in part.** Per-subject baseline normalisation matches practice in Kim et al. (2023) and Prajod et al. (2024), but it requires calibration data from every new user and should be reported as such. Do not plan around semi-supervised labelling on the strength of Başaran et al. (2024); the demonstrated gain is 2 to 4 points without significance testing. Validating wrist pulse rate variability against an ECG subset is more justified than the reading list suggests, because Xu et al. (2026) show that existing agreement evidence covers only resting RMSSD and SDNN in healthy adults; cite a primary validation study for frequency-domain features.

**Reading order.** Vos et al. (2023) and Schmidt et al. (2019) remain the right starting point, for validation practice and the feature inventory respectively, not for the taxonomy.

**Cross-cutting pattern.** Three of the ten papers rest on chest or clinical ECG: Čukić et al. (2023) included ECG recordings only, King et al. (2019) used a chest ECG patch, and the failing transfer pair in Prajod et al. (2024) used chest ECG. Conclusions about a wrist device drawn from them are extrapolations and should be labelled as such. The recording modality of the primary studies pooled by Cheng et al. (2022) and Schneider and Schwerdtfeger (2020) was not checked.

## 5. Corrections to the reading list

1. Kim 2023: the title is "Machine learning-based classification analysis of knowledge worker mental stress." Replace "needs windows longer than 2–3 minutes" with "performance improves gradually and plateaus near 2.5–3 min; authors recommend at least 2–3 min."
2. Başaran 2024: the title ends "by using semi-supervised learning." Replace "Fewer labels needed" with the figures in Section 3.7.
3. Xu 2026: the title is "Accuracy of Photoplethysmography-Derived Pulse Rate Variability Compared with Electrocardiography-Derived Heart Rate Variability: A Systematic Review and Meta-Analysis." Remove it as the source for the frequency-domain caution.
4. arXiv 2405.09563: cite as Prajod et al. (2024), ICMI '24. Remove the WESAD-to-nurse prediction attributed to it.
5. Cheng 2022: replace "no difference in HRV reactivity" with "no significant pooled group difference in HRV measured during challenge tasks (26 studies; PTSD 6 studies), except in generalised anxiety disorder."
6. Schneider 2020: replace "at rest and under stress" with "at rest; under stress only heart rate differed significantly, and reactivity relative to baseline was not analysed."
7. King 2019: replace "Self-reported stress does not match physiology" with "single self-report items reached at most fair agreement (κ = 0.34) with ECG-derived physiology."
8. Vos 2023: 33 papers reviewed, not 24.
9. WESAD: add the two meditation periods and "17 recorded, 15 usable."
10. Nurse dataset: add that labels are model-generated and nurse-confirmed, with 83 validated hours and no validated non-stress class.
11. LifeSnaps: replace "4 months of Fitbit data" with "up to 9 weeks per participant across two rounds; no depression measure; device aggregates only."
12. "What the abstracts imply" and "Recommended next steps": mark the taxonomy and two-timescale design as the author's proposal.

## 6. Limitations

- Only 13 of roughly 70 sources were verified. Every other description on the reading list remains abstract-level.
- Table bodies were not retrieved for Čukić et al. (2023) and Xu et al. (2026); per-study details such as recording length were not checked.
- Table values for King et al. (2019) were read through a summarising fetch and are less reliable than the body-text quotes.
- Cheng et al. (2022) was read from an Internet Archive snapshot dated 2025-07-07; supplementary tables were not obtained.
- Vos et al. (2023) was read as the arXiv v3 author preprint and Prajod et al. (2024) as arXiv v1, not the publishers' versions of record. The ordinal of the ICMI '24 conference was not confirmed and is omitted from the reference.
- Several source papers contain internal inconsistencies (confidence intervals in Cheng et al., 2022, and Schneider & Schwerdtfeger, 2020; sampling rates and sheet counts in Hosseini et al., 2022; label percentages in Başaran et al., 2024). Figures here follow the tables where text and table disagree.
- Two statements are the verifiers' inferences and not the authors' claims: that 77% accuracy in Başaran et al. (2024) sits near a majority-class baseline, and that sample-level splitting likely inflated its accuracy. Both are flagged in the text.
- Verdicts assess whether a source supports a claim. A "not supported by this source" verdict does not mean the claim is false.
- No devil's-advocate or editorial review pass was run; this is a fact-check, not a full systematic review.

## 7. AI disclosure

This report was produced with AI assistance (Claude, Anthropic). Full texts were retrieved and read by AI agents, which extracted the quotations and figures; the compilation and verdicts were also AI-generated. No human has checked the quotations against the sources. Verify any figure against the original paper before citing it in a manuscript.

## References

Başaran, O. T., Can, Y. S., André, E., & Ersoy, C. (2024). Relieving the burden of intensive labeling for stress monitoring in the wild by using semi-supervised learning. *Frontiers in Psychology, 14*, Article 1293513. https://doi.org/10.3389/fpsyg.2023.1293513

Cheng, Y.-C., Su, M.-I., Liu, C.-W., Huang, Y.-C., & Huang, W.-L. (2022). Heart rate variability in patients with anxiety disorders: A systematic review and meta-analysis. *Psychiatry and Clinical Neurosciences, 76*(7), 292–302. https://doi.org/10.1111/pcn.13356

Čukić, M., Savić, D., & Sidorova, J. (2023). When heart beats differently in depression: Review of nonlinear heart rate variability measures. *JMIR Mental Health, 10*, Article e40342. https://doi.org/10.2196/40342

Hosseini, S., Gottumukkala, R., Katragadda, S., Bhupatiraju, R. T., Ashkar, Z., Borst, C. W., & Cochran, K. (2022). A multimodal sensor dataset for continuous stress detection of nurses in a hospital. *Scientific Data, 9*, Article 255. https://doi.org/10.1038/s41597-022-01361-y

Kim, H., Kim, M., Park, K., Kim, J., Yoon, D., Kim, W., & Park, C. H. (2023). Machine learning-based classification analysis of knowledge worker mental stress. *Frontiers in Public Health, 11*, Article 1302794. https://doi.org/10.3389/fpubh.2023.1302794

King, Z. D., Moskowitz, J., Egilmez, B., Zhang, S., Zhang, L., Bass, M., Rogers, J., Ghaffari, R., Wakschlag, L., & Alshurafa, N. (2019). micro-Stress EMA: A passive sensing framework for detecting in-the-wild stress in pregnant mothers. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3*(3), Article 91. https://doi.org/10.1145/3351249

Prajod, P., Mahesh, B., & André, E. (2024). Stressor type matters! — Exploring factors influencing cross-dataset generalizability of physiological stress detection. In *Proceedings of the International Conference on Multimodal Interaction (ICMI '24)* (pp. 508–517). Association for Computing Machinery. https://doi.org/10.1145/3678957.3685738

Schmidt, P., Reiss, A., Dürichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. In *Proceedings of the 2018 International Conference on Multimodal Interaction (ICMI '18)*. Association for Computing Machinery. https://doi.org/10.1145/3242969.3242985

Schmidt, P., Reiss, A., Dürichen, R., & Van Laerhoven, K. (2019). Wearable-based affect recognition—A review. *Sensors, 19*(19), Article 4079. https://doi.org/10.3390/s19194079

Schneider, M., & Schwerdtfeger, A. (2020). Autonomic dysfunction in posttraumatic stress disorder indexed by heart rate variability: A meta-analysis. *Psychological Medicine, 50*(12), 1937–1948. https://doi.org/10.1017/S003329172000207X

Vos, G., Trinh, K., Sarnyai, Z., & Rahimi Azghadi, M. (2023). Generalizable machine learning for stress monitoring from wearable devices: A systematic literature review. *International Journal of Medical Informatics, 173*, Article 105026. https://doi.org/10.1016/j.ijmedinf.2023.105026

Xu, S., Liu, H., Liu, Z., Su, P., & Gu, Z. (2026). Accuracy of photoplethysmography-derived pulse rate variability compared with electrocardiography-derived heart rate variability: A systematic review and meta-analysis. *Sensors, 26*(16), Article 5192. https://doi.org/10.3390/s26165192

Yfantidou, S., Karagianni, C., Efstathiou, S., Vakali, A., Palotti, J., Giakatos, D. P., Marchioro, T., Kazlouski, A., Ferrari, E., & Girdzijauskas, Š. (2022). LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive snapshots of our lives in the wild. *Scientific Data, 9*, Article 663. https://doi.org/10.1038/s41597-022-01764-x
