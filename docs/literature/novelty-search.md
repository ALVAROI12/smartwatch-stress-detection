# Novelty search for the JBHI rewrite, 2026-09-21

Brief: `docs/literature/novelty-search-brief.md`. Method: `academic-research-skills:deep-research` in lit-review mode. Four search agents worked in parallel, one per question group: cross-dataset, labels and confounds, personalisation, and benchmarks with wrist HRV. They searched WebSearch, the arXiv API, Europe PMC/PubMed, Crossref and Unpaywall. The Semantic Scholar API was rate-limited (HTTP 429) for most of the run. Every paper used below as close prior work was read in full unless marked **Unverifiable**. Full texts are in `sources/novelty/` (git-ignored). The per-question agent notes, with more quotes and line numbers, were kept outside the repo; everything load-bearing is repeated here.

Papers already verified in `dataset-and-claims-verification.md` were not redone: Prajod & André 2022, Prajod et al. 2024, Sahu et al. 2025, Islam & Washington 2023, the Vos et al. 2023 review, and Schmidt et al. 2019. Vos et al. 2023 in *J Biomed Inform* is a different, empirical paper and is covered here.

## Summary

| # | Contribution | Verdict | Closest prior work |
|---|---|---|---|
| 1 | Five-dataset E4 wrist LODO, transfer cost 0.015–0.05 BA | **Partly anticipated** | Kwon et al. (2026): LODO on 3 E4 corpora |
| 2 | Label quality drives apparent transfer failure | **Novel as evidence**; the idea is voiced but untested | Kwon et al. (2026); Mishra et al. (2020); Singh et al. (2026, FEEL) |
| 3a | Window-level split leakage (92% vs 49%) | **Already published** | Saeb et al. (2017); Bahameish et al. (2024) |
| 3b | Baseline-order confound | **Partly anticipated** (concept stated qualitatively, never quantified on wrist stress data) | Richer et al. (2024) |
| 3c | Missing channel posing as a domain gap | **Novel for wearable stress** (general concept exists) | Zhou et al. (2023); Mishra et al. (2020) |
| 4a | Few-shot personalisation, 5 windows per class | **Partly anticipated**; survives the chronological rerun on WESAD (0.82–0.95 BA with first-in-time windows and a 30 s gap; random selection inflated it by up to 0.08). See Section 4 | Tervonen et al. (2026); Stewart et al. (2020); Akkaya (2026) |
| 4b | Per-user few-shot calibration on an unseen dataset | **Novel as far as found** | Dahal (2026, SSRN preprint; read in full 2026-09-22: adapts with 10–30% of target-dataset subjects, not per-user labels) |
| 5 | Wrist HR/HRV validated against chest ECG in the pipeline | **Partly anticipated**; present as a methods check | Watanabe et al. (2025); Milstein & Gordon (2020) |

**Read before claiming novelty.** None of these could be retrieved in full:
- ~~Schreiber & Maleshkova (2026), AutoStress Benchmark~~ **Read in full on 2026-09-22 (UTSA copy): no longer a risk.** See "AutoStress and Cui (read 2026-09-22)" below.
- ~~Dahal (2026)~~ and ~~Calza-Metre & Borzì (2026)~~ **read in full on 2026-09-22**: no verdict changes. See "Papers read 2026-09-22" below.
- **Akkaya (2026), *BMC Med Inform Decis Mak*.** Abstract only.
- **Menghini et al. (2019), *Psychophysiology*.** Abstract only.

## 1. Five-dataset leave-one-dataset-out on the wrist E4

**Verdict: partly anticipated.** Pooled LODO on public E4 wrist corpora is already published. Pairwise transfer across four E4 corpora, and a five-dataset E4 "cross-dataset" paper, also exist. No verified paper has five public E4 wrist datasets in pooled LODO, uses Hongn 2025 PhysioNet as a target, uses stress vs all non-stress with per-subject z-scored HR/HRV+EDA, or reports that no target shows a large external cost. (Per-target costs of 0.015–0.05 are within the error bars expected for 15–35 test subjects; Varoquaux, 2018. Claim "no large cost", not "equal performance".)

**Kwon, Yoon, Hur & Kang (2026), *Healthcare*: the closest paper.** Full text: `sources/novelty/kwon-2026-iso-hr-cross-corpus.txt`.
- Data: three E4 wrist corpora (WESAD, Stress-Predict, Nurse). PhysioNet (Hongn 2025) is used only to set a movement threshold.
- Model and preprocessing: 64 s windows, logistic regression, histogram gradient boosting, a single SCR count, and four unsupervised domain-adaptation methods. Standardisation is fitted per fold on the training data, with no per-subject normalisation. Metric: AUROC.
- Validation, §2.6: "cross-corpus generalization uses leave-one-dataset-out (LODO; train on the union of the other corpora, test on the held-out corpus)".
- Results, abstract: "the transfer area under the receiver operating characteristic curve (AUROC) was approximately 0.90 with WESAD as the target but 0.52–0.57 with Stress-Predict or Nurse." §3.2 is headed "Cross-Corpus Transfer Is Poor for Every Model Evaluated".
- **Our reading of their numbers.** Within Stress-Predict (Table 4, LOSO) they already reach only 0.628 (LR) and 0.633 (GB). The LODO values are 0.564 and 0.558, so the transfer *cost* is 0.06–0.08 AUROC. Their "poor transfer" is mostly low absolute performance on Stress-Predict, which is also our weakest dataset (0.697 BA within). Say this in the paper rather than presenting our result as a contradiction of theirs. Where we genuinely differ is the absolute level after label fixes and per-subject z-scoring.
- **Activity features.** Adding activity lowered their transfer "because the movement–stress association reverses across protocols" (abstract). This supports using physiology features only.

**Other multi-dataset work.**
- **Calza-Metre (2025 thesis; *Smart Health* 2026).** Four E4 wrist corpora: WESAD, Campanella, VerBIO, AffectiveROAD. Pairwise zero-shot, not pooled LODO; the scaler is fitted on the source, with no per-subject normalisation. §4.2: "Zero-shot cross-test exposes sizeable domain gaps across the board". WESAD XGBoost macro-F1 falls from 0.8655 within to 0.6303 when trained on Campanella. The journal paper was read in full on 2026-09-22 (`calzametre-2026-smart-health.txt`); its Table 7 matches the thesis.
- **Xiao et al. (2025), IMWUT.** Seven E4/EmbracePlus datasets, mostly private. Training on one or two sources, not LODO; per-subject change scores. Trained on WESAD, ERM falls from 0.877 BA to 0.53–0.69 on the other datasets. §7.1.5: "merging datasets with different distributions, like our Control set and WESAD, does not enhance OOD performance."
- **Vos et al. (2023), *J Biomed Inform*.** SWELL, NEURO, WESAD, UBFC-Phys, EXAM; HR+EDA; XGBoost+ANN ensemble. WESAD is the only held-out dataset. §4.2: "when training StressData with the WESAD dataset excluded (Experiment 8), predictive accuracy reduced to 59% when testing on WESAD as an unseen validation set." Training on resampled "synthesized" subjects reached 85%. Metric is accuracy.
- **Mishra et al. (2020), IMWUT: the only small-cost precedent.** Two private E4 studies, both with mental-arithmetic stressors. Per-participant z-scored HR and min–max EDA. E4→E4 median AUROC costs about 0.01 (0.94 → 0.93; 0.98 → 0.97). §5.2: "this threshold changes with device type (i.e., data quality) and the data distribution in training and test sets."
- **Can, Benouis & André (2026), *IEEE Access*.** Five datasets, four of them E4. Self-supervised pretraining on the source, then a linear head trained on *target labels* under LOSO. This is not zero-shot and not LODO; VERBIO→WESAD weighted F1 is 96.38 vs 97.63 within. Reviewers may cite it, so explain the difference.
- **Amin et al. (2025), EMBC.** One new study scored with a model pretrained on the Mishra 2020 studies. E4 AUROC 0.953 within the study vs 0.723 with the pretrained model.

**Comparison table of the closest cross-dataset works**

| Work | Datasets (E4 wrist) | Design | Per-subject normalisation | Task | Metric | Transfer cost |
|---|---|---|---|---|---|---|
| **Ours** | 5 public (5) | pooled LODO, all 5 targets | z-score HR/HRV+EDA | stress vs all non-stress | BA | 0.015–0.05 |
| Kwon 2026 | 3 public (3) | pooled LODO | no | stress vs baseline/relax (Nurse: survey extremes) | AUROC | WESAD ≈ 0 (LR); Stress-Predict 0.06–0.08; Nurse near chance throughout |
| Calza-Metre 2025/26 | 4 public (4) | pairwise zero-shot | no | stress vs baseline | macro-F1 | best source 0–0.15; many pairs 0.2–0.6 |
| Xiao 2025 | 7, 2 public (all wrist) | 1–2 sources → rest | change score | binary stress | BA | 0.19–0.35 (ERM from WESAD) |
| Vos 2023 JBI | 6 (5 E4 + SWELL) | one held-out target (WESAD) | not described | stress vs non-stress | accuracy | 80% → 59% (85% with synthesis) |
| Mishra 2020 | 4 private (2 E4) | pairwise | z-score HR, min–max EDA | arithmetic stress vs rest | median AUROC | E4↔E4 ≈ 0.01 |
| Can 2026 | 5 (4 E4) | SSL transfer, target labels used | SD scaling | stress vs non-stress | weighted F1 | 1–3 points, not zero-shot |
| AutoStress 2026 | 3 (WESAD, PhysioNet Hongn, VitaStress; 2 E4, 1 Corsano) | no: pooled LOSO + 10× 80/20 subject splits; LODO named as future work | none described | baseline vs cognitive/social stress | accuracy, BA | not measured (pooled LOSO acc 0.84; per-dataset BA 0.74–0.92) |
| Dahal 2026 (SSRN) | 4 (WESAD, PhysioNet, SWELL-KW, UBFC-Phys) | pairwise, then fine-tuning on 30% of target subjects | participant-wise on SWELL | stress | ROC-AUC | WESAD→PhysioNet 0.413 zero-shot; Unverifiable |

## 2. Label quality drives apparent transfer failure

**Verdict: novel as evidence.** Several papers name label noise as a cause but none tests it. No paper audits the dataset descriptors, fixes the labels and measures transfer before and after. Two of our seven fixes are standard labelling practice elsewhere: excluding post-stressor rest and trimming baseline.

- **Kwon et al. (2026) tested a global label shift and found it does not rescue Stress-Predict.** §3.5: "Across shifts from −64 to +64 s, single-SCR AUROC ranges from 0.548 to 0.582 … Alignment uncertainty can therefore contribute to the point estimate but does not explain the weak discrimination by itself." §4: the evidence points "toward context and label quality", but they fix no labels.
  - Our errors are different in kind. They are per subject: 10 boundaries off by 92–442 s, one protocol-violating subject (S01), open-ended task spans, a bad-fit subject kept, and recovery counted as rest. A uniform shift cannot correct them.
  - Reviewers will know this paper. Cite it and make this distinction explicitly.
- **Mishra et al. (2020)** already exclude post-stressor rest. §4.4, footnote 6: "such rest periods will contain some residual physiological arousal of the preceding stressor, and hence labeling them not-stressed might not be appropriate". They also drop the first 6 min of baseline.
  - They never measure the effect on transfer.
  - Our PhysioNet second-half-rest result (external 0.746 → 0.797) is the first measurement of this effect that we found.
  - The Hongn 2025 descriptor also uses second-half rests (see `dataset-and-claims-verification.md`, fix 6).
- **Singh et al. (2026, FEEL; arXiv, marked NeurIPS 2025 D&B)** cover 19 datasets, including WESAD, UBFC-Phys and PhysioNet. §5.2: "Labeling strategy emerged as a major factor influencing model generalization". The evidence is correlational (grouping datasets by labelling method), and the task is arousal/valence, not stress.
- **Benchekroun et al. (2023)** only speculate that the labels in their field dataset "may not be as accurate". The label-noise-robust learning literature, e.g. Saeed et al. (2025), models random synthetic noise, not systematic protocol errors.
- No paper documents label-timing misalignment in WESAD.

## 3. Pitfalls

**3a. Window-level split leakage: already published.**
- Saeb et al. (2017): "record-wise CV often massively overestimates the prediction accuracy"; about 45% of reviewed studies used it.
- For stress, Bahameish et al. (2024) find K-fold exceeds leave-one-group-out by a mean of 5%.
- Campanella et al. (2023) is an example of the pitfall (see the earlier verification file).
- Use: present our 92% vs 49% as a confirmation on our data, in methods or the introduction, not as a contribution. The size of our gap is larger than published stress examples and is worth stating.

**3b. Baseline-order confound: partly anticipated.**
- **Richer et al. (2024), *Sci Rep*, Discussion:** comparing the TSST to a pre-stress baseline "can potentially induce a classification bias … parts of the observed differences … could be attributed to the different tasks, or to sequence effects, and not to the stress situation." They avoided the problem by design (a friendly-TSST control on another day). Their data are body motion, and the argument is qualitative.
- **Kriklenko & Kovaleva (2026)** concede that fixed task order confounds their workload labels with "protocol position".
- **Tervonen et al. (2023)** describe carryover ("physiological leakage") between stressors.
- **Tognotti et al. (2026)** quantify a related but different mechanism. Letting the normalisation baseline overlap the test windows inflates accuracy by 3–13 points.
- **Bahameish et al. (2024)** drop 8 WESAD participants whose meditation preceded stress. This shows awareness of order in WESAD, handled by exclusion rather than analysis.
- Not found in any paper:
  - a demonstration on public wrist stress datasets that baseline-first recording confounds baseline vs stress with time in session;
  - a measure of how baseline-referenced features label post-stressor rest (our 51–85%).
- Also not found: any work on EDA or skin-temperature warm-up after putting on the E4 as a confound.

**3c. Missing channel posing as a domain gap: novel for wearable stress.**
- The general problem is named "missingness shift" by Zhou et al. (2023, arXiv; EHR data, linear models).
- Mishra et al. (2020) saw thresholds vary across studies but attributed this to device and data quality.
- No paper shows a whole target dataset missing a channel, with the tree model's missing-value branch shifting scores while ranking stays intact. Our UBFC-Phys numbers show exactly that: AUROC 0.928, oracle-threshold BA 0.909, fixed by dropping temperature.

## 4. Few-shot personalisation

**4a. Verdict: partly anticipated, and our current number is at risk.**
- The idea of adding a few of a new person's labelled samples to a generic model is well established: Nkurikiyeyezu et al. (2020), Stewart et al. (2020), Sah & Ghasemzadeh (2021), Han et al. (2024), Simon & Chetouani (2026).
- Gains as large as ours appear mainly where calibration windows are drawn at random, or selection is not stated, and windows overlap heavily.
- Where calibration windows are taken chronologically, gains are small or vanish:
  - **Tervonen et al. (2026), *UMUAI*: the closest setup.** WESAD wrist E4 BVP+EDA, NeuroKit2 features, XGBoost, 60 s windows with a 30 s slide, 1–3 labelled windows per class taken "from the beginning of the first occurrence of each task".
    - Result, averaged over four datasets: retraining XGB gained 1.7 balanced-accuracy points and fine-tuning 0.7 (§4.2.2).
    - §5.2: "Incorporating some of the test person's data in model-level personalization introduced no certain benefit over the non-personalized model."
    - Their WESAD tasks exclude baseline (stress vs amusement and/or meditation).
  - **Akkaya (2026), abstract only.** Three E4 datasets. The few-shot recalibration benefit held "when calibration windows were sampled across the recording, but this did not survive a prospective chronological protocol".
  - **Stewart et al. (2020).** WESAD, but chest signals. Six context windows gave AUC 0.970 when taken from baseline and 0.984 when drawn at random. The authors warn that random context windows are correlated with neighbouring test windows.
  - **Han et al. (2024), IMWUT.** Chronological (the "initial sequence of data points from each label") and it does help on WESAD (AUROC about 0.91 → 0.97). Its budget is at least 20% of the person's data, about 40 windows.

**Update, 2026-09-21 evening: the rerun was done** in the main JBHI session (commit `bae4e5e`, `domain_adaptation.py --support chronological --gap 30`). On WESAD, the first 5 windows per class as support with a 30 s gap reach 0.82–0.95 balanced accuracy, +0.16 to +0.26 over source-only; random support had inflated this by up to 0.08. PhysioNet cannot be evaluated this way (a median of 5 Baseline windows per subject, all consumed as support). So the claim survives the chronological protocol on WESAD, unlike Tervonen et al. (2026). Per-subject z-scoring in that run still uses the whole recording, which the normalisation experiment on branch `jbhi-novelty-experiments` addresses.

**Our original protocol matched the risky pattern** (as found in the literature pass):
- `scripts/domain_adaptation.py:121` draws the 5 support windows per class with `rng.permutation`, anywhere in the recording.
- Windows are 60 s with a 30 s step (`scripts/extract_features.py:33`), so each support window shares half its signal with a neighbouring test window.
- The model there is a small neural network, not XGBoost.

**Whole-recording per-subject z-scoring is itself personalisation.** Tervonen et al. (2026) call it "personal normalization". In their study it beat every minimal-data method. It uses the test subject's unlabelled windows, including the stress windows (see also Section 6 of `dataset-and-claims-verification.md`). This applies to the main LODO result too, not only the few-shot one.

**Before claiming 4a, rerun with:**
- the first k windows of each class in time as support;
- a gap of at least one window (30 s) before any test window;
- random selection reported as the contrast.

Also report one normalisation variant that takes statistics only from calibration or baseline windows. Tognotti et al. (2026) quantify how test-inclusive normalisation inflates accuracy. If the gain survives this protocol on five datasets, that is a genuine, leakage-controlled result. If it does not, drop the 0.85–0.95 claim.

**4b. Few-shot calibration of a model trained on other datasets, tested on an unseen dataset: novel as far as found.**
- Dahal (2026, SSRN, **Unverifiable**; snippets only) adapts across WESAD, PhysioNet, SWELL-KW and UBFC-Phys using 30% of the target dataset's *subjects*. That is dataset-level fine-tuning, not a few labels per new user. It must be read before claiming priority.
- Simon & Chetouani (2026) borrowed *unlabelled* K-EmoCon windows to personalise WESAD, with no gain.
- Xu et al. (2024) propose domain adaptation plus a few user labels but report no results.

## 5. Wrist HR/HRV against chest ECG

**Verdict: partly anticipated.** Present this as a methods sanity check supporting the HR/HRV/EDA feature choice and quality gating, not as a contribution.

**Already published:**
- **Watanabe et al. (2025, UbiComp Companion) compared WESAD chest ECG with E4 BVP per condition, including the TSST.** With a fixed filter, the stress task is worst: beat F1 60.96, IBI MAE 43.36 ms, RMSSD MAE 158.7 ms. This is the direct WESAD precedent.
- **Milstein & Gordon (2020), speaking task:** HR r = 0.992; RMSSD 61.71 ms (E4) vs 42.00 ms (ECG), r = 0.417. That is the same direction and similar size as our r = 0.53 and +34 ms bias.
- **Schuurmans et al. (2020), rest and meditation:** HR ICC .99; E4 RMSSD about 16 ms higher than ECG. "The Empatica E4 has good predictive value for all ANS parameters except for RMSSD."
- **Menghini et al. (2019), abstract only:** HRV accuracy was "diminished by wrist movements, cognitive and emotional stress". Do not quote numbers from it.
- **Hu et al. (2024):** only 26.14% of E4 PPG was valid in the lab. Compare our 37% of stress windows usable.
- **Amin et al. (2025):** the same stress pipeline in the same people gives HRV-only AUROC 0.61 on the E4 vs 0.73 on a Polar H10 chest strap.

**Not found in any paper:** validation at the classifier's own windows, a usable-window yield for stress windows specifically, and a link from agreement to the classifier's features.

**Caveat:** our r is window-level, while most prior studies correlate segment or subject means. State this when comparing.

## Fair comparison table (subject-independent, 2023–2026, plus anchor)

Only papers read in full. "OK" means LOSO or subject-grouped with no tuning on test subjects that we could see. "Optimistic" means design choices were made on LOSO test results. No paper uses our task and metric (stress vs all non-stress, balanced accuracy, corrected tests), so this is context, not a leaderboard.

| Paper | Dataset / sensors | Task | Validation | Result | Flag |
|---|---|---|---|---|---|
| Schmidt et al. 2018 (anchor) | WESAD wrist physiology, RF | binary (stress vs baseline+amusement) | LOSO | acc 88.33, F1 86.10 | OK |
| Schmidt et al. 2018 | same | 3-class | LOSO | acc 76.17, F1 66.33 | OK |
| Stržinar et al. 2023 | WESAD wrist EDA | binary | LOSO | acc 84.87, F1 82.13 | OK |
| Wu et al. 2024 (TAFFC) | WESAD wrist BVP+EDA+TEMP, SSL transformer | binary / 3-class | LOSO | acc 96.29, F1 95.11 / acc 84.94, F1 82.60 | OK as reported |
| Farahani et al. 2026 (arXiv) | WESAD wrist, RF | binary (all non-stress) | LOSO | acc 0.930, F1 0.799 | OK |
| Farahani et al. 2026 | Stress-Predict | binary | LOSO | acc 0.739, F1 0.154 | OK |
| Kwon et al. 2026 | WESAD / Stress-Predict, E4 | binary | LOSO | AUROC 0.876–0.897 / 0.563–0.633 | OK |
| Simon & Chetouani 2026 (arXiv) | WESAD wrist | 3-class | LOSO | acc 83.99, macro-F1 77.07 | OK (no calibration) |
| Rashid et al. 2023 (IEEE IoT J) | WESAD wrist | binary / 3-class | LOSO | acc 94.12 / 86.34 | Optimistic |
| Gil-Martín et al. 2022 (IEEE AESM) | WESAD wrist, all signals / wrist physiology | binary | LOSO | acc 92.70, F1 92.55 / acc 87.30 | Optimistic (preprocessing chosen on LOSO results); the often-quoted 96.62 uses chest + wrist |
| Gil-Martín et al. 2022 | WESAD wrist, all signals | 3-class | LOSO | acc 74.90, F1 74.60 | Optimistic |
| Hosseini, E. et al. 2023 (IEEE BIBM) | WESAD wrist EDA | stress vs baseline only | LOSO | acc 97.03 | Optimistic (feature subset chosen on LOSO results; amusement dropped) |
| Zhao et al. 2025 (PULSE, arXiv) | PhysioNet Hongn 2025, wrist BVP/ACC/TEMP | binary | LOSO | AUROC 0.965 (no-teacher baseline 0.827) | OK-ish; pretraining exposure not stated |

Excluded as leaky or not subject-grouped:
- Hongn et al. 2025 descriptor (93%, 10-fold over protocol blocks).
- UBFC-Phys descriptor (stratified 7-fold over samples).
- Campanella et al. 2023 (random 10-fold).
- Birkenmaier et al. 2026 (98.62%; hyperparameters "conducted on the complete dataset prior to LOSO evaluation"; chest EDA).
- Hagos et al. 2026 (internally inconsistent numbers).
- Can et al. 2019, *Sensors* (own programming-contest dataset, 3-class; 88.20% general model; participant not respected in cross-validation according to the full-text check).
- Naegelin et al. 2023, *J Biomed Inform* (lab office simulation; stratified 10-fold over 1-min segments with hyperparameters selected "over the 10 test data folds"; chest-ECG HRV).
- Dahal, Bogue-Jimenez & Doblas 2023, *Sensors* (chest ECG; 70% of each subject's windows in training; their own WESAD LOSO check ranges 0.31–0.95).

Unverifiable (no full text): Zhu et al. 2023 (JBHI, wrist EDA), Al Dossary et al. 2025 (ICMI). Ladakis et al. (2025) was read in round 2 and is not a benchmark for us: its datasets are PhysioNet Non-EEG (Affectiva Q), Drivers and Nurses (only Nurses is E4), and the "74.1% balanced accuracy" in the round-1 snippet is its description of Siirtola & Röning (2020), not its own result.

## Round 2 (2026-09-21 evening): gaps filled

Three more agents searched the unread papers and forward citations, domain adaptation / threshold transfer / mild stress, and wrist EDA / exercise. Nothing overturns a verdict above. Load-bearing figures were re-checked against the saved texts.

**Forward citations.** None of the papers citing Kwon (2026; 0 citers so far), Prajod et al. (2024), Hongn et al. (2025) or Campanella et al. (2024) runs pooled LODO on wrist E4. Worth citing:
- **Aydoğan & Villagra Povina (2026), *Med Eng Phys* (abstract only).** On PhysioNet, XGBoost reaches BA 0.703 for rest vs stress but labels 82.6% of exercise-session windows as stress. WESAD→PhysioNet transfer "remained poor".
- **Fecke & Rehof (2026), *IEEE Access* (abstract only).** Normalises against short baseline sections to prevent "normalization data leakage". Cite next to Tognotti (2026).
- **Moon et al. (2026), ReliaGate.** Uses WESAD, UBFC-Phys, Campanella and PhysioNet, but only within each dataset.

**Domain adaptation does not beat source-only: partly anticipated.**
- Kwon (2026), §3.3 is headed "Four Unsupervised Adaptation Methods Do Not Recover Transfer" (CORAL, subspace alignment, TCA, importance weighting; no per-subject normalisation). Their §4.4 says "Adversarial representation learning … was not evaluated".
- Xiao et al. (2025): subject-adversarial DANN on normalised wrist data gains about 0.01 out-of-distribution (0.6843 vs 0.6712) and loses in distribution.
- Sigcha et al. (2026, *Applied Sciences*): EDA-only CORAL across two datasets is direction-dependent.
- Not found: a head-to-head of per-subject normalisation vs DA for stress. Claim that part (our DANN and subject-DANN arms, all on top of z-scoring). State the scope if DA was run on WESAD↔PhysioNet only.

**Ranking vs threshold: partly anticipated.**
- Mishra et al. (2020, §5.2) treat "building a model and choosing a decision (classification) threshold" as "two separate components". Their label-free clustering threshold always beat a fixed one, whereas our label-free stress-rate cut hurt Campanella (0.821 → 0.733). Report that contrast.
- Not found: a split of cross-dataset loss into ranking and threshold parts, or prior-shift correction (Saerens EM, black-box shift estimation) applied to wearable stress.
- Argument to add: for a binary task, prior-shift correction only rescales the odds, so its gain is capped by the oracle-threshold gain we already report (at most +0.04, or +0.08 on UBFC-Phys). It also assumes p(x|y) is unchanged, which fails across protocols.

**Mild stress: open, but not named as such in reviews.**
- Siirtola & Röning (2020, AffectiveROAD, E4): training on continuous stress targets gave BA 82.3% vs 74.1% for binary training. This is the closest method precedent.
- Kaya et al. (2026): psychological stress was confused with rest in 50% of windows.
- No paper reports mild-vs-strong performance under subject-independent or cross-dataset validation.
- A defensible design would be an intensity-stratified evaluation plus one simple method (ordinal or regression training). PhysioNet's per-task self-rated stress (1–10) is the best graded label already in our pipeline. UBFC-Phys ctrl/test cannot be used: it is between-subject, so per-subject z-scoring removes the difference. Graded levels run in increasing order reintroduce the order confound.

**Wrist EDA validity: weak agreement, relative changes tracked for strong stressors.**
- Milstein & Gordon (2020): SCL r = 0.298 during conversation.
- van Lier et al. (2020): mean cross-correlation 0.25 with the reference.
- Liang et al. (2026): wrist SCR rate still gives LOSO BA 0.71 for TSST vs all non-stress.
- Adopt Kleckner et al. (2018) quality rules (EDA 0.05–60 μS, slope within ±10 μS/s, 5 s padding; κ = 0.74 against experts). Make the temperature rule optional, because UBFC-Phys has no temperature. Report invalid and flat-signal rates per dataset.
- Soften "EDA carries most of the stress signal" to strong social stressors.

**Exercise as a confound: known, with numbers.**
- Gjoreski et al. (2017, E4): a lab-trained detector in daily life flagged 1,630 of 4,938 no-stress events as stress; context features raised precision to 0.95.
- Sevil et al. (2021): stress detection held at 87.16% on subjects not used in training, with activity recognised separately.
- Kwon (2026): accelerometer features lower cross-corpus transfer.
- Gap we can fill: subject-held-out, stress-vs-all-non-stress on wrist data with and without exercise among the negatives. Running on branch `jbhi-novelty-experiments`.

## Papers read 2026-09-22 (UTSA and free copies)

### AutoStress and Cui

**Schreiber & Maleshkova (2026), AutoStress Benchmark, IEEE CAI: no leave-one-dataset-out.** Full text: `sources/novelty/schreiber-2026-autostress.txt`.
- **Data (§III-A, B):** WESAD (15, E4), the PhysioNet "Wearable Dataset" of Hongn et al. (36, E4) and VitaStress (21, Corsano CardioWatch 287-2B); 72 subjects. The round-2 inference was right. No UBFC-Phys, Stress-Predict or Campanella.
- **Task (§III-D):** "baseline vs. mental stress". Exercise and amusement are excluded (§IV). This is not our stress vs all non-stress task.
- **Validation (§III-D, E):** classifier selection on "10× 80/20 subject-aware splits", then leave-one-subject-out on the pooled data. Also LOSO on WESAD + VitaStress only, and LOSO with dataset- and class-balancing weights. No per-subject normalisation is described.
- **Results (§IV):**
  - Pooled XGBoost LOSO accuracy is 0.84 (SD 0.18).
  - Per-dataset LOSO balanced accuracy (Table IV): VitaStress 0.915, WESAD 0.807, PhysioNet 0.739.
  - WESAD + VitaStress only: 0.87 accuracy.
  - Balancing: no gain (0.83).
- **They name LODO as future work.** §IV-B: "domain differences remain and should be explored with harmonization, LODO (leave-one-dataset-out) experiments, and domain-adaptive methods in future work."
- **Use in the paper.** Cite it as pooled multi-dataset work that did not hold out datasets. It supports two of our points:
  - PhysioNet is the hardest target there too (0.739 BA vs our 0.776 within).
  - They attribute the gap to "mismatches in stressor tasks".
- **Verdict for contribution 1 is unchanged.** Kwon et al. (2026) stay the closest prior work, and they remain the only verified pooled LODO on wrist E4.

**Cui, Sun, Chen & Peng (2025), *Biomedical Signal Processing and Control* 110, 108149: not a transfer study.** Full text: `sources/novelty/cui-2025-ppg-cross-dataset.txt`.
- The title says "cross-dataset", but each of SIPD (own; E4 and Honor Band 5), WESAD (E4) and CLAS (Shimmer3, earlobe PPG) is trained and tested separately. The paper uses "k-fold cross-validation with k = 5" (§4.2). It does not say that folds are grouped by subject.
- On WESAD it uses "a 2-min sliding window with a stride of 2-sec" (§4.2), giving 5,818 windows. The reported 94.8–98.9% accuracy is therefore probably inflated by leakage between neighbouring windows (our inference).
- Use in the paper: related work on PPG features only, or as another example of record-wise evaluation. No bearing on contributions 1–5.

### Dahal (2026), SSRN: 4b stays novel

Full text: `sources/novelty/dahal-2026-shift-aware.txt` (6 pages, not peer-reviewed).
- **Data and features:** WESAD, PhysioNet (Hongn), SWELL-KW and UBFC-Phys, labelled binary stress/non-stress. Random Forest on EDA and HR/BVP summary statistics.
- **Adaptation is at dataset level.** §H: "fine-tune on 10/20/30% of target subjects selected stratified by stress prevalence. Target subjects not in the adaptation set are evaluated zero-shot." That is not a few labels per new user, so 4b (per-user few-shot calibration on an unseen dataset) stays novel.
- **Transfer is pairwise, not pooled LODO.** WESAD→PhysioNet ROC-AUC goes from 0.413 (zero-shot, below chance) to 0.842 with 30% of target subjects; PhysioNet→WESAD goes from 0.614 to 0.691.
- **Other results.** Participant-wise normalisation lifts SWELL R vs T from 0.601 to 0.857. Cost-aware thresholds differ by dataset (WESAD 0.37, PhysioNet 0.41 at 1:1). These are consistent with our z-scoring and threshold findings.
- **Weak source.** It describes UBFC-Phys as "camera-based physiological proxies", and its extended characterisation credits WESAD's stress data to "chest-worn devices". Cite as a preprint only.

### Calza-Metre & Borzì (2026), *Smart Health*: matches the thesis

Full text: `sources/novelty/calzametre-2026-smart-health.txt`.
- **Design (§3, "Evaluation strategies"):** four E4 wrist datasets (WESAD, Campanella, VerBio, AffectiveRoad); within-dataset LOSO, then pairwise zero-shot transfer.
  - The source model is "the LOSO fold with the highest macro-F1", which is an optimistic choice.
  - Target windows are "transformed using the source preprocessing". There is no per-subject normalisation.
- **Results (Table 7):** the same as the thesis. WESAD XGBoost macro-F1 is 0.866 within and 0.630 from Campanella. The abstract reports an average F1 drop of −21% for deep models.
- **Labels (Table 3):** Campanella is labelled "rest vs. cognitive/social/combined stress", i.e. every task counts as stress, which differs from our subtraction-only label.
- **Mild stress.** VerBio stress is thresholded at 0.20. §4.2 says this corresponds "to mild perceived arousal rather than strong and discrete autonomic responses", and VerBio is the worst source. Cite this for the mild-stress question.
- The verdict on contribution 1 is unchanged.

### Cahoon & Garcia (2023), ACM BCB: no threat

Full text: `sources/novelty/cahoon-2023-healthcare-workers.txt`.
- WESAD, Nurse (CSDN) and TILES-2019, using heart rate and step count only.
- "All models were trained using five-fold stratified cross-validation", not grouped by subject.
- Within-dataset XGBoost reaches ROC-AUC 0.83 (WESAD), 0.65 (CSDN) and 0.84 (TILES-2019). Pairwise transfer without adaptation gives 0.46–0.52 (§4, Table 5). Supervised TrAdaBoost, which uses target labels, improves it.
- Use as related work on healthcare-worker transfer.

### ISPAAD (Schreiber & Maleshkova 2025), *BIO Web of Conferences*

Full text: `sources/novelty/schreiber-2025-ispaad.txt`. It is the dataset paper behind AutoStress (WESAD, PhysioNet Hongn, VitaStress; 71 subjects) and reports no transfer experiments.

## Papers cited in the MSc thesis, checked 2026-09-22

The author's thesis cites several papers the novelty search had not covered. Eleven were checked, ten against full text (`sources/novelty/`). None changes a verdict. Details are in the agent notes (outside the repo).

- **Li & Washington (2024), *JMIR AI*.**
  - WESAD **chest RespiBAN** only, despite "consumer wearable" in the title.
  - 3 classes; personalised models trained on the first 70% of each class.
  - Personalised 95.06% vs participant-exclusive 67.65%.
  - Supports 4a only as a large-budget, chest-based contrast. Do not cite it as wrist or consumer-device evidence.
- **Albaladejo-González et al. (2023), *J Ambient Intell Humaniz Comput*** (https://doi.org/10.1007/s12652-022-04365-z).
  - WESAD E4 wrist BVP vs SWELL-KW chest ECG, 50 HRV features, 5-min windows.
  - Pairwise transfer; MLP macro-F1 falls 99.03 → 28.41 (WESAD→SWELL) and 82.75 → 57.28 (SWELL→WESAD).
  - Normalises on "the first 50% of" each subject's non-stress windows, so its non-stress test windows are the second half of baseline: the order confound of 3b, not discussed.
  - Add to "Other multi-dataset work"; no verdict change.
- **Dahal, Bogue-Jimenez & Doblas (2023), *Sensors*.** Chest ECG, WESAD + SWELL pooled, windows split within subjects, no cross-dataset test. A leakage example (3a), not a benchmark.
- **Bent et al. (2020), *npj Digit Med*.**
  - Includes the E4, using its onboard 1 Hz HR against ECG.
  - MAE 11.3 bpm at rest and 12.8 bpm during activity; error during activity "on average, 30% higher than during rest".
  - Supports the motion caveat. Not comparable to our 2.4 bpm, which is BVP-derived with artefact rejection.
- **Gillinov et al. (2017), *MSSE*.** Abstract only. No E4; exercise background only.
- **Varoquaux (2018), *NeuroImage*.**
  - Expected error bars for binary classification: ±10% at n = 100, ±15% at n = 30.
  - "The standard error across folds strongly underestimates them."
  - Hence the softened wording for contribution 1 above.
- **Gil-Martín et al. (2022)** and **E. Hosseini et al. (2023):** added to the fair comparison table as Optimistic.
- **Can et al. (2019)**, **Naegelin et al. (2023)** and **Dahal et al. (2023):** added to the excluded list.
- **Mattern et al. (2023).** Wrist E4, four recalled emotions (not stress); 75% is window-level 10-fold vs 32% LOSO. Use only in the leakage discussion.
- **Sharma & Gedeon.**
  - The thesis cites "2021, CMPB 208, 106219"; that DOI does not exist in Crossref.
  - The real paper is Sharma & Gedeon (2012), *Computer Methods and Programs in Biomedicine, 108*(3), 1287–1301, https://doi.org/10.1016/j.cmpb.2012.07.003.

**Thesis bibliography corrections for the correction note:**
- Gil-Martín's 96.6% is chest + wrist; wrist-only is 92.70%.
- Can et al. (2019) is neither SWELL nor LOSO.
- Naegelin et al. (2023) is a lab simulation, not free-living.
- Li & Washington (2024) is chest data.
- Dahal et al. (2023) is not subject-independent.
- Fix the Sharma & Gedeon year, volume and DOI.
- Add the Albaladejo-González DOI.

## Experiments run after this search

See `docs/novelty_experiments.md`, branch `jbhi-novelty-experiments`. Three results change the framing below:
- **The small transfer cost does not depend on transductive z-scoring.** With raw features the cost is −0.01 to 0.05. A causal (past-windows-only) scaling keeps it on long recordings (PhysioNet, Stress-Predict, WESAD) but loses about 0.10 externally on short recordings (Campanella, UBFC-Phys).
- **Exercise breaks cross-dataset transfer.** A model trained on datasets without exercise calls 69.5% of PhysioNet exercise windows stress, and external BA falls from 0.746 to 0.583 (p_holm = 0.011). With exercise among the training negatives, only 4–6% are called stress.
- **The label fixes are not what separates us from Kwon et al.** In their WESAD + Stress-Predict setting the fixes change transfer by at most 0.02 (not significant). Our pipeline beats their Stress-Predict AUROC (0.67 vs 0.56) with or without the fixes. Contribution 2 should be framed as data quality, not as the cause of the small transfer cost.

## Recommended framing

1. **Headline:** the largest pooled leave-one-dataset-out evaluation on public wrist-E4 stress data (five datasets), where no target shows a large external cost (0.015–0.05 balanced accuracy, within small-sample error bars).
   - Do not say "first cross-dataset" or "first LODO": Kwon et al. (2026) did LODO on three E4 corpora.
   - Frame the contrast with Kwon, Calza-Metre, Xiao and Vos as "reported failures shrink after label audit and per-subject normalisation".
   - Note that Stress-Predict is hard within-dataset for everyone.
2. **Label audit as the main evidential contribution.** Claim "first descriptor-level label audit with measured effect on cross-dataset transfer". Do not claim that label noise as an explanation is new: cite Kwon (2026), FEEL and Benchekroun (2023) for the idea, and Mishra (2020) for recovery exclusion.
   - Contrast our per-subject fixes with Kwon's global-shift test.
3. **Order confound.** Cite Richer et al. (2024) for the concept. Claim the quantification on public wrist datasets and the post-stressor-rest mechanism.
4. **Missing channel.** Cite Zhou et al. (2023) for missingness shift and Mishra et al. (2020) for thresholds varying across studies. Claim the concrete diagnosis: ranking intact, threshold shifted, fixed by dropping the channel.
5. **Window-level leakage and wrist HRV validation:** move both to methods as confirmations. Cite Saeb (2017) and Bahameish (2024), and Watanabe (2025), Milstein & Gordon (2020) and Schuurmans (2020).
6. **Few-shot personalisation:** hold the claim until it is rerun chronologically (Section 4). If it survives, frame it as a leakage-controlled test of a known idea, citing Tervonen (2026), Akkaya (2026) and Stewart (2020). Cross-dataset per-user calibration is the part with a novelty claim, pending Dahal (2026).

**Claims to drop or soften**
- "First cross-dataset / LODO wrist stress study": drop.
- "Prior work reports near-chance cross-dataset transfer" (citing only Prajod): soften. Kwon's Stress-Predict *cost* is 0.06–0.08 AUROC, and Mishra found about 0.01 between E4 studies. Near-chance results come from low within-dataset ceilings (Kwon on Stress-Predict and Nurse) or from chest-ECG studies (Prajod). Even chest-strap→E4 heart-rate transfer in Mishra et al. (2020, Table 4) keeps AUROC at 0.75–0.80.
- "Few-shot personalisation reaches 0.85–0.95": do not state it until the chronological rerun is done.
- The wrist HR/HRV validation as a listed contribution: drop it and keep it as a methods check.
- Per-subject z-scoring over the whole session: state openly that it is transductive (it uses unlabelled test-subject windows), and report one variant that is not.

## References

Akkaya, A. (2026). Calibration, not architecture, limits cross-subject wearable stress detection: A multi-dataset, decision-focused evaluation with label-efficient recalibration. *BMC Medical Informatics and Decision Making*. https://doi.org/10.1186/s12911-026-03842-1 [Abstract only]

Amin, O. B., Mishra, V., Tapera, T. M., Volpe, R., & Sathyanarayana, A. (2025). Extending stress detection reproducibility to consumer wearable sensors. In *2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*. IEEE. https://doi.org/10.1109/EMBC58623.2025.11252853

Bahameish, M., Stockman, T., & Requena Carrión, J. (2024). Strategies for reliable stress recognition: A machine learning approach using heart rate variability features. *Sensors, 24*(10), Article 3210. https://doi.org/10.3390/s24103210

Benchekroun, M., Velmovitsky, P. E., Istrate, D., Zalc, V., Morita, P. P., & Lenne, D. (2023). Cross dataset analysis for generalizability of HRV-based stress detection models. *Sensors, 23*(4), Article 1807. https://doi.org/10.3390/s23041807

Calza-Metre, M. (2025). *Machine learning for stress detection based on wearable sensor data* [Master's thesis, Politecnico di Torino]. https://webthesis.biblio.polito.it/38110/

Calza-Metre, M., & Borzì, L. (2026). Machine learning-based automatic stress detection: Performance and generalization across datasets. *Smart Health, 41*, Article 100693. https://doi.org/10.1016/j.smhl.2026.100693 

Can, Y. S., Benouis, M., & André, E. (2026). Cross-dataset generalizability analysis of multimodal self-supervised learning for stress recognition across lab and daily contexts. *IEEE Access, 14*, 35930–35943. https://doi.org/10.1109/ACCESS.2026.3670764

Cui, X., Sun, H., Chen, Z., & Peng, C.-K. (2025). Enhanced PPG-based stress detection: A multivariate cross-dataset analysis across devices and tasks. *Biomedical Signal Processing and Control, 110*, Article 108149. https://doi.org/10.1016/j.bspc.2025.108149

Cahoon, J. L., & Garcia, L. A. (2023). Continuous stress monitoring for healthcare workers: Evaluating generalizability across real-world datasets. In *Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB '23)*. ACM. https://doi.org/10.1145/3584371.3612974

Dahal, S. (2026). *A shift-aware deployment framework for wearable stress AI: Cross-dataset phenotype audit, few-shot adaptation, statistical reliability, and cost-aware policy* [Preprint]. SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7406759 

Farahani, S. A., Cao, H., & Rahmani, A. M. (2026). *When clean signals are not enough: Detecting structural ambiguity for safe wearable stress classification* (arXiv:2608.18397) [Preprint]. arXiv.

Han, Y., Zhang, P., Park, M., & Lee, U. (2024). Systematic evaluation of personalized deep learning models for affect recognition. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 8*(4), Article 206. https://doi.org/10.1145/3699724

Hu, X., Sgherza, T. R., Nothrup, J. B., Fresco, D. M., Naragon-Gainey, K., & Bylsma, L. M. (2024). From lab to life: Evaluating the reliability and validity of psychophysiological data from wearable devices in laboratory and ambulatory settings. *Behavior Research Methods, 56*(7). https://doi.org/10.3758/s13428-024-02387-3

Kriklenko, E., & Kovaleva, A. (2026). Machine learning classification of physiological dynamics during standardized task-demand transitions. *Bioengineering, 13*(8), Article 898. https://doi.org/10.3390/bioengineering13080898

Kwon, N., Yoon, S., Hur, J., & Kang, H. S. (2026). When wellness stress detection enters healthcare workflows: An iso-heart-rate, cross-corpus evaluation of wrist wearables. *Healthcare, 14*(15), Article 2434. https://doi.org/10.3390/healthcare14152434

Menghini, L., Gianfranchi, E., Cellini, N., Patron, E., Tagliabue, M., & Sarlo, M. (2019). Stressing the accuracy: Wrist-worn wearable sensor validation over different conditions. *Psychophysiology, 56*(11), Article e13441. https://doi.org/10.1111/psyp.13441 [Abstract only]

Milstein, N., & Gordon, I. (2020). Validating measures of electrodermal activity and heart rate variability derived from the Empatica E4 utilized in research settings that involve interactive dyadic states. *Frontiers in Behavioral Neuroscience, 14*, Article 148. https://doi.org/10.3389/fnbeh.2020.00148

Mishra, V., Sen, S., Chen, G., Hao, T., Rogers, J., Chen, C.-H., & Kotz, D. (2020). Evaluating the reproducibility of physiological stress detection models. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 4*(4), Article 147. https://doi.org/10.1145/3432220

Nkurikiyeyezu, K., Yokokubo, A., & Lopez, G. (2020). Effect of person-specific biometrics in improving generic stress predictive models. *Sensors and Materials, 32*(2), 703–722. https://doi.org/10.18494/SAM.2020.2650

Rashid, N., Mortlock, T., & Al Faruque, M. A. (2023). Stress detection using context-aware sensor fusion from wearable devices. *IEEE Internet of Things Journal, 10*(16), 14114–14127. https://doi.org/10.1109/JIOT.2023.3265768

Richer, R., Koch, V., Abel, L., Hauck, F., Kurz, M., Ringgold, V., Müller, V., Küderle, A., Schindler-Gmelch, L., Eskofier, B. M., & Rohleder, N. (2024). Machine learning-based detection of acute psychosocial stress from body posture and movements. *Scientific Reports, 14*, Article 8251. https://doi.org/10.1038/s41598-024-59043-1

Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience, 6*(5), Article gix019. https://doi.org/10.1093/gigascience/gix019

Saeed, A., Spathis, D., Oh, J., Choi, E., & Etemad, A. (2025). Learning under label noise through few-shot human-in-the-loop refinement. *Scientific Reports, 15*(1), Article 4276. https://doi.org/10.1038/s41598-025-87046-z

Sah, R. K., & Ghasemzadeh, H. (2021). *Stress classification and personalization: Getting the most out of the least* (arXiv:2107.05666) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2107.05666

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. In *Proceedings of the 20th ACM International Conference on Multimodal Interaction* (pp. 400–408). ACM. https://doi.org/10.1145/3242969.3242985

Schreiber, P., & Maleshkova, M. (2025). ISPAAD: Integrated stress, physical activity, and amusement dataset. *BIO Web of Conferences, 195*, Article 01003. https://doi.org/10.1051/bioconf/202519501003

Schreiber, P., & Maleshkova, M. (2026). AutoStress benchmark: Evaluating factors that influence cross dataset generalizabilty in stress recognition. In *2026 IEEE Conference on Artificial Intelligence (CAI)* (pp. 1335–1341). IEEE. https://doi.org/10.1109/CAI68641.2026.11536323

Schuurmans, A. A. T., de Looff, P., Nijhof, K. S., Rosada, C., Scholte, R. H. J., Popma, A., & Otten, R. (2020). Validity of the Empatica E4 wristband to measure heart rate variability (HRV) parameters: A comparison to electrocardiography (ECG). *Journal of Medical Systems, 44*, Article 190. https://doi.org/10.1007/s10916-020-01648-w

Simon, L., & Chetouani, M. (2026). *Retrieval-augmented personalization with foundation models for wearable stress detection* (arXiv:2606.24985) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2606.24985

Singh, P., Gupta, A., Jalan, S., Kumar, M., & Singh, P. (2026). *FEEL: Quantifying heterogeneity in physiological signals for generalizable emotion recognition* (arXiv:2604.05926) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2604.05926

Stewart, C. L., Folarin, A., & Dobson, R. (2020). *Personalized acute stress classification from physiological signals with neural processes* (arXiv:2002.04176) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2002.04176

Stržinar, Ž., Sanchis, A., Ledezma, A., Sipele, O., Pregelj, B., & Škrjanc, I. (2023). Stress detection using frequency spectrum analysis of wrist-measured electrodermal activity. *Sensors, 23*(2), Article 963. https://doi.org/10.3390/s23020963

Tervonen, J., Amparore, E., Botta, M., Närväinen, J., Pettersson, K., & Mäntyjärvi, J. (2026). Data-constrained personalization of cognitive state detection with feature-based and foundation models. *User Modeling and User-Adapted Interaction*. https://doi.org/10.1007/s11257-026-09444-w

Tervonen, J., Närväinen, J., Mäntyjärvi, J., & Pettersson, K. (2023). Explainable stress type classification captures physiologically relevant responses in the Maastricht Acute Stress Test. *Frontiers in Neuroergonomics, 4*, Article 1294286. https://doi.org/10.3389/fnrgo.2023.1294286

Tognotti, A., Otesteanu, C. F., Anceschi, L., & Menon, C. (2026). Baseline normalization choices inflate classification performance in wearable health monitoring: Quantification and mitigation strategies. *Frontiers in Digital Health, 8*, Article 1827279. https://doi.org/10.3389/fdgth.2026.1827279

Vos, G., Trinh, K., Sarnyai, Z., & Rahimi Azghadi, M. (2023). Ensemble machine learning model trained on a new synthesized dataset generalizes well for stress prediction using wearable devices. *Journal of Biomedical Informatics, 148*, Article 104556. https://doi.org/10.1016/j.jbi.2023.104556

Watanabe, Y., Yamane, N., Sathyanarayana, A., Mishra, V., & Goodwin, M. S. (2025). Beyond motion artifacts: Optimizing PPG preprocessing for accurate pulse rate variability estimation. In *Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing* (pp. 1176–1182). ACM. https://doi.org/10.1145/3714394.3756241

Wu, Y., Daoudi, M., & Amad, A. (2024). Transformer-based self-supervised multimodal representation learning for wearable emotion recognition. *IEEE Transactions on Affective Computing, 15*(1), 157–172. https://doi.org/10.1109/TAFFC.2023.3263907

Xiao, Y., Sharma, H., Kaur, S., Bergen-Cico, D., & Salekin, A. (2025). Human heterogeneity invariant stress sensing. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 9*(3). https://doi.org/10.1145/3749465

Zhao, Z., Pendiyala, K., Mortazavi, M., & Yan, N. (2025). *PULSE: Privileged knowledge transfer from rich to deployable sensors for embodied multi-sensory learning* (arXiv:2510.24058) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2510.24058

Zhou, H., Balakrishnan, S., & Lipton, Z. C. (2023). *Domain adaptation under missingness shift* (arXiv:2211.02093) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2211.02093

### References added 2026-09-22 (thesis-cited papers; DOIs checked in Crossref)

Albaladejo-González, M., Ruipérez-Valiente, J. A., & Gómez Mármol, F. (2023). Evaluating different configurations of machine learning models and their transfer learning capabilities for stress detection using heart rate. *Journal of Ambient Intelligence and Humanized Computing, 14*(8), 11011–11021. https://doi.org/10.1007/s12652-022-04365-z

Bent, B., Goldstein, B. A., Kibbe, W. A., & Dunn, J. P. (2020). Investigating sources of inaccuracy in wearable optical heart rate sensors. *npj Digital Medicine, 3*, Article 18. https://doi.org/10.1038/s41746-020-0226-6

Can, Y. S., Chalabianloo, N., Ekiz, D., & Ersoy, C. (2019). Continuous stress detection using wearable sensors in real life: Algorithmic programming contest case study. *Sensors, 19*(8), Article 1849. https://doi.org/10.3390/s19081849

Dahal, K., Bogue-Jimenez, B., & Doblas, A. (2023). Global stress detection framework combining a reduced set of HRV features and random forest model. *Sensors, 23*(11), Article 5220. https://doi.org/10.3390/s23115220

Gil-Martín, M., San-Segundo, R., Mateos, A., & Ferreiros-López, J. (2022). Human stress detection with wearable sensors using convolutional neural networks. *IEEE Aerospace and Electronic Systems Magazine, 37*(1), 60–70. https://doi.org/10.1109/MAES.2021.3115198

Gillinov, S., Etiwy, M., Wang, R., Blackburn, G., Phelan, D., Gillinov, A. M., Houghtaling, P., Javadikasgari, H., & Desai, M. Y. (2017). Variable accuracy of wearable heart rate monitors during aerobic exercise. *Medicine & Science in Sports & Exercise, 49*(8), 1697–1703. https://doi.org/10.1249/MSS.0000000000001284 [Abstract only]

Hosseini, E., Fang, R., Zhang, R., Rafatirad, S., & Homayoun, H. (2023). Emotion and stress recognition utilizing galvanic skin response and wearable technology: A real-time approach for mental health care. In *2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)* (pp. 1125–1131). IEEE. https://doi.org/10.1109/BIBM58861.2023.10386049

Li, J., & Washington, P. (2024). A comparison of personalized and generalized approaches to emotion recognition using consumer wearable devices: Machine learning study. *JMIR AI, 3*, Article e52171. https://doi.org/10.2196/52171

Mattern, E., Jackson, R. R., Doshmanziari, R., Dewitte, M., Varagnolo, D., & Knorn, S. (2023). Emotion recognition from physiological signals collected with a wrist device and emotional recall. *Bioengineering, 10*(11), Article 1308. https://doi.org/10.3390/bioengineering10111308

Naegelin, M., Weibel, R. P., Kerr, J. I., Schinazi, V. R., La Marca, R., von Wangenheim, F., Hoelscher, C., & Ferrario, A. (2023). An interpretable machine learning approach to multimodal stress detection in a simulated office environment. *Journal of Biomedical Informatics, 139*, Article 104299. https://doi.org/10.1016/j.jbi.2023.104299

Sharma, N., & Gedeon, T. (2012). Objective measures, sensors and computational techniques for stress recognition and classification: A survey. *Computer Methods and Programs in Biomedicine, 108*(3), 1287–1301. https://doi.org/10.1016/j.cmpb.2012.07.003 [Abstract only]

Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage, 180*, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061

AI disclosure: the searches, full-text retrieval and quotations were done by AI agents (Claude). The orchestrating agent re-checked every quote and figure in this file against the saved full texts, except those marked abstract-only. One agent error was caught and corrected: the Q1 notes misread Mishra et al.'s Table 4 chest-strap→E4 values. No human has yet checked the quotations against the sources.
