# Closable gaps: novelty the JBHI paper can add with existing data, in days

Date: 2026-09-23. Scope: gaps closable with the five E4 datasets already harmonised (`harmonized_windows_v2.csv`, raw signals in the thesis worktree) and existing scripts. Read: `CLAUDE.md`, `STATUS.md`, `docs/contribution_map.md` (§4, §7, §9), `docs/novelty_experiments.md`, `docs/literature/contribution-gap-scan.md`. Web scan 2026-09-23 (evidence level noted: FT = full text or HTML fetched, AO = abstract/snippet only).

Scoring: novelty 1–5 × feasibility 1–5. Effort in working days for one person.

## 0. Two things found during this scan that the paper must handle first

**(a) The matched-arousal novelty claim is overstated as written.** `paper/main.tex` line 33 says "Whether a detector trained on stressor labels has learned anything beyond arousal has not, to our knowledge, been tested", and `contribution_map.md` §1 says "no paper does the matched-arousal test". Kwon et al. (2026, *Healthcare* 14:2434; FT via PMC) already did an HR-matched version: "Within each subject, windows are binned into 5 bpm HR bins, and the majority class is randomly subsampled so that the two classes have identical HR distributions per bin." They also report an "HR-leakage index" and found that "matched non-cardiac signal remained strong only in WESAD". The repo cites Kwon, and `contribution-gap-scan.md` line 27 even calls iso-HR "the closest existing specificity control", but the manuscript does not. Their test is stress vs baseline, HR only, no non-stress arousal classes. Our novelty is narrower but real: a two-channel (HR + tonic EDA) index, **non-stress arousal classes** (Lego, UBFC control, hyperventilation, emotion clips, exercise), a fixed detector under LODO, and subject-clustered permutation inference. Reword to "extends the iso-HR test of Kwon et al. from stress-versus-baseline to a panel of non-stress arousal states". https://pmc.ncbi.nlm.nih.gov/articles/PMC13465799/

**(b) A quick probe (read-only, 2 minutes) exposed a trivial baseline the paper does not report.** A label-free, untrained arousal index (mean of per-subject z-scored `hr_mean` and `eda_tonic_mean`) scored directly as a stress score, stress vs baseline/rest/amusement, against the external (LODO) model's `p_ext` from `contribution_probes/arousal/scores.csv`:

| Dataset | Windows | Arousal index alone, AUROC | External model, AUROC |
|---|---|---|---|
| WESAD | 1042 | 0.945 | 0.965 |
| Campanella | 218 | 0.948 | 0.918 |
| UBFC-Phys | 315 | 0.840 | 0.906 |
| Stress-Predict | 2546 | **0.739** | 0.708 |
| PhysioNet | 1599 | 0.662 | 0.807 |

(Merge on window_id + dataset + subject; 294 duplicate keys in both tables, so treat to ±0.01. Same transductive whole-session z-scoring as the model.) Two features with no training match or beat the trained detector on three of five datasets. This is the strongest single piece of evidence for the arousal thesis and costs one table. PhysioNet is the one dataset where the model adds real signal beyond the index (+0.15); that residual is worth explaining (gap 3 below).

## 1. Ranked list

| Rank | Gap | Novelty | Feasibility | Score | Effort | Script to extend |
|---|---|---|---|---|---|---|
| 1 | Arousal-controlled evaluation kit: arousal-index baseline + matched-arousal AUROC + iso-HR reproduction, packaged as a reporting standard | 4 | 5 | 20 | 2–4 d | `matched_arousal_hardening.py` → new `arousal_controlled_eval.py` |
| 2 | Informative PPG missingness as a cross-dataset shortcut, plus quality-gated selective classification under LODO | 4 | 5 | 20 | 2–3 d | `threshold_transfer_probe.py`, `leave_one_dataset_out.py` (new flag) |
| 3 | Arousal + residual decomposition of the stress score (what the model adds beyond arousal, per dataset) | 3 | 5 | 15 | 1–2 d | `matched_arousal_probe.py` Part A |
| 4 | Frozen public PPG encoder (PaPaGei, Pulse-PPG) under LODO + matched arousal | 4 | 3 | 12 | 4–7 d | new `foundation_encoder_probe.py`, feeding `leave_one_dataset_out.py` |
| 5 | Benchmark artefact release (labels, frozen splits, panel, loader) deposited with the JBHI paper | 3 | 4 | 12 | 2–3 d (deposit) / 1–2 mo (standalone paper) | `generate_frozen_splits.py`, `harmonization_audit.py` |
| 6 | Construct validity against self-report (SAM arousal vs valence, per-subject response vs self-rated rise) | 2–3 | 4 | 10 | 1–2 d | `mild_stress_probe.py`, `self_report_labels.py` |

## 2. Details per gap

### Rank 1. Arousal-controlled evaluation kit (candidate 1, merged with part of 4)

- **Gap.** Kwon (2026) matched HR for stress vs baseline. Aydoğan & Villagra Povina (2026) and Bosch (2026) score exercise one confounder at a time. Liu & Ning (2026, arXiv 2609.22622, FT HTML) benchmark five EDA datasets with LOSO and LODO but "the evaluation focuses on binary stress classification" with no non-stress arousal classes. Nobody proposes a reusable protocol with (i) a label-free arousal baseline every paper should beat, (ii) within-arousal-bin AUROC against each non-stress class, (iii) subject-clustered inference.
- **Closest prior.** Kwon et al. 2026 iso-HR, code public (github.com/RURUGURU/isohr-wearable-stress, Zenodo): https://pmc.ncbi.nlm.nih.gov/articles/PMC13465799/ . Bosch 2026, 2×3 stress × {sit, walk, cycle}, n = 19, chest/Shimmer sensors, no arousal matching: "Tonic electrodermal activity showed a robust, additive response to both cognitive stress (r = 0.48) and physical exertion (r = 0.67), with no interaction" (FT HTML) https://arxiv.org/html/2605.15756 . Arxiv 2604.12671 (physical vs psychological stress with cortisol; AO) states that models "tend to conflate physiologically distinct states that share similar arousal levels" but tests one study with cortisol, not a detector across datasets.
- **What to add.** (a) The arousal-index baseline table from §0(b), per dataset, with subject bootstrap CIs, for both stress-vs-rest and stress-vs-panel. (b) Kwon's iso-HR metric run on our data (HR-only, 5 bpm bins, within subject) next to our two-channel index, to show the iso-HR test passes against baseline (reference cell 0.58) yet fails against seated non-stress arousal. (c) One function `arousal_controlled_auroc(scores, index, labels, subjects)` with a README-level spec, so others can adopt it. This turns rank 1 of the contribution map into a citable method, not only a finding.
- **Feasibility.** Everything is in `scores.csv` and the feature table; no retraining except iso-HR (reuse existing splits).
- **Risk.** Low. Main risk is the circularity already noted (index built from model features); the feature-disjoint model in `hardening/partA_disjoint.csv` answers it.

### Rank 2. Informative missingness and quality-gated abstention (candidate 5, reframed)

- **New finding from this scan.** Cardiac coverage alone predicts the label, with opposite signs across datasets. AUROC of (1 − `cardiac_coverage`) for stress vs everything else: WESAD 0.81, UBFC-Phys 0.74, Stress-Predict 0.49, Campanella 0.38, PhysioNet 0.34. Fraction of windows with no HR: WESAD 0.63 stress vs 0.16 non-stress; UBFC 0.39 vs 0.15; PhysioNet 0.09 vs 0.22 (exercise loses PPG). XGBoost's missing-value branch can learn "PPG lost = stress" from speech datasets and "PPG lost = not stress" from exercise. This is the same mechanism already proven for temperature on UBFC-Phys (`threshold_transfer_probe.py`) but for the core cardiac channel, and it bears on the "37% usable PPG" fact.
- **Closest prior.** Farahani, Cao & Rahmani (2026, arXiv 2608.18397, FT HTML): conformal coupling gate, WESAD + Stress-Predict, LOSO only, "No dedicated PPG (BVP) signal-quality index is applied", false positives 29→27 and 94→92 (n.s.). Kwon 2026 HR-leakage index (cardiac level, not missingness). No paper found that treats PPG dropout as a label shortcut in wearable stress. https://arxiv.org/html/2608.18397
- **What to add.** (a) LODO with three missingness treatments: native NaN (current), NaN + explicit indicator dropped / cardiac features imputed from subject median, and cardiac-available windows only. Report external BA change per target. (b) Coverage–accuracy curve under LODO with abstention gated by `cardiac_coverage` (and by motion, `acc_mag_std`), using existing out-of-sample scores. (c) Specificity panel re-scored under (a) to see whether exercise false-stress depends on missingness.
- **Effort.** 2–3 days; one opt-in flag in `leave_one_dataset_out.py` (keep defaults unchanged, as for the novelty flags).
- **Risk.** Medium on direction: the effect may be small because EDA dominates. Either result is reportable: a large effect is a new shortcut finding; a null closes a reviewer question ("is the model detecting speech-induced PPG loss?").

### Rank 3. Arousal + residual decomposition (candidate 4)

- **Gap.** Kwon's HR-leakage index is the only decomposition, cardiac only. Nobody reports, per dataset, how much of a stress score is explained by a label-free arousal index and what the residual carries.
- **What to add.** Regress the logit of `p_ext` on a spline of the arousal index per dataset (subject random intercept or subject-demeaned); report R² and the residual's AUROC for stress vs rest and vs each panel class. §0(b) predicts: residual near 0.5 on WESAD/Campanella/Stress-Predict, sizeable on PhysioNet. Then attribute PhysioNet's residual with SHAP on the residual (HRV? SCR count?). This answers "what does the model detect that arousal does not" in one figure.
- **Effort.** 1–2 days on existing scores. **Risk.** Low; it is a reframing of rank 1, so present it as part of the same section, not a separate contribution.

### Rank 4. Frozen public PPG encoders (candidate 3)

- **Gap still open.** Foundation-model stress results remain WESAD-only with weak protocols: Pulse-PPG (UbiComp 2025) on WESAD; PaPaGei (ICLR 2025) on WESAD; Geenjaar et al. (arXiv 2606.07365, June 2026, FT HTML) multimodal-guided PPG FM, WESAD "cross-subject linear probing protocol (5-fold cross-validation)", AUC 0.94 vs Pulse-PPG 0.92, no LOSO. UME EDA FM (Alchieri et al., arXiv 2603.16878, FT HTML): "no statistical difference between using our UME and the EDA-specific handcrafted features", and weights only "following peer review", so an EDA encoder cannot be run yet. No FM has been scored on PhysioNet (Hongn), Stress-Predict, UBFC-Phys or Campanella under LODO, and none under matched arousal.
- **Weights available.** PaPaGei: https://github.com/Nokia-Bell-Labs/papagei-foundation-model (weights on Zenodo; 125 Hz, 10 s segments; 5M parameters). Pulse-PPG: https://github.com/maxxu05/pulseppg (weights Zenodo 10.5281/zenodo.17270930).
- **Feasibility.** Medium. Raw E4 BVP at 64 Hz is in the thesis worktree for all five datasets; resample to 125 Hz, cut each 60 s window into six 10 s segments, mean-pool embeddings, then logistic regression / XGBoost under the frozen LODO splits. About 11.4k windows × 6 segments is minutes on the local MPS backend (torch 2.14, MPS available). Re-windowing must reproduce `extract_features.py` boundaries (60 s, 30 s step) so window_ids align.
- **Risk.** High uncertainty, both outcomes useful. Specific trap: raw-BVP encoders see motion artefact, and PPG loss is itself label-informative on WESAD/UBFC (rank 2), so an FM "win" may be a speech-motion shortcut. Always report FM results with the matched-arousal and missingness controls. Expected: no gain on PhysioNet/Stress-Predict, which strengthens the stressor-potency ceiling.

### Rank 5. Benchmark release (candidate 2)

- **Status of competition.** Liu & Ning (arXiv 2609.22622, posted 2026-09-18, FT HTML): 26-dataset EDA survey, benchmark on Neuro, WESAD, MAUS, AffectiveROAD, EmpaticaE4Stress (95 subjects), LOSO + LODO, "The code will be made publicly available upon acceptance", no fixed split or harmonised-label release stated, no specificity panel. Kwon 2026 released code but for three corpora and no label audit. Our five datasets overlap only on WESAD, so the niche (audited protocol-stage labels, frozen splits, specificity panel, E4 all channels) is still open, but the window is closing.
- **Closable in days.** Deposit label files, split files and probe scripts on Zenodo with a DOI, cite in the JBHI data-availability statement. The standalone Scientific Data / D&B paper is months, not days.
- **Risk.** Low scientific risk; licence check needed per dataset (release labels and code, not raw data).

### Rank 6. Self-report construct validity (candidate 6)

- **What is left.** Detectability-as-trait and mild-stress intensity are already negative (`mild_stress_probe.py`, contribution map §3). Coverage of self-report is thin: stress delta in WESAD (75% of windows) and PhysioNet (30%), SAM arousal/valence in WESAD and EPM-E4 only.
- **Quick probe result (not a finding yet).** Spearman of `p_ext` with clip-level SAM arousal on EPM-E4 is 0.08 (valence −0.11, n = 855 windows, not clustered by subject). If the detector detected arousal, it should track self-rated arousal across emotion clips; it barely does. This could cut against the arousal thesis, or reflect that SAM is subjective arousal while the model sees autonomic arousal, and that E4 clip responses are small. It needs a subject-level analysis before it goes in the paper. WESAD's 0.57 is confounded by condition.
- **Closest prior.** No 2025–2026 paper found that tests a stress detector's score against self-rated arousal vs valence across non-stress emotion blocks; searches returned only dataset descriptors (Hongn 2025 *Sci Data*: https://www.nature.com/articles/s41597-025-04845-9).
- **Risk.** High that the result is ambiguous; worth 1 day because a reviewer will ask whether "arousal" means physiological or subjective arousal. Frame the paper's claim as autonomic arousal and report this as a limitation or a within-subject correlation.

## 3. Candidates rejected or already closed

- Warm-up / time in session: done (contribution map §2, §9).
- Few-shot, DA, per-user thresholds, detectability trait: done, negative.
- Recovery-aware labels: payoff ≤ +0.02 per the error budget; sensitivity analysis only.
- EDA foundation model: blocked until UME weights are released.
- Counterbalanced UTSA study: new data, out of scope for "days".

## 4. Suggested order for one week

1. Day 1: §0(a) wording fix; §0(b) table with CIs (rank 1a, rank 3).
2. Days 2–3: rank 2 missingness flag + coverage–accuracy curve.
3. Day 4: iso-HR reproduction and packaged `arousal_controlled_auroc` (rank 1b–c).
4. Days 5–7 (optional): PaPaGei row under LODO with matched-arousal and missingness controls (rank 4); Zenodo deposit (rank 5) in parallel.

## Sources

- Kwon et al. 2026, iso-HR: https://pmc.ncbi.nlm.nih.gov/articles/PMC13465799/ (FT)
- Bosch 2026, stress × exertion: https://arxiv.org/html/2605.15756 (FT)
- Liu & Ning 2026, EDA survey and benchmark: https://arxiv.org/abs/2609.22622 (FT HTML)
- Farahani, Cao & Rahmani 2026, ICCM: https://arxiv.org/html/2608.18397 (FT HTML)
- Geenjaar et al. 2026, multimodal-guided PPG FM: https://arxiv.org/html/2606.07365v1 (FT HTML)
- Alchieri et al. 2026, UME EDA FM: https://arxiv.org/html/2603.16878 (FT HTML)
- Physical vs psychological stress with cortisol: https://arxiv.org/html/2604.12671v2 (AO)
- PaPaGei: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
- Pulse-PPG: https://github.com/maxxu05/pulseppg ; https://arxiv.org/html/2502.01108
- Akkaya 2026, calibration not architecture: https://link.springer.com/article/10.1186/s12911-026-03842-1 (AO)
- Hongn et al. 2025 dataset: https://www.nature.com/articles/s41597-025-04845-9 (AO)
