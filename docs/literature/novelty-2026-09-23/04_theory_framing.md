# 04. Theory framing: from "negative result" to a measurement-validity contribution

Written 2026-09-23 for the JBHI revision (`paper/main.tex`, branch `jbhi-revision`). Read-only on the repo. Full texts were extracted to ``sources/novelty-2026-09-23-fulltexts/` (git-ignored, jbhi-revision checkout)` (Siegel 2018, Kreibig 2010, Quigley & Barrett 2014, Geirhos 2020 arXiv, Cacioppo/Tassinary/Berntson Handbook 3rd ed. ch. 1, Epel 2018 via Europe PMC, V3 via PubMed Central, Smets 2018 and Sharma 2026 via Europe PMC). Quote status is marked on each entry: **[full text]** means the quote was read in the fetched full text; **[abstract]** means only the abstract was read; **[memory, unverified]** means not fetched and must be checked before citing.

---

## 0. The one-paragraph reframe

The paper now reads as "detectors learn arousal, not stress", which a reviewer can file as a negative result. Psychophysiology already has a name for this: an **inference problem**. Cacioppo and Tassinary showed that a physiological response can be used to infer a psychological state only when the mapping is close to one-to-one within the context of use. Otherwise P(physiology | state), which is what a lab protocol measures, differs from P(state | physiology), which is what a deployed detector claims. Every stress dataset estimates the first. Our specificity panel and matched-arousal test are, as far as we can find, **the first direct estimate of the second for wrist stress detectors, and of how far it falls short of the first.** The mapping is many-to-one: evaluative stress, non-evaluative tasks, hyperventilation, manual work and anger all reach the same wrist state. Exercise is the only state that reliably maps elsewhere. In V3 terms (Goldsack 2020), the field has done analytical validation (wrist HR against ECG) and convergent validation (stressor recall) but no **discriminant** clinical validation. In Geirhos's terms, the arousal shortcut is **shared by every training distribution**, so leave-one-dataset-out cannot expose it; only a construct-shift test can. Framed this way the contribution is positive and reusable: (i) an operational test of psychophysiological specificity for any wearable "state" detector, (ii) a quantified mapping (which states collide and which separate), and (iii) a reporting standard. The emotion-specificity debate (Kreibig vs Siegel/Barrett) is the precedent that makes the null *expected* and therefore interpretable rather than disappointing.

Proposed one-sentence thesis for the abstract or introduction:
> "Laboratory stress datasets estimate how physiology responds to a stressor; a deployed detector claims the reverse, and the two coincide only when the psychophysiological mapping is one-to-one. We provide the first direct test of that assumption for wrist stress detection."

---

## 1. Framework by framework

### 1.1 Cacioppo & Tassinary: psychophysiological inference (one-to-many, many-to-one)

**Sources.** Cacioppo & Tassinary (1990), *Am Psychol* 45:16–28 [not fetched; the book chapter restates it]. Cacioppo, Tassinary & Berntson (2007), *Handbook of Psychophysiology* 3rd ed., ch. 1 **[full text]**.

**What it says (verbatim, Handbook ch. 1):**
- "A many-to-one relation, meaning that two or more psychological elements are associated with the same physiological element."
- "In the idealized case, an outcome is defined as a many-to-one, situation-specific (context-dependent) relationship … Note that this if [sic] often the first attribute of a psychophysiological relationship that is established in laboratory practice."
- "In its idealized form, a psychophysiological marker is defined as a one-to-one, situation-specific (i.e., context-dependent) relationship … Thus, markers are characterized by limited ranges of validity."
- "The idealized invariant relationship refers to an isomorphic (one-to-one), context-independent (cross-situational) association."
- "Unfortunately, invariant relationships are often assumed rather than formally established, and, as we have argued, such an approach leads to erroneous psychophysiological inferences and vacuous theoretical advances."
- On the direction of inference: research with psychological factors as the independent variable estimates P(physiology | psychology), and research with physiology as the independent variable estimates P(psychology | physiology). "These conditional probabilities are equal only when the relationship between [them] is 1:1 (Cacioppo & Tassinary, 1990)." (The Greek symbols were lost in text extraction; the meaning is clear from context.)

**Why it helps.** This is the backbone of the reframe. It gives the paper a four-cell taxonomy (outcome, marker, concomitant, invariant) for classifying what a wrist stress detector is. Our results place it precisely:
- Stressor-versus-rest classification within dataset establishes an **outcome** relation. That is the first step, and the one the field stops at.
- Small leave-one-dataset-out cost shows the relation generalises across lab contexts, which is the "context-independent" axis. But only for stressor-versus-rest.
- The specificity panel and the matched-arousal test show the relation is **many-to-one** (several psychological states map to one wrist state). So wrist "stress" is at best a **concomitant**, not a marker or an invariant.
- Exercise and fear clips (once trained) are the only places where many-to-one starts to separate.

This turns "the detector fails on Lego" into "we classified the psychophysiological relation, which the field had assumed rather than tested". The Handbook sentence on invariants "assumed rather than formally established" is almost a description of the stress-detection literature.

**Proposed sentences.**
> "In Cacioppo and Tassinary's terms, a stressor-versus-rest classifier establishes an outcome relation, P(signal | stressor); deployment requires the inverse, P(stress | signal), and the two are equal only when the mapping is one-to-one. The specificity panel and matched-arousal test estimate the inverse directly and find the mapping many-to-one: evaluative stress, the same tasks without evaluation, hyperventilation, manual work and anger clips occupy the same region of wrist feature space."

> "Wrist stress detection has treated an invariant psychophysiological relation as an assumption; we test it."

---

### 1.2 The autonomic specificity debate: Kreibig (2010), Siegel et al. (2018), Quigley & Barrett (2014)

**Kreibig (2010), *Biol Psychol* 84:394–421 [full text].**
- Abstract: "Positions on the degree of specificity of ANS activation in emotion, however, greatly diverge, ranging from undifferentiated arousal, over acknowledgment of strong response idiosyncrasies, to highly specific predictions of autonomic response patterns for certain emotions. A review of 134 publications … suggests considerable ANS response specificity in emotion when considering subtypes of distinct emotions."
- Anger: "the anger response is characterized by α- and β-adrenergically mediated cardiovascular effects: increased HR, increased SBP and DBP, and increased TPR …"
- Fear: "Various of the studies investigating fear report increased HR … or increased electrodermal activity … indicating a general arousal response." And: "Only a few studies report HR deceleration in the context of laboratory fear elicitation: decreased HR along with signs of increased vasoconstriction (decreased FPA …) has been found in response to a film clip eliciting fear of falling."
- Anxiety: "… decreased FPA [finger pulse amplitude] … decreased FT [finger temperature] …"
- Happiness: "The autonomic response pattern of happiness is characterized by increased cardiac activity due to vagal withdrawal, vasodilation, increased electrodermal activity, and increased respiratory activity."
- On measures: "Due to a limited number of studies considered, a restricted range of physiological variables (only cardiovascular and electrodermal, but no respiratory measures), and the univariate nature of the meta-analytic approach, such results give only an imperfect answer to the question of autonomic patterning in emotion."

**Siegel et al. (2018), *Psychol Bull* 144:343–393 [full text].**
- Abstract: "We present a meta-analysis of 202 studies … We found increases in mean effect size for 59.4% of ANS variables across emotion categories, but the pattern of effect sizes did not clearly distinguish 1 emotion category from another. We also observed significant variation within emotion categories; heterogeneity accounted for a moderate to substantial percentage (i.e., I² ≥ 30%) of variability in 54% of these effect sizes."
- "These findings suggest that autonomic nervous system changes during emotion are less like a bodily fingerprint and more like a population of variable, context sensitive instances."
- On classifiers: "Pattern classification uses algorithms to develop patterns that distinguish categories from one another, but correct classification of an instance requires only that it is statistically closer to one pattern than another." and "However, pattern classification, even when successful, does not reveal a single body state for an emotion category."

**Quigley & Barrett (2014), *Biol Psychol* 98:82–94 [full text].**
- "The CAT predicts that there will be ANS variation across instances within an emotion category and similarities in instances across categories."
- Reanalysing classifier studies: "The largest discriminant function accounted for 62.5% of the explained variance and distinguished emotions that varied in subjective arousal." A second study: "The largest discriminant function accounted for 58.3% of the variance and distinguished emotions that that [sic] varied in subjective arousal. This interpretation was bolstered by large factor loadings for mean skin conductance level … that were positively related to activation …"
- "… both of these studies found that discriminant functions were better able to predict emotion categories using self-reports than ANS outcomes."

**Why it helps.** Two things.
1. **Precedent and prediction.** After decades of emotion research, the best-supported reading is that ANS measures mainly carry arousal, and that category specificity, where it exists, needs more channels (vascular resistance, respiration, pre-ejection period) than a wrist provides. In Quigley & Barrett's reanalysis the *first* discriminant function of emotion classifiers was arousal, with skin conductance loading on it. So our result is what theory predicts for a wrist with HR, HRV and EDA. That makes it a confirmatory test of a stated hypothesis, not an unlucky model. Siegel's point that a classifier being "closer to one pattern than another" is not a fingerprint answers the reviewer who says "but your classifier gets 0.88 on WESAD".
2. **Where specificity should appear, and our data agree.** Kreibig's reliable differentiators are vascular (TPR, finger pulse amplitude, finger temperature) and, for fear, a cardiac deceleration or immobilisation pattern. Our only separable non-exercise state is fear clips, and the feature-disjoint rerun puts that residual in the cardiac features (0.79 with an EDA-only index and a cardiac-only model). That matches Kreibig's cardiac account of film-induced fear better than an evaluative-EDA account. It is a theory-consistent positive finding the paper already has but does not frame.

Caveat for the text: Kreibig and Siegel reach different conclusions on specificity. The paper does not need to take sides. It only needs the point both accept: with cardiac rate and electrodermal measures alone, the dominant shared dimension is arousal, and category differences depend on vascular and respiratory channels the wrist lacks or measures poorly.

**Proposed sentences.**
> "Our result is the wearable counterpart of a long-standing finding in emotion psychophysiology: with heart rate and electrodermal measures, the first discriminant dimension across emotional states is arousal [Quigley & Barrett 2014], and a meta-analysis of 202 studies found that autonomic effect sizes 'did not clearly distinguish 1 emotion category from another' [Siegel et al. 2018]. Specificity, where it has been reported, rests on vascular and respiratory measures [Kreibig 2010] that a wrist either lacks or records poorly."

> "Consistent with Kreibig's review, which describes film-induced fear with cardiac deceleration and vasoconstriction, the only emotional state our detector separates at matched arousal is fear, and the residual sits in the cardiac features."

---

### 1.3 Stress-measurement taxonomy: Epel et al. (2018); Dickerson & Kemeny (2004)

**Epel et al. (2018), *Front Neuroendocrinol*, "More than a feeling" [full text, PMC6345505].**
- "Acute stress is often measured by examining short term changes in physiology with the general principle being that events or psychological states that increase arousal can be characterized as 'stress.' This is an over-simplification … what we consider the two primary physiological 'stress systems' – the sympathetic adrenal medullary and the hypothalamic pituitary adrenal cortical axes – are associated with many psychological states including positive and negative emotion, cognitive effort, and approach and avoidant motivational states. Thus, examining acute psychological or physiologic responses and assuming that changes from a baseline or resting state indicate that a person is feeling 'stressed' is problematic."
- "The most important distinction identified in the Typology is between stressor exposure ('stressors') and psychological responses to the stressor."
- "All of these different measures of stressor exposures, perceptions of stress, and psychological and biological stress responses are at best loosely related (Mauss et al., 2005)."

**Dickerson & Kemeny (2004), *Psychol Bull* 130:355 [abstract].** "… motivated performance tasks elicited cortisol responses if they were uncontrollable or characterized by social-evaluative threat … These findings … contradict the belief that cortisol is responsive to all types of stressors."

**Why it helps.** Epel's paragraph is close to the paper's thesis, written by stress scientists for population science. It gives the paper an authoritative source for the three-construct split already in `docs/what_wrist_detectors_detect.md` (exposure, autonomic response, appraisal). It also lets the paper say that the ML field's operational definition ("stress = rise from baseline") is the one stress science calls "problematic". Dickerson & Kemeny give the *positive* definition of what is missing: social-evaluative threat plus uncontrollability is the stressor-specific ingredient, and it is defined on the HPA axis, which a wrist does not see. The UBFC-Phys contrast (same tasks, evaluation removed) is exactly a Dickerson–Kemeny manipulation. Our null there is therefore a test of whether the evaluative ingredient leaves a *sympathetic* trace on the wrist.

**Proposed sentences.**
> "Stress science distinguishes stressor exposure, the physiological response and the appraisal, and notes that treating 'changes from a baseline or resting state' as evidence that a person is 'stressed' is 'problematic', because the sympathetic system responds to 'positive and negative emotion, cognitive effort, and approach and avoidant motivational states' [Epel et al. 2018]. Wearable stress datasets label exposure, and wearable detectors read the response; our results measure the gap between the two."

> "The UBFC-Phys contrast removes the ingredient that Dickerson and Kemeny identified as specific to psychological stress, social-evaluative threat, while keeping the task. At matched arousal the wrist detector does not register its removal."

---

### 1.4 Construct, criterion and discriminant validity; V3 and DiMe

**Goldsack et al. (2020), *npj Digit Med* 3:55, V3 [full text, PMC7156507].**
- "Verification is a bench evaluation that demonstrates that sensor technologies are capturing data with a minimum defined accuracy and precision when compared against a ground-truth reference standard …"
- "Analytical validation involves evaluation of a BioMeT for generating physiological- and behavioral metrics." "During the process of analytical validation, the metric produced by the algorithm must be evaluated against an appropriate reference standard."
- "Clinical validation is the process that evaluates whether the BioMeT acceptably identifies, measures, or predicts a meaningful clinical, biological, physical, functional state, or experience in the specified context of use."
- "… if no reference standard exists, then the need for evidence of clinical validity and utility increases." (paraphrase of the fetched sentence ", no reference standard exists), then the need for evidence of clinical validity and utility increases.")

**V3+ (2024), *npj Digit Med*** adds usability validation [search result only; not read].

**Classical construct validity:** Cronbach & Meehl (1955), Campbell & Fiske (1959) multitrait–multimethod convergent and discriminant validity [memory, unverified; standard references, check before citing].

**Why it helps.** V3 gives JBHI reviewers a familiar audit structure, and the paper fills one of its cells. Mapping:

| V3 layer | Question | What the paper has | Status |
|---|---|---|---|
| Verification | Does the E4 capture BVP/EDA/TEMP/ACC? | Vendor spec; cardiac coverage (only 37% of WESAD stress windows keep usable PPG) | Partial, reported |
| Analytical validation | Do wrist metrics match a reference? | HR vs chest ECG 2.4 bpm, r = 0.94; RMSSD r = 0.53, +34 ms; wrist vs palm EDA r ≈ 0.3 (Milstein & Gordon 2020) | Done for HR and HRV; EDA from literature |
| Clinical validation, convergent | Does the output rise under stressors? | Within-dataset and LODO BA 0.70–0.92; transfer cost 0.015–0.05 | Done |
| Clinical validation, **discriminant** | Does it stay low under non-stress arousal? | Specificity panel (false-stress 0.24–0.69); matched-arousal AUROC 0.36–0.58 | **New in this paper** |
| Clinical validation, criterion (experience) | Does it track the felt state? | Self-rated intensity does not predict misses (mild-stress probe); SAM in WESAD and EPM-E4 unused so far | Partial; see Analysis A |
| Context of use | Which states will the device meet? | Laboratory only; exercise-aware vs not | Stated as a limitation |

This recasts the paper as filling the **discriminant-validity cell of clinical validation**, which the wearable-stress literature has left empty. It also lets the paper state a context of use precisely: "stressor-versus-rest within a seated laboratory protocol". That is where the detector is valid. "Stress in daily life" is not.

**Proposed sentences.**
> "Under the V3 framework [Goldsack et al. 2020], wrist stress detectors have analytical validation (wrist heart rate against ECG) and convergent clinical validation (recall of stressor periods), but no discriminant validation: evidence that the output stays low under states that share the physiology but are not stress. The specificity panel supplies that evidence, and it is negative for every seated non-evaluative state we could test."

> "We therefore report the detector's valid context of use as stressor-versus-rest discrimination in seated laboratory protocols, not the detection of stress."

---

### 1.5 Shortcut learning and confounds in ML for health: Geirhos et al. (2020)

**Geirhos et al. (2020), *Nat Mach Intell* 2:665 [full text, arXiv 2004.07780].**
- "Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios."
- The pneumonia example: a classifier "had unexpectedly learned to identify particular hospital systems with near-perfect accuracy (e.g. by detecting a hospital-specific metal token on the scan …). Together with the hospital's pneumonia prevalence rate it was able to achieve a reasonably good prediction—without learning much about pneumonia at all."
- "… the difference between intended and shortcut solution is not something a neural network can possibly infer from the training data."
- "We believe that good o.o.d. tests should fullfill [sic] at least the following three conditions: First, … a clear distribution shift … Second, it should have a well-defined intended solution. … Third, a good o.o.d. test is a test where the majority of current models struggle."
- Morgan's Canon, quoted: "In no case is an animal activity to be interpreted in terms of higher psychological processes if it can be fairly interpreted in terms of processes which stand lower on the scale of psychological evolution and development."

**Why it helps.** This adds the most novelty for an ML-facing venue, because our case **inverts the usual shortcut story**:
- The usual story: a shortcut is exposed by a dataset shift (a new hospital). Leave-one-dataset-out is the standard remedy.
- Our case: the shortcut (arousal) is present in **every** training dataset, because every protocol contrasts an arousing stressor with a calm rest. It therefore *survives* leave-one-dataset-out (cost 0.015–0.05). A small LODO cost is **evidence for the shortcut, not against it.** Only a test that shifts the **construct** while holding the shortcut feature fixed can expose it. The matched-arousal test does exactly that, and meets all three of Geirhos's conditions for a good o.o.d. test: a clear shift (new states), a well-defined intended answer (these states are not stress), and a test where the model struggles.
- Morgan's Canon gives the paper a principled default: do not read "stress" into a detector whose behaviour is fully explained by "arousal". The matched-arousal test is the formal check of that canon.
- Other confounds in the paper fit the same frame: time in session (a time-only classifier beats physiology on PhysioNet and Stress-Predict), sensor warm-up, and the missing-temperature threshold shift (a dataset-specific shortcut, which LODO *does* expose). This gives a useful two-way split: **dataset-specific shortcuts** (temperature, time, label noise) are exposed by LODO; **construct-shared shortcuts** (arousal) need a specificity panel.

**Proposed sentences.**
> "Shortcut learning is usually exposed by a shift in data source [Geirhos et al. 2020]. The arousal shortcut in stress detection is different: every stress protocol contrasts an arousing stressor with a calm rest, so the shortcut is shared by all training sets and survives leave-one-dataset-out evaluation. A small cross-dataset transfer cost is therefore consistent with the shortcut rather than evidence against it. Exposing it needs a shift in construct at fixed arousal, which is what the matched-arousal test provides."

> "We distinguish dataset-specific shortcuts (missing temperature, time in session), which leave-one-dataset-out exposes, from construct-shared shortcuts, which only a specificity panel can."

---

### 1.6 Challenge versus threat (Blascovich & Tomaka; Seery) and what a wrist can see

**Seery (2011), *Neurosci Biobehav Rev* 35:1603 [abstract].** "According to the biopsychosocial model of challenge and threat, evaluations of personal resources and situational demands determine to what extent individuals experience a relatively positive (challenge) versus negative (threat) psychological state … Because challenge and threat reliably result in distinct patterns of physiological changes, assessing cardiovascular responses in particular can provide valuable insight into underlying psychological processes."

**Uphill et al. (2019), *Front Psychol* 10:1255 [full text via WebFetch summary; quote as returned]:** "Challenge is characterized by relatively greater cardiac reactivity (increased CO) and a decrease in TPR. In contrast threat is characterized by no change or an increase in TPR and no change or a small increase in CO."

**Porter & Goolkasian (2019), *Front Psychol* 10:967 [full text]:** "… both challenge and threat appraisals showed increased heart rate, which indicates SAM activation; however, challenge appraisals decreased total peripheral resistance and had no effect on blood pressure, while threat appraisals increased total peripheral resistance and blood pressure (Tomaka et al. …)."

**Epel et al. (2018) [full text]:** "In this model, both challenge and threat states occur during acute 'stressful' situations; however, the states differ in their antecedent appraisal processes and subsequent downstream cardiovascular reactivity."

**Why it helps.** This is the strongest mechanistic account of *why* the wrist cannot separate evaluative stress from engagement. In the biopsychosocial model, **heart rate rises in both challenge and threat**. The distinguishing signals are cardiac output and total peripheral resistance (usually from impedance cardiography plus continuous blood pressure) and pre-ejection period. The E4 has none of these. So a wrist that reads HR, HRV and EDA is, by this theory, blind to the appraisal dimension that separates distress from engaged effort. A Lego task under countdown, or arithmetic without an evaluator, should look like challenge. A TSST should look more like threat. Both produce the HR and EDA rise the detector reads.

What a wrist *could* see, with caveats (Section 2, Analysis B):
- **Pulse amplitude** of the PPG (a local vasoconstriction proxy; Kreibig lists decreased finger pulse amplitude under anxiety). Wrist PPG amplitude is dominated by contact pressure and motion, and wrist vascular beds differ from the finger. At best it proxies peripheral constriction, not TPR.
- **Pulse arrival time** needs an ECG R-peak. That is impossible on the E4 alone, but possible on **WESAD**, which has synchronised chest ECG.
- Cardiac output and PEP: not available.

No published study we could find estimates challenge versus threat from a wrist PPG (two searches, no hits). This gap is itself citable as a limit of wrist sensing, and a cheap proxy test (Analysis B) is novel.

**Proposed sentences.**
> "The biopsychosocial model of challenge and threat predicts our null. Heart rate rises under both challenge and threat; what separates them is cardiac output relative to total peripheral resistance [Seery 2011; Uphill et al. 2019], which a wrist band does not measure. A detector built on heart rate, heart-rate variability and electrodermal activity should therefore respond to engagement whether or not it is appraised as a threat, which is what the specificity panel shows."

> "This locates the missing information: not in the classifier, but in the vascular channel of the sympathetic response."

---

### 1.7 Allostasis: arousal as metabolic mobilisation, not distress

**Epel et al. (2018) [full text]:** "… changes in psychological and physiologic functioning support behavior, and in situations in which the stressor is active, recruitment of metabolic resources are often necessary to meet the situational demands."

**Quigley & Barrett (2014) [full text]:** ANS changes are part of the somatovisceral state that is "combined with sensory information … and with stored conceptual representations … to instantiate a current mental state". That is, the same autonomic change becomes different experiences depending on context.

**McEwen (1998), allostasis and allostatic load** [memory; McEwen 2002 in Europe PMC is a related review with no abstract; verify before citing].

**Why it helps.** Allostasis explains the *function* of what the wrist sees: anticipatory mobilisation of energy for a demand. Stressors, exercise, effortful Lego, hyperventilation and anger all need mobilisation, so all look alike. This framing reads as physiology, not failure. It also points to a positive use: a validated **wrist mobilisation or load index** with known specificity is a legitimate digital measure. It is just not a stress measure. This also matters for the allostatic-load use case: cumulative arousal burden may be the health-relevant quantity anyway, and the paper can end on it without overclaiming.

**Proposed sentence.**
> "Read through allostasis, the detector measures anticipatory metabolic mobilisation, which stressors share with exercise, effortful manual work and strong emotion; its outputs are better reported as a mobilisation index with a published specificity profile than as stress."

---

### 1.8 Ecological and real-world validity critiques

**Smets et al. (2018), *npj Digit Med* 1:67, SWEET, 1,002 subjects [full text].**
- "Identifying the concept of stress in ambulant conditions is challenging as the gold standard is based on self-reports …"
- "This illustrates that physical activity is associated with changes in physiology, highlighting the challenge of differentiating physiological changes caused by physical activity from those caused by stress."
- "Further research should compare the link between physiological signals and self-reported stress-responses on the one hand and self-reported pleasure, arousal and control levels based on the SAM on the other hand to better differentiate stress and arousal levels."

**Sharma et al. (2026), *JMIR*, scoping review of 34 naturalistic studies [full text]:** "… there are no obvious ground truth labels, unlike in controlled environments where the stressor and stress-inducing periods are known with high levels of certainty." It proposes a model-card reporting framework.

**Can, Arnrich & Ersoy (2019), *J Biomed Inform* [abstract]:** "Although there are a number of works related to stress detection in controlled laboratory conditions, the number of studies examining stress detection in daily life is limited."

**Campbell & Ehlert (2012), *Psychoneuroendocrinology* [abstract]:** in 49 TSST studies, "Significant correlations between cortisol responses and perceived emotional stress variables were found in approximately 25% of the studies." (Cortisol, not ANS, so cite only for response–appraisal dissociation.)

**Close recent competitors (arXiv, [abstract]):** Bosch (2026, arXiv 2605.15756): 19 participants, n-back stress × sitting/walking/cycling, chest and trapezius sensors, exercise only. Kaya et al. (2026, arXiv 2604.12671): 6 participants, TSST vs cycling with cortisol; "Wearable-only classification achieved 77.8% accuracy but struggled distinguishing psychological stress." Both are exercise-only and within-study. Neither scores a fixed detector on a panel of seated arousal states or matches arousal. This supports the novelty claim.

**Why it helps.** The ecological literature assumes the lab is the clean case and the field adds noise. Our result is sharper: **the lab ground truth is already ambiguous about the construct**, so field failure is predictable from lab data. The specificity panel is a laboratory proxy for the field's confusion states, and it can be run before anyone collects ambulatory data. Smets et al. explicitly called for separating stress from arousal with SAM ratings; Analysis A does that on public data. The PPV illustration below turns this into a deployment number.

**Proposed sentences.**
> "The gap between laboratory and daily-life stress detection is usually attributed to noisier labels and sensors in the field [Can et al. 2019; Smets et al. 2018]. Our results suggest that a larger part is predictable in the laboratory: a detector that cannot separate stress from non-evaluative arousal will fire on the many arousing non-stress episodes of daily life, whatever the label quality."

> "Smets et al. called for separating stress from arousal with self-assessment ratings; the specificity panel and the matched-arousal test are a controlled version of that separation."

---

## 2. Theory-driven analyses the five E4 datasets can support

All three reuse existing features, splits and the LODO model. None needs new data.

### Analysis A (cheapest, strongest): felt arousal versus valence, and physiological versus felt arousal

**Theory.** Quigley & Barrett (first discriminant function = arousal), Siegel (no fingerprints), Smets (call to separate stress from arousal via SAM), Epel (stress systems respond to positive and negative emotion).

**Data in hand.** SAM valence and arousal are in `harmonized_windows_v2.csv` for EPM-E4 (100% of windows) and WESAD (75%). EPM-E4 class means: Anger valence 1.2 / arousal 7.0; Sadness 1.3 / 6.7; Happiness 7.4 / 6.5; Fear 3.3 / 4.7. WESAD: stress 4.5 / 6.9; amusement 7.5 / 3.0; baseline 6.7 / 2.5. So **happiness is a high-arousal, positive-valence state rated nearly as arousing as anger and sadness.** It is the natural test of "arousal, not negative affect". The committed false-stress rates are happiness 0.24 vs anger 0.42, sadness 0.37 and fear 0.35 (external model).

**Test.**
1. Regress the external detector's stress score on within-subject-centred SAM arousal and SAM valence (mixed model, subject random intercept; clip as the unit). The arousal-only reading predicts arousal > 0 and valence ≈ 0. A valence effect would mean the detector carries some negative-affect information, which is a partly positive finding.
2. Repeat with the label-free physiological arousal index (z(HR) + z(tonic EDA)) as the outcome. This asks whether the wrist even tracks *felt* arousal.
3. Happiness vs anger/sadness at matched physiological arousal (same quintile AUROC as the matched-arousal probe).

**Feasibility check done here (descriptive, not a result).** At clip level (132 subject × clip units in EPM-E4, within-subject centred), wrist physiology barely tracks SAM: HR r = 0.05 with arousal and 0.07 with valence; tonic EDA r = 0.00 and −0.15; SCR count r = −0.22 and +0.23. SAM arousal and valence are uncorrelated within subject (r = 0.06), so the two effects are identifiable. The likely outcome is that the wrist tracks **neither** felt arousal nor valence within the emotion set. That would sharpen the paper's language: the detector reads **physiological mobilisation**, which is dissociated even from felt arousal (consistent with Campbell & Ehlert and with Mauss, as cited by Epel). Power is limited (33 subjects, about 4 clips each), so report CIs and treat it as supporting evidence. Main risk: the EPM-E4 clock starts at donning, with a 7 °C temperature swing, so clip order is confounded with warm-up. Include clip position as a covariate, and drop temperature features (already done).

**Cost.** Under a day. Reuse `specificity_panel_probe.py` per-window scores (regenerated) and `matched_arousal_probe.py` binning.

**Paper sentence if the null holds.** "Within the emotion set, neither the detector nor the physiological arousal index tracked self-rated arousal or valence (Table X); what the wrist reads is physiological mobilisation, which is dissociated even from felt arousal."

### Analysis B: a vascular channel as the missing challenge–threat dimension

**Theory.** Blascovich/Seery (TPR separates threat from challenge; HR does not), Kreibig (pulse amplitude and finger temperature carry the specificity in anxiety and fear).

**B1. PPG pulse amplitude at matched arousal (all five datasets).** Compute beat-level PPG amplitude (peak minus preceding trough, median per window) from the filtered BVP on still windows only (low accelerometer SD, cardiac coverage ≥ 0.5), z-scored per subject. Then (i) test whether it separates stress from Lego with countdown, the UBFC control group and hyperventilation **within arousal quintiles**; (ii) add it as a feature and rerun the matched-arousal test.

Descriptive feasibility using the existing, motion-contaminated `bvp_std` (per-subject z, all windows): seated cognitive stressors show lower amplitude (Campanella subtraction −0.88, PhysioNet Stroop −0.62 and TMCT −0.55, Stress-Predict Stroop −0.53). Lego with countdown sits at +0.43 **at similar arousal** (index 1.10 vs 1.26 for subtraction). That is a 1.3 SD gap where the HR/HRV/EDA model sees none. But speech raises it (UBFC speech +0.64/+0.78, WESAD TSST +0.66), which is almost certainly motion and talking. In the UBFC evaluated vs non-evaluated contrast there is no threat-like sign (arithmetic: test −0.11 vs control −0.35). So the Lego gap may be hand motion, not vasoconstriction. That is why the beat-level, still-window version is needed. If the gap survives, it is a **positive, theory-predicted finding**: the first wrist evidence that a vascular channel carries stress-specific information that HR/HRV/EDA lacks. If it does not survive, the challenge–threat account of the null is strengthened (the relevant channel is unmeasurable at the wrist).

**B2. Pulse arrival time on WESAD only.** WESAD has chest ECG (700 Hz) alongside wrist BVP (64 Hz). PAT = R-peak to wrist pulse foot, per beat, on still segments. Under threat, rising TPR and blood pressure should shorten PAT beyond what HR explains. Test stress vs amusement vs baseline, with PAT residualised on HR. **Caveats:** the RespiBAN–E4 synchronisation is by double-tap alignment, so absolute PAT is unreliable, but *within-session changes* are usable if the offset is constant. 64 Hz gives 15.6 ms resolution, so average over at least 30 beats (stress-related PAT changes are of the order of 10–30 ms [memory, unverified]). Only 37% of TSST windows keep usable PPG.

**Cost.** B1: 2–3 days (beat amplitude needs a pass over the raw BVP; the peak detector exists in the NeuroKit2 pipeline). B2: 2–3 days for WESAD only.

### Analysis C: a V3-style validity table plus a deployment PPV, with no new model

**Theory.** V3 (analytical vs clinical validation), Cacioppo (P(state | signal) vs P(signal | state)), classical discriminant validity.

**C1.** Turn the table in Section 1.4 into a paper table: layers (verification, analytical, clinical-convergent, clinical-discriminant, criterion-experience, context of use) × evidence × numbers × verdict. It costs nothing and makes the contribution legible to JBHI's digital-health reviewers.

**C2. The inverse probability, made concrete.** Apply Bayes' rule to the committed specificity-panel rates to estimate the positive predictive value of a "stress" alert under an explicit deployment mix. Illustration (assumptions stated, *not* a result): a 16 h waking day with 30 min of genuine stressor, 60 min exercise, 120 min seated effortful non-evaluative tasks, 60 min emotional media and 690 min rest. Recall 0.75. False-stress 0.69 for exercise (0.08 if trained), 0.50 for seated tasks, 0.30–0.35 for emotion, 0.15 for rest. PPV ≈ **0.09** without exercise in training and **0.11** with it. Training on exercise barely helps, because seated non-evaluative arousal and rest dominate the false positives. Report as a sensitivity curve over the rest false-stress rate and the stressor fraction. This is exactly the quantity Cacioppo says lab designs never estimate, and it turns the specificity panel into a deployment-relevant number.

**Cost.** Hours.

---

## 3. Suggested edits to `paper/main.tex`

1. **Introduction, paragraph 1, after "Whether a detector trained on stressor labels has learned anything beyond arousal has not … been tested":** add the Cacioppo sentence (1.1) and the Epel sentence (1.3). That turns the gap from "not tested" into "a known inference problem never tested for wearables".
2. **Related work:** add a short subsection, "Psychophysiological specificity", with Kreibig, Siegel, Quigley & Barrett, Blascovich/Seery and Epel (about 6 sentences, 1.2 + 1.6).
3. **Contributions list:** rephrase (2) as "the first discriminant-validity test of wrist stress detection, a specificity panel and a matched-arousal test, showing a many-to-one psychophysiological mapping", and add the LODO-cannot-expose-construct-shared-shortcuts point (1.5) as an explicit methodological contribution.
4. **Discussion, "Arousal transfers, stress does not":** add the shortcut inversion sentence (1.5), the challenge–threat mechanism (1.6) and Kreibig for the fear result (1.2).
5. **Discussion, "Reporting norms":** anchor the proposed specificity panel in V3 discriminant validation, and add the PPV-under-a-deployment-mix as a required companion number (C2).
6. **Limitations:** state the context of use in V3 language (1.4).

## 4. Citation hygiene

- Verified in full text for this report: Cacioppo et al. Handbook ch. 1 (2007), Kreibig (2010), Siegel et al. (2018), Quigley & Barrett (2014), Geirhos et al. (2020), Goldsack et al. (2020), Epel et al. (2018), Smets et al. (2018), Sharma et al. (2026), Porter & Goolkasian (2019).
- Abstract only: Seery (2011), Dickerson & Kemeny (2004), Campbell & Ehlert (2012), Can et al. (2019), Bosch (2026), Kaya et al. (2026). Uphill et al. (2019) came through a WebFetch summariser; re-check the quote before use.
- Memory, unverified: Cacioppo & Tassinary (1990) original wording (use the Handbook chapter instead), Cronbach & Meehl (1955), Campbell & Fiske (1959), McEwen (1998), Mauss et al. (2005), PAT effect sizes.
- Per the repo rule "label wrist conclusions drawn from chest or clinical ECG studies as extrapolations": the challenge–threat literature is impedance cardiography plus blood pressure, not wrist. Say "predicts" rather than "shows" for the wrist.
