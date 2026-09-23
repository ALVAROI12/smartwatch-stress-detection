# Project status: read first, update last

Last updated: 2026-09-23 (chat "add academic research skills").

## How to use this file

- Every chat reads this file before anything else, instead of searching old chats.
- Before ending a work block, update **Now**, **Open decisions**, **Next steps** and **Running**, and add one dated line to **Log** (newest first).
- Settled findings go to `CLAUDE.md` (short) or a `docs/*.md` write-up; this file holds only the current state and points to them.
- Before editing shared files, check **Running** so two chats do not edit the same files.

## Now

The paper's direction changed on 2026-09-22. Working title: **"What Do Wrist-Worn Stress Detectors Detect?"** The thesis: wrist detectors transfer across datasets because they detect autonomic arousal, and the within-dataset ceiling is stressor potency. The draft manuscript is `paper/main.tex` (IEEEtran, built as `paper/main.pdf`). A new method is no longer a goal for this paper: few-shot personalisation, domain adaptation and per-user thresholds are reported as leakage-controlled negatives. `main` on GitHub now holds the corrected pipeline (PRs #4, #6 and #7 merged).

**2026-09-23 revision pass (branch `gitignore-graphify`, pushed).** A claim-to-code audit (`docs/claim_to_code_audit.md`) found 26 numbers in `main.tex` that disagreed with committed tables and 10 with no source; all are now fixed or backed. A simulated JBHI review (`docs/review_simulation/jbhi_review.md`, five seats, same model family) returned **Major Revision** with a 15-item roadmap, all on existing data. Reanalyses done (`scripts/reviewer_reanalyses.py`, `outputs/tables/jbhi_v2/reviewer_reanalyses/`) change the thesis:
- A label-free HR+EDA arousal index, untrained, matches the trained detector under LODO (no model-minus-index BA difference survives Holm; largest +0.075, UBFC-Phys). This is now the direct evidence for "detects arousal".
- The "non-stress" probes are not non-stress by their descriptors: UBFC-Phys control is an "easier" social task that "imply[s] a high stress level in both scenarios"; Campanella Lego tasks were built to "create the mental strain" of work; Stress-Predict tags hyperventilation as stress-inducing. The title's "Evaluative Stress Does Not" cannot stand.
- Matched-arousal nulls are underpowered (minimum detectable deviation 0.12–0.22; only hyperventilation equivalent to 0.5 at ±0.10). Transfer costs are not significant after Holm but upper CI bounds reach 0.08–0.13.
- Kwon et al. (2026) already ran an iso-heart-rate test (WESAD stays separable, 0.955 to 0.938); the intro claim is narrowed and Kwon, Schmidt 2019 and Vos 2023 are cited.

**Manuscript rewritten on this evidence (commits `37d3342`, `894a661`).** New title: "What Do Wrist-Worn Stress Detectors Detect? An Untrained Arousal Index Matches Them Across Five Empatica E4 Datasets". Abstract, contributions, results (Tables II–IV), discussion and conclusion follow the reanalyses; probes are described as their descriptors do; nulls are stated with their power. 10 pages. The title and thesis change still needs sign-off from the user and Dr. Pan (open decision 6).

## Key numbers (details: `CLAUDE.md`, `docs/contribution_map.md`, `docs/novelty_experiments.md`)

- Leave-one-dataset-out over five E4 datasets: transfer cost 0.015–0.05 balanced accuracy (HR/HRV/EDA).
- Within WESAD + PhysioNet + EPM-E4: 0.831 balanced accuracy (95% CI 0.813–0.850) on subject-grouped splits, 0.849 LOSO.
- Exercise is called stress 69% of the time when exercise is not in training, 8% when it is. Seated non-evaluative tasks, hyperventilation, Lego and anger are not separable from stress at matched arousal (AUROC 0.36–0.58, none significant after Holm); exercise is (0.89).
- Minutes since start alone beats physiology within dataset on PhysioNet (0.83 vs 0.78) and Stress-Predict (0.95 vs 0.70).

## Deliverables

| What | Path | Note |
|---|---|---|
| Manuscript | `paper/main.tex` → `paper/main.pdf` | Build: `tectonic paper/main.tex`; figures: `paper/make_paper_figures.py` |
| Committee report | `docs/contribution_report_ieee.pdf` | Newest (2026-09-22 19:28); follows the new direction |
| Advisor report | `~/Desktop/JBHI_revision_report_for_Dr_Pan_2026-09-22_IEEE.pdf` | Source on local branch `jbhi-revision-with-memo`. Predates the contribution map, so it still proposes personalisation as the method |
| Findings | `docs/contribution_map.md`, `docs/novelty_experiments.md`, `docs/literature/` | |

## Open decisions (user, Dr. Pan)

1. Which document goes to Dr. Pan: the committee report plus manuscript (recommended), or the 2026-09-22 advisor report (partly superseded)?
2. Adopt the "what detectors detect" thesis for the JBHI paper (`docs/contribution_map.md` §5)?
3. Write the benchmark-release paper in parallel (§4, rank 2)?
4. Start the IRB for a counterbalanced UTSA lab study (§4, rank 4)?
5. File a correction note for the thesis and poster?
6. New title and thesis after the 2026-09-23 review (roadmap REV-10). Proposal: drop "Evaluative Stress Does Not"; lead with "an untrained arousal index matches trained wrist stress detectors across five E4 datasets".

## Next steps

1. Remaining roadmap items (`docs/review_simulation/jbhi_review.md` §4). Done: REV-1 (except the author e-mail placeholder at L18 and a manipulation-check table), REV-2 text, REV-4, REV-5 core, REV-10, REV-11 (Kwon, Schmidt, Vos, Milstein, Richer). Open: REV-3 rest (graded synthetic positive control, missingness-stratified matched test, HR-only matching on WESAD), REV-6 (normalisation/weighting sensitivity), REV-7 (EDA quality screening), REV-8 (bootstrap the Shapley budget), REV-9, REV-12 checklist box, REV-13 (cut to main text + supplement, release feature table decision), REV-14 (HR/HRV-only arm, demographics), REV-15 (EPM-E4 settling control). Then re-run the simulated review in re-review mode.
2. Merge `gitignore-graphify` into `jbhi-revision`: `git merge --ff-only gitignore-graphify`.
3. After decision 1, send the chosen documents to Dr. Pan.
2. `docs/contribution_map.md` §5, items 2–3: add the Stress-Predict hyperventilation relabelling to the label audit; correct the PULSE premise in the advisor sheet.
3. Optional check: one leave-one-dataset-out row with a frozen public PPG encoder (§4, rank 8).

## Running

Nothing running (checked 2026-09-23).

## Branches and places

- `jbhi-revision`: active branch, this worktree; in sync with origin.
- `main`: corrected pipeline (merged 2026-09-23).
- `jbhi-revision-with-memo`: local only; advisor report sources.
- `thesis-final-figures`: worktree `~/Projects/smartwatch-stress-detection`. The raw datasets and feature tables (`data/processed/combined/`) live there; treat them as read-only.

## Log (newest first)

- 2026-09-23 afternoon: graphify knowledge graph (`graphify-out/`, ignored); claim-to-code audit and fixes; unbacked numbers backed; simulated JBHI review (Major Revision); reviewer reanalyses A–F; Kwon/Schmidt/Vos citations. Branch `gitignore-graphify`.
- 2026-09-23: created this file.
- 2026-09-22 evening: hardening probes; manuscript `paper/main.tex`; committee report `docs/contribution_report_ieee.pdf`.
- 2026-09-22 afternoon: contribution-map probes (specificity panel, time in session, matched arousal); few-shot on an unseen dataset and the mild-stress pilot came out negative; AutoStress, Dahal and other flagged papers read in full.
- 2026-09-22: repo refresh; README corrected; PRs #4, #6 and #7 merged into `main`.
- 2026-09-22 morning: novelty experiments merged (`29afa47`); IEEE advisor report on the memo branch.
- 2026-09-21: dataset-paper audit and seven label fixes (`6070b6b`); tuning and domain adaptation rerun; leakage-controlled few-shot (`bae4e5e`).
