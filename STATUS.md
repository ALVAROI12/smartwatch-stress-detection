# Project status: read first, update last

Last updated: 2026-09-23 (chat "novelty research agents", step 1 of the novelty plan).

## How to use this file

- Every chat reads this file before anything else, instead of searching old chats.
- Before ending a work block, update **Now**, **Open decisions**, **Next steps** and **Running**, and add one dated line to **Log** (newest first).
- Settled findings go to `CLAUDE.md` (short) or a `docs/*.md` write-up; this file holds only the current state and points to them.
- Before editing shared files, check **Running** so two chats do not edit the same files.

## Now

The paper's direction changed on 2026-09-22. Working title: **"What Do Wrist-Worn Stress Detectors Detect?"** The thesis: wrist detectors transfer across datasets because they detect autonomic arousal, and the within-dataset ceiling is stressor potency. The draft manuscript is `paper/main.tex` (IEEEtran, built as `paper/main.pdf`). A new method is no longer a goal for this paper: few-shot personalisation, domain adaptation and per-user thresholds are reported as leakage-controlled negatives. `main` on GitHub now holds the corrected pipeline (PRs #4, #6 and #7 merged).

Novelty review of 2026-09-23 (`docs/novelty_plan_2026-09-23.md`): not scooped, but Zhou et al. (BIBM 2023) and Kwon et al. (2026, iso-HR test) must be credited, and the intro claim at `paper/main.tex` line 33 is false as stated. Simulated reviewer: major revision. Ten paper-versus-table mismatches are listed there.

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
6. Novelty plan (`docs/novelty_plan_2026-09-23.md` §8): critical-evaluation framing and new title, three to five contributions, which optional steps to fund, autonomic-only arousal claim or self-report analysis first?

## Next steps

0. Novelty plan steps 1 and 2 are done (step 2: `docs/arousal_kit.md`; the untrained arousal index matches the external model on all five datasets; the model's non-arousal part is shared by active tasks; nulls detect within-bin AUROC ≥ 0.60–0.64). Next: step 3 (HRV-direction script) and step 4 (stressor-potency dose–response) of `docs/novelty_plan_2026-09-23.md` §6.
1. After decision 1, send the chosen documents to Dr. Pan.
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

- 2026-09-23: novelty plan step 2, arousal-controlled evaluation kit (`scripts/arousal_kit.py`, tables in `contribution_probes/arousal_kit/`); results added to the paper as Table V and two paragraphs.
- 2026-09-23: novelty plan step 1 applied to `paper/main.tex` (new title; Zhou 2023 cited; Kwon iso-HR test credited and reconciled with the HR-only disjoint variant; ten paper-versus-table mismatches fixed; second-half-rest and position-matching caveats added).
- 2026-09-23: five novelty agents (collision check, simulated reviewer, unused findings, theory framing, closable gaps); reports in `docs/literature/novelty-2026-09-23/`, consolidated plan in `docs/novelty_plan_2026-09-23.md`.
- 2026-09-23: created this file.
- 2026-09-22 evening: hardening probes; manuscript `paper/main.tex`; committee report `docs/contribution_report_ieee.pdf`.
- 2026-09-22 afternoon: contribution-map probes (specificity panel, time in session, matched arousal); few-shot on an unseen dataset and the mild-stress pilot came out negative; AutoStress, Dahal and other flagged papers read in full.
- 2026-09-22: repo refresh; README corrected; PRs #4, #6 and #7 merged into `main`.
- 2026-09-22 morning: novelty experiments merged (`29afa47`); IEEE advisor report on the memo branch.
- 2026-09-21: dataset-paper audit and seven label fixes (`6070b6b`); tuning and domain adaptation rerun; leakage-controlled few-shot (`bae4e5e`).
