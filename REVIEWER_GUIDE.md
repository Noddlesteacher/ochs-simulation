# Reviewer guide: follow one replicate

The public entry point is START_HERE.py. The scientific pipeline is separated
from the watchdog, saving, and verification helpers so that a reader can follow
the method without first reading the experiment-management code.

The two experiments are named `main_study` (random-length histories) and
`fixed_history_study` (fixed all-state-1 target histories). Older identifiers
are retained only for scientific/archive compatibility; the precise mapping
is in [docs/EXPERIMENT_NAMES.md](docs/EXPERIMENT_NAMES.md).

## Main study

| Question | Start reading here | What to check |
|---|---|---|
| Where do states and lengths come from? | simulation/generation.py: positive_geometric, generate_episode | K starts at 1; T=h+K; no length truncation |
| What can the predictor observe? | simulation/generation.py: generate_observation | Target sees H only; each reference sees H plus its own realised Y |
| How is randomness reproduced? | simulation/random_sources.py: seed_words, Factory | Fixed master entropy and stable per-purpose addresses |
| Which references are selected? | simulation/ochs.py: pool | History-only grouping, 199 potential references, no target included |
| How are scores evaluated? | simulation/ochs.py: predict | References use their own Y; candidates use the target H |
| How is the outer rank formed? | simulation/ranks.py: ranks | Count reference scores greater than or equal to target; add target's own rank |
| How are ties randomised? | simulation/iblock.py: exact_q | Only the separate i-block benchmark uses U |
| How is coverage measured? | simulation/main_study.py: evaluate | The truth is supplied after prediction, not during construction |

The important mathematical objects retain familiar short names: H is a
history, y a candidate, T the local history length, h the forecast horizon,
and N the number of selected references. Short mathematical names do not
denote hidden global data.

In ranks.py, binary search is an exact, faster way to count
reference_score >= candidate_score. It does not approximate a score or
replace exact ties. The number of retained candidates is calculated using
integer cross-multiplication: for p=n/d and alpha=k/20, retain if 20*n > k*d.

## Fixed-history study

Read these three files in order:

1. simulation/fixed_prediction.py: **template** computes exact scores for every
   candidate at each fixed (h,L); **reference_view** selects G1 references.
2. simulation/fixed_sources.py: **generate_input** generates one new future
   and one uniform per candidate. Its addresses deliberately omit L and alpha.
3. simulation/fixed_study.py: **work** shares that input across all four lengths
   and evaluates the newly constructed sets.

The templates are computed from the scoring functions by
simulation/build_templates.py. They are not shipped as precomputed input.
No main-study target history, target future, or G0 selection is used to score
the new all-state-1 target.

## Figures and tables

| Manuscript item | Code | Newly computed source |
|---|---|---|
| Figure 1: reliability | figures/figure1_reliability.py | main_study_summary/tables/SUMMARY_BY_H_GROUP_ALPHA.csv |
| Figure 2: fixed-length performance | figures/figure2_fixed_lengths.py | fixed_history_summary/tables/SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv |
| Figures 3–4: retained-set composition | figures/figures3_4_composition.py | fixed_history_summary/RETAINED_PATHS_AND_COUNTS.json |
| Tables 1–7 | tables/make_tables.py | Main/fixed summaries, paired contrasts, reference availability, diagnostics |

collect_reproduction.py builds the composition JSON from the newly calculated
h=6, repeat-0 set members. It does not choose an attractive example.
State-composition percentages count retained paths equally; they are not
predictive state probabilities.

Figure 1 presents primary OCHS only, by horizon, and i-block only overall.
G1 OCHS curves are not repeated for the three identical weights. The plotting
scripts fix colours, fonts, sizes, markers, labels and export dimensions.

## Evidence limits

The 114 Poisson–binomial diagnostics apply only to primary OCHS, 3 horizons,
2 groups and 19 interior alpha levels. They are not i-block diagnostics.
They do not independently certify the scoring implementation.

Selected-seed replays test compatibility with the original saved computations.
They are not an independent rederivation of every full-run score. Full raw-file
comparison, when the entire entry point has run, establishes exact output
agreement with the archived run, not a new proof of mathematical validity.
