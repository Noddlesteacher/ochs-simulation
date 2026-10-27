# OCHS: simulation code and reproduction guide

Code for the simulation studies in **Outer-Conformal Hybrid Scoring for
Finite-State Markov Forecasting**, by Mengfan Fu.

Full reproduction completed on 26 October 2026. See
[VALIDATION.md](VALIDATION.md) for its scope and [validation/](validation/README.md)
for the saved comparison records.

The repository regenerates the simulated data from the original seeds, constructs
the prediction sets, computes the reported summaries, and produces the four
figures and seven tables in the manuscript checked on 26 October 2026. Readers do
not need the author's saved simulation data.

The manuscript version covered by the current verification has SHA-256
`66fe0b65cbb8a52474950b5d2ae27f877ca827c7e7d2c164c4dec1e7a8c9898f`.
This identifies the checked PDF; it is not a claim that later manuscript versions
will have the same tables or figures.

## Run the complete reproduction

Clone the repository and enter its directory:

```bash
git clone https://github.com/Noddlesteacher/ochs-simulation.git
cd ochs-simulation
```

You can also use GitHub's **Code > Download ZIP**, extract the archive, and
open the extracted directory. While the repository is private, access requires
the owner's or an authorized collaborator's GitHub account.

Use Python **3.13.0**. From the repository directory, create a separate Python
environment and install the versions listed in `requirements.txt`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python START_HERE.py
```

If using VS Code, select `.venv/bin/python` as the Python interpreter, open
`START_HERE.py`, and run that file. The remaining scripts are called by the
launcher; they do not need to be started individually.

The tested software environment is Python 3.13.0 on macOS 14.6, Apple Silicon.
Package versions, including plotting dependencies, are pinned. Exact results
and rendering on other platforms have not been established by this validation.

The launcher performs these stages in sequence:

1. Generate and score all 15,000 main-study panels.
2. Compute the 12 exact score templates for the fixed-history study.
3. Run the fixed-history study using the reference pools regenerated in stage 1.
4. Compare the generated scientific files and templates with archived checksums.
5. Compute and verify the aggregate summaries.
6. Generate and verify the manuscript figures and tables.

The terminal prints the study, horizon, and task ID as each task completes.
Later stages print their own status messages. The final success message is:

```text
Completed and verified. Figures and tables: .../paper
```

Successful simulation alone does not mean that all comparisons have passed.
The complete pipeline writes `FULL_REPRODUCTION_VERIFIED.json` only after the
data, summary, figure, and table checks succeed.

## The two studies

| Study | Target histories | Replicates | Purpose |
|---|---|---|---|
| Main study | Random-length Markov histories | 5,000 independent panels at each of h = 3, 6, 9 | Overall and history-group coverage, set size, and reference availability |
| Fixed-history study | All-state-1 histories of lengths 10, 50, 100, and 200 | The 5,000 main-study reference pools at each horizon, paired across four lengths | Performance at specified target histories and within-pool length comparisons |

Each main-study panel contains 200 independent countries: one target and 199
potential references. For every country, the scoring-history length is
`T = h + K`, where `K` has a geometric distribution with success probability
`1/78` and support `1, 2, ...`. The target's observed history has length `T`.
Each reference's observed record has length `T + h` and contains its completed
history and realized continuation.

The state space is `{1, 2, 3, 4}`. The initial distribution is uniform, and the
transition matrix is

```text
0.990  0.010  0.000  0.000
0.000  0.000  0.500  0.500
0.000  0.000  0.722  0.278
0.653  0.347  0.000  0.000
```

Group 1 contains histories consisting entirely of state 1. Group 0 contains
the remaining histories. OCHS uses only references in the target's history
group, selected without looking at their continuations. Each selected reference
is scored using its own realized continuation.

The four methods are OCHS with lambda = 0.90, count-only OCHS, OCHS with
lambda = 0.50, and randomized i-block. All methods use the 21 miscoverage
levels `alpha = 0, 0.05, ..., 1` and the complete admissible candidate space.
There are `2^h` candidates under the specified transition graph.

OCHS uses conservative outer ranks. Its internal block coordinate is also
conservative. The separate i-block benchmark uses

```text
q = (greater_count + U * equal_count) / permutation_count
```

and retains a candidate when `q > alpha`. If there is at most one permutable
middle block, `q = U`. The same candidate-specific uniform is reused at every
alpha level.

In the fixed-history study, every reference pool is used, including pools whose
original target was in group 0. A new target continuation is generated from
state 1. Within a pool, the reference records, target continuation, and
candidate-specific uniforms are shared across the four lengths. The original
main-study target's continuation is not used as the new target's truth.

The three OCHS specifications coincide in group 1. The fixed-history study
evaluates conditional performance at specified histories; its results should
not be interpreted as an additional fixed-history coverage guarantee.

## Output files

By default, the launcher creates a new directory beside the repository:

```text
OCHS_REPRODUCED_<date_and_time>/
    main_study/
    templates/
    fixed_history_study/
    main_study_summary/
    fixed_history_summary/
    paper/
        figures/
        tables/
    RAW_CHECK.json
    SUMMARY_CHECK.json
    FULL_REPRODUCTION_VERIFIED.json
```

The study directories retain generated inputs, predictions, evaluations, and
task records. The summary directories contain CSV files derived from those
new records. `paper/figures/` contains the four PDF figures and PNG previews;
`paper/tables/` contains the seven LaTeX tables.

The launcher also accepts an explicit **new** output location:

```bash
python START_HERE.py --out ../OCHS_MY_REPRODUCTION
```

It does not overwrite an existing reproduction directory.

## Match code to the manuscript

| Manuscript item | Code to read | Generated input |
|---|---|---|
| Main-study design | `simulation/generation.py`, `simulation/main_study.py` | Original seeds and frozen settings |
| Fixed-history design | `simulation/fixed_sources.py`, `simulation/fixed_prediction.py`, `simulation/fixed_study.py` | Newly generated main-study reference pools |
| OCHS calibration | `simulation/ochs.py`, `simulation/ranks.py` | Same-group reference scores and target-candidate scores |
| Randomized i-block | `simulation/iblock.py` | Exact permutation tallies and candidate uniforms |
| Figure 1: reliability | `figures/figure1_reliability.py` | `main_study_summary/tables/SUMMARY_BY_H_GROUP_ALPHA.csv` |
| Figure 2: fixed-history performance | `figures/figure2_fixed_lengths.py` | `fixed_history_summary/tables/SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv` |
| Figures 3 and 4: retained-path composition | `figures/figures3_4_composition.py` | `fixed_history_summary/RETAINED_PATHS_AND_COUNTS.json` |
| Tables 1-7 | `tables/make_tables.py` | Newly computed summary, paired-comparison, availability, and diagnostic CSVs |
| Coverage diagnostics | `analysis/aggregate_main.py` | Replicate-level reference counts and coverage indicators |

Figure 1 shows primary OCHS reliability overall and within each history group.
Randomized i-block is displayed overall. The identical group-1 OCHS curves are
shown once. Colors distinguish horizons.

Figures 3 and 4 use the prespecified horizon-6, repeat-0 example. Examples are
not chosen by how their sets look. State-composition bars give each retained
path equal weight; they are not predictive probabilities.

## Read the scientific code

Start with `START_HERE.py` to see the sequence of operations. Then follow one
main-study replicate through these files:

1. `simulation/generation.py`: generate the country episodes and separate
   observed information from evaluation truth.
2. `simulation/ochs.py`: select references and score their own continuations.
3. `simulation/ranks.py`: form conservative outer p-values and apply the strict
   inclusion rule.
4. `simulation/iblock.py`: form the separate randomized benchmark p-values.
5. `simulation/main_study.py`: evaluate coverage and prediction-set size.

The exact block calculation is in `simulation/scoring_backend/`. Its additional
code avoids enumerating every indexed permutation while preserving the full
permutation counts. See `docs/SCORING_GUIDE.md` for the calculation and
`REVIEWER_GUIDE.md` for a function-by-function reading guide.

`analysis/` computes summaries; `figures/` controls plotting; `tables/` formats
the manuscript tables. Saving, resource monitoring, and comparison routines are
separate from the scoring functions.

## Seeds and reference files

`protocol/step5_seed.json` and `protocol/step6_seed.json` contain the original
research seeds. The labels `step5` and `step6` are retained in archived random
stream identities. The public study names are `main_study` and
`fixed_history_study`; see `docs/EXPERIMENT_NAMES.md`.

Reproduction depends on the seeds, PCG64 implementation, stream addresses,
candidate ordering, and innovation rules. These are fixed together. Changing
an archived identifier can change the simulated data even if the seed is left
unchanged.

`expected/` contains comparison hashes and four reference figure PDFs. It does
not supply raw panels, prediction sets, score templates, or aggregate result
tables to the simulation. The workers regenerate these objects. Table templates
provide the manuscript's LaTeX layout and reference cell text; numerical cells
are derived from the new CSVs and checked against the templates.

## Completed verification

The full reproduction was completed on 26 October 2026. It regenerated all
15,000 main-study panels and all 60,000 fixed-history scenarios. Checks found:

| Comparison | Result |
|---|---|
| 120,000 compressed scientific input and result files | Exact SHA-256 agreement with the archived experiment |
| 12 freshly regenerated fixed-history score templates | Exact agreement with archived hashes |
| 9 aggregate CSV files | Byte-for-byte agreement |
| 7 manuscript tables, containing 378 numerical items | Agreement with the manuscript table snippets |
| 4 standalone PDF figures | Zero differing RGB pixels at both 144 and 288 dpi |
| Figures embedded in the current manuscript PDF | Identical drawing streams and zero differing pixels after isolating each figure at its native size |

The completed run included a stop caused by insufficient disk space and a later
author-requested pause for laptop travel. Both were handled with recorded,
restricted engineering recovery. Completed outputs, original random stream
identities, and cumulative resource accounting were retained. This was a full
reproduction completed across those interruptions, not an uninterrupted test
of a fresh launch in a single session.

Exact agreement with the archived outputs is evidence of computational
reproduction. It is not an independent mathematical proof or an independent
implementation of every score. The full manuscript was not recompiled as part
of these checks.

All reported outcomes are retained, including the fixed-history results at
h = 9, L = 200 and 95% nominal coverage: OCHS coverage 94.34% and randomized
i-block coverage 90.90%.

The 114 coverage diagnostics apply to primary OCHS only: three horizons, two
history groups, and 19 interior alpha levels. They use individual replicate
reference counts, not an average count. Six raw p-values are below 0.05; none
is below 0.05 after Bonferroni adjustment. The smallest raw and adjusted
p-values are 0.007075 and 0.806523, respectively.

## Runtime and storage

Allow several hours. In the completed local run, recorded task execution times
totaled approximately 6.02 hours for the main study and 1.84 hours for the
fixed-history study. These totals exclude some preparation, template building,
verification, aggregation, plotting, and time spent paused. They are not runtime
promises for another machine.

The launcher requires at least 8 GiB free space before starting. During
computation it stops if free space falls below 4 GiB. The controller uses one
worker, a 120-second per-task limit, a 2 GiB sampled worker-RSS limit, and
cumulative task-time limits of 24 hours for the main study and 8 hours for the
fixed-history study. Study output limits are 12 GiB and 2 GiB, respectively.
Resource limits may stop a reproduction on a slower or more constrained machine.

Keep the laptop connected to power and awake while it runs. Leave additional
disk space for the operating system and other applications; starting just above
the minimum does not reserve that space for the experiment.

## If execution stops

Keep the output directory and its logs. Do not delete a failed task, change its
seed, or start a second copy in the same directory. The absence of
`FULL_REPRODUCTION_VERIFIED.json` means that the full pipeline has not passed.

The lower-level `run_simulation.py --resume` option checks completed task
identities and hashes and preserves cumulative task-time accounting. It does
not automatically repair failed or partially written tasks. The author's
disk-recovery scripts retained in this repository are incident-specific and
are not the normal entry point for a new reader. See
[the recovery note](RECOVERY_README_ZH.md). A separate local travel-pause
continuation was used for the completed author run; it is not required by a
new reader.

For a bug report, include the Python version, operating system, command, last
terminal message, and relevant task or comparison record. Do not substitute
archived results for generated results to make a comparison pass.

## Citation and license

Please cite the accompanying manuscript when using this code. Final publication
details and a versioned software citation will be added when available.

The repository's software license is awaiting author approval. A publication
DOI and versioned software citation have not yet been assigned.
