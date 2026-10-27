# OCHS simulation studies

Simulation code for **Outer-Conformal Hybrid Scoring for Finite-State Markov
Forecasting**, by Mengfan Fu.

This repository contains the two simulation studies in the paper: a main study
with random-length histories and a paired study with fixed all-state-1 target
histories. Running `START_HERE.py` generates the data from the original seeds,
computes the prediction sets, and builds the paper's four figures and seven
tables. No previously generated experimental data need to be downloaded.

A full reproduction was completed on 26 October 2026. Its results and figures
were also checked against the manuscript. The records and scope of those checks
are described in [VALIDATION.md](VALIDATION.md).

## Getting started

Download the repository using **Code > Download ZIP**, or clone it:

```bash
git clone https://github.com/Noddlesteacher/ochs-simulation.git
cd ochs-simulation
```

The checked environment is **Python 3.13.0 on macOS 14.6, Apple Silicon**. From
the repository directory, create an environment and install the pinned packages:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python START_HERE.py
```

In VS Code, select `.venv/bin/python` as the interpreter, open `START_HERE.py`,
and run that file. It calls the other scripts in sequence; there is no need to
run each file separately.

Allow several hours and keep the computer awake and connected to power.
The launcher requires at least 8 GiB of free disk space. The exact dependency
versions are required by the environment check, including the plotting
dependencies used for fonts and figure rendering. Other operating systems and
architectures have not been verified for identical outputs.

The terminal prints a line when each task completes. Once both studies finish,
the program checks the generated records, computes the summaries, and builds
the figures and tables. The final message is:

```text
Completed and verified. Figures and tables: .../paper
```

At that point, `FULL_REPRODUCTION_VERIFIED.json` records that the data, summary,
figure, and table comparisons all passed.

## Experiments

### Main study: random-length histories

There are 5,000 independent panels at each horizon `h = 3, 6, 9`, for 15,000
panels in total. Each panel contains 200 independent countries: one target and
199 potential references.

For each country, the scoring-history length is `T = h + K`, where `K` is
geometric with success probability `1/78` and support `1, 2, ...`. The
target's observed record has length `T`. A reference's observed record has
length `T + h`, containing both its scoring history and its realized
continuation.

The state space is `{1, 2, 3, 4}`, the initial distribution is uniform, and the
transition matrix is

```text
0.990  0.010  0.000  0.000
0.000  0.000  0.500  0.500
0.000  0.000  0.722  0.278
0.653  0.347  0.000  0.000
```

Group 1 consists of all-state-1 histories; group 0 contains the other histories.
OCHS selects references by history group and scores each reference using its
own realized continuation. Coverage and prediction-set size are reported
overall and by group.

### Fixed-history study: all-state-1 targets

The second study sets the target history to all 1s, with lengths
`L = 10, 50, 100, 200`. It uses all 5,000 main-study reference pools at each
horizon, including pools whose original target belonged to group 0. This gives
15,000 pool replicates and 60,000 length scenarios.

For each pool, a new target continuation is generated from state 1. The
reference records, new continuation, and candidate-specific uniforms are shared
across the four lengths. The new target is evaluated against that continuation,
rather than the original main-study target's future. The selected group-1
references and their count are the same across lengths.

This study examines performance at specified histories. The paper's groupwise
coverage theorem does not give a separate coverage guarantee at each fixed
history.

### Methods

Both studies compare OCHS with lambda = 0.90, count-only OCHS (lambda = 1),
OCHS with lambda = 0.50, and randomized i-block. All use the full admissible
candidate space of `2^h` paths and the 21 miscoverage levels
`alpha = 0, 0.05, ..., 1`. The three OCHS specifications give identical
prediction sets in group 1.

OCHS uses conservative outer ranks and a conservative internal block
coordinate. The separate randomized i-block benchmark uses

```text
q = (greater_count + U * equal_count) / permutation_count
```

and retains a candidate when `q > alpha`. With at most one permutable middle
block, `q = U`. Each candidate's uniform is reused across all alpha levels
and, in the fixed-history study, across all four lengths.

## Where the results are saved

The program creates a new directory beside the repository:

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

The study directories retain the generated inputs, predictions, evaluations,
and task records. The summary directories contain the CSV results calculated
from those records. The four PDF figures and PNG previews are in
`paper/figures/`; the seven LaTeX tables are in `paper/tables/`.

To choose a different output location, pass a new directory:

```bash
python START_HERE.py --out ../OCHS_MY_REPRODUCTION
```

An existing output directory is not overwritten.

## Finding your way through the code

Start with `START_HERE.py` for the order of operations. It runs the main study,
computes the 12 exact fixed-history score templates, runs the paired study,
then verifies, aggregates, and plots the results.

For the scientific calculations, the following files are the main places to
read:

| Part of the method | Files |
|---|---|
| Generate country episodes and separate observed data from evaluation truth | `simulation/generation.py` |
| Select same-group references and score references and candidates | `simulation/ochs.py` |
| Form conservative outer ranks and prediction-set membership | `simulation/ranks.py` |
| Calculate the randomized i-block benchmark | `simulation/iblock.py` |
| Evaluate a main-study replicate | `simulation/main_study.py` |
| Generate shared fixed-study inputs and evaluate the four lengths | `simulation/fixed_sources.py`, `simulation/fixed_prediction.py`, `simulation/fixed_study.py` |

The exact block calculation is in `simulation/scoring_backend/`. It aggregates
permutation suffixes while retaining the full indexed permutation counts.
[SCORING_GUIDE.md](docs/SCORING_GUIDE.md) explains this calculation.
[REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) gives a more detailed reading guide.

The `analysis/`, `figures/`, and `tables/` directories handle summaries and
paper outputs:

| Paper output | Script | Input generated by the reproduction |
|---|---|---|
| Figure 1: reliability | `figures/figure1_reliability.py` | `main_study_summary/tables/SUMMARY_BY_H_GROUP_ALPHA.csv` |
| Figure 2: fixed-history performance | `figures/figure2_fixed_lengths.py` | `fixed_history_summary/tables/SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv` |
| Figures 3 and 4: retained-path composition | `figures/figures3_4_composition.py` | `fixed_history_summary/RETAINED_PATHS_AND_COUNTS.json` |
| Tables 1-7 | `tables/make_tables.py` | Summary, paired-comparison, reference-availability, and diagnostic CSVs |

Figure 1 shows primary OCHS overall and within each group, with randomized
i-block shown overall. Colors distinguish horizons. Figures 3 and 4 use the
prespecified horizon-6, repeat-0 example. Their bars describe the state
composition of equally weighted retained paths, not predictive state
probabilities. Plotting scripts specify the colors, fonts, markers, and export
dimensions.

## Seeds and reproducibility checks

The original seeds are in [protocol/step5_seed.json](protocol/step5_seed.json)
and [protocol/step6_seed.json](protocol/step6_seed.json). The older labels remain
part of the archived random-stream identities. The public study names are
`main_study` and `fixed_history_study`; their mapping is documented in
[EXPERIMENT_NAMES.md](docs/EXPERIMENT_NAMES.md).

PCG64, stream addresses, candidate order, and integer innovation rules are
preserved alongside the seeds. They jointly determine the generated data.
Changing an identifier used in a stream address can change the data even with
the same master seed.

The `expected/` directory holds archived checksums and four reference PDFs for
comparison. Raw panels, score templates, predictions, and aggregate CSVs are
generated by the program. The LaTeX table templates contain the paper's layout
and reference cell text; each numerical cell is obtained from a newly computed
CSV and checked before the table is written.

The completed reproduction matched all 120,000 compressed scientific files,
12 score templates, and nine aggregate CSVs. The seven tables matched all 378
numerical items. The four PDF figures had zero differing RGB pixels at both
144 and 288 dpi. Their embedded versions in the checked manuscript also
matched. See [VALIDATION.md](VALIDATION.md) and the [saved check
records](validation/README.md) for the manuscript fingerprint and comparison
details.

The author run completed across a disk-space interruption and a deliberate
pause, with recorded engineering recovery. That history and the limits of the
verification are retained in the validation record. These checks establish
agreement with the archived calculations; they are not an independent
implementation of every score. The full manuscript was not recompiled.

All reported results are retained. At `h = 9`, `L = 200`, and 95% nominal
coverage, empirical coverage is 94.34% for OCHS and 90.90% for randomized
i-block. The 114 coverage diagnostics concern primary OCHS only, across three
horizons, two groups, and 19 interior alpha levels. They use each replicate's
reference count. Six raw p-values are below 0.05; none remains below 0.05 after
Bonferroni adjustment. The minimum raw and adjusted p-values are 0.007075 and
0.806523.

## Runtime, storage, and interrupted runs

In the completed local run, recorded task execution totaled about 6.02 hours
for the main study and 1.84 hours for the fixed-history study. Template building,
verification, aggregation, plotting, some preparation, and paused time are
additional. Runtime on another computer will depend on its hardware and load.

The controller enforces the following limits:

| Resource | Limit |
|---|---|
| Free disk space at startup | At least 8 GiB |
| Free disk space during computation | Stop below 4 GiB |
| Concurrent workers | One |
| Execution time per task | 120 seconds |
| Sampled worker RSS | 2 GiB |
| Cumulative task execution | 24 hours for the main study; 8 hours for the fixed-history study |
| Study output size | 12 GiB for the main study; 2 GiB for the fixed-history study |

Leave room for other applications as well as the generated files. A slow or
constrained machine may reach a resource limit before completing the run.

If the program stops, keep the output directory and logs for inspection.
The lower-level `run_simulation.py --resume` checks completed task identities
and hashes and retains cumulative execution time; it does not repair failed or
partially written tasks. `RESUME_HERE.py` is specific to the author's recorded
disk interruption. New reproductions start with `START_HERE.py`.

For a problem report, include the command, Python and operating-system
versions, the last terminal message, and the relevant task or comparison record.

## Citation and license

Please cite **Outer-Conformal Hybrid Scoring for Finite-State Markov
Forecasting**, by Mengfan Fu, when using this code. Publication details and a
versioned software citation will be added when available.

The software license is awaiting author approval.
