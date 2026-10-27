# Experiment names

Readers do not need to know the development-stage numbering.

| Public name | Scientific question | Internal archival ID |
|---|---|---|
| `main_study` — Main study: random-length histories | Whole-path coverage and set size overall and by history group; 5,000 panels for each h=3,6,9 | `step5` |
| `fixed_history_study` — Fixed-history study: all-state-1 target histories | Conditional performance at L=10,50,100,200; all main-study reference pools reused, paired across lengths | `step6` |

## What readers run

Run `python START_HERE.py`. It generates data from the original seeds, runs
both experiments, aggregates their new results, and builds the manuscript
figures and tables. It is an hours-long full reproduction, not a plot-only
command. Do not run it just to view these instructions.

The public `run_simulation.py --study` choices are `main_study` and
`fixed_history_study`. New output folders have those same descriptive names;
summary folders are `main_study_summary` and `fixed_history_summary`.

## Why some original identifiers remain

`study_names.py` is the small public-to-archival mapping. Numerical workers
still receive the original study IDs. The frozen seed files, run/model IDs,
random-stream namespaces, panel IDs, saved scientific identities and expected
hashes have not been renamed. In particular, changing a run ID that participates
in random-stream addressing can change data even with an unchanged master seed.

The seed records remain `protocol/step5_seed.json` and
`protocol/step6_seed.json`. Historical QA logs and the bounded replay helpers
retain their original labels so that their evidence is not silently rewritten.
They refer to the two experiments above, not additional public run stages.

Manuscript figure filenames such as `step6_length_performance_nominal90.pdf`
and existing LaTeX labels are also retained for compatibility with the author's
manuscript and figure checks. The drawing code, colours, numerical data mapping,
and figure verification rules are unchanged.

The public-name mapping itself changed no scientific computation. The historical
naming checks are in `checks/NAMING_QA.json`. A full reproduction subsequently
completed on 26 October 2026; see [the current validation record](../VALIDATION.md).
Original experiment folders, packages and manuscript files are untouched.
