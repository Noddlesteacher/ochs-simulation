# Validation status — 25 October 2026

## Scoring-library directory rename

`simulation/method3/` is now `simulation/scoring_backend/`. Only the directory
name, nine import paths in four callers, and the scoring-guide references were
changed. The ten moved library files are byte-identical to their originals.
Historical QA records retain their original paths; checks/BACKEND_RENAME.json
records the path mapping, source comparison and current check scope. No full
simulation was run, and the end-to-end validation limitation below still applies.

## Public-name edition

The public entry point, run_simulation.py CLI, progress labels, new output
directories and reader-facing instructions now use `main_study` and
`fixed_history_study`. The small study_names.py mapping passes the unchanged
archival IDs to the workers. Original seeds, scientific modules, aggregation,
plotting, table construction and expected scientific fingerprints are unchanged.

This edition passed 27 unit/orchestration tests: the existing 18 deterministic
tests, 3 updated entry-point tests, and 6 public-name tests. The public-name
launch tests stop at a mocked worker boundary; they verify both mappings,
seed-file hashes, budgets and paths without generating scientific panels.
The entry-point test mocks all long-running subprocesses. File-by-file SHA-256
comparisons and the exact text diff are saved in checks/NAMING_QA.json and
checks/PUBLIC_NAMING.diff. Historical checks below are retained without rewriting
their records. No new full-study run, aggregate recomputation or figure rendering
was performed for this naming-only change. The full-run limitation below remains.

## What passed in this correction

| Check | Actual scope and result |
|---|---|
| From-seed regeneration | 15 predetermined main-study addresses and the corresponding 15 fixed-study pool addresses |
| Scientific file comparison | All 120 regenerated input/result files match original SHA-256 values exactly |
| Fresh score templates | All 12 templates recomputed; all candidate scores/tallies match original template file hashes |
| No saved-answer dependency | Calculating workers were prohibited from reading expected/ or data/ using a Python audit hook |
| Small deterministic identities | 18 tests, including a bounded independent indexed-block enumeration |
| Completed-task resume | 12 checks: no RNG/scoring on completed resume; wrong identities, altered files and prior failure rejected |
| Main-entry wiring | 3 tests with subprocesses mocked; checks stage order/new-data paths, stop-on-error, and disk gate |
| Aggregate regression | All original evaluation arrays reaggregated; nine CSVs match byte for byte |
| Paper tables | 7 tables; all 378 numerical cells checked; emitted LaTeX equals author's table snippets |
| Paper figures | 4 PDFs, exact page dimensions and zero different RGB pixels at both 144 and 288 dpi |
| Visual inspection | All four generated PNG previews inspected for labels, legends and layout |
| Scientific source preservation | Scientific function/class bodies unchanged after excluding docstrings; exact backend and aggregation modules unchanged bytewise |

Machine-readable evidence is in checks/. The fresh-templates and replay checks
used no supplied raw panel or score-template files. Only original hashes were
read afterwards for comparison.

## What the figure/aggregate checks do and do not establish

The aggregate regression used the original frozen evaluation arrays as external
test fixtures; those arrays are **not included** in this package and are not
inputs to START_HERE.py.

Figure 1, Figure 2 and the table regression used the original complete
aggregate tables as external test fixtures. Figures 3–4 used composition
reconstructed from the freshly regenerated h=6, repeat-0 result. The figure
tests establish unchanged plotting style and data mapping, not full new-run
completion.

A mock orchestration test checks command wiring only. It does not run 15,000
panels, and its temporary MOCK_ONLY success marker is not release evidence.

## Not yet completed

**The complete new entry point has not run from an empty output directory
through both full studies and all final checks.**

Therefore this package is a from-seed reproduction **candidate**, not a fully
validated submission release. The current machine had approximately 5 GiB
available during preparation, below this launcher's 8 GiB safety gate. No full
reproduction was launched or claimed.

Other operating systems, CPU architectures and dependency versions have not
been certified. Tight checksum checks may reject platform-dependent rendering
or floating-point aggregation changes. Such a failure must be explained; it
must not be hidden by substituting original answers.

No full manuscript LaTeX compilation was performed. The four standalone
figures and seven table snippets were checked, not Introduction prose or
unrelated manuscript material.

## Engineering corrections recorded during QA

- Replaced the old artifact-only default with a full from-seed entry point.
- Removed the Step6 worker's dependence on supplied old score templates.
- Routed all plots/tables to explicitly provided new-run outputs.
- Corrected a test path comparison to resolve macOS temporary-directory
  symlinks. This affected the test assertion, not scientific computation.
- Corrected the summary fingerprint inventory: the fixed-study aggregator
  produces TABLE4_FIXED_LENGTH.csv, not a standalone REFERENCE_COUNTS.csv.
  The original CSVs and all numerical algorithms were unchanged.

The earlier artifact-only packages are not evidence of a full from-seed run.
Before submission, perform a clean complete run, retain its logs/check records,
and only then update this status and tag a reviewed release.
