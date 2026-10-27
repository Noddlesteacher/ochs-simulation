# Completed-run evidence

These records describe the reproduction completed on 26 October 2026. They are
comparison records, not saved experimental inputs used to run the simulations.

| File | What it records |
|---|---|
| `FULL_REPRODUCTION_VERIFIED.json` | Final completed-run marker: 15,000 main panels and 60,000 fixed-history scenarios |
| `RAW_CHECK.json` | Exact original scientific file hashes for 120,000 files and 12 templates |
| `SUMMARY_CHECK.json` | Exact comparison of nine freshly computed aggregate CSVs |
| `FIGURE_CHECK.json` | Four standalone PDFs, page dimensions and exact pixels at two resolutions |
| `TABLE_CHECK.json` | Seven tables, numerical cells and their aggregate-source keys |
| `MANUSCRIPT_COMPARISON.json` | Latest 22-page manuscript: four embedded figures, 51 table rows, 378 numeric items and selected prose results |
| `SOURCE_COPY_CHECK.json` | Publication-copy checks: Python files, scientific assets, seeds and reference fingerprints preserved |
| `PACKAGING_TESTS.txt` | Deterministic tests and mocked launch checks run on the publication copy |

The first five records are exact copies of the completed run's records. The
manuscript comparison is a portable copy: the author's absolute paths have been
replaced, while all comparison values are retained. It includes the original
report's SHA-256 and the checked manuscript's SHA-256.

The raw simulated panels and full manuscript are not included here. The
repository generates the panels from seed; `expected/` supplies the comparison
fingerprints after computation. The author retains the full local outputs and
the original check records.

Packaging tests are not another full simulation. Small deterministic fixtures
and mocked orchestration checks have separate scopes from the completed-run
comparisons above. This copy was not subjected to another complete simulation
after its directory name and documentation were changed.
