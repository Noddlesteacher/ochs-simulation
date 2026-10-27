# Validation record

Updated 26 October 2026.

## Completed reproduction

The author completed a from-seed reproduction of 15,000 main-study panels and
60,000 fixed-history scenarios. The resulting inputs and scientific outputs
matched the frozen experiment. The newly computed summaries were then used to
build and check the manuscript figures and tables.

| Check | Scope | Outcome |
|---|---|---|
| Scientific file comparison | 120,000 compressed scientific files across both studies | Exact SHA-256 match |
| Score-template comparison | All 12 freshly calculated fixed-history templates | Exact hash match |
| Summary comparison | Nine newly calculated aggregate CSVs | Byte-for-byte match |
| Table comparison | Seven LaTeX tables, 51 rows and 378 numerical items | Exact match to manuscript snippets |
| Figure comparison | Four standalone PDFs at 144 and 288 dpi | Equal page sizes and zero differing RGB pixels |
| Latest manuscript comparison | Four embedded figures and all seven tables in the checked 22-page PDF | Figure drawing streams and isolated renders match; all 378 numerical items match |

The checked manuscript PDF has SHA-256
`66fe0b65cbb8a52474950b5d2ae27f877ca827c7e7d2c164c4dec1e7a8c9898f`.
It is identified here without distributing the manuscript itself.

The actual check records are in [validation/](validation/README.md).

## Interruptions and the completed run

The reproduction began in a new output directory. During the main study it
stopped when free disk space fell below the runtime limit. An author-approved
engineering recovery preserved the interruption evidence and regenerated only
the same interrupted input address; it checked the exact original hashes before
subsequent scoring. Completed tasks were retained and checked. During the
fixed-history study, the author requested a safe pause for laptop travel.
That continuation resumed from a completed-task boundary.

The final checks cover the complete generated results after both interruptions.
This is a completed full reproduction, not a claim of an uninterrupted run in
one session. Scientific scoring, seeds and random-stream identities were not
changed by those engineering steps.

The original run directories, failed-attempt logs, and incident-specific
continuation records remain in the author's local research archive. They are
not inputs required by a reader who starts a new reproduction.

## This publication copy

This directory is a separate publication copy of the validated recovery edition.
The Python files, scientific settings, seeds, exact numerical backend, plotting
code, expected fingerprints and table templates are retained byte for byte.
Reader-facing documentation and the file manifest are updated. Completed-run
verification records are included.

The packaging checks compare the files, exercise deterministic identities and
test the entry-point wiring with long-running subprocesses mocked. They do not
start another simulation. A new full simulation has not been run in this renamed
publication directory.

The historical 25 October validation report is retained in
[docs/history/VALIDATION_20261025.md](docs/history/VALIDATION_20261025.md). Its
then-current statement that a full reproduction had not yet completed is a
historical record, not the present status.

## What these checks establish

They establish computational agreement with the frozen experimental outputs
and agreement of the reproduced figures and tables with the checked manuscript.
They do not independently rederive every production score or prove the
statistical coverage theorem. The independent numerical checks described in
the manuscript retain their original, more limited scopes.

The full manuscript was not recompiled. Other operating systems, CPU
architectures and dependency versions have not been certified for bitwise
scientific output or identical figure rendering.

No outcomes were dropped for falling below nominal coverage. In particular,
h = 9, L = 200, 95% nominal results remain 94.34% for OCHS and 90.90% for
randomized i-block.
