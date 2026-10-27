"""Compare NEW computations with hash-only evidence from the original run.

Hashes cannot supply any simulated path, score, set, or numerical table. This
module is called only AFTER computation. Nothing is copied into a result.
"""

import argparse
from pathlib import Path

from simulation.settings import PACKAGE, HS, REPLICATES
from simulation.storage import load, save, sha, require


def check_case(study, h, panel, folder, expected):
    key = f"h{h}/{panel}"
    files = expected["studies"][study][key]
    for name, checksum in files.items():
        require(
            sha(folder / name) == checksum, f"Scientific mismatch: {study}/{key}/{name}"
        )
    return len(files)


def check_raw(main, fixed, templates, report):
    expected = load(PACKAGE / "expected/original_scientific_hashes.json.gz")
    checked = 0
    for study, run in (("step5", main), ("step6", fixed)):
        require(
            load(run / "COMPLETE.json")["completed_panels"] == 15000, "Incomplete run"
        )
        keys = {f"h{h}/FORMAL_{i:06d}" for h in HS for i in range(REPLICATES)}
        actual = {
            str(p.parent.relative_to(run / "panels"))
            for p in (run / "panels").glob("h*/FORMAL_*/DONE.json")
        }
        require(actual == keys == set(expected["studies"][study]), "Missing/extra task")
        for h in HS:
            for i in range(REPLICATES):
                panel = f"FORMAL_{i:06d}"
                checked += check_case(
                    study, h, panel, run / "panels" / f"h{h}" / panel, expected
                )
    for name, checksum in expected["templates"].items():
        require(sha(templates / name) == checksum, "Exact template mismatch: " + name)
    save(
        report,
        {
            "status": "PASS",
            "scientific_files": checked,
            "templates": 12,
            "comparison": "Exact original compressed scientific file SHA-256; no tolerance",
        },
    )


def check_summaries(main, fixed, report):
    expected = load(PACKAGE / "expected/summary_hashes.json")
    checks = []
    for study, directory in (("step5", main), ("step6", fixed)):
        for name, checksum in expected[study].items():
            require(
                sha(directory / name) == checksum,
                "Aggregate mismatch: " + study + "/" + name,
            )
            checks.append({"study": study, "file": name, "sha256": checksum})
    save(report, {"status": "PASS", "files": checks, "comparison": "Exact CSV bytes"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--summaries", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.summaries:
        check_summaries(args.main, args.fixed, args.report)
    else:
        require(args.templates is not None, "Template path required")
        check_raw(args.main, args.fixed, args.templates, args.report)
    print("PASS: computed results match the original evidence.", flush=True)


if __name__ == "__main__":
    main()
