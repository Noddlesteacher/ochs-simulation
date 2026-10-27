#!/usr/bin/env python3
"""Draw the paper artifacts from the NEW run's aggregated results.

Called by START_HERE.py after data generation, scoring, and aggregation.
Each figure runs in a fresh process so its Matplotlib defaults cannot leak into
another figure. This matters for exact fonts, line widths and page dimensions.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    out.mkdir(parents=True)
    jobs = [
        (
            "figures/figure1_reliability.py",
            "figure1",
            ["--source-csv", str(args.main_results / "SUMMARY_BY_H_GROUP_ALPHA.csv")],
        ),
        (
            "figures/figure2_fixed_lengths.py",
            "figure2",
            [
                "--source-csv",
                str(args.fixed_results / "SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv"),
            ],
        ),
        (
            "figures/figures3_4_composition.py",
            "figures3_4",
            ["--source-json", str(args.composition)],
        ),
        (
            "tables/make_tables.py",
            "tables",
            [
                "--main-results",
                str(args.main_results),
                "--fixed-results",
                str(args.fixed_results),
            ],
        ),
    ]
    logs = []
    for program, folder, inputs in jobs:
        command = [
            sys.executable,
            str(ROOT / program),
            "--out",
            str(out / folder),
        ] + inputs
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
        logs.append(
            {
                "program": program,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        (out / "BUILD_LOG.json").write_text(json.dumps(logs, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"{program} failed; see {out}/BUILD_LOG.json")
    # A flat figures folder has the exact four names used by main.tex.
    figure_dir = out / "figures"
    figure_dir.mkdir()
    for _, folder, _ in jobs[:3]:
        for path in (out / folder).iterdir():
            if path.suffix in (".pdf", ".png", ".svg"):
                shutil.copy2(path, figure_dir / path.name)
    shutil.copy2(ROOT / "paper_figures.tex", out / "paper_figures.tex")
    from verify_figures import compare

    compare(figure_dir, out / "FIGURE_CHECK.json")
    print(f"Four figures and seven tables built from the supplied results: {out}")


if __name__ == "__main__":
    main()
