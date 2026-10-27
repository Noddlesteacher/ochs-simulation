#!/usr/bin/env python3
"""Run the main study and fixed-history study from seed, then build paper artifacts.

Run this file in your IDE, or type: python START_HERE.py
This is the FULL experiment, not a quick plot of supplied data. Allow hours.
All output goes into a new sibling directory; original files are never changed.
"""

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_environment():
    """Fail early instead of silently using a different scientific environment."""
    if sys.version_info[:3] != (3, 13, 0):
        raise RuntimeError("Use Python 3.13.0, the author's verified version.")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Install the exact requirements.txt versions first.")
    from verify_package import main as verify_package

    verify_package()
    if shutil.disk_usage(ROOT.parent).free < 8 * 2**30:
        raise RuntimeError(
            "STOP: at least 8 GiB free required by this reproduction launcher; no simulation started."
        )


def run(program, *arguments):
    """Run one clearly named stage; propagate any error without retrying it."""
    command = [sys.executable, "-B", str(ROOT / program), *map(str, arguments)]
    print("\nRunning: " + program, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional NEW output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the explicitly approved main-study disk interruption")
    args = parser.parse_args()
    check_environment()
    output = (
        args.out
        or ROOT.parent / ("OCHS_REPRODUCED_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ).resolve()
    if args.resume:
        if args.out is None:
            raise ValueError("Resume requires the existing --out directory")
        from recovery import prepare_resume

        prepare_resume(output)
    elif output.exists() or output.is_relative_to(ROOT):
        raise FileExistsError("Choose a new directory outside the code package.")
    else:
        output.mkdir(parents=True)
    print(
        "Full reproduction: 15,000 main panels and 60,000 fixed-history scenarios.",
        flush=True,
    )
    print("Output: " + str(output), flush=True)
    try:
        # 1. Simulate and score every main-study panel with the original seed.
        run(
            "run_simulation.py",
            "--study",
            "main_study",
            "--out",
            output / "main_study",
            "--run-full-study",
            *(["--resume"] if args.resume else []),
        )

        # 2. Compute exact candidate scores for all 12 fixed histories ourselves.
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "simulation.build_templates",
                "--out",
                str(output / "templates"),
            ],
            cwd=ROOT,
            check=True,
        )

        # 3. Reuse the NEW main-study pools; generate the fixed-history futures
        #    and candidate uniforms with its own original seed.
        run(
            "run_simulation.py",
            "--study",
            "fixed_history_study",
            "--out",
            output / "fixed_history_study",
            "--main-inputs",
            output / "main_study/panels",
            "--templates",
            output / "templates",
            "--run-full-study",
        )

        # 4. Check all newly generated inputs and scientific results. A mismatch
        #    stops the build; original saved results are NEVER substituted.
        run(
            "verify_reproduction.py",
            "--main",
            output / "main_study",
            "--fixed",
            output / "fixed_history_study",
            "--templates",
            output / "templates",
            "--report",
            output / "RAW_CHECK.json",
        )

        # 5. Aggregate the NEW results, including the prespecified repeat-0 sets.
        run(
            "collect_reproduction.py",
            "--run",
            output / "main_study",
            "--out",
            output / "main_study_summary",
        )
        run(
            "collect_reproduction.py",
            "--run",
            output / "fixed_history_study",
            "--out",
            output / "fixed_history_summary",
        )
        run(
            "verify_reproduction.py",
            "--summaries",
            "--main",
            output / "main_study_summary/tables",
            "--fixed",
            output / "fixed_history_summary/tables",
            "--report",
            output / "SUMMARY_CHECK.json",
        )

        # 6. Draw the same four figures and build the same seven LaTeX tables.
        run(
            "render_paper.py",
            "--main-results",
            output / "main_study_summary/tables",
            "--fixed-results",
            output / "fixed_history_summary/tables",
            "--composition",
            output / "fixed_history_summary/RETAINED_PATHS_AND_COUNTS.json",
            "--out",
            output / "paper",
        )
        (output / "FULL_REPRODUCTION_VERIFIED.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "main_panels": 15000,
                    "fixed_history_scenarios": 60000,
                    "checks": [
                        "RAW_CHECK.json",
                        "SUMMARY_CHECK.json",
                        "paper/FIGURE_CHECK.json",
                        "paper/tables/TABLE_CHECK.json",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "\nCompleted and verified. Figures and tables: " + str(output / "paper"),
            flush=True,
        )
    except BaseException as error:
        # A resumed attempt must never overwrite the original failure evidence.
        stop_name = ("PIPELINE_STOPPED_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                     + ".json") if args.resume else "PIPELINE_STOPPED.json"
        (output / stop_name).write_text(
            json.dumps(
                {
                    "status": "STOPPED",
                    "error": repr(error),
                    "action": "All input and output retained. No automatic retry or replacement.",
                },
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
