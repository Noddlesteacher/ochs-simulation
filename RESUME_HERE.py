#!/usr/bin/env python3
"""Continue the approved disk-interrupted run, then finish all paper checks.

Open this file and use Run Python File, or run: python -B RESUME_HERE.py
This starts a long computation. It does not start a new seed or a new study.
For a genuinely NEW reproduction use START_HERE.py instead.
"""

from pathlib import Path
import sys

from simulation.storage import load
import START_HERE


def main():
    root = Path(__file__).resolve().parent
    run = root.parent / load(root / "recovery_approval.json")["run_name"]
    print("Continue existing output: " + str(run), flush=True)
    sys.argv = [str(root / "START_HERE.py"), "--resume", "--out", str(run)]
    START_HERE.main()


if __name__ == "__main__":
    main()
