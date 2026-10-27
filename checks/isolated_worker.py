"""Test-only read guard: a computing worker cannot read the answer directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def refuse_answer_reads(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        path = Path(arguments[0]).resolve()
        if path.is_relative_to(ROOT / "expected") or path.is_relative_to(ROOT / "data"):
            raise AssertionError(
                "Computing worker attempted to read saved answers: " + str(path)
            )


sys.addaudithook(refuse_answer_reads)
from simulation.run_worker import main

main()
