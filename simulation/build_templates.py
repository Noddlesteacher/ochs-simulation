"""Compute all 12 exact fixed-history templates from the scoring functions.

These are deterministic calculations, not saved experimental inputs. No RNG is
used. Each child computes every admissible path for one (h, L) combination.
"""

import argparse
import sys
import time
from pathlib import Path

from .fixed_prediction import template
from .settings import HS, LS
from .storage import load, save, sha, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--one", type=int, nargs=2, metavar=("H", "L"))
    args = parser.parse_args()
    if args.one:
        h, length = args.one
        save(args.out / f"h{h}_L{length}.json.gz", template(h, length))
        return

    from run_simulation import run_task

    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    checks = []
    for h in HS:
        for length in LS:
            folder = args.out / "resources" / f"h{h}_L{length}"
            command = [
                sys.executable,
                "-B",
                "-m",
                "simulation.build_templates",
                "--out",
                str(args.out.resolve()),
                "--one",
                str(h),
                str(length),
            ]
            seconds, _ = run_task(
                command, folder, 1800 - (time.monotonic() - start), args.out
            )
            path = args.out / f"h{h}_L{length}.json.gz"
            result = load(path)
            require(result["h"] == h and result["L"] == length, "Template identity")
            require(len(result["rows"]) == 2**h, "Incomplete candidate space")
            checks.append(
                {
                    "h": h,
                    "L": length,
                    "candidates": 2**h,
                    "seconds": seconds,
                    "sha256": sha(path),
                }
            )
            print(f"Exact template: h={h}, L={length}, {2**h} candidates", flush=True)
    save(
        args.out / "COMPLETE.json",
        {"status": "PASS", "templates": checks, "random_draws": 0},
    )


if __name__ == "__main__":
    main()
