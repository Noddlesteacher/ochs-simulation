"""Bounded engineering QA: regenerate only 15 existing addresses per study.

This is NOT the full experiment and is never used to estimate its coverage.
No saved raw data, scores, or templates are needed. Expected hashes are read
only by this parent after the isolated calculating child has exited.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.storage import load, save, sha, require
from verify_reproduction import check_case

CASES = {
    3: (0, 1, 13, 2499, 4999),
    6: (0, 2, 12, 2499, 4999),
    9: (0, 6, 49, 2499, 4999),
}


def bounded(command, log, deadline):
    started = time.monotonic()
    peak = 0
    with log.open("x") as stream:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    peak = max(peak, process.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if (
                    time.monotonic() - started > 120
                    or time.monotonic() > deadline
                    or peak > 2 * 2**30
                ):
                    raise RuntimeError("Bounded QA resource limit reached")
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    require(child.returncode == 0, "QA worker failed; retained log: " + str(log))
    return {"seconds": time.monotonic() - started, "peak_sampled_rss_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--templates",
        type=Path,
        help="Optional templates freshly computed in THIS QA session",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 1800
    completed = []
    try:
        unit = bounded(
            [sys.executable, "-B", "-m", "tests.unit_checks"],
            out / "unit.log",
            deadline,
        )
        templates = args.templates or out / "templates"
        if args.templates is None:
            bounded(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "simulation.build_templates",
                    "--out",
                    str(templates),
                ],
                out / "templates.log",
                deadline,
            )
        require(
            load(templates / "COMPLETE.json")["status"] == "PASS",
            "Incomplete fresh templates",
        )
        expected = load(ROOT / "expected/original_scientific_hashes.json.gz")
        for name, checksum in expected["templates"].items():
            require(sha(templates / name) == checksum, "Template mismatch: " + name)
        for h, ids in CASES.items():
            for repeat in ids:
                panel = f"FORMAL_{repeat:06d}"
                for study in ("step5", "step6"):
                    command = [
                        sys.executable,
                        "-B",
                        str(ROOT / "checks/isolated_worker.py"),
                        "--study",
                        study,
                        "--h",
                        str(h),
                        "--panel",
                        panel,
                        "--out",
                        str(out / study),
                    ]
                    if study == "step6":
                        command += [
                            "--main-inputs",
                            str(out / "step5"),
                            "--templates",
                            str(templates),
                        ]
                    resource = bounded(
                        command, out / f"{study}_h{h}_{panel}.log", deadline
                    )
                    count = check_case(
                        study, h, panel, out / study / f"h{h}" / panel, expected
                    )
                    completed.append(
                        {
                            "study": study,
                            "h": h,
                            "panel": panel,
                            "exact_files": count,
                            **resource,
                        }
                    )
                    print(study, h, panel, "exact match", flush=True)
        save(
            out / "FROM_SEED_CHECK.json",
            {
                "status": "PASS",
                "cases": completed,
                "unit": unit,
                "templates_recomputed_and_hash_matched": 12,
                "scientific_files_exact": sum(r["exact_files"] for r in completed),
                "worker_saved_answer_reads": "forbidden by audit hook",
                "new_random_addresses": 0,
                "scope": "15 existing addresses per study, not all 15000; same-implementation compatibility, not a full independent scoring audit",
            },
        )
    except BaseException as error:
        save(out / "FAILED.json", {"error": repr(error), "completed": completed})
        raise


if __name__ == "__main__":
    main()
