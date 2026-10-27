#!/usr/bin/env python3
"""Run one complete study from its original seeds (called by START_HERE.py).

Explicitly request --run-full-study. There is no pilot, seed search, or sample
size option. The main study uses 15,000 original scientific addresses; the
fixed-history study uses those same pools and four lengths. No original files
are written. Public names are mapped to unchanged archival IDs internally.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from simulation.settings import PACKAGE, HS, REPLICATES
from simulation.storage import load, save, require, sha
from study_names import LEGACY_STUDY_IDS, STUDY_TITLES


def directory_bytes(folder):
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def consumed_seconds(out):
    """Charge successful work, the archived disk-stop attempt and input replay."""
    records = list((out / "panels").glob("h*/FORMAL_*/RESOURCE_RECORD.json"))
    records += list((out / "recovery_disk_001").rglob("RESOURCE_RECORD.json"))
    return sum(load(path)["elapsed_seconds"] for path in records)


def run_task(command, folder, remaining_seconds, root):
    """Real parent watchdog. No timeout exception is caught and retried."""
    log = root / "logs" / folder.parent.name / (folder.name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    require(not log.exists(), "Existing attempt log; inspect before resuming this task")
    start = time.monotonic()
    peak = 0
    failure = None
    with log.open("x") as output:
        child = subprocess.Popen(
            command, cwd=PACKAGE, stdout=output, stderr=subprocess.STDOUT
        )
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    peak = max(peak, process.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if elapsed > min(120, remaining_seconds):
                    failure = "Wall-time budget exceeded"
                elif peak > 2 * 2**30:
                    failure = "Worker exceeded 2 GiB RSS"
                elif shutil.disk_usage(root).free < 4 * 2**30:
                    failure = "Available disk below 4 GiB"
                if failure:
                    child.kill()
                    child.wait()
                    break
                time.sleep(0.2)
        except BaseException:
            child.kill()
            child.wait()
            raise
    elapsed = time.monotonic() - start
    record = {
        "elapsed_seconds": elapsed,
        "peak_sampled_rss_bytes": peak,
        "returncode": child.returncode,
        "failure": failure,
    }
    save(folder / "RESOURCE_RECORD.json", record)
    require(
        not failure and child.returncode == 0,
        f'Execution stopped: {failure or "worker error"}; input retained',
    )
    return elapsed, directory_bytes(folder) + log.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(LEGACY_STUDY_IDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--main-inputs",
        type=Path,
        help="Main-study /panels folder, required by the fixed-history study",
    )
    parser.add_argument("--run-full-study", action="store_true")
    parser.add_argument(
        "--templates", type=Path, help="Fresh fixed-history exact templates"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue untouched tasks; completed tasks are verified",
    )
    args = parser.parse_args()
    # Only the public label changes. Workers, seed files and saved scientific
    # identities keep the original ID; never rename a random-stream namespace.
    study = LEGACY_STUDY_IDS[args.study]
    require(
        args.run_full_study,
        "No run started. START_HERE.py runs the complete reproduction pipeline.",
    )
    print(STUDY_TITLES[args.study], flush=True)
    out = args.out.resolve()
    require(
        not out.is_relative_to(PACKAGE),
        "Use an output directory outside the publication package",
    )
    budget = (24 if study == "step5" else 8) * 3600
    byte_limit = (12 if study == "step5" else 2) * 2**30
    if not args.resume:
        require(not out.exists(), "Output exists; never replace a run")
        require(
            shutil.disk_usage(out.parent).free >= 8 * 2**30,
            "At least 8 GiB free recommended for a fresh reproduction",
        )
        out.mkdir()
        save(
            out / "REPRODUCTION_IDENTITY.json",
            {
                "study": study,
                "panels_per_h": 5000,
                "h": [3, 6, 9],
                "seed_manifest_sha256": sha(
                    PACKAGE / "protocol" / f"{study}_seed.json"
                ),
                "template_sha256": (
                    template_hashes(args.templates) if study == "step6" else None
                ),
                "code_sha256": {
                    str(p.relative_to(PACKAGE)): sha(p)
                    for p in sorted((PACKAGE / "simulation").rglob("*.py"))
                },
            },
        )
    identity = load(out / "REPRODUCTION_IDENTITY.json")
    require(identity["study"] == study, "Study identity mismatch")
    require(
        identity["seed_manifest_sha256"]
        == sha(PACKAGE / "protocol" / f"{study}_seed.json"),
        "Seed changed",
    )
    require(
        identity["code_sha256"]
        == {
            str(p.relative_to(PACKAGE)): sha(p)
            for p in sorted((PACKAGE / "simulation").rglob("*.py"))
        },
        "Code changed",
    )
    require(
        not list(out.rglob("FAILED.json")),
        "A failed task is present; no automatic retry",
    )
    if study == "step6":
        require(
            identity["template_sha256"] == template_hashes(args.templates),
            "Templates changed",
        )
        require(
            args.main_inputs is not None,
            "The fixed-history study needs complete main-study reproduction output",
        )
        expected = {f"h{h}/FORMAL_{i:06d}" for h in HS for i in range(REPLICATES)}
        actual = {
            str(p.parent.relative_to(args.main_inputs))
            for p in args.main_inputs.glob("h*/FORMAL_*/DONE.json")
        }
        require(
            actual == expected,
            "The fixed-history study requires ALL 15,000 main-study pools",
        )
    elapsed = consumed_seconds(out)
    used_bytes = directory_bytes(out)
    for h in HS:
        for i in range(REPLICATES):
            panel = f"FORMAL_{i:06d}"
            folder = out / "panels" / f"h{h}" / panel
            if (folder / "DONE.json").exists():
                commit = load(folder / "DONE.json")
                require(
                    commit["identity"] == {"study": study, "h": h, "panel": panel},
                    "Completed identity",
                )
                for name, expected_hash in commit["files"].items():
                    require(
                        sha(folder / name) == expected_hash, "Completed data changed"
                    )
                continue
            require(
                elapsed < budget,
                "Cumulative compute budget reached; not reset on resume",
            )
            command = [
                sys.executable,
                "-B",
                "-m",
                "simulation.run_worker",
                "--study",
                study,
                "--h",
                str(h),
                "--panel",
                panel,
                "--out",
                str(out / "panels"),
            ]
            if args.main_inputs:
                command.extend(["--main-inputs", str(args.main_inputs.resolve())])
            if args.templates:
                command.extend(["--templates", str(args.templates.resolve())])
            seconds, written_bytes = run_task(command, folder, budget - elapsed, out)
            elapsed += seconds
            used_bytes += written_bytes
            require(
                used_bytes <= byte_limit, "Output budget exceeded; results retained"
            )
            print(f"{STUDY_TITLES[args.study]} | h={h}, {panel} completed", flush=True)
    save(
        out / "COMPLETE.json",
        {"study": study, "completed_panels": 15000, "elapsed_seconds": elapsed},
    )
    print("Finished. Use collect_reproduction.py to aggregate these saved outputs.")


def template_hashes(folder):
    require(folder is not None, "Missing freshly generated template directory")
    from simulation.settings import LS

    expected = {f"h{h}_L{length}.json.gz" for h in HS for length in LS}
    require(
        {p.name for p in folder.glob("h*_L*.json.gz")} == expected,
        "Expected 12 templates",
    )
    return {name: sha(folder / name) for name in sorted(expected)}


if __name__ == "__main__":
    main()
