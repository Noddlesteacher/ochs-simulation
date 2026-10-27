"""One explicitly approved disk-interruption recovery; no scientific changes.

The original attempt is moved, never deleted, into recovery_disk_001/archived.
We replay the SAME input address, compare exact bytes, and only then publish a
new INPUT_COMMIT. The unchanged worker does all subsequent scoring. This is not
a general retry policy: any different failure, changed file, or second recovery
attempt stops for inspection.
"""

import argparse
from pathlib import Path
import shutil
import sys

import psutil

from simulation.settings import PACKAGE, HS, REPLICATES, RUN, frozen_seed
from simulation.storage import load, save, sha, require

APPROVAL = PACKAGE / "recovery_approval.json"
RECOVERY_DIRECTORY = "recovery_disk_001"
INPUT_NAMES = ("visible.json.gz", "truth.json.gz", "generation.json.gz",
               "uniforms.json.gz")
COMPUTE_LIMIT = 24 * 3600
OUTPUT_LIMIT = 12 * 2**30


def approved_panel(run):
    approval = load(APPROVAL)
    return run / "main_study/panels" / f"h{approval['h']}" / approval["panel"]


def verify_evidence(run):
    """Check every byte recorded BEFORE the repair was written."""
    approval = load(APPROVAL)
    require(run.name == approval["run_name"], "Not the approved interrupted run")
    for relative, row in approval["evidence"].items():
        path = run / relative
        require(path.is_file() and not path.is_symlink(), "Missing evidence: " + relative)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Changed interruption evidence: " + relative)
    folder = approved_panel(run)
    allowed = {"visible.json.gz", "truth.json.gz", "generation.json.gz",
               "uniforms.json.gz.pending", "RESOURCE_RECORD.json"}
    require({p.name for p in folder.iterdir()} == allowed,
            "Unexpected partial task files; no automatic repair")
    record = load(folder / "RESOURCE_RECORD.json")
    require(record["failure"] == "Available disk below 4 GiB"
            and record["returncode"] == -9, "Only the approved disk stop is recoverable")
    return approval


def check_no_other_runner(run):
    """Do not resume while another launcher/worker owns this output."""
    current = psutil.Process()
    own_family = {current.pid, *(p.pid for p in current.parents())}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] in own_family:
            continue
        command = " ".join(process.info["cmdline"] or [])
        if str(run) in command and any(name in command for name in
                ("START_HERE.py", "RESUME_HERE.py", "run_simulation.py", "simulation.run_worker")):
            raise ValueError("Another reproduction process uses this output; do not start twice")


def inspect_run(run):
    """Read-only full audit of saved completed tasks, identity and resource gates."""
    run = Path(run).resolve()
    approval = verify_evidence(run)
    require(not run.is_relative_to(PACKAGE), "Output cannot be inside the code package")
    check_no_other_runner(run)
    study = run / "main_study"
    require(not (study / RECOVERY_DIRECTORY).exists(),
            "Recovery already attempted; inspect its evidence before any further action")
    require(not (study / "COMPLETE.json").exists(), "Main study already complete")
    for name in ("templates", "fixed_history_study", "main_study_summary",
                 "fixed_history_summary", "paper", "FULL_REPRODUCTION_VERIFIED.json"):
        require(not (run / name).exists(), "Later-stage output exists: " + name)
    require(not list(study.rglob("FAILED.json")), "A failed task exists; no automatic retry")
    require(set(study.rglob("*.pending")) == {approved_panel(run) / "uniforms.json.gz.pending"},
            "Unexpected interrupted file")
    identity = load(study / "REPRODUCTION_IDENTITY.json")
    require(identity["study"] == "step5" and identity["h"] == list(HS)
            and identity["panels_per_h"] == REPLICATES, "Run identity mismatch")
    require(identity["seed_manifest_sha256"] == sha(PACKAGE / "protocol/step5_seed.json"),
            "Frozen seed changed")
    require(identity["code_sha256"] == {
        str(p.relative_to(PACKAGE)): sha(p) for p in sorted((PACKAGE / "simulation").rglob("*.py"))
    }, "Scientific source changed")

    expected = load(PACKAGE / "expected/original_scientific_hashes.json.gz")["studies"]["step5"]
    expected_keys = {f"h{h}/FORMAL_{i:06d}" for h in HS for i in range(REPLICATES)}
    require(set(expected) == expected_keys, "Frozen hash evidence has wrong task IDs")
    complete_keys = {f"h{h}/FORMAL_{i:06d}" for h in HS
                     for i in range(approval["completed_counts"][str(h)])}
    partial_key = f"h{approval['h']}/{approval['panel']}"
    panels = study / "panels"
    require({p.name for p in panels.iterdir()} == {f"h{h}" for h in HS}, "Unexpected horizon")
    actual_keys = {str(p.relative_to(panels)) for p in panels.glob("h*/*")}
    require(actual_keys == complete_keys | {partial_key}, "Missing/extra panel directory")

    checked_files = 0
    elapsed = 0.0
    for number, key in enumerate(sorted(complete_keys), 1):
        folder = panels / key
        h = int(folder.parent.name[1:])
        expected_identity = {"study": "step5", "h": h, "panel": folder.name}
        files = expected[key]
        require(set(files) == set(INPUT_NAMES) | {"prediction.json.gz", "evaluation.json.gz"},
                "Incomplete frozen scientific hash keys")
        require({p.name for p in folder.iterdir()} == set(files) |
                {"INPUT_COMMIT.json", "DONE.json", "RESOURCE_RECORD.json"},
                "Unexpected completed-task files: " + key)
        for name, checksum in files.items():
            require(sha(folder / name) == checksum, "Scientific file mismatch: " + key + "/" + name)
            checked_files += 1
        for commit_name, names in (("INPUT_COMMIT.json", set(INPUT_NAMES)), ("DONE.json", set(files))):
            commit = load(folder / commit_name)
            require(commit["identity"] == expected_identity, "Commit identity mismatch: " + key)
            require(commit["files"] == {name: files[name] for name in names},
                    "Commit hashes mismatch: " + key)
        record = load(folder / "RESOURCE_RECORD.json")
        require(record["failure"] is None and record["returncode"] == 0,
                "Completed task has failed resource record: " + key)
        require(0 <= record["elapsed_seconds"] <= 120
                and record["peak_sampled_rss_bytes"] <= 2 * 2**30,
                "Completed task exceeded resource limit: " + key)
        elapsed += record["elapsed_seconds"]
        if number % 1000 == 0:
            print(f"Verified {number:,}/{len(complete_keys):,} saved tasks (no recomputation)", flush=True)
    elapsed += load(approved_panel(run) / "RESOURCE_RECORD.json")["elapsed_seconds"]
    output_bytes = sum(p.stat().st_size for p in study.rglob("*") if p.is_file())
    free = shutil.disk_usage(run).free
    require(elapsed < COMPUTE_LIMIT, "Cumulative 24-hour budget exhausted")
    require(output_bytes < OUTPUT_LIMIT, "12 GiB output limit reached")
    require(free >= 8 * 2**30, "At least 8 GiB free required before resuming")
    return {"status": "PASS", "completed_panels": len(complete_keys),
            "per_h_completed": approval["completed_counts"], "scientific_files_checked": checked_files,
            "comparison": "Exact compressed scientific SHA-256 against original frozen hashes",
            "elapsed_seconds_including_killed_attempt": elapsed,
            "remaining_main_compute_seconds": COMPUTE_LIMIT - elapsed,
            "current_output_bytes": output_bytes, "free_bytes": free,
            "incomplete_task": partial_key, "new_simulations": 0}


def reconstruct_input(run, destination):
    """Replay exactly one frozen input; do not compute any predictions or metrics."""
    approval = verify_evidence(run)
    require(not destination.exists(), "Input replay destination already exists")
    from simulation import generation, iblock
    from simulation.main_study import generate
    from simulation.run_worker import commit

    h, panel = approval["h"], approval["panel"]
    visible, truth, audit, uniforms = generate(
        h, panel, frozen_seed("step5"), RUN, generation, iblock, stage="formal")
    values = dict(zip(INPUT_NAMES, (visible, truth, audit, uniforms)))
    expected = load(PACKAGE / "expected/original_scientific_hashes.json.gz")["studies"]["step5"][f"h{h}/{panel}"]
    for name, value in values.items():
        save(destination / name, value)
        require(sha(destination / name) == expected[name], "Frozen input replay mismatch: " + name)
        existing = approved_panel(run) / name
        if existing.exists():
            require(existing.read_bytes() == (destination / name).read_bytes(),
                    "Saved input disagrees with deterministic replay: " + name)
    commit(destination, "INPUT_COMMIT.json", INPUT_NAMES,
           {"study": "step5", "h": h, "panel": panel})
    print("PASS: original input bytes and all candidate uniforms reproduced exactly; no scoring", flush=True)


def publish_prepared(run, area):
    """Preserve original bytes and log before placing the validated input commit."""
    approval = verify_evidence(run)
    prepared = area / "prepared"
    from simulation.run_worker import verify_commit
    expected_identity = {"study": "step5", "h": approval["h"], "panel": approval["panel"]}
    commit = verify_commit(prepared, "INPUT_COMMIT.json")
    require(commit["identity"] == expected_identity and set(commit["files"]) == set(INPUT_NAMES),
            "Prepared input identity or file set mismatch")
    expected = load(PACKAGE / "expected/original_scientific_hashes.json.gz")["studies"]["step5"][f"h{approval['h']}/{approval['panel']}"]
    require(commit["files"] == {name: expected[name] for name in INPUT_NAMES},
            "Prepared input differs from frozen hashes")
    require({p.name for p in prepared.iterdir()} == set(INPUT_NAMES) | {"INPUT_COMMIT.json"},
            "Unexpected prepared output; never accept precomputed prediction")
    archive = area / "archived"
    archive.mkdir(exist_ok=False)
    original = approved_panel(run)
    log = run / "main_study/logs" / f"h{approval['h']}" / (approval["panel"] + ".log")
    # Same-volume renames preserve every original byte, including the empty .pending.
    original.rename(archive / "panel")
    log.rename(archive / "worker.log")
    prepared.rename(original)
    saved_evidence = {}
    for relative, row in approval["evidence"].items():
        if relative.startswith(f"main_study/panels/h{approval['h']}/{approval['panel']}/"):
            path = archive / "panel" / Path(relative).name
        elif relative.endswith(".log"):
            path = archive / "worker.log"
        else:
            path = run / relative
        require(sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
                "Preserved evidence mismatch: " + relative)
        saved_evidence[str(path.relative_to(run))] = row
    save(area / "RECOVERY_RECEIPT.json", {
        "status": "INPUT_RECOVERED_NOT_YET_SCORED", "identity": expected_identity,
        "preserved_evidence": saved_evidence, "input_files": commit["files"],
        "seed_changed": False, "completed_tasks_recomputed": 0,
        "budget_policy": "Original failed attempt and replay resources remain charged",
    })


def prepare_resume(run):
    """Called only by an explicit user launch; preflight failures do not touch the run."""
    from run_simulation import run_task

    report = inspect_run(run)
    area = run / "main_study" / RECOVERY_DIRECTORY
    area.mkdir(exist_ok=False)
    save(area / "PREFLIGHT.json", report)
    command = [sys.executable, "-B", str(PACKAGE / "recovery.py"),
               "--replay-input", "--run", str(run), "--out", str(area / "prepared")]
    run_task(command, area / "replay", report["remaining_main_compute_seconds"], area)
    publish_prepared(run, area)
    print("Recovery prepared. Continuing the SAME run; completed panels are skipped after verification.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-input", action="store_true")
    args = parser.parse_args()
    if args.replay_input:
        reconstruct_input(args.run.resolve(), args.out.resolve())
    else:
        # Inspection is read-only; its report must be outside the interrupted run.
        require(not args.out.resolve().is_relative_to(args.run.resolve()),
                "Read-only audit report must be outside the interrupted run")
        save(args.out, inspect_run(args.run))


if __name__ == "__main__":
    main()
