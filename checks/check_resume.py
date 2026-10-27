"""Check completed-task resume and rejection of corrupted QA copies only."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation import run_worker
from simulation.storage import save, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks = []
    for study in ("step5", "step6"):
        folder = args.replay / study / "h3/FORMAL_000000"
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with patch.object(
            run_worker, "generate", side_effect=AssertionError("Unexpected generation")
        ), patch.object(
            run_worker, "generate_input", side_effect=AssertionError("Unexpected RNG")
        ), patch.object(
            run_worker, "predict", side_effect=AssertionError("Unexpected scoring")
        ), patch.object(
            run_worker, "work", side_effect=AssertionError("Unexpected scoring")
        ):
            run_worker.run(
                study,
                3,
                "FORMAL_000000",
                args.replay / study,
                args.replay / "step5",
                args.templates,
            )
        require(
            before == {p.name: p.read_bytes() for p in folder.iterdir()},
            "Resume modified output",
        )
        checks.append(study + ": completed resume reads/verifies only")
        for mutation in ("h", "panel", "input", "result", "failed"):
            root = args.out / (study + "_" + mutation)
            copy = root / "h3/FORMAL_000000"
            shutil.copytree(folder, copy)
            if mutation in ("h", "panel"):
                commit = json.loads((copy / "DONE.json").read_text())
                commit["identity"][mutation] = 6 if mutation == "h" else "FORMAL_000001"
                (copy / "DONE.json").write_text(json.dumps(commit))
            elif mutation in ("input", "result"):
                name = (
                    ("visible.json.gz" if study == "step5" else "input.json.gz")
                    if mutation == "input"
                    else (
                        "prediction.json.gz" if study == "step5" else "result.json.gz"
                    )
                )
                with (copy / name).open("ab") as file:
                    file.write(b"DELIBERATE_QA_CORRUPTION")
            else:
                save(copy / "FAILED.json", {"deliberate_negative_test": True})
            with patch.object(
                run_worker,
                "generate",
                side_effect=AssertionError("Unexpected generation"),
            ), patch.object(
                run_worker,
                "generate_input",
                side_effect=AssertionError("Unexpected RNG"),
            ):
                try:
                    run_worker.run(
                        study,
                        3,
                        "FORMAL_000000",
                        root,
                        args.replay / "step5",
                        args.templates,
                    )
                except ValueError:
                    checks.append(study + ": rejects " + mutation)
                else:
                    raise AssertionError("Invalid completed task accepted")
    save(
        args.out / "RESUME_CHECK.json",
        {
            "status": "PASS",
            "checks": checks,
            "scope": "Completed-task reuse and deliberately corrupted copies of existing QA results; no new random addresses",
        },
    )


if __name__ == "__main__":
    main()
