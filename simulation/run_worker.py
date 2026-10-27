"""One reproducibility task in a child process, with immutable input commits.

The parent enforces wall time and memory. A failure leaves the input and error
record in place. This module does not choose seeds or retry failed tasks.
"""

import argparse
import traceback
from pathlib import Path

from . import generation, iblock, ochs
from .fixed_sources import generate_input
from .fixed_study import work
from .main_study import generate, predict, evaluate
from .settings import A, MODEL, RUN, LS, PACKAGE, frozen_seed
from .storage import load, save, require, sha


def verify_commit(folder, filename):
    commit = load(folder / filename)
    for name, expected in commit["files"].items():
        require(sha(folder / name) == expected, "Committed file changed: " + name)
    return commit


def commit(folder, filename, files, identity):
    save(
        folder / filename,
        {"identity": identity, "files": {name: sha(folder / name) for name in files}},
    )


def run(study, h, panel, output, main_inputs=None, templates_dir=None):
    """Replay a fixed scientific address; never silently skip a corrupt result."""
    folder = output / f"h{h}" / panel
    identity = {"study": study, "h": h, "panel": panel}
    if study == "step6":
        require(templates_dir is not None, "Step6 requires freshly computed templates")
    require(
        h in (3, 6, 9) and panel in {f"FORMAL_{i:06d}" for i in range(5000)},
        "Task identity",
    )
    require(
        not (folder / "FAILED.json").exists(),
        "Failed task needs explicit investigation",
    )
    if (folder / "DONE.json").exists():
        require(
            verify_commit(folder, "DONE.json")["identity"] == identity,
            "Completed task identity",
        )
        return
    if (folder / "INPUT_COMMIT.json").exists():
        require(
            verify_commit(folder, "INPUT_COMMIT.json")["identity"] == identity,
            "Input task identity",
        )
        require(
            not (folder / "prediction.json.gz").exists()
            and not (folder / "result.json.gz").exists(),
            "Partial output exists; do not replace it",
        )
        require(
            not list(folder.glob("*.pending")),
            "Interrupted publication needs investigation",
        )
    else:
        require(
            not folder.exists() or not list(folder.iterdir()),
            "Uncommitted evidence; no automatic replacement",
        )
        folder.mkdir(parents=True, exist_ok=True)
        if study == "step5":
            visible, truth, audit, uniforms = generate(
                h,
                panel,
                frozen_seed("step5"),
                RUN,
                generation,
                iblock,
                failure_folder=folder,
                stage="formal",
            )
            inputs = {
                "visible.json.gz": visible,
                "truth.json.gz": truth,
                "generation.json.gz": audit,
                "uniforms.json.gz": uniforms,
            }
        else:
            require(main_inputs is not None, "Step6 requires ALL Step5 reference pools")
            source = main_inputs / f"h{h}" / panel
            require(
                verify_commit(source, "DONE.json")["identity"]
                == {"study": "step5", "h": h, "panel": panel},
                "Wrong source pool",
            )
            visible = load(source / "visible.json.gz")
            # The original target is never forwarded to fixed-history prediction.
            references = dict(visible)
            references["records"] = visible["records"][1:]
            candidates = [
                row["candidate"]
                for row in load(templates_dir / f"h{h}_L10.json.gz")["rows"]
            ]
            inputs = {
                "references.json.gz": references,
                "input.json.gz": generate_input(
                    h, panel, candidates, frozen_seed("step6"), "formal"
                ),
            }
        for name, value in inputs.items():
            save(folder / name, value)
        commit(folder, "INPUT_COMMIT.json", inputs, identity)

    input_files = tuple(load(folder / "INPUT_COMMIT.json")["files"])
    if study == "step5":
        visible = load(folder / "visible.json.gz")
        require(visible["h"] == h and visible["panel_id"] == panel, "Visible identity")
        uniforms = load(folder / "uniforms.json.gz")
        prediction, _, _ = predict(visible, uniforms["entries"], RUN, ochs, iblock)
        save(folder / "prediction.json.gz", prediction)
        # Target truth enters only after prediction has been saved.
        save(
            folder / "evaluation.json.gz",
            evaluate(prediction, load(folder / "truth.json.gz")),
        )
        output_files = ("prediction.json.gz", "evaluation.json.gz")
    else:
        inputs = load(folder / "input.json.gz")
        require(
            inputs["h"] == h and inputs["pool_id"] == panel,
            "Fixed-history input identity",
        )
        templates = {
            length: load(templates_dir / f"h{h}_L{length}.json.gz") for length in LS
        }
        result, _ = work(load(folder / "references.json.gz"), inputs, templates)
        save(folder / "result.json.gz", result)
        output_files = ("result.json.gz",)
    commit(folder, "DONE.json", input_files + output_files, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("step5", "step6"), required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-inputs", type=Path)
    parser.add_argument("--templates", type=Path)
    args = parser.parse_args()
    folder = args.out / f"h{args.h}" / args.panel
    try:
        run(args.study, args.h, args.panel, args.out, args.main_inputs, args.templates)
    except BaseException:
        if not (folder / "FAILED.json").exists():
            save(folder / "FAILED.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
