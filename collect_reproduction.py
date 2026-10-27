#!/usr/bin/env python3
"""Aggregate newly simulated panels. No original scientific data are read."""

import argparse
import json
from pathlib import Path

import numpy as np

from simulation.settings import HS, LS, METHODS, A
from simulation.ochs import legal_candidates
from simulation.storage import load, save, require, sha
from analysis.aggregate_main import aggregate as aggregate_main
from analysis.aggregate_fixed import aggregate as aggregate_fixed


def collect_main(root, destination):
    records = []
    for h in HS:
        for repeat in range(5000):
            panel = f"FORMAL_{repeat:06d}"
            folder = root / "panels" / f"h{h}" / panel
            check_commit(folder, "step5", h, panel)
            evaluation = load(folder / "evaluation.json.gz")
            prediction = load(folder / "prediction.json.gz")["ochs"]
            generation = load(folder / "generation.json.gz")
            indexed = {(r["method"], r["alpha"][0]): r for r in evaluation["rows"]}
            require(
                len(indexed) == len(evaluation["rows"]) == 84, "Evaluation row identity"
            )
            values = [
                [
                    [indexed[m, k]["covered"], indexed[m, k]["set_size"]]
                    for k in range(21)
                ]
                for m in METHODS
            ]
            episodes = generation["episodes"]
            records.append(
                {
                    "h": h,
                    "panel": panel,
                    "group": prediction["group"],
                    "N": prediction["N"],
                    "target_T": prediction["T"],
                    "country_K_sum": sum(e["K"] for e in episodes),
                    "country_K_sum_squares": sum(e["K"] ** 2 for e in episodes),
                    "coverage_and_size": values,
                }
            )
    path = destination / "PANEL_SUMMARIES.json.gz"
    save(path, {"methods": list(METHODS), "alpha_denominator": 20, "rows": records})
    aggregate_main(path, destination / "tables")


def collect_fixed(root, destination):
    evaluation = np.zeros((3, 5000, 4, 2, 21, 2), dtype=np.uint16)
    references = np.zeros((3, 5000), dtype=np.uint16)
    for horizon_index, h in enumerate(HS):
        candidates = [list(y) for y in legal_candidates([1], h, A)]
        for repeat in range(5000):
            panel = f"FORMAL_{repeat:06d}"
            folder = root / "panels" / f"h{h}" / panel
            check_commit(folder, "step6", h, panel)
            result = load(folder / "result.json.gz")
            inputs = load(folder / "input.json.gz")
            truth = candidates.index(inputs["future"])
            references[horizon_index, repeat] = result["N"]
            for length_index, length in enumerate(LS):
                for method_index, method in enumerate(("ochs090", "randomized_iblock")):
                    for k, hex_bits in enumerate(
                        result["lengths"][str(length)]["members"][method]
                    ):
                        bits = int(hex_bits, 16)
                        calculated = [int(bool(bits & (1 << truth))), bits.bit_count()]
                        require(
                            calculated
                            == result["evaluation"][length_index][method_index][k],
                            "Member/evaluation mismatch",
                        )
                        evaluation[
                            horizon_index, repeat, length_index, method_index, k
                        ] = calculated
    path = destination / "EVALUATION_ARRAYS.npz"
    with path.open("xb") as file:
        np.savez_compressed(
            file,
            evaluation=evaluation,
            N=references,
            horizons=np.array(HS),
            lengths=np.array(LS),
        )
    aggregate_fixed(path, destination / "tables")
    collect_composition(root, destination / "RETAINED_PATHS_AND_COUNTS.json")


def collect_composition(root, destination):
    """Figures 3--4: the prespecified h=6, repeat-0 actual sets, not a chosen example."""
    folder = root / "panels/h6/FORMAL_000000"
    check_commit(folder, "step6", 6, "FORMAL_000000")
    result = load(folder / "result.json.gz")
    candidates = [list(y) for y in legal_candidates([1], 6, A)]
    panels = []
    for nominal, k in ((80, 4), (90, 2)):
        for method in ("ochs090", "randomized_iblock"):
            for length in LS:
                hex_bits = result["lengths"][str(length)]["members"][method][k]
                bits = int(hex_bits, 16)
                indices = [j for j in range(len(candidates)) if bits & (1 << j)]
                paths = [candidates[j] for j in indices]
                panels.append(
                    {
                        "nominal_percent": nominal,
                        "alpha_index": k,
                        "method": method,
                        "L_star": length,
                        "set_size": len(paths),
                        "membership_hex": hex_bits,
                        "retained_indices_zero_based": indices,
                        "retained_paths": paths,
                        "counts": [
                            [sum(y[ell] == s for y in paths) for s in range(1, 5)]
                            for ell in range(6)
                        ],
                        "status": "PASS",
                        "source_pool": "FORMAL_000000",
                        "N_star": result["N"],
                    }
                )
    with destination.open("x") as file:
        file.write(json.dumps(panels, indent=2) + "\n")


def check_commit(folder, study, h, panel):
    record = load(folder / "DONE.json")
    require(
        record["identity"] == {"study": study, "h": h, "panel": panel},
        "Commit identity",
    )
    for name, expected in record["files"].items():
        require(sha(folder / name) == expected, "Modified committed file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = load(args.run / "COMPLETE.json")
    require(
        status["completed_panels"] == 15000,
        "Partial run cannot replace complete paper results",
    )
    expected = {f"h{h}/FORMAL_{i:06d}" for h in HS for i in range(5000)}
    actual = {
        str(p.parent.relative_to(args.run / "panels"))
        for p in (args.run / "panels").glob("h*/FORMAL_*/DONE.json")
    }
    require(actual == expected, "Missing or extra completed tasks")
    args.out.mkdir(parents=True, exist_ok=False)
    if status["study"] == "step5":
        collect_main(args.run, args.out)
    else:
        collect_fixed(args.run, args.out)


if __name__ == "__main__":
    main()
