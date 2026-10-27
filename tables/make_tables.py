"""Build Tables 1--7 from newly computed results, preserving the author's layout.

Only numerical cells are substituted. Every formatted cell must equal the
submitted manuscript. A disagreement raises an error; nothing is silently
rounded differently, relabelled, or corrected.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def read_table(directory, filename, keys):
    with (directory / filename).open(newline="") as file:
        rows = list(csv.DictReader(file))
    result = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity in result:
            raise ValueError(f"Duplicate scientific key: {identity}")
        result[identity] = row
    return result


def build_rows(main_results, fixed_results):
    summary = read_table(
        main_results,
        "SUMMARY_BY_H_GROUP_ALPHA.csv",
        ("h", "group", "method", "alpha_k"),
    )
    references = read_table(
        main_results, "REFERENCE_AVAILABILITY.csv", ("h", "group", "alpha_k")
    )
    paired = read_table(
        main_results,
        "PAIRED_METHOD_CONTRASTS.csv",
        ("h", "group", "contrast", "alpha_k"),
    )
    fixed = read_table(
        fixed_results,
        "SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv",
        ("h", "L", "method", "alpha_index"),
    )
    diagnostics = read_table(
        main_results, "COVERAGE_DIAGNOSTICS_114.csv", ("h", "group", "alpha_k")
    )
    tables = {number: [] for number in range(1, 8)}
    sources = []

    def main_row(h, group, method, k):
        key = (str(h), group, method, str(k))
        sources.append({"file": "step5/SUMMARY_BY_H_GROUP_ALPHA.csv", "key": key})
        return summary[key]

    def fixed_row(h, length, method, k):
        key = (str(h), str(length), method, str(k))
        sources.append({"file": "step6/SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv", "key": key})
        return fixed[key]

    for h in (3, 6, 9):
        tables[1].append([str(h), "200", str(2**h), "5,000"])

    # Tables 2 and 5: the author's G1 panel precedes the G0 panel.
    for number, levels in ((2, (4, 2)), (5, (1,))):
        for group in ("G1", "G0"):
            methods = (
                ("ochs090", "randomized_iblock")
                if group == "G1"
                else ("ochs090", "count", "ochs050", "randomized_iblock")
            )
            for h in (3, 6, 9):
                for k in levels:
                    records = [main_row(h, group, method, k) for method in methods]
                    cells = [str(h)]
                    if number == 2:
                        cells.append(str(100 - 5 * k))
                    cells.append(f"{int(records[0]['n']):,}")
                    cells.extend(f"{float(row['size_mean']):.3f}" for row in records)
                    tables[number].append(cells)

    for h in (3, 6, 9):
        group1 = references[str(h), "G1", "1"]
        group0 = references[str(h), "G0", "1"]
        sources.extend(
            [
                {"file": "step5/REFERENCE_AVAILABILITY.csv", "key": (h, "G1", 1)},
                {"file": "step5/REFERENCE_AVAILABILITY.csv", "key": (h, "G0", 1)},
            ]
        )
        tables[3].append(
            [
                str(h),
                group1["n"],
                f"{float(group1['N_mean']):.2f}",
                f"{float(group0['N_mean']):.2f}",
                f"{100*float(main_row(h,'G1','ochs090',2)['full_rate']):.2f}",
                f"{100*float(group1['forced_full_rate']):.2f}",
                f"{100*float(main_row(h,'G1','ochs090',1)['full_rate']):.2f}",
            ]
        )
        for length in (10, 50, 100, 200):
            ochs80, block80 = [
                fixed_row(h, length, m, 4) for m in ("ochs090", "randomized_iblock")
            ]
            ochs90, block90 = [
                fixed_row(h, length, m, 2) for m in ("ochs090", "randomized_iblock")
            ]
            tables[4].append(
                [str(h), str(length)]
                + [
                    f"{float(row['mean_size']):.3f}"
                    for row in (ochs80, block80, ochs90, block90)
                ]
                + [f"{100*float(row['coverage']):.2f}" for row in (ochs90, block90)]
            )
        for k in (4, 2, 1):
            key = (str(h), "G0", "ochs090-minus-count", str(k))
            row = paired[key]
            sources.append({"file": "step5/PAIRED_METHOD_CONTRASTS.csv", "key": key})
            cells = [str(h), str(100 - 5 * k)]
            for prefix, factor, decimals in (
                ("size_difference", 1, 4),
                ("normalized_size_difference", 100, 4),
                ("coverage_difference", 100, 3),
            ):
                cells.extend(
                    f"{factor*float(row[prefix+suffix]):+.{decimals}f}"
                    for suffix in ("_mean", "_lo", "_hi")
                )
            tables[6].append(cells)

    selected = [
        row
        for row in diagnostics.values()
        if float(row["p_raw_failure_upper_tail"]) < 0.05
    ]
    selected.sort(key=lambda row: (int(row["h"]), -int(row["alpha_k"])))
    if (
        len(diagnostics) != 114
        or len(selected) != 6
        or any(r["method"] != "ochs090" or r["group"] != "G0" for r in selected)
    ):
        raise ValueError("Unexpected diagnostic family or displayed selection")
    for row in selected:
        sources.append(
            {
                "file": "step5/COVERAGE_DIAGNOSTICS_114.csv",
                "key": (row["h"], row["group"], row["alpha_k"]),
            }
        )
        tables[7].append(
            [
                row["h"],
                str(100 - 5 * int(row["alpha_k"])),
                f"{int(row['n']):,}",
                f"{int(row['misses']):,}",
                f"{100*float(row['coverage']):.2f}",
                f"{100*float(row['mean_finite_rank_coverage_lower_bound']):.2f}",
                f"{float(row['p_raw_failure_upper_tail']):.6f}",
                f"{float(row['p_bonferroni_114']):.6f}",
            ]
        )
    return tables, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--fixed-results", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tables, provenance = build_rows(args.main_results, args.fixed_results)
    checks = []
    for number, numerical_rows in tables.items():
        template = ROOT / "tables/templates" / f"table_{number}.tex"
        lines = template.read_text().splitlines(keepends=True)
        row_number = 0
        for line_number, line in enumerate(lines):
            if not re.match(r"^\s*\d+\s*&", line):
                continue
            actual = NUMBER.findall(line)
            expected = numerical_rows[row_number]
            if actual != expected:
                raise ValueError(
                    f"Table {number}, row {row_number+1}: manuscript={actual}, data={expected}"
                )
            values = iter(expected)
            lines[line_number] = NUMBER.sub(lambda _: next(values), line)
            row_number += 1
        if row_number != len(numerical_rows):
            raise ValueError(f"Table {number}: missing or extra numeric rows")
        generated = "".join(lines).encode()
        if generated != template.read_bytes():
            raise ValueError(f"Table {number}: layout unexpectedly changed")
        (args.out / f"table_{number}.tex").write_bytes(generated)
        checks.append(
            {
                "table": number,
                "numeric_rows": row_number,
                "numeric_cells": sum(map(len, numerical_rows)),
                "byte_identical_to_author_table": True,
                "sha256": hashlib.sha256(generated).hexdigest(),
            }
        )
    (args.out / "TABLE_CHECK.json").write_text(
        json.dumps(
            {"status": "PASS", "tables": checks, "source_rows": provenance}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "tables": 7,
                "numeric_cells": sum(r["numeric_cells"] for r in checks),
            }
        )
    )


if __name__ == "__main__":
    main()
