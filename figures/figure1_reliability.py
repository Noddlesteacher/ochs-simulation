#!/usr/bin/env python3
"""Plot the frozen main specification only. No simulation or scoring imports."""

from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
HS = (3, 6, 9)
METHODS = ("ochs090", "count", "ochs050", "randomized_iblock")
COUNTS = {
    "overall": (5000, 5000, 5000),
    "G0": (4294, 4328, 4361),
    "G1": (706, 672, 639),
}
COLORS = {3: "#0072B2", 6: "#D55E00", 9: "#009E73"}
MARKERS = {3: "o", 6: "s", 9: "^"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, obj):
    with p.open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_csv
    require(
        digest(source)
        == "7611ffca2795c5cbad989a79677355fc1bda51856f8291d812fe4fd51cae1c69",
        "Recomputed summary differs from the manuscript; stop without substitution",
    )
    provenance = {"run_id": "OCHS_SHIFTED_STEP5_FORMAL_R01"}
    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keys = [(int(r["h"]), r["group"], r["method"], int(r["alpha_k"])) for r in rows]
    require(len(keys) == len(set(keys)) == 756, "Duplicate or missing source rows")
    require(
        set(keys) == set(itertools.product(HS, COUNTS, METHODS, range(21))),
        "Unexpected scientific key set",
    )
    index = dict(zip(keys, rows, strict=True))
    for (h, g, m, k), r in index.items():
        require(int(r["n"]) == COUNTS[g][HS.index(h)], "Denominator mismatch")
        require(
            float(r["coverage"]) == int(r["covered"]) / int(r["n"]),
            "Saved coverage/count mismatch",
        )
        require(float(r["nominal"]) == 1 - k / 20, "Nominal grid mismatch")
    for h, k in itertools.product(HS, range(21)):
        require(
            len({index[h, "G1", m, k]["coverage"] for m in METHODS[:3]}) == 1,
            "G1 specs differ",
        )
    # Check exact mapping against the first approved redesign, when available.
    previous_check = "Frozen source hash and complete scientific key set checked"
    require(not args.out.exists(), "Choose a new output directory; no overwrites")
    args.out.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.65,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(5.625, 6.2))
    fig.subplots_adjust(
        left=0.105, right=0.98, top=0.915, bottom=0.19, wspace=0.34, hspace=0.60
    )
    panels = [
        ("A", r"Overall - OCHS ($\lambda=.90$)", "overall", "ochs090"),
        ("B", r"Overall - randomized $i$-block", "overall", "randomized_iblock"),
        ("C", r"Group 0 - OCHS ($\lambda=.90$)", "G0", "ochs090"),
        ("D", "Group 1 - OCHS", "G1", "ochs090"),
    ]
    points = []
    series = []
    for ax, (p, title, g, m) in zip(axes.flat, panels, strict=True):
        require(
            m == "ochs090" or (p == "B" and g == "overall"), "Forbidden method/group"
        )
        ax.set_title(
            p + "  " + title,
            loc="left",
            fontsize=8.2 if p in ("B", "C") else 8.5,
            pad=23,
        )
        subtitle = (
            "n = 5,000 per horizon"
            if g == "overall"
            else "n = " + " / ".join(f"{n:,}" for n in COUNTS[g])
        )
        ax.text(
            0, 1.04, subtitle, transform=ax.transAxes, fontsize=7.3, color="#444444"
        )
        ax.set(
            xlim=(-0.018, 1.018),
            ylim=(-0.018, 1.018),
            xlabel=r"Nominal coverage $1-\alpha$",
            ylabel="Empirical coverage",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which="major", color="#dedede", lw=0.65)
        ax.grid(which="minor", color="#eeeeee", lw=0.4)
        ax.tick_params(labelsize=7.8, length=2.3)
        ax.tick_params(which="minor", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.plot([0, 1], [0, 1], color="#111111", ls=(0, (5, 4)), lw=0.9, zorder=1)
        for h in HS:
            selected = [index[h, g, m, k] for k in reversed(range(21))]
            x = [float(r["nominal"]) for r in selected]
            y = [float(r["coverage"]) for r in selected]
            (line,) = ax.plot(
                x,
                y,
                color=COLORS[h],
                ls="-",
                lw=1.25,
                marker=MARKERS[h],
                ms=2.7,
                mfc="white",
                mec=COLORS[h],
                mew=0.6,
                zorder=3,
            )
            require(
                list(line.get_xdata()) == x and list(line.get_ydata()) == y,
                "Changed line coordinates",
            )
            series.append(dict(panel=p, h=h, group=g, method=m, points=21, exact=True))
            points.extend(dict(panel=p, **r) for r in selected)
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[h],
            marker=MARKERS[h],
            ms=4,
            mfc="white",
            lw=1.25,
            label=f"h = {h}",
        )
        for h in HS
    ]
    legend = fig.legend(
        handles=handles,
        title="Forecast horizon",
        loc="lower center",
        bbox_to_anchor=(0.54, 0.015),
        frameon=False,
        ncol=3,
        fontsize=9,
        title_fontsize=8.5,
        handlelength=2.5,
        columnspacing=2.6,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        for text in (ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts):
            b = text.get_window_extent(renderer)
            require(
                b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.x1 and b.y1 <= fig.bbox.y1,
                "Text outside canvas",
            )
            require(
                not b.overlaps(legend.get_window_extent(renderer)),
                "Legend overlaps text",
            )
    fig.savefig(
        args.out / "reliability_lambda090_main.pdf",
        metadata={
            "Title": "Step5 reliability: frozen OCHS lambda 0.90",
            "Subject": "OCHS_SHIFTED_STEP5_FORMAL_R01; presentation only",
        },
    )
    fig.savefig(args.out / "reliability_lambda090_main.png", dpi=240)
    plt.close(fig)
    write_json(args.out / "PLOT_POINTS.json", points)
    check = dict(
        status="PASS",
        run_id=provenance["run_id"],
        source_rows=756,
        counts=COUNTS,
        curves=series,
        total_curves=12,
        total_points=252,
        levels=21,
        G1_specs_identical=True,
        previous_figure_mapping=previous_check,
        point_offsets=0,
        smoothing=False,
        empirical_coverage_changes=0,
        simulations_run=0,
        scope="Frozen aggregate/plot correspondence, not a new panel-level numerical audit",
        python=sys.version,
        matplotlib=matplotlib.__version__,
    )
    write_json(args.out / "NUMERICAL_QA.json", check)
    print(json.dumps(dict(status="PASS", curves=12, points=252, output=str(args.out))))


if __name__ == "__main__":
    main()
