#!/usr/bin/env python3
"""Presentation-only revision: neutral State 1-4 labels and readable Set size.

Run: python figures/figures3_4_composition.py --out NEW_EMPTY_DIRECTORY
Reads composition derived from the newly computed repeat-0 prediction sets.
No simulation, scoring, RNG, new membership or scientific imports.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

EXPECTED_DATA_SHA256 = (
    "cb3694eade75cf431a7a9f8e1e255383010e878e4f6622c37147734c66deb08b"
)
STATES = {s: f"State {s}" for s in range(1, 5)}
COLORS = {1: "#4C78A8", 2: "#F2A541", 3: "#C85A5A", 4: "#5B9B78"}
LS = (10, 50, 100, 200)
METHODS = ("ochs090", "randomized_iblock")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_path = args.source_json
    require(
        hashlib.sha256(data_path.read_bytes()).hexdigest() == EXPECTED_DATA_SHA256,
        "Recomputed composition differs from the manuscript; no substitution",
    )
    panels = json.loads(data_path.read_text())
    require(len(panels) == 16, "Expected 16 panels")
    look = {}
    for p in panels:
        key = (p["nominal_percent"], p["method"], p["L_star"])
        require(key not in look, "Duplicate panel key")
        look[key] = p
        require(
            p["source_pool"] == "FORMAL_000000" and p["N_star"] == 33, "Wrong example"
        )
        require(p["set_size"] == len(p["retained_paths"]), "Wrong set size")
        for ell in range(6):
            for state in STATES:
                count = sum(y[ell] == state for y in p["retained_paths"])
                require(count == p["counts"][ell][state - 1], "Changed composition")
            require(
                sum(Fraction(c, p["set_size"]) for c in p["counts"][ell]) == 1,
                "Composition must sum exactly to 100 percent",
            )
    require(
        set(look) == {(n, m, L) for n in (80, 90) for m in METHODS for L in LS},
        "Missing or extra panel",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ochs-step6-neutral-labels-v2",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for nominal in (80, 90):
        stem = f"step6_h6_state_composition_nominal{nominal}"
        for ext in ("pdf", "png", "svg"):
            require(
                not (out / f"{stem}.{ext}").exists(),
                "Refusing to overwrite an existing figure",
            )
        fig = plt.figure(figsize=(405 / 72, 3.5), facecolor="white")
        fig.text(
            0.515,
            0.982,
            "State composition of retained prediction paths",
            ha="center",
            va="top",
            fontsize=10.5,
        )
        fig.text(
            0.515,
            0.926,
            f"h = 6  |  nominal coverage = {nominal}%  |  fixed all-state-1 histories",
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.text(
            0.018,
            0.465,
            "State composition (%)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
        )
        left, right, gap = 0.112, 0.992, 0.029
        width = (right - left - 3 * gap) / 4
        for col, L in enumerate(LS):
            fig.text(
                left + col * (width + gap) + width / 2,
                0.863,
                f"L* = {L}",
                fontsize=9,
                ha="center",
                va="center",
            )
        for row, method in enumerate(METHODS):
            bottom = 0.538 if row == 0 else 0.151
            fig.text(
                left,
                0.816 if row == 0 else 0.429,
                "OCHS" if row == 0 else "Randomized i-block",
                fontsize=8.5,
                fontweight="bold",
                va="center",
            )
            for col, L in enumerate(LS):
                p = look[nominal, method, L]
                ax = fig.add_axes([left + col * (width + gap), bottom, width, 0.215])
                ax.set_ylim(0, 100)
                ax.set_xlim(0.45, 6.55)
                ax.set_xticks(range(1, 7), [f"T+{ell}" for ell in range(1, 7)])
                ax.tick_params(axis="x", labelsize=7, length=0, pad=3)
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=55,
                    ha="right",
                    rotation_mode="anchor",
                )
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.tick_params(
                    axis="y",
                    labelsize=7,
                    length=2,
                    width=0.45,
                    pad=2,
                    labelleft=(col == 0),
                )
                ax.spines["left"].set_linewidth(0.45)
                ax.spines["bottom"].set_linewidth(0.45)
                ax.grid(axis="y", color="#dddddd", linewidth=0.4)
                ax.set_axisbelow(True)
                ax.set_title(f"Set size = {p['set_size']}", fontsize=8, pad=4)
                lower = [0.0] * 6
                for state in STATES:
                    values = [100 * c[state - 1] / p["set_size"] for c in p["counts"]]
                    ax.bar(
                        range(1, 7),
                        values,
                        bottom=lower,
                        width=0.79,
                        color=COLORS[state],
                        linewidth=0,
                    )
                    lower = [b + v for b, v in zip(lower, values)]
        fig.legend(
            handles=[Patch(facecolor=COLORS[s], label=STATES[s]) for s in STATES],
            loc="lower center",
            bbox_to_anchor=(0.51, 0.003),
            ncol=4,
            frameon=False,
            fontsize=7.5,
            handlelength=1.1,
            handletextpad=0.4,
            columnspacing=1.8,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(renderer)
                require(
                    box.x0 >= -0.5
                    and box.y0 >= -0.5
                    and box.x1 <= fig.bbox.x1 + 0.5
                    and box.y1 <= fig.bbox.y1 + 0.5,
                    f"Text clipping: {text.get_text()}",
                )
        fig.savefig(
            out / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
        fig.savefig(out / f"{stem}.svg", metadata={"Date": None})
        fig.savefig(out / f"{stem}.png", dpi=320)
        plt.close(fig)
    print(
        json.dumps(
            dict(
                status="PASS",
                source_sha256=EXPECTED_DATA_SHA256,
                unchanged_panels=16,
                unchanged_composition_cells=384,
                exact_column_sum_checks=96,
                numerical_changes=False,
            )
        )
    )


if __name__ == "__main__":
    main()
