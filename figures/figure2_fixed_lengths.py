"""Figure 2: the original fixed-length performance plot, unchanged style."""

from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HS = (3, 6, 9)
LS = (10, 50, 100, 200)
LABELS = {
    "ochs090": "OCHS (all three labels)",
    "randomized_iblock": "Randomized i-block",
}
COLORS = {"ochs090": "#1d5c91", "randomized_iblock": "#be5831"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with args.source_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    look = {
        (int(r["h"]), int(r["L"]), r["method"], int(r["alpha_index"])): r for r in rows
    }
    assert len(look) == len(rows) == 1008
    assert all(int(r["R"]) == 5000 for r in rows)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    def figure_save(fig, name):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(
                args.out / (name + "." + ext),
                bbox_inches="tight",
                dpi=180,
                facecolor="white",
            )
        plt.close(fig)

    for k in (2,):
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
        for col, h in enumerate(HS):
            for method in LABELS:
                rr = [look[h, L, method, k] for L in LS]
                sizes = [float(r["mean_size"]) for r in rr]
                ses = [float(r["size_mcse"]) for r in rr]
                axes[0, col].errorbar(
                    LS,
                    sizes,
                    yerr=np.array(ses) * 1.959963984540054,
                    color=COLORS[method],
                    marker="o",
                    capsize=3,
                    label=LABELS[method],
                    lw=1.5,
                )
                cv = [float(r["coverage"]) for r in rr]
                axes[1, col].plot(LS, cv, color=COLORS[method], marker="o", lw=1.5)
                axes[1, col].vlines(
                    LS,
                    [float(r["coverage_low95"]) for r in rr],
                    [float(r["coverage_high95"]) for r in rr],
                    color=COLORS[method],
                    lw=1.1,
                )
            axes[0, col].set_title(f"h = {h} | {2**h} candidates")
            axes[1, col].axhline(1 - k / 20, color="black", linestyle="--", lw=1)
            axes[1, col].set_ylim(0, 1.025)
            for row in range(2):
                axes[row, col].set_xticks(LS)
                axes[row, col].grid(axis="y", color="#e8e8e8")
                axes[row, col].set_axisbelow(True)
            axes[1, col].set_xlabel("Fixed all-one history length L")
        axes[0, 0].set_ylabel("Mean prediction-set size")
        axes[1, 0].set_ylabel("Empirical coverage")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
        )
        fig.suptitle(
            f"Fixed all-one histories | alpha = {k/20:.2f}, nominal coverage = {1-k/20:.0%}",
            y=1.055,
        )
        fig.text(
            0.5,
            -0.02,
            "5,000 source pools per h; paired across four lengths. Pointwise 95% Monte Carlo intervals. 200 countries per scenario.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout()
        figure_save(fig, "step6_length_performance_nominal90")


if __name__ == "__main__":
    main()
