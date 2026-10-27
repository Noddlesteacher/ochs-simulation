"""Fixed-history aggregation. The pool is the independent replication unit."""

import itertools
import math
import numpy as np
from simulation.settings import HS, LS, METHODS
from simulation.storage import require
from analysis.common import csvout

R = 5000
Z = 1.959963984540054


def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def wilson(c, n):
    p = c / n
    d = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def aggregate(input_file, out):
    out.mkdir(parents=True, exist_ok=False)
    with np.load(input_file, allow_pickle=False) as saved:
        arrays = saved["evaluation"]
        Ns = saved["N"]
        require(saved["horizons"].tolist() == list(HS), "horizon order")
        require(saved["lengths"].tolist() == list(LS), "length order")
    require(arrays.shape == (3, 5000, 4, 2, 21, 2), "evaluation shape")
    require(Ns.shape == (3, 5000), "reference shape")
    summaries = []
    for hi, h in enumerate(HS):
        for li, L in enumerate(LS):
            for method in METHODS:
                mi = 1 if method == "randomized_iblock" else 0
                for k in range(21):
                    x = arrays[hi, :, li, mi, k]
                    covered = int(x[:, 0].sum())
                    size, se = mean_se(x[:, 1])
                    lo, up = wilson(covered, R)
                    r = dict(
                        h=h,
                        L=L,
                        method=method,
                        alpha_index=k,
                        alpha=k / 20,
                        nominal_coverage=1 - k / 20,
                        R=R,
                        covered=covered,
                        coverage=covered / R,
                        coverage_low95=lo,
                        coverage_high95=up,
                        mean_size=size,
                        size_mcse=se,
                        size_low95=size - Z * se,
                        size_high95=size + Z * se,
                        normalized_mean_size=size / 2**h,
                        N_mean=float(Ns[hi].mean()),
                        N_min=int(Ns[hi].min()),
                        N_max=int(Ns[hi].max()),
                    )
                    for label, test in (
                        ("empty", x[:, 1] == 0),
                        ("singleton", x[:, 1] == 1),
                        ("full", x[:, 1] == 2**h),
                    ):
                        rate, se2 = mean_se(test)
                        r.update(
                            {
                                label + "_count": int(test.sum()),
                                label + "_rate": rate,
                                label + "_mcse": se2,
                            }
                        )
                    summaries.append(r)
    csvout(out / "SUMMARY_FIXED_LENGTH_ALL_ALPHA.csv", summaries)
    csvout(
        out / "TABLE4_FIXED_LENGTH.csv",
        [r for r in summaries if r["alpha_index"] in (4, 2, 1)],
    )
    length_rows = []
    method_rows = []
    for hi, h in enumerate(HS):
        for ai, bi in itertools.combinations(range(4), 2):
            for method in METHODS:
                mi = 1 if method == "randomized_iblock" else 0
                for k in range(21):
                    for measure, xi in (("coverage", 0), ("size", 1)):
                        d = arrays[hi, :, bi, mi, k, xi].astype(np.int64) - arrays[
                            hi, :, ai, mi, k, xi
                        ].astype(np.int64)
                        mean, se = mean_se(d)
                        length_rows.append(
                            dict(
                                h=h,
                                L_small=LS[ai],
                                L_large=LS[bi],
                                method=method,
                                alpha_index=k,
                                alpha=k / 20,
                                R=R,
                                measure=measure,
                                direction="large-minus-small",
                                mean_difference=mean,
                                paired_mcse=se,
                                low95=mean - Z * se,
                                high95=mean + Z * se,
                            )
                        )
        for li, L in enumerate(LS):
            for k in range(21):
                for measure, xi in (("coverage", 0), ("size", 1)):
                    d = arrays[hi, :, li, 1, k, xi].astype(np.int64) - arrays[
                        hi, :, li, 0, k, xi
                    ].astype(np.int64)
                    mean, se = mean_se(d)
                    method_rows.append(
                        dict(
                            h=h,
                            L=L,
                            alpha_index=k,
                            alpha=k / 20,
                            R=R,
                            measure=measure,
                            direction="randomized_iblock-minus-ochs090",
                            mean_difference=mean,
                            paired_mcse=se,
                            low95=mean - Z * se,
                            high95=mean + Z * se,
                        )
                    )
    csvout(out / "PAIRED_LENGTH_DIFFERENCES.csv", length_rows)
    csvout(out / "PAIRED_METHOD_DIFFERENCES.csv", method_rows)
