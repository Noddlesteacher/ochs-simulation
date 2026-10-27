"""Recompute main-study summaries from saved per-panel coverage and sizes.
No generator or scorer is imported. Diagnostic bounds use every panel's N.
"""

import math
from collections import Counter
from fractions import Fraction as F
import numpy as np
from simulation.storage import load, require
from simulation.statistics import (
    moments,
    wilson,
    failure_bound,
    forced,
    poisson_binomial,
)
from analysis.common import csvout

METHODS = ["ochs090", "count", "ochs050", "randomized_iblock"]


def stats(x, prefix):
    n = len(x)
    total = int(x.sum())
    squares = int(np.square(x.astype(np.int64)).sum())
    return {
        prefix + "_" + key: value for key, value in moments(n, total, squares).items()
    }


def aggregate(input_file, out):
    out.mkdir(parents=True, exist_ok=False)
    saved = load(input_file)
    require(
        saved["methods"] == METHODS and saved["alpha_denominator"] == 20, "array schema"
    )
    rows = saved["rows"]
    expected = {(h, f"FORMAL_{i:06d}") for h in (3, 6, 9) for i in range(5000)}
    index = {(row["h"], row["panel"]): row for row in rows}
    require(
        len(index) == len(rows) == 15000 and set(index) == expected,
        "panel identity set",
    )
    arrays = {}
    meta = {}
    for h in (3, 6, 9):
        meta[h] = [index[h, f"FORMAL_{i:06d}"] for i in range(5000)]
        arrays[h] = np.array(
            [row["coverage_and_size"] for row in meta[h]], dtype=np.int16
        )
        require(arrays[h].shape == (5000, 4, 21, 2), "evaluation shape")
    summary = []
    paired = []
    availability = []
    dist = []
    diagnostics = []
    model = []
    for h, arr in arrays.items():
        groups = np.array([r["group"] for r in meta[h]])
        Ns = np.array([r["N"] for r in meta[h]])
        targetK = np.array([r["target_T"] - h for r in meta[h]], dtype=np.int64)
        rho = F(1, 78)
        q1 = F(1, 4) * F(99, 100) ** h * rho / (1 - (1 - rho) * F(99, 100))
        model.append(
            {
                "h": h,
                "n": 5000,
                "G1_targets": int(groups.sum()),
                "G1_fraction": float(groups.mean()),
                "G1_probability_theory": float(q1),
                "target_K_mean": float(targetK.mean()),
                "target_K_mcse": float(targetK.std(ddof=1) / math.sqrt(5000)),
                "K_mean_theory": 78,
                "target_T_mean": float(targetK.mean()) + h,
                "T_mean_theory": 78 + h,
                "all_countries_K_mean": sum(r["country_K_sum"] for r in meta[h])
                / 1000000,
            }
        )
        for label, g in [("overall", None), ("G0", 0), ("G1", 1)]:
            mask = np.ones(5000, dtype=bool) if g is None else groups == g
            data = arr[mask]
            ns = Ns[mask]
            n = len(data)
            for value, count in sorted(Counter(map(int, ns)).items()):
                dist.append(
                    {
                        "h": h,
                        "group": label,
                        "N": value,
                        "panels": count,
                        "denominator": n,
                    }
                )
            for k in range(21):
                forced_count = sum(forced(int(N), k) for N in ns)
                availability.append(
                    {
                        "h": h,
                        "group": label,
                        "alpha_k": k,
                        "alpha": k / 20,
                        "nominal": 1 - k / 20,
                        "n": n,
                        "N_mean": float(ns.mean()) if n else None,
                        "N_mcse": (
                            float(ns.std(ddof=1) / math.sqrt(n)) if n > 1 else None
                        ),
                        "N_mean_theory": (
                            199 * float(q1 if g == 1 else 1 - q1)
                            if g is not None
                            else None
                        ),
                        "N_min": int(ns.min()) if n else None,
                        "N_max": int(ns.max()) if n else None,
                        "empty_pool": int((ns == 0).sum()),
                        "forced_full_count": forced_count,
                        "forced_full_rate": forced_count / n if n else None,
                        "actual_full_ochs090": int((data[:, 0, k, 1] == 2**h).sum()),
                        "actual_full_count": int((data[:, 1, k, 1] == 2**h).sum()),
                        "actual_full_ochs050": int((data[:, 2, k, 1] == 2**h).sum()),
                        "actual_full_randomized_iblock": int(
                            (data[:, 3, k, 1] == 2**h).sum()
                        ),
                    }
                )
                for mi, m in enumerate(METHODS):
                    hits = data[:, mi, k, 0]
                    sizes = data[:, mi, k, 1]
                    covered = int(hits.sum())
                    lo, hi = wilson(covered, n)
                    s = stats(sizes, "size")
                    norm = {
                        key.replace("size_", "normalized_size_"): (
                            v / 2**h if v is not None else None
                        )
                        for key, v in s.items()
                    }
                    summary.append(
                        {
                            "h": h,
                            "group": label,
                            "method": m,
                            "alpha_k": k,
                            "alpha": k / 20,
                            "nominal": 1 - k / 20,
                            "n": n,
                            "covered": covered,
                            "coverage": covered / n if n else None,
                            "coverage_lo": lo,
                            "coverage_hi": hi,
                            **s,
                            **norm,
                            "empty_count": int((sizes == 0).sum()),
                            "singleton_count": int((sizes == 1).sum()),
                            "full_count": int((sizes == 2**h).sum()),
                            "empty_rate": float((sizes == 0).mean()) if n else None,
                            "singleton_rate": float((sizes == 1).mean()) if n else None,
                            "full_rate": float((sizes == 2**h).mean()) if n else None,
                        }
                    )
                for j, other in enumerate(METHODS[1:], 1):
                    delta_size = data[:, 0, k, 1] - data[:, j, k, 1]
                    delta_cover = data[:, 0, k, 0] - data[:, j, k, 0]
                    ss = stats(delta_size, "size_difference")
                    norm = {
                        key.replace("size_difference", "normalized_size_difference"): (
                            v / 2**h if v is not None else None
                        )
                        for key, v in ss.items()
                    }
                    denom = int(data[:, j, k, 1].sum())
                    paired.append(
                        {
                            "h": h,
                            "group": label,
                            "contrast": "ochs090-minus-" + other,
                            "alpha_k": k,
                            "alpha": k / 20,
                            "nominal": 1 - k / 20,
                            "n": n,
                            **ss,
                            **norm,
                            **stats(delta_cover, "coverage_difference"),
                            "relative_mean_size_difference": (
                                int(delta_size.sum()) / denom if denom else None
                            ),
                        }
                    )
                if g is not None and 0 < k < 20:
                    if n:
                        probs = [failure_bound(int(N), k) for N in ns]
                        pm = poisson_binomial(
                            [np.longdouble(p.numerator) / p.denominator for p in probs]
                        )
                        misses = n - int(data[:, 0, k, 0].sum())
                        tail = float(pm[misses:].sum())
                        bound = 1 - float(sum(probs, F(0)) / n)
                    else:
                        misses = 0
                        tail = None
                        bound = None
                    diagnostics.append(
                        {
                            "h": h,
                            "group": label,
                            "method": "ochs090",
                            "alpha_k": k,
                            "alpha": k / 20,
                            "nominal": 1 - k / 20,
                            "n": n,
                            "covered": int(data[:, 0, k, 0].sum()),
                            "misses": misses,
                            "coverage": float(data[:, 0, k, 0].mean()) if n else None,
                            "mean_finite_rank_coverage_lower_bound": bound,
                            "p_raw_failure_upper_tail": tail,
                            "p_bonferroni_114": (
                                min(1.0, 114 * tail) if tail is not None else None
                            ),
                            "family_size": 114,
                            "alert_adjusted_05": tail is not None and 114 * tail < 0.05,
                            "status": "COMPLETE" if n else "NA_EMPTY_GROUP",
                        }
                    )
            print(f"Aggregated h={h} {label}, n={n}", flush=True)
    require(
        len(summary) == 756 and len(paired) == 567 and len(diagnostics) == 114,
        "fixed analysis row counts",
    )
    csvout(out / "SUMMARY_BY_H_GROUP_ALPHA.csv", summary)
    csvout(out / "PAIRED_METHOD_CONTRASTS.csv", paired)
    csvout(out / "REFERENCE_AVAILABILITY.csv", availability)
    csvout(out / "REFERENCE_N_DISTRIBUTION.csv", dist)
    csvout(out / "COVERAGE_DIAGNOSTICS_114.csv", diagnostics)
    csvout(out / "GENERATOR_DISTRIBUTION_CHECK.csv", model)
