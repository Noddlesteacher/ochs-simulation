"""Wilson intervals, paired MCSE and the 114 Poisson-binomial diagnostics."""

import math
from fractions import Fraction as F
from itertools import product
import numpy as np
from .storage import require

Z = 1.959963984540054


def moments(n, s, ss):
    if not n:
        return {"mean": None, "mcse": None, "lo": None, "hi": None}
    mean = s / n
    se = math.sqrt(max(0.0, (ss - s * s / n) / (n * (n - 1)))) if n > 1 else None
    return {
        "mean": mean,
        "mcse": se,
        "lo": mean - Z * se if se is not None else None,
        "hi": mean + Z * se if se is not None else None,
    }


def wilson(c, n):
    if not n:
        return None, None
    p = c / n
    d = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def failure_bound(N, k):
    return F(((N + 1) * k) // 20, N + 1)


def forced(N, k):
    return 20 > k * (N + 1)


def poisson_binomial(probs):
    # Same nonnegative coefficient recursion as the previously audited R01
    # independent.py poisson_binomial; no old observations or family reused.
    p = np.asarray(probs, dtype=np.longdouble)
    require(np.all((p >= 0) & (p <= 1)), "PB probabilities")
    mass = np.zeros(len(p) + 1, dtype=np.longdouble)
    mass[0] = 1
    for j, q in enumerate(p):
        mass[1 : j + 2] = mass[1 : j + 2] * (1 - q) + mass[: j + 1] * q
        mass[0] *= 1 - q
    require(abs(mass.sum() - 1) < np.longdouble("2e-11"), "PB mass normalization")
    return mass


def deterministic_tests():
    for N in range(200):
        require(forced(N, 0) and not forced(N, 20), "forced endpoint")
        require(
            forced(N, 4) == (N <= 3) and forced(N, 2) == (N <= 8),
            "forced interior boundary",
        )
        for k in range(21):
            require(
                failure_bound(N, k)
                == F(
                    (F(N + 1) * F(k, 20)).numerator
                    // (F(N + 1) * F(k, 20)).denominator,
                    N + 1,
                ),
                "exact floor",
            )
    checked = 0
    for probabilities in (
        [],
        [F(0)],
        [F(1)],
        [F(1, 3), F(1, 2), F(2, 3)],
        [F(0), F(1), F(1, 10)],
    ):
        pm = [F(0)] * (len(probabilities) + 1)
        for bits in product((0, 1), repeat=len(probabilities)):
            value = F(1)
            for q, b in zip(probabilities, bits):
                value *= q if b else 1 - q
            pm[sum(bits)] += value
        got = poisson_binomial(
            [np.longdouble(q.numerator) / q.denominator for q in probabilities]
        )
        require(
            all(abs(float(a) - float(b)) < 1e-14 for a, b in zip(pm, got, strict=True)),
            "PB independent enumeration",
        )
        for misses in range(len(pm) + 1):
            require(
                abs(float(sum(pm[misses:])) - float(got[misses:].sum())) < 1e-14,
                "PB upper failure tail",
            )
        checked += 1
    require(
        moments(2, 2, 4)["mean"] == 1 and moments(2, 2, 4)["mcse"] == 1,
        "paired variance",
    )
    return {
        "status": "PASS",
        "PB_exact_enumerations": checked,
        "floor_N_values": 200,
        "alpha_values": 21,
        "diagnostic_family": 114,
    }
