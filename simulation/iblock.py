"""Independent randomized i-block baseline. OCHS B stays conservative."""

from fractions import Fraction
from math import factorial
from .storage import canon as canonical, require
from .scoring_backend.ochs_v81_stage2a import ExactScoreCache, _validated_episode
from .scoring_backend.ochs_v81 import legal_candidates

NAMESPACE = "OCHS_SINGLE_PATH_IBLOCK_TIE_V1"


def address(study, model, h, panel, target, candidate):
    require(
        all(isinstance(x, str) and x for x in (study, model, panel)),
        "address identifiers",
    )
    require(
        type(h) is int and h >= 1 and type(target) is int and target >= 0,
        "address h/target",
    )
    require(
        len(candidate) == h and all(type(x) is int and x >= 1 for x in candidate),
        "address candidate",
    )
    return canonical([NAMESPACE, study, model, h, panel, target, list(candidate)])


def validate_tally(t):
    require(set(t) == {"D", "gt", "eq", "total"}, "tally fields")
    require(all(type(v) is int for v in t.values()), "integer tallies required")
    require(t["D"] >= 0 and t["gt"] >= 0 and t["eq"] >= 1, "invalid counts")
    require(t["total"] == factorial(t["D"]), "total must equal D factorial")
    require(t["gt"] + t["eq"] <= t["total"], "rank mass exceeds total")


def exact_q(tally, u):
    """q=(strictly-greater + U*equal)/D!, with exact integer/Fraction arithmetic."""
    validate_tally(tally)
    require(type(u) is Fraction and 0 < u < 1, "U must be an exact open-unit Fraction")
    return (tally["gt"] + u * tally["eq"]) / tally["total"]


def inclusion_probability(tally, alpha):
    """Ideal continuous-U law, arithmetic only; NOT a coverage estimator."""
    validate_tally(tally)
    require(type(alpha) is Fraction and 0 <= alpha <= 1, "exact alpha in [0,1]")
    return min(
        Fraction(1),
        max(
            Fraction(0),
            (tally["gt"] + tally["eq"] - alpha * tally["total"]) / tally["eq"],
        ),
    )


def pair_tally(history, candidate, adjacency, *, cache=None):
    require(len(history) >= 1 and len(candidate) >= 1, "nonempty pair required")
    require(
        all(type(x) is int for x in tuple(history) + tuple(candidate)),
        "integer states required",
    )
    array, H, y, key = _validated_episode(history, candidate, adjacency)
    z = H + y
    require(
        all(array[a - 1, b - 1] == 1 for a, b in zip(z, z[1:])),
        "graph-admissible pair required",
    )
    local = cache if cache is not None else ExactScoreCache()
    # Deliberately NOT prepare_exact_score: G1 OCHS bypasses block, baseline must not.
    diag, _, _ = local.block_component(
        key, history=H, future=y, adjacency=array, budget=None
    )
    t = {
        "D": diag.n_blocks,
        "gt": diag.n_strictly_greater,
        "eq": diag.n_equal_identity,
        "total": diag.n_indexed_orders,
    }
    validate_tally(t)
    require(
        Fraction(t["gt"] + t["eq"], t["total"]) == diag.q_conservative,
        "backend conservative rank mismatch",
    )
    return t


def parse_uniform_entries(entries, expected_addresses):
    require(type(entries) is list, "U entries must be a list")
    require(
        len(expected_addresses) == len(set(expected_addresses)),
        "duplicate expected candidate address",
    )
    result = {}
    for entry in entries:
        require(
            type(entry) is dict
            and set(entry) == {"address", "numerator", "denominator"},
            "U entry fields",
        )
        a, n, d = entry["address"], entry["numerator"], entry["denominator"]
        require(isinstance(a, str) and a not in result, "duplicate/malformed U address")
        require(
            type(n) is int and type(d) is int and 0 < n < d, "U rational components"
        )
        result[a] = Fraction(n, d)
    require(result.keys() == set(expected_addresses), "missing/extra U address")
    return result


def predict(
    history,
    horizon,
    adjacency,
    entries,
    *,
    study,
    model,
    panel,
    target=0,
    cache=None,
    reverse=False
):
    """Only public predictive inputs plus randomizer identity; no reference argument."""
    require(type(horizon) is int and horizon >= 1, "positive horizon")
    candidates = tuple(legal_candidates(history, horizon, adjacency))
    require(len(candidates) == len(set(candidates)), "duplicate candidate")
    keys = [address(study, model, horizon, panel, target, y) for y in candidates]
    uniforms = parse_uniform_entries(entries, keys)
    local = cache if cache is not None else ExactScoreCache()
    by_candidate = {}
    for y in reversed(candidates) if reverse else candidates:
        key = address(study, model, horizon, panel, target, y)
        u = uniforms[key]
        tally = pair_tally(history, y, adjacency, cache=local)
        q = exact_q(tally, u)
        qc = Fraction(tally["gt"] + tally["eq"], tally["total"])
        by_candidate[y] = dict(
            candidate=list(y),
            address=key,
            **tally,
            U=[u.numerator, u.denominator],
            q=[q.numerator, q.denominator],
            q_cons=[qc.numerator, qc.denominator]
        )
    rows = [by_candidate[y] for y in candidates]
    members = [
        [i for i, r in enumerate(rows) if 20 * r["q"][0] > k * r["q"][1]]
        for k in range(21)
    ]
    conservative = [
        [i for i, r in enumerate(rows) if 20 * r["q_cons"][0] > k * r["q_cons"][1]]
        for k in range(21)
    ]
    return {
        "candidates": [list(y) for y in candidates],
        "rows": rows,
        "members": members,
        "conservative_members": conservative,
    }
