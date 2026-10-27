"""OCHS: select references by history, score their OWN futures, then rank."""

from dataclasses import asdict
from .settings import A, MODEL_ID, LAMBDAS
from .storage import require
from . import ranks as rr
from .scoring_backend.ochs_v81_stage2a import prepare_exact_score, ExactScoreCache
from .scoring_backend.hybrid import history_group
from .scoring_backend.ochs_v81 import legal_candidates


def validate_visible(obj, adjacency):
    require(
        tuple(tuple(row) for row in adjacency) == A
        and all(type(v) is int for row in adjacency for v in row),
        "frozen binary adjacency",
    )
    require(
        set(obj)
        == {
            "model_id",
            "panel_id",
            "h",
            "target_id",
            "total_countries",
            "potential_references",
            "records",
        },
        "unexpected predictor fields",
    )
    require(obj["model_id"] == MODEL_ID, "wrong model")
    require(type(obj["h"]) is int and obj["h"] in (3, 6, 9), "horizon")
    require(type(obj["target_id"]) is int and obj["target_id"] == 0, "target ID")
    require(
        type(obj["total_countries"]) is int
        and obj["total_countries"] == 200
        and type(obj["potential_references"]) is int
        and obj["potential_references"] == 199,
        "country/reference count",
    )
    require(isinstance(obj["panel_id"], str) and obj["panel_id"], "panel ID")
    records, h = obj["records"], obj["h"]
    require(
        type(records) is list and len(records) == 200, "exactly 200 observed records"
    )
    require(
        [r["unit_id"] for r in records] == list(range(200)),
        "ordered unique IDs; target excluded from refs",
    )
    for r in records:
        require(set(r) == {"unit_id", "L", "start", "end", "states"}, "record fields")
        require(
            all(type(r[k]) is int for k in ("unit_id", "L", "start", "end")),
            "integer metadata",
        )
        v = r["states"]
        minimum = h + 1 if r["unit_id"] == 0 else 2 * h + 1
        require(
            type(v) is list and len(v) == r["L"] >= minimum,
            "shifted-model minimum length",
        )
        require(r["start"] == 1 - len(v) and r["end"] == 0, "calendar")
        require(all(type(x) is int and 1 <= x <= 4 for x in v), "states")
        require(
            all(adjacency[a - 1][b - 1] for a, b in zip(v, v[1:])),
            "illegal observed transition",
        )


def pool(obj, adjacency=A):
    """Split reference records and select the target's history group.

    Each selected tuple contains (unit_id, its history, its OWN realised future).
    The selection is unchanged across target candidates.
    """
    validate_visible(obj, adjacency)
    h = obj["h"]
    H = tuple(obj["records"][0]["states"])
    g = history_group(H)
    refs, selection = [], []
    for r in obj["records"][1:]:
        Hi, Yi = tuple(r["states"][:-h]), tuple(r["states"][-h:])
        mark = history_group(Hi)
        selected = mark == g
        selection.append(
            {
                "unit_id": r["unit_id"],
                "eligible": True,
                "group": mark,
                "scoring_T": len(Hi),
                "selected": selected,
            }
        )
        if selected:
            refs.append((r["unit_id"], Hi, Yi))
    return H, g, refs, selection


def legacy_view(obj, adjacency=A):
    validate_visible(obj, adjacency)
    return {k: obj[k] for k in ("panel_id", "h", "target_id", "records")} | {"M": 199}


def membership(pairs):
    for n, d in pairs:
        require(
            type(n) is int and type(d) is int and 1 <= n <= d,
            "exact positive p numerator/denominator",
        )
    return [[j for j, (n, d) in enumerate(pairs) if 20 * n > k * d] for k in range(21)]


def predict(obj, adjacency=A):
    """Construct all three OCHS sets for one observed panel, without its truth."""
    H, g, refs, selection = pool(obj, adjacency)
    h = obj["h"]
    candidates = legal_candidates(H, h, adjacency)
    require(len(candidates) == len(set(candidates)) == 2**h, "complete candidate space")
    cache = ExactScoreCache()  # Never global / never shared between panels.
    # These scores stay fixed while the target candidate varies.
    refc = [
        prepare_exact_score(Hi, Yi, adjacency, lam="9/10", cache=cache).components
        for _, Hi, Yi in refs
    ]
    # One score per admissible continuation, using the target's full history.
    targetc = [
        prepare_exact_score(H, y, adjacency, lam="9/10", cache=cache).components
        for y in candidates
    ]
    ps = {name: rr.ranks(refc, targetc, lam) for name, lam in LAMBDAS.items()}
    if g == 1:
        require(ps["ochs090"] == ps["count"] == ps["ochs050"], "G1 invariance")
    result = {
        "model_id": MODEL_ID,
        "panel_id": obj["panel_id"],
        "h": h,
        "total_countries": 200,
        "potential_references": 199,
        "legacy_M": 199,
        "group": g,
        "T": len(H),
        "N": len(refs),
        "eligible_count": 199,
        "selection": selection,
        "reference_ids": [i for i, _, _ in refs],
        "candidates": [list(y) for y in candidates],
        "references": [
            {"unit_id": i, "H": list(Hi), "Y": list(Yi), "component": rr.encode(c)}
            for (i, Hi, Yi), c in zip(refs, refc)
        ],
        "components": [rr.encode(c) for c in targetc],
        "p": ps,
        "members": {m: membership(p) for m, p in ps.items()},
    }
    return result, asdict(cache.snapshot())
