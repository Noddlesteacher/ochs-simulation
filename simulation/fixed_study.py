"""Compute all four fixed lengths before evaluating their shared future."""

from .settings import (
    A,
    HS,
    LS,
    METHODS,
    FIXED_RUN as RUN,
    FIXED_MODEL as MODEL,
    MODEL as REF_MODEL,
)
from .storage import require, digest
from .fixed_prediction import reference_view, reference_components, predict
from .random_sources import midpoint


def bits(indices):
    return format(sum(1 << i for i in indices), "x")


def indices(bits_hex, J):
    value = int(bits_hex, 16)
    require(value >= 0 and value < 1 << J, "bitset extent")
    return [j for j in range(J) if value >> j & 1]


def evaluate(h, pool, pred, future, candidates, N):
    require(future in candidates, "new future must belong to candidate space")
    truth = candidates.index(future)
    rows = []
    for L in LS:
        for method in METHODS:
            base = "ochs090" if method != "randomized_iblock" else method
            for k, mm in enumerate(pred[str(L)]["members"][base]):
                rows.append(
                    dict(
                        h=h,
                        pool_id=pool,
                        L=L,
                        method=method,
                        alpha_index=k,
                        covered=int(truth in mm),
                        size=len(mm),
                        N=N,
                    )
                )
    for k in range(21):
        coverage = [int(truth in pred[str(L)]["members"]["ochs090"][k]) for L in LS]
        require(
            coverage == sorted(coverage, reverse=True),
            "shared-truth length coverage nesting",
        )
    return rows


def row_key(r):
    return (r["h"], r["pool_id"], r["L"], r["method"], r["alpha_index"])


def work(visible, inp, templates):
    """Predict all four histories with shared references/U; evaluate afterwards."""
    h = inp["h"]
    pool = inp["pool_id"]
    records, selected = reference_view(visible, h)
    refs = reference_components(selected)
    candidates = [r["candidate"] for r in templates[10]["rows"]]
    require(
        inp["candidate_order_sha256"] == digest(candidates),
        "saved input candidate identity",
    )
    require(len(inp["J"]) == len(candidates), "saved U count")
    uniforms = [midpoint(j) for j in inp["J"]]
    # No truth passed into the predictor.
    pred = predict(refs, templates, uniforms)
    rows = evaluate(h, pool, pred, inp["future"], candidates, len(refs))
    compact = {}
    for L in LS:
        p = pred[str(L)]
        compact[str(L)] = dict(
            p=p["p"],
            members={m: [bits(mm) for mm in v] for m, v in p["members"].items()},
        )
    evaluation = [
        [
            [
                [r["covered"], r["size"]]
                for r in rows
                if r["L"] == L and r["method"] == m
            ]
            for m in ("ochs090", "randomized_iblock")
        ]
        for L in LS
    ]
    result = dict(
        h=h,
        pool_id=pool,
        run_id=RUN,
        model_id=MODEL,
        reference_model=REF_MODEL,
        reference_pool_sha256=digest(records),
        references=refs,
        reference_ids=[r["unit_id"] for r in refs],
        N=len(refs),
        lengths=compact,
        evaluation=evaluation,
        aliases={"count": "ochs090", "ochs050": "ochs090"},
        q_reconstruction="template tally + saved J; (gt + ((2*J+1)/2**65)*eq)/total",
        logical_evaluation_rows=len(rows),
    )
    require(
        result["logical_evaluation_rows"] == 336, "four L x four methods x 21 levels"
    )
    return result, rows
