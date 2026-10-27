"""Fixed-history study. Same G1 reference pool and U across four lengths."""

from fractions import Fraction as F
from .settings import A, HS, LS, MODEL as REF_MODEL
from .storage import require, digest
from . import ranks as rr, iblock as baseline
from .scoring_backend.ochs_v81_stage2a import (
    prepare_exact_score as prepare,
    ExactScoreCache as Cache,
)
from .scoring_backend.hybrid import history_group as group
from .scoring_backend.ochs_v81 import legal_candidates as candidates


def reference_view(visible, h):
    require(type(h) is int and h in HS and visible["h"] == h, "wrong horizon")
    require(visible["model_id"] == REF_MODEL, "reference model")
    records = [r for r in visible["records"] if r["unit_id"] != 0]
    require(
        len(records) == 199 and [r["unit_id"] for r in records] == list(range(1, 200)),
        "199 unique ordered references",
    )
    selected = []
    for r in records:
        require(
            set(r) == {"unit_id", "L", "start", "end", "states"}, "reference fields"
        )
        z = r["states"]
        require(
            len(z) == r["L"] >= 2 * h + 1
            and r["start"] == 1 - len(z)
            and r["end"] == 0,
            "reference window",
        )
        require(all(type(x) is int and 1 <= x <= 4 for x in z), "state domain")
        require(all(A[a - 1][b - 1] for a, b in zip(z, z[1:])), "reference support")
        H, y = z[:-h], z[-h:]
        if group(H) == 1:
            selected.append((r["unit_id"], H, y))
    return records, selected


def reference_components(selected):
    cache = Cache()
    return [
        dict(
            unit_id=i,
            T=len(H),
            Y=list(y),
            component=rr.encode(prepare(H, y, A, lam="9/10", cache=cache).components),
        )
        for i, H, y in selected
    ]


def template(h, L):
    require(h in HS and L in LS, "template specification")
    H = [1] * L
    ys = list(candidates(H, h, A))
    cache = Cache()
    require(
        len(ys) == 2**h and ys == sorted(set(ys)),
        "lexicographic complete candidate space",
    )
    rows = []
    for y in ys:
        c = prepare(H, y, A, lam="9/10", cache=cache).components
        require(c.group == 1 and c.block is None, "G1 score branch unchanged")
        t = baseline.pair_tally(H, y, A, cache=cache)
        qc = F(t["gt"] + t["eq"], t["total"])
        key = dict(
            H=H,
            y=list(y),
            A=A,
            T=L,
            h=h,
            score_version=c.scoring_version,
            baseline="full_indexed_exact_suffix",
        )
        rows.append(
            dict(
                candidate=list(y),
                component=rr.encode(c),
                tally=t,
                q_cons=[qc.numerator, qc.denominator],
                key_sha256=digest(key),
            )
        )
    return dict(h=h, L=L, H=H, rows=rows)


def members(pairs):
    return [[j for j, (n, d) in enumerate(pairs) if 20 * n > k * d] for k in range(21)]


def predict(ref_components, templates, uniforms):
    """Uniforms are injected Fractions; no truth, RNG, numeric P, or old target."""
    refs = [rr.decode(r["component"]) for r in ref_components]
    output = {}
    for L, t in templates.items():
        rows = t["rows"]
        require(len(rows) == len(uniforms), "candidate U count")
        ps = rr.ranks(refs, [rr.decode(r["component"]) for r in rows], "1")
        qs = [
            baseline.exact_q(r["tally"], u) for r, u in zip(rows, uniforms, strict=True)
        ]
        qp = [[q.numerator, q.denominator] for q in qs]
        for r, q in zip(rows, qs, strict=True):
            require(q <= F(*r["q_cons"]), "randomized subset of conservative")
        om, bm = members(ps), members(qp)
        require(
            len(om[0]) == len(bm[0]) == len(rows) and om[20] == bm[20] == [],
            "alpha endpoints",
        )
        for mm in (om, bm):
            require(
                all(set(mm[k + 1]) <= set(mm[k]) for k in range(20)), "alpha nesting"
            )
        output[str(L)] = dict(
            p=ps,
            q=qp,
            members=dict(ochs090=om, randomized_iblock=bm),
            aliases={"count": "ochs090", "ochs050": "ochs090"},
        )
    if len(templates) == 4:
        for a, b in zip(LS, LS[1:]):
            require(
                all(
                    set(output[str(b)]["members"]["ochs090"][k])
                    <= set(output[str(a)]["members"]["ochs090"][k])
                    for k in range(21)
                ),
                "OCHS length nesting",
            )
    return output
