"""Exact components and conservative outer ranks: (1 + #reference >= target)/(N+1)."""

from fractions import Fraction
from functools import cmp_to_key
from .storage import require
from .scoring_backend.ochs_v81_exact_numeric import (
    compare_exact_scores,
    ExactScoreComponents,
    ExactCountWeight,
    ExactBlockCoordinate,
)


def encode(c):
    require(c.count.weight is not None, "non-finite legal component")
    b = c.block
    return dict(
        group=c.group,
        h=c.horizon,
        W=str(c.count.weight),
        factors=[str(f) for f in c.count.factors],
        block=(
            None
            if b is None
            else dict(
                D=b.D,
                gt=str(b.n_strict_greater),
                eq=str(b.n_equal),
                identity=None if b.identity_score is None else str(b.identity_score),
            )
        ),
    )


def decode(c):
    block = c["block"]
    count = ExactCountWeight.finite([Fraction(x) for x in c["factors"]])
    require(str(count.weight) == c["W"], "stored W/factors mismatch")
    b = (
        None
        if block is None
        else ExactBlockCoordinate.from_rank_counts(
            D=block["D"],
            n_strict_greater=int(block["gt"]),
            n_equal=int(block["eq"]),
            identity_score=(
                None if block["identity"] is None else Fraction(block["identity"])
            ),
        )
    )
    return ExactScoreComponents(group=c["group"], horizon=c["h"], count=count, block=b)


def relation(a, b, lam):
    r = compare_exact_scores(a, b, lam=lam)
    require(r.certified, "UNCERTIFIED comparison; no epsilon fallback")
    return int(r.relation)


def ranks(refs, targets, lam):
    """Sort exact reference scores then lower-bound; ties included in upper rank.
    This is algebraically sum 1{ref>=target}; QA compares against full summation.
    """
    ordered = sorted(refs, key=cmp_to_key(lambda a, b: relation(a, b, lam)))
    nums = []
    for star in targets:
        lo, hi = 0, len(ordered)
        while lo < hi:
            mid = (lo + hi) // 2
            if relation(ordered[mid], star, lam) < 0:
                lo = mid + 1
            else:
                hi = mid
        nums.append(1 + len(ordered) - lo)
    return [[n, len(refs) + 1] for n in nums]
