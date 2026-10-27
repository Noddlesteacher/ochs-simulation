"""Marks, groups, eligibility, and the Method 3 hybrid score.

Spec sections 2 (mark), 3.7 (hybrid score) and 4 (eligibility as an extra
mark).  Everything here must be a function of the *history* and the candidate
only -- never of a future that has not been revealed at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .states import Path, SequenceLike, check_adjacency, check_states
from .counts import INF, count_score
from .blockdiag import DEFAULT_SALT, ScoreConvention, block_feature

#: Group labels.  Exactly two groups, fixed in advance, history-only.
GROUP_ALL_STATE_1 = 1
GROUP_OTHER = 0

EligibilityFn = Callable[[Sequence[int]], bool]


def always_eligible(history: Sequence[int]) -> bool:
    """Default eligibility rule: every unit enters the reference pool."""
    return True


def history_group(history: SequenceLike) -> int:
    """Return the prespecified mark ``G(H) = 1{H == (1, 1, ..., 1)}``.

    Fixed in advance and computed from the history alone, so it is available
    for the target before its future is revealed.  This is what makes the
    groupwise conformal rank of :mod:`method3.conformal` legitimate.
    """
    values = [int(state) for state in history]
    if not values:
        raise ValueError("history must be nonempty.")
    return GROUP_ALL_STATE_1 if all(state == 1 for state in values) else GROUP_OTHER


def grouping_key(
    history: SequenceLike, eligibility_fn: EligibilityFn
) -> tuple[int, bool]:
    """Return the full mark ``(G(H), eligible(H))`` used for the rank.

    [CRITICAL] Spec 4.  If a filter decides which units enter the reference
    pool, it is not a data-cleaning step but an additional deterministic mark:
    the rank must be taken within ``{same G, same eligibility}``.  Carrying it
    in the key -- rather than silently dropping ineligible rows -- is what
    keeps the target and the references on the same footing.

    ``eligibility_fn`` must be a function of that unit's own pre-origin history
    only.  Inheriting a full-record cleaning rule (for instance ">= 5 non-peace
    months over the whole series") inspects futures and invalidates the
    calibration; :func:`assert_history_only_eligibility` guards against it in
    the tests.
    """
    values = [int(state) for state in history]
    return history_group(values), bool(eligibility_fn(values))


def assert_history_only_eligibility(
    histories: Sequence[Sequence[int]],
    flags: Sequence[bool],
    eligibility_fn: EligibilityFn,
) -> None:
    """Recompute eligibility from pre-origin data and assert it matches.

    Guarantee implemented: the eligibility mark used in the rank is a function
    of ``H`` alone.  Raises ``AssertionError`` on any mismatch.
    """
    if len(histories) != len(flags):
        raise ValueError("histories and flags must have equal length.")
    for index, (history, flag) in enumerate(zip(histories, flags)):
        recomputed = bool(eligibility_fn([int(state) for state in history]))
        assert recomputed == bool(flag), (
            f"eligibility flag for unit {index} is not a function of its history: "
            f"stored {bool(flag)}, recomputed {recomputed}"
        )


def combine_extended(weights: Sequence[float], values: Sequence[float]) -> float:
    """Weighted sum on ``R u {+inf}`` with the convention ``0 * inf := 0``.

    Spec 3.7: any ``+inf`` component carrying strictly positive weight makes
    the whole score ``+inf``; a component whose weight is exactly zero is
    dropped before summing, so ``lambda = 0`` does not inherit an infinite
    count score it is not using.
    """
    if len(weights) != len(values):
        raise ValueError("weights and values must have equal length.")
    total = 0.0
    for weight, value in zip(weights, values):
        if weight == 0.0:
            continue
        if value == INF:
            return INF
        total += weight * value
    return total


@dataclass(frozen=True)
class HybridScore:
    """The Method 3 score for one ``(H, y)`` plus the features behind it."""

    candidate: Path
    group: int
    eligible: bool
    count_score: float
    block_feature: float
    score: float


def hybrid_score(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    beta: float = 1.0,
    lam: float = 0.5,
    max_permutations: int | None = 100,
    convention: ScoreConvention = "matrix_power",
    salt: str = DEFAULT_SALT,
    eligibility_fn: EligibilityFn = always_eligible,
    random_seed: int | None = None,
) -> HybridScore:
    """Return the group-specific Method 3 nonconformity score ``A_g(H, y)``.

    Spec 3.7, prespecified and group-specific:

    * group 1 (all-state-1 history):  ``A_1 = R_count(H, y)`` -- count only.
      [CRITICAL] The block feature is *not* evaluated here and cannot change
      this score.  On an all-state-1 history the block orbit is degenerate, so
      including it would add noise without information.
    * group 0:  ``A_0 = lam * R_count(H, y) / h + (1 - lam) * B_block(H, y)``.

    Larger means more nonconforming.  ``lam`` is fixed in advance, never tuned
    on the deployment calibration set (spec 4, calibration quarantine).

    Guarantee implemented: none by itself -- this is the score map ``A_g`` fed
    to the outer conformal rank.  Assumption relied on: ``A_g`` is a fixed
    measurable function of ``(H, y)`` chosen before seeing the calibration
    data, and is applied identically to references and to target candidates.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1].")

    array = check_adjacency(adjacency)
    history_values = check_states(history, array)
    candidate_values = check_states(candidate, array)
    horizon = len(candidate_values)
    if horizon < 1:
        raise ValueError("candidate must be nonempty.")

    group = history_group(history_values)
    eligible = bool(eligibility_fn(history_values))
    count = count_score(history_values, candidate_values, array, beta=beta)

    if group == GROUP_ALL_STATE_1:
        feature = 0.0
        score = count
    else:
        feature = block_feature(
            history_values,
            candidate_values,
            array,
            max_permutations=max_permutations,
            convention=convention,
            salt=salt,
            random_seed=random_seed,
        )
        score = combine_extended(
            (lam, 1.0 - lam),
            (count / horizon, feature),
        )

    return HybridScore(
        candidate=tuple(candidate_values),
        group=group,
        eligible=eligible,
        count_score=count,
        block_feature=feature,
        score=score,
    )


def count_only_score(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    beta: float = 1.0,
) -> float:
    """Return the ``lam = 1`` score used by the count-only baseline.

    Equal to ``R_count / h`` in group 0 and ``R_count`` in group 1; since the
    outer rank is taken *within* a group and ``h`` is fixed, the two differ by
    a positive constant and induce the same ordering.
    """
    array = check_adjacency(adjacency)
    history_values = check_states(history, array)
    candidate_values = check_states(candidate, array)
    count = count_score(history_values, candidate_values, array, beta=beta)
    if history_group(history_values) == GROUP_ALL_STATE_1:
        return count
    return count / len(candidate_values) if count != INF else INF
