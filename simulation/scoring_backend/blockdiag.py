"""Block non-conformity score and the identity-anchored Monte Carlo p-value.

Spec sections 3.3, 3.4 and 3.5.  :func:`block_p_value` is the literal
implementation of the proved theorem; :func:`INVALID_legacy_block_p_value` is
the deliberately broken variant kept only as a labelled negative control for
Experiment 4.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import permutations
from typing import Literal, Sequence

import numpy as np

from .states import Path, SequenceLike, check_adjacency, check_states
from .counts import transition_count_matrix
from .iblocks import (
    IBlockDecomposition,
    block_orbit_id,
    decompose_i_blocks,
    derive_random_seed,
    permutation_group_summary,
    rebuild_suffix,
    sample_block_orders,
    unit_rng,
)

ScoreConvention = Literal["one_step", "matrix_power"]

#: Number of significant digits kept before the ``>`` / ``==`` comparisons of
#: spec 3.4.  Mathematically equal scores must compare equal: misclassifying a
#: true tie as "below" removes it from ``n_equal`` and is anti-conservative.
SCORE_DIGITS = 10

#: Domain separator for the externally seeded inner RNG.  This fixed public
#: value is recorded in every configuration and result file.
DEFAULT_SALT = "method3-block-diagnostic-v1"


@dataclass(frozen=True)
class BlockDiagnostic:
    """Everything Experiment 4 and the hybrid score need from one candidate."""

    candidate: Path
    anchor_state: int
    p_conservative: float
    p_randomized: float
    n_blocks: int
    n_eval: int
    exhaustive: bool
    log_group_size: float
    group_size: int | None
    n_greater: int
    n_equal: int
    n_distinct_scores: int
    identity_score: float
    score_gap: float
    """``S(identity) - max_{pi != id} S(pi)``; ``-inf`` when ``D <= 1``."""


def round_score(value: float) -> float:
    """Round to :data:`SCORE_DIGITS` significant digits for tie detection."""
    if not math.isfinite(value):
        return value
    if value == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, SCORE_DIGITS - 1 - magnitude)


def empirical_transition_matrix(
    sequence: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Return the raw-count ``P^`` of spec 3.3, with a documented zero-row rule.

    ``P^[a, b] = #{t : X_t = a, X_{t+1} = b} / #{t : X_t = a}``.

    Zero-denominator rows -- states that never occur as a source in the
    augmented sequence -- are filled deterministically with the uniform
    distribution over the ``A``-allowed successors of that state.  The choice
    is arbitrary but must be *deterministic and identical for every unit*; a
    data-dependent fallback would break the score map.

    ``P^`` depends on the sequence only through its transition counts, and
    legal block permutations preserve those exactly (see
    :func:`method3.iblocks.assert_count_invariance`), so ``P^`` is invariant
    over the orbit and is computed once per candidate.
    """
    array = check_adjacency(adjacency)
    counts = transition_count_matrix(sequence, array)
    m = array.shape[0]

    matrix = np.zeros((m, m), dtype=float)
    for a in range(m):
        row_total = float(counts[a].sum())
        if row_total > 0:
            matrix[a] = counts[a] / row_total
        else:
            allowed = np.flatnonzero(array[a])
            matrix[a, allowed] = 1.0 / float(len(allowed))
    return matrix


def _matrix_powers(matrix: np.ndarray, horizon: int) -> list[np.ndarray]:
    """Return ``[P, P^2, ..., P^h]``, computed once per candidate."""
    powers: list[np.ndarray] = []
    current = np.eye(matrix.shape[0], dtype=float)
    for _ in range(horizon):
        current = current @ matrix
        powers.append(current)
    return powers


def block_score(
    suffix: Sequence[int],
    horizon: int,
    matrix: np.ndarray,
    powers: Sequence[np.ndarray] | None,
    convention: ScoreConvention,
) -> float:
    """Return ``S(pi)`` for one arrangement, given its final ``h + 1`` states.

    ``suffix[0]`` is ``X_T`` and ``suffix[1:]`` is ``X_{T+1..T+h}``.

    * ``one_step``    -- ``1 - (1/h) sum_j P^[X_{T+j-1}, X_{T+j}]``
    * ``matrix_power`` -- ``1 - (1/h) sum_j (P^^j)[X_T, X_{T+j}]``

    The two differ in what each forecast step conditions on: the immediate
    predecessor, or the state at the forecast origin.  Which convention is in
    force determines which impossibility statement the paper can make; see
    Experiment 4.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive for a block score.")
    if len(suffix) != horizon + 1:
        raise ValueError("suffix must contain exactly horizon + 1 states.")

    if convention == "one_step":
        total = 0.0
        for step in range(1, horizon + 1):
            total += float(matrix[suffix[step - 1] - 1, suffix[step] - 1])
    elif convention == "matrix_power":
        if powers is None:
            powers = _matrix_powers(matrix, horizon)
        origin = suffix[0]
        total = 0.0
        for step in range(1, horizon + 1):
            total += float(powers[step - 1][origin - 1, suffix[step] - 1])
    else:
        raise ValueError(f"unknown score convention: {convention!r}")

    return 1.0 - total / horizon


def _orbit_scores(
    decomposition: IBlockDecomposition,
    orders: Sequence[Sequence[int]],
    horizon: int,
    matrix: np.ndarray,
    powers: Sequence[np.ndarray] | None,
    convention: ScoreConvention,
) -> list[float]:
    """Evaluate every arrangement in ``V`` after the whole set has been drawn."""
    suffix_length = horizon + 1
    return [
        round_score(
            block_score(
                rebuild_suffix(decomposition, order, suffix_length),
                horizon,
                matrix,
                powers,
                convention,
            )
        )
        for order in orders
    ]


def block_diagnostic(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    max_permutations: int | None = 100,
    convention: ScoreConvention = "matrix_power",
    salt: str = DEFAULT_SALT,
    xi: float | None = None,
    random_seed: int | None = None,
) -> BlockDiagnostic:
    """Identity-anchored Monte Carlo block p-value for one ``(H, y)`` pair.

    Guarantee implemented: for an exchangeable block orbit, ``p_conservative``
    is super-uniform and ``p_randomized`` is exactly uniform, *for the sampled
    set as well as for full enumeration*, because the evaluated set

    ``V = {identity} u W``,  ``W ~ Uniform without replacement from S_D \\ {id}``

    is drawn independently of the data and of the scores, and the identity's
    own self-tie is counted in ``n_equal``.  Assumption relied on: the block
    orbit generated by :func:`method3.iblocks.decompose_i_blocks` is
    exchangeable, which holds because every legal permutation preserves the
    initial state and the full transition-count matrix.

    The naive alternative -- sample ``n`` orders, do *not* force the identity
    into the set, divide by ``n`` -- is **invalid**.  For all-distinct scores
    it puts mass ``P(p^ = 0) = (N - n) / (N (n + 1))`` with ``N = D!`` on the
    value zero, so it rejects at level ``alpha`` strictly more often than
    ``alpha``.  It is implemented only as
    :func:`INVALID_legacy_block_p_value`, purely as a labelled negative
    control, and is never used by any method in this package.

    ``p_conservative`` uses ``xi = 1`` and therefore satisfies
    ``p >= 1 / n_eval > 0`` surely.  It does not depend on ``xi`` at all, so
    every Method 3 quantity -- ``B_block`` and hence every hybrid score -- is
    untouched by the choice below.

    ``random_seed`` identifies a random field fixed independently of the
    observed histories.  A canonical identifier for the complete block orbit
    selects one coordinate in that field.  Consequently every observed block
    arrangement in the same orbit uses the same initial RNG state; neither its
    history/candidate split nor processing order selects a different stream.
    Passing ``None`` requests fresh entropy; simulations pass a reproducible
    externally generated seed.

    ``xi`` controls only ``p_randomized``.  When omitted it is drawn from the
    orbit-indexed external stream; an explicit independent value in
    ``[0, 1]`` is also accepted.  Method 3 uses ``p_conservative`` and hence is
    unaffected by this optional tie randomizer.
    """
    array = check_adjacency(adjacency)
    history_values = check_states(history, array)
    candidate_values = check_states(candidate, array)
    horizon = len(candidate_values)
    if horizon < 1:
        raise ValueError("candidate must be nonempty.")

    augmented = history_values + candidate_values
    anchor = candidate_values[-1]
    decomposition = decompose_i_blocks(augmented, anchor)

    # P^ is orbit-invariant, so it is estimated once from the observed
    # arrangement and reused for every permutation.
    matrix = empirical_transition_matrix(augmented, array)
    powers = _matrix_powers(matrix, horizon) if convention == "matrix_power" else None

    orbit_seed = (
        None
        if random_seed is None
        else derive_random_seed(
            random_seed,
            horizon,
            convention,
            block_orbit_id(decomposition),
            salt="block-orbit",
        )
    )
    rng = unit_rng(orbit_seed, salt)
    orders, exhaustive = sample_block_orders(
        decomposition.n_blocks, max_permutations, rng
    )

    # The whole evaluated set exists before any score is computed: no
    # score-dependent stopping, filtering, or reweighting (spec 3.4).
    scores = _orbit_scores(decomposition, orders, horizon, matrix, powers, convention)
    identity_score = scores[0]

    n_eval = len(scores)
    n_greater = sum(1 for score in scores if score > identity_score)
    n_equal = sum(1 for score in scores if score == identity_score)
    assert n_equal >= 1, "the identity's own self-tie must be counted"

    tie_randomizer = float(rng.random()) if xi is None else float(xi)
    if not 0.0 <= tie_randomizer <= 1.0:
        raise ValueError("xi must lie in [0, 1].")
    p_conservative = (n_greater + 1.0 * n_equal) / n_eval
    p_randomized = (n_greater + tie_randomizer * n_equal) / n_eval

    others = scores[1:]
    score_gap = identity_score - max(others) if others else -math.inf
    log_group_size, group_size = permutation_group_summary(decomposition.n_blocks)

    return BlockDiagnostic(
        candidate=tuple(candidate_values),
        anchor_state=anchor,
        p_conservative=p_conservative,
        p_randomized=p_randomized,
        n_blocks=decomposition.n_blocks,
        n_eval=n_eval,
        exhaustive=exhaustive,
        log_group_size=log_group_size,
        group_size=group_size,
        n_greater=n_greater,
        n_equal=n_equal,
        n_distinct_scores=len(set(scores)),
        identity_score=identity_score,
        score_gap=score_gap,
    )


def block_feature(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    max_permutations: int | None = 100,
    convention: ScoreConvention = "matrix_power",
    salt: str = DEFAULT_SALT,
    random_seed: int | None = None,
) -> float:
    """Return ``B_block(H, y) = 1 - p_block^cons(H, y)`` (spec 3.5).

    [CRITICAL] The diagnostic is total: it is defined for *every* ``(H, y)``,
    including ``D = 0`` and singleton orbits.  No special case is needed --
    when ``D <= 1`` the group has a single element, so ``n_eval = 1``,
    ``n_greater = 0``, ``n_equal = 1``, hence ``p^cons = 1`` and
    ``B_block = 0``.  That is exactly the deterministic fallback the spec
    prescribes, reached by the ordinary formula rather than by a branch, so no
    unit can ever receive a "missing" feature.

    The same ``max_permutations`` must be used for references and for target
    candidates; callers pass a single configured value.
    """
    return (
        1.0
        - block_diagnostic(
            history,
            candidate,
            adjacency,
            max_permutations=max_permutations,
            convention=convention,
            salt=salt,
            random_seed=random_seed,
        ).p_conservative
    )


def _distinct_multiset_permutations(items: Sequence[tuple[int, ...]], limit: int):
    """Yield each *distinct* arrangement of a multiset exactly once.

    Standard next-permutation enumeration over the sorted multiset.  Raises if
    more than ``limit`` distinct arrangements exist, so a caller never silently
    receives a truncated enumeration.
    """
    current = sorted(items)
    produced = 0
    while True:
        produced += 1
        if produced > limit:
            raise ValueError(
                f"more than {limit} distinct arrangements; exact enumeration refused."
            )
        yield tuple(current)

        pivot = len(current) - 2
        while pivot >= 0 and current[pivot] >= current[pivot + 1]:
            pivot -= 1
        if pivot < 0:
            return
        successor = len(current) - 1
        while current[successor] <= current[pivot]:
            successor -= 1
        current[pivot], current[successor] = current[successor], current[pivot]
        current[pivot + 1 :] = reversed(current[pivot + 1 :])


def exact_block_p_value(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    convention: ScoreConvention = "matrix_power",
    limit: int = 200_000,
) -> tuple[float, int, int, int]:
    """Exact ``(p^cons, n_greater, n_equal, n_total)`` over the *whole* group.

    Enumerating ``D!`` directly is impossible for large ``D``, but every
    distinct arrangement of the block multiset has exactly the same number of
    index-order preimages, namely ``prod_j m_j!`` over the multiplicities of
    identical blocks.  That constant cancels in the ratio, so the exact
    conservative p-value equals the count over *distinct arrangements*:

    ``p^cons = (#{distinct arr : S > S_id} + #{distinct arr : S = S_id})
               / #{distinct arr}``.

    This is an exact reference value used by Experiment 4 -- in particular for
    the all-state-1 one-step case where ``D = T`` makes ``D!`` astronomically
    large but the number of distinct arrangements is only ``T``.  It agrees
    with :func:`block_diagnostic` whenever that function enumerates fully.
    """
    array = check_adjacency(adjacency)
    history_values = check_states(history, array)
    candidate_values = check_states(candidate, array)
    horizon = len(candidate_values)

    augmented = history_values + candidate_values
    decomposition = decompose_i_blocks(augmented, candidate_values[-1])
    matrix = empirical_transition_matrix(augmented, array)
    powers = _matrix_powers(matrix, horizon) if convention == "matrix_power" else None

    identity_score = round_score(
        block_score(
            rebuild_suffix(
                decomposition, tuple(range(decomposition.n_blocks)), horizon + 1
            ),
            horizon,
            matrix,
            powers,
            convention,
        )
    )

    n_greater = n_equal = n_total = 0
    for arrangement in _distinct_multiset_permutations(
        list(decomposition.blocks), limit
    ):
        suffix: list[int] = []
        remaining = horizon + 1
        pieces = [decomposition.tail, *reversed(arrangement), decomposition.prefix]
        collected: list[tuple[int, ...]] = []
        for piece in pieces:
            if len(piece) >= remaining:
                collected.append(piece[len(piece) - remaining :])
                remaining = 0
                break
            collected.append(piece)
            remaining -= len(piece)
        for piece in reversed(collected):
            suffix.extend(piece)

        score = round_score(block_score(suffix, horizon, matrix, powers, convention))
        n_total += 1
        if score > identity_score:
            n_greater += 1
        elif score == identity_score:
            n_equal += 1

    return (n_greater + n_equal) / n_total, n_greater, n_equal, n_total


def INVALID_legacy_block_p_value(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    max_permutations: int = 100,
    convention: ScoreConvention = "matrix_power",
    salt: str = DEFAULT_SALT,
    random_seed: int | None = None,
) -> float:
    """NEGATIVE CONTROL ONLY -- the invalid unanchored Monte Carlo p-value.

    Samples ``n`` block orders uniformly **without replacement from the whole
    group** -- so the identity may or may not land in the sample -- does *not*
    force the identity in, and divides by ``n``.

    When all orbit scores are distinct this places mass exactly

    ``P(p^ = 0) = (N - n) / (N (n + 1))``,   ``N = D!``

    on the value zero: with probability ``(N - n) / N`` the identity is absent
    from the sample, and given that, it is the largest of the ``n + 1``
    exchangeable scores with probability ``1 / (n + 1)``.  A test that can
    return zero rejects at *any* level, so the procedure over-rejects.

    Used exclusively by Experiment 4 to exhibit that anti-conservatism; never
    call it from a method under evaluation.
    """
    array = check_adjacency(adjacency)
    history_values = check_states(history, array)
    candidate_values = check_states(candidate, array)
    horizon = len(candidate_values)

    augmented = history_values + candidate_values
    decomposition = decompose_i_blocks(augmented, candidate_values[-1])
    matrix = empirical_transition_matrix(augmented, array)
    powers = _matrix_powers(matrix, horizon) if convention == "matrix_power" else None

    orbit_seed = (
        None
        if random_seed is None
        else derive_random_seed(
            random_seed,
            horizon,
            convention,
            block_orbit_id(decomposition),
            salt="legacy-block-orbit",
        )
    )
    rng = unit_rng(orbit_seed, salt + "|legacy")
    n_blocks = decomposition.n_blocks
    base = np.arange(n_blocks)
    n_draw = min(max_permutations, math.factorial(n_blocks))

    identity_score = round_score(
        block_score(
            rebuild_suffix(decomposition, tuple(range(n_blocks)), horizon + 1),
            horizon,
            matrix,
            powers,
            convention,
        )
    )
    # Uniform without replacement from the WHOLE group: the identity is not
    # forced in, but it is not excluded either.
    full_size = math.factorial(n_blocks)
    if full_size <= n_draw:
        drawn = list(permutations(range(n_blocks)))
        scores = _orbit_scores(
            decomposition, drawn, horizon, matrix, powers, convention
        )
        n_greater = sum(1 for score in scores if score > identity_score)
        n_equal = sum(1 for score in scores if score == identity_score)
        return (n_greater + n_equal) / len(drawn)

    drawn: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(drawn) < n_draw:
        order = tuple(int(value) for value in rng.permutation(base))
        if order not in seen:
            seen.add(order)
            drawn.append(order)
    scores = _orbit_scores(decomposition, drawn, horizon, matrix, powers, convention)

    n_greater = sum(1 for score in scores if score > identity_score)
    n_equal = sum(1 for score in scores if score == identity_score)
    return (n_greater + n_equal) / n_draw
