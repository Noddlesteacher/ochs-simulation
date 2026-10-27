"""Transition counts, adjacency-smoothed ``P~``, and the count feature ``R_count``.

Spec section 3.1.  The single load-bearing convention in this module is the
**extended real** one: scores live in ``R u {+inf}`` with ``-log 0 := +inf``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .states import SequenceLike, check_adjacency, check_states

#: The extended-real value taken by ``-log 0``.  Kept as a module constant so
#: that every comparison in the package refers to the same object.
INF = float("inf")


def transition_count_matrix(
    sequence: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Return the ``m x m`` matrix of adjacent transition counts ``N_ab``.

    Counts every ``t`` with ``X_t = a, X_{t+1} = b``, including transitions
    forbidden by ``A`` (those are legitimate observations under the extended
    real convention, and the i-block count-invariance check needs them).
    """
    array = check_adjacency(adjacency)
    values = check_states(sequence, array)
    m = array.shape[0]

    counts = np.zeros((m, m), dtype=np.int64)
    for first, second in zip(values[:-1], values[1:]):
        counts[first - 1, second - 1] += 1
    return counts


def smoothed_transition_matrix(
    history: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Return the adjacency-smoothed transition matrix.

    For state ``a``, define the *allowed-transition* row count and degree as

    ``N_a^(A) = sum_{b: A_ab = 1} N_ab`` and
    ``k_a = sum_b A_ab``.

    Then, for every ``b``,

    ``P~_ab(H) = A_ab (N_ab + beta) / (N_a^(A) + beta k_a)``.

    Thus the denominator sums only transitions to ``A``-allowed columns.  An
    ``A``-forbidden transition keeps probability exactly zero no matter what
    was observed, and every ``A``-allowed transition is strictly positive.
    Consequently ``R_count`` is ``+inf`` exactly on paths that use a forbidden
    transition (spec 3.1).
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number.")

    array = check_adjacency(adjacency)
    counts = transition_count_matrix(history, array)
    m = array.shape[0]

    matrix = np.zeros((m, m), dtype=float)
    for a in range(m):
        allowed = np.flatnonzero(array[a])
        row_total = float(counts[a, allowed].sum())
        denominator = row_total + beta * float(len(allowed))
        matrix[a, allowed] = (counts[a, allowed] + beta) / denominator

    return matrix


def count_score(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    beta: float = 1.0,
) -> float:
    """Return ``R_count(H, y) = -sum_l log P~_{z_{l-1}, z_l}(H)``, ``z_0 = H_T``.

    Guarantee implemented: none by itself; this is one nonconformity *feature*.
    Assumption relied on: the transition matrix is estimated from ``H`` alone,
    so ``R_count`` is a measurable function of ``(H, y)`` and nothing else --
    which is what lets the outer conformal argument of :mod:`method3.conformal`
    treat it as a fixed score map.

    Extended reals (spec 3.1, [CRITICAL]): a path that uses an ``A``-forbidden
    transition returns ``+inf`` rather than raising or returning ``nan``.  This
    applies identically to a reference country's *realised* future, which then
    counts against every finite candidate -- conservative, and never dropped.
    """
    array = check_adjacency(adjacency)
    history_values = check_states(history, array)
    candidate_values = check_states(candidate, array)

    matrix = smoothed_transition_matrix(history_values, array, beta=beta)

    total = 0.0
    previous = history_values[-1]
    for nxt in candidate_values:
        probability = float(matrix[previous - 1, nxt - 1])
        if probability <= 0.0:
            return INF
        total -= math.log(probability)
        previous = nxt

    return total


def count_scores_for_units(
    histories: Sequence[SequenceLike],
    futures: Sequence[SequenceLike],
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    beta: float = 1.0,
) -> list[float]:
    """Score the realised future of every completed unit ``(H_i, Y_i)``.

    Rows with ``+inf`` are retained by construction; see :func:`count_score`.
    """
    if len(histories) != len(futures):
        raise ValueError("histories and futures must have equal length.")
    return [
        count_score(history, future, adjacency, beta=beta)
        for history, future in zip(histories, futures)
    ]
