"""State-space utilities: adjacency handling and candidate enumeration.

States are labelled ``1..m``.  The default four-state space carries the
conflict-state meaning of the conflict-forecasting literature:

``1 = peacetime, 2 = escalation, 3 = war, 4 = deescalation``.

Nothing in this module is randomised; every function here is a pure map.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

State = int
Path = tuple[State, ...]
SequenceLike = Sequence[State]


#: Default adjacency for the four conflict states.  Row ``a`` marks which
#: states may follow ``a``.  The pattern is forced by the state-labelling rule
#: of Experiment 3: states 1 and 4 both have ``Y_t = 0`` so they may only be
#: followed by 1 or 2, while states 2 and 3 both have ``Y_t > 0`` so they may
#: only be followed by 3 or 4.
DEFAULT_ADJACENCY: tuple[tuple[int, ...], ...] = (
    (1, 1, 0, 0),
    (0, 0, 1, 1),
    (0, 0, 1, 1),
    (1, 1, 0, 0),
)

STATE_NAMES: tuple[str, ...] = ("peacetime", "escalation", "war", "deescalation")


def check_adjacency(adjacency: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Return a validated square 0/1 adjacency matrix with ``k_a >= 1``."""
    array = np.asarray(adjacency, dtype=int)

    if array.ndim != 2:
        raise ValueError("adjacency must be a two-dimensional square matrix.")
    if array.shape[0] != array.shape[1]:
        raise ValueError("adjacency must be square.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("adjacency must contain only 0/1 entries.")
    if np.any(array.sum(axis=1) == 0):
        raise ValueError("each state must have at least one allowed next state.")

    return array


def n_states(adjacency: Sequence[Sequence[int]] | np.ndarray) -> int:
    """Return the size ``m`` of the state space."""
    return int(check_adjacency(adjacency).shape[0])


def row_degrees(adjacency: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Return ``k_a = sum_b A[a, b]`` for every state ``a``."""
    return check_adjacency(adjacency).sum(axis=1).astype(int)


def check_states(sequence: SequenceLike, adjacency: np.ndarray) -> list[State]:
    """Validate state *labels* only; transitions are not checked.

    Transition legality is deliberately not enforced here.  Under the extended
    real convention of :mod:`method3.counts` an ``A``-forbidden transition is a
    legitimate input that receives score ``+inf``; rejecting it would force the
    caller to drop such rows, which is exactly what the finite-sample argument
    forbids (see :func:`method3.counts.count_score`).
    """
    array = check_adjacency(adjacency)
    m = array.shape[0]
    values = [int(state) for state in sequence]

    if not values:
        raise ValueError("sequence must be nonempty.")
    for state in values:
        if state < 1 or state > m:
            raise ValueError(f"state must be between 1 and {m}; got {state}.")

    return values


def is_admissible(sequence: SequenceLike, adjacency: np.ndarray) -> bool:
    """Return ``True`` when every consecutive transition is allowed by ``A``."""
    array = check_adjacency(adjacency)
    values = check_states(sequence, array)
    return all(
        array[values[t] - 1, values[t + 1] - 1] == 1 for t in range(len(values) - 1)
    )


def allowed_next(state: State, adjacency: np.ndarray) -> list[State]:
    """Return every state that may follow ``state``."""
    array = check_adjacency(adjacency)
    m = array.shape[0]
    if state < 1 or state > m:
        raise ValueError(f"state must be between 1 and {m}; got {state}.")
    return [b + 1 for b in range(m) if array[state - 1, b] == 1]


def enumerate_paths(
    start_state: State,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> list[Path]:
    """Enumerate the candidate space ``Y(H)`` of allowed futures of length ``h``.

    The candidate space is anchored at ``start_state = H_T`` and contains only
    ``A``-admissible paths, so ``|Y(H)| <= m ** h``.
    """
    array = check_adjacency(adjacency)
    m = array.shape[0]

    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    if start_state < 1 or start_state > m:
        raise ValueError(f"start_state must be between 1 and {m}.")

    paths: list[Path] = [()]
    current_last: list[State] = [int(start_state)]

    for _ in range(horizon):
        next_paths: list[Path] = []
        next_last: list[State] = []
        for path, last in zip(paths, current_last):
            for nxt in allowed_next(last, array):
                next_paths.append(path + (nxt,))
                next_last.append(nxt)
        paths, current_last = next_paths, next_last

    return paths


def stationary_distribution(matrix: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return the stationary distribution of an irreducible stochastic matrix.

    Used by the data-generating processes so that a rolling window of length
    ``T`` taken from the interior of a country path is marginally stationary;
    this is what makes DGP-1 "origin-stable".  Falls back to the uniform
    distribution if the linear solve is degenerate.
    """
    array = np.asarray(matrix, dtype=float)
    m = array.shape[0]

    # Solve pi (P - I) = 0 with sum(pi) = 1.
    system = np.vstack([array.T - np.eye(m), np.ones(m)])
    target = np.zeros(m + 1)
    target[-1] = 1.0

    try:
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - numerically defensive
        return np.full(m, 1.0 / m)

    solution = np.clip(solution, 0.0, None)
    total = solution.sum()
    if not np.isfinite(total) or total <= tol:
        return np.full(m, 1.0 / m)
    return solution / total


def sequence_key(values: Iterable[int]) -> str:
    """Return the canonical comma-joined text form of a state sequence."""
    return ",".join(str(int(value)) for value in values)
