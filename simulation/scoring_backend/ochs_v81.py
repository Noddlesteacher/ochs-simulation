"""OCHS v8.1: direct augmented counts and exact block diagnostics.

This module is deliberately separate from the historical Method 3 modules.
It implements the Stage 1 v8.1 specification without changing the older
history-smoothed score, sampled block diagnostic, simulations, or baselines.

All state labels are one-based.  Larger scores are more nonconforming.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
import math
from typing import Hashable, Iterator, Mapping, Sequence

import numpy as np

from .counts import INF, transition_count_matrix
from .hybrid import GROUP_ALL_STATE_1, GROUP_OTHER, history_group
from .iblocks import IBlockDecomposition, decompose_i_blocks, rebuild_sequence
from .states import (
    Path,
    SequenceLike,
    check_adjacency,
    check_states,
    enumerate_paths,
)

# Used only to distinguish harmless floating-point roundoff from a violated
# probability bound.  It is never used to turn near-ties into exact ties.
_BOUND_CHECK_TOLERANCE = 1e-12

# Floating expressions for the same mathematical score can differ in their
# last machine digits because logarithms are added in a different order.  The
# outer rank therefore compares one fixed 14-decimal canonicalization of every
# reference and target score.  This is a symmetric implementation convention,
# not part of the mathematical definition, and retains roughly two guard
# digits above ordinary double-precision roundoff at the score scales used here.
_OUTER_RANK_DECIMALS = 14


@dataclass(frozen=True)
class HistoryGraphDiagnostic:
    """History-only metadata; it never changes counts or eligibility."""

    graph_compatible: bool
    illegal_edges: tuple[tuple[int, int, int], ...]
    """Triples ``(t, x_t, x_{t+1})`` with one-based transition position."""


@dataclass(frozen=True)
class AugmentedCounts:
    """The unmasked directed counts of ``z = H || y`` (Eq. 3.A1)."""

    sequence: Path
    counts: np.ndarray
    outgoing_counts: np.ndarray
    n_transitions: int


@dataclass(frozen=True)
class ObservedStateMatrix:
    """The closed empirical matrix on ``V(z)`` only (Eq. 3.A7)."""

    states: Path
    matrix: np.ndarray
    outgoing_counts: tuple[int, ...]
    transition_counts: tuple[tuple[int, ...], ...]

    @property
    def state_to_local(self) -> Mapping[int, int]:
        """Map each original one-based state label to its local matrix index."""
        return {state: index for index, state in enumerate(self.states)}


@dataclass(frozen=True)
class BlockOrderScore:
    """The matrix-power score for one indexed block order (Eq. 3.A5)."""

    order: tuple[int, ...]
    score: float
    exact_score: Fraction


@dataclass(frozen=True)
class ConservativeBlockDiagnostic:
    """Exact conservative terminal-anchor diagnostic (Eq. 3.A6)."""

    n_blocks: int
    n_indexed_orders: int
    q_conservative: float
    block_feature: float
    identity_score: float | None
    n_at_least_identity: int
    n_equal_identity: int
    observed_states: Path


@dataclass(frozen=True)
class CompletedReference:
    """One ordinary reference and its own observed realized future."""

    reference_id: Hashable
    history: SequenceLike
    realized_future: SequenceLike


@dataclass(frozen=True)
class ReferenceTrueScore:
    """A same-group reference score from that reference's own future (3.9)."""

    reference_id: Hashable
    history: Path
    realized_future: Path
    group: int
    score: float
    history_graph_compatible: bool


@dataclass(frozen=True)
class OCHSCandidateResult:
    """One target candidate's score, outer p-value, and strict decision."""

    candidate: Path
    score: float
    p_value: float
    included: bool


@dataclass(frozen=True)
class OCHSPredictionResult:
    """The complete v8.1 result for one fixed target and one fixed weight."""

    target_id: Hashable
    target_group: int
    target_history_graph_compatible: bool
    alpha: float
    lam: float
    n_same_group: int
    reference_scores: tuple[float, ...]
    reference_details: tuple[ReferenceTrueScore, ...]
    candidates: tuple[OCHSCandidateResult, ...]

    @property
    def prediction_set(self) -> tuple[Path, ...]:
        """Return retained legal futures in deterministic enumeration order."""
        return tuple(row.candidate for row in self.candidates if row.included)


def _validated_pair(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, Path, Path]:
    """Validate labels and nonempty episode parts, but not historical edges."""
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(history, array))
    candidate_values = tuple(check_states(candidate, array))
    return array, history_values, candidate_values


def history_graph_diagnostic(
    history: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> HistoryGraphDiagnostic:
    """Flag adjacency conflicts in ``H`` without deleting any observations.

    This is diagnostic metadata for v8.1 Section D2.  It is deliberately
    separate from candidate legality and from every scoring calculation.
    """
    array = check_adjacency(adjacency)
    values = tuple(check_states(history, array))
    illegal = tuple(
        (position, source, target)
        for position, (source, target) in enumerate(
            zip(values[:-1], values[1:]), start=1
        )
        if array[source - 1, target - 1] == 0
    )
    return HistoryGraphDiagnostic(
        graph_compatible=not illegal,
        illegal_edges=illegal,
    )


def future_is_legal(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> bool:
    """Return membership in ``Y_A(H)`` (Eq. 3.A2).

    Only the bridge ``x_T -> y_1`` and future-internal edges are checked.
    Earlier history edges do not enter this decision.
    """
    array, history_values, candidate_values = _validated_pair(
        history, candidate, adjacency
    )
    previous = history_values[-1]
    for state in candidate_values:
        if array[previous - 1, state - 1] == 0:
            return False
        previous = state
    return True


def legal_candidates(
    history: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[Path, ...]:
    """Enumerate the reported candidate space ``Y_A(H)`` (Eq. 3.A2)."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(history, array))
    return tuple(enumerate_paths(history_values[-1], horizon, array))


def augmented_transition_counts(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> AugmentedCounts:
    """Count all ``T+h-1`` unmasked transitions of ``H || y`` (Eq. 3.A1).

    The bridge is an ordinary adjacent pair in the concatenated sequence.  No
    outgoing edge is added after the terminal future state.
    """
    array, history_values, candidate_values = _validated_pair(
        history, candidate, adjacency
    )
    sequence = history_values + candidate_values
    counts = transition_count_matrix(sequence, array)
    outgoing = counts.sum(axis=1, dtype=np.int64)
    expected = len(sequence) - 1
    if int(counts.sum()) != expected:
        raise RuntimeError("augmented transition count is inconsistent with T+h-1")

    counts.setflags(write=False)
    outgoing.setflags(write=False)
    return AugmentedCounts(
        sequence=sequence,
        counts=counts,
        outgoing_counts=outgoing,
        n_transitions=expected,
    )


def augmented_count_score(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> float:
    """Return the direct augmented count score ``R_count_aug`` (Eq. 3.4).

    A graph-illegal future returns ``+inf`` before any ratio is evaluated.
    For a legal future, every scored edge occurs in ``H || y`` and therefore
    has a positive numerator and denominator.  Encountering zero is a hard
    implementation error rather than a row-completion case.
    """
    array, history_values, candidate_values = _validated_pair(
        history, candidate, adjacency
    )
    if not future_is_legal(history_values, candidate_values, array):
        return INF

    augmented = augmented_transition_counts(history_values, candidate_values, array)
    score = 0.0
    previous = history_values[-1]
    for state in candidate_values:
        numerator = int(augmented.counts[previous - 1, state - 1])
        denominator = int(augmented.outgoing_counts[previous - 1])
        if numerator < 1 or denominator < 1:
            raise RuntimeError(
                "legal candidate produced a zero augmented-count denominator"
            )
        if numerator > denominator:
            raise RuntimeError("transition count exceeds its outgoing row count")
        score -= math.log(numerator / denominator)
        previous = state

    bound = len(candidate_values) * math.log(augmented.n_transitions)
    if score < -_BOUND_CHECK_TOLERANCE or score > bound + _BOUND_CHECK_TOLERANCE:
        raise RuntimeError("direct augmented count score violates Eq. (3.A4)")
    return max(0.0, score)


def terminal_anchor_blocks(
    sequence: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> IBlockDecomposition:
    """Return ``I_0``, indexed ``I_1,...,I_D``, and fixed tail ``I_K``.

    This is the terminal-anchor construction in v8.1 Section 3.5.1.  Equal
    block contents remain separate tuple positions and therefore retain their
    distinct indices.
    """
    array = check_adjacency(adjacency)
    values = tuple(check_states(sequence, array))
    return decompose_i_blocks(values, values[-1])


def indexed_block_orders(n_blocks: int) -> Iterator[tuple[int, ...]]:
    """Yield the full symmetric group on ``D`` block indices in fixed order.

    The iterator contains exactly ``D!`` members and begins with the identity.
    In particular, ``D=0`` and ``D=1`` each yield one order.  No cap,
    deduplication, sampling, or randomization is applied.
    """
    if n_blocks < 0:
        raise ValueError("n_blocks must be nonnegative.")
    yield from permutations(range(n_blocks))


def observed_state_matrix(
    sequence: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> ObservedStateMatrix:
    """Construct the closed row-stochastic matrix on ``V(z)`` (Eq. 3.A7).

    This function is called by the primary block path only when ``D>=2``.
    It assigns no row to an absent state and fails if any observed state has no
    outgoing count.  Historical counts are not masked by adjacency.

    This empirical transition estimate is a scoring object, not the DGP.
    """
    array = check_adjacency(adjacency)
    values = tuple(check_states(sequence, array))
    states = tuple(sorted(set(values)))
    state_to_local = {state: index for index, state in enumerate(states)}
    full_counts = transition_count_matrix(values, array)

    local = np.zeros((len(states), len(states)), dtype=float)
    local_counts = np.zeros((len(states), len(states)), dtype=np.int64)
    outgoing_counts: list[int] = []
    observed_columns = {state - 1 for state in states}
    all_columns = set(range(array.shape[0]))
    absent_columns = sorted(all_columns - observed_columns)

    for source in states:
        denominator = int(full_counts[source - 1].sum())
        if denominator < 1:
            raise RuntimeError(
                f"observed state {source} has no outgoing count in D>=2 branch"
            )
        if absent_columns and int(full_counts[source - 1, absent_columns].sum()) != 0:
            raise RuntimeError("an observed-state transition leaves V(z)")
        outgoing_counts.append(denominator)
        row = state_to_local[source]
        for target in states:
            column = state_to_local[target]
            count = int(full_counts[source - 1, target - 1])
            local_counts[row, column] = count
            local[row, column] = count / denominator

    row_sums = local.sum(axis=1)
    if not np.allclose(
        row_sums,
        np.ones(len(states)),
        rtol=0.0,
        atol=_BOUND_CHECK_TOLERANCE,
    ):
        raise RuntimeError("observed-state empirical matrix is not row-stochastic")
    if np.any(local < 0.0):
        raise RuntimeError("observed-state empirical matrix has a negative entry")

    local.setflags(write=False)
    return ObservedStateMatrix(
        states=states,
        matrix=local,
        outgoing_counts=tuple(outgoing_counts),
        transition_counts=tuple(
            tuple(int(value) for value in row) for row in local_counts
        ),
    )


def matrix_powers(matrix: np.ndarray, horizon: int) -> tuple[np.ndarray, ...]:
    """Return ``P^1,...,P^h`` for the matrix-power block score (3.A5)."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")

    powers: list[np.ndarray] = []
    current = array.copy()
    for _ in range(horizon):
        current.setflags(write=False)
        powers.append(current)
        current = current @ array
    return tuple(powers)


ExactMatrix = tuple[tuple[Fraction, ...], ...]


def _multiply_exact_matrices(left: ExactMatrix, right: ExactMatrix) -> ExactMatrix:
    """Multiply two small rational matrices without floating-point roundoff."""
    size = len(left)
    return tuple(
        tuple(
            sum(
                (left[row][middle] * right[middle][column] for middle in range(size)),
                start=Fraction(0, 1),
            )
            for column in range(size)
        )
        for row in range(size)
    )


def _exact_matrix_powers(
    observed: ObservedStateMatrix,
    horizon: int,
) -> tuple[ExactMatrix, ...]:
    """Compute the same powers as Eq. (3.A5) using exact count ratios.

    Counts and row totals are integers, so every entry is rational.  Exact
    arithmetic is used for the conservative rank to prevent a mathematical
    tie from being split by two neighboring floating-point representations.
    """
    base: ExactMatrix = tuple(
        tuple(
            Fraction(count, observed.outgoing_counts[row])
            for count in observed.transition_counts[row]
        )
        for row in range(len(observed.states))
    )
    powers: list[ExactMatrix] = []
    current = base
    for _ in range(horizon):
        powers.append(current)
        current = _multiply_exact_matrices(current, base)
    return tuple(powers)


def _exact_matrix_power_score_for_order(
    decomposition: IBlockDecomposition,
    order: tuple[int, ...],
    history_length: int,
    horizon: int,
    powers: Sequence[ExactMatrix],
    state_to_local: Mapping[int, int],
) -> Fraction:
    """Evaluate Eq. (3.A5) exactly for one validated indexed order."""
    permuted = rebuild_sequence(decomposition, order)
    start = permuted[history_length - 1]
    probability_sum = Fraction(0, 1)
    for step in range(1, horizon + 1):
        future_state = permuted[history_length + step - 1]
        probability_sum += powers[step - 1][state_to_local[start]][
            state_to_local[future_state]
        ]
    score = Fraction(1, 1) - probability_sum / horizon
    if score < 0 or score > 1:
        raise RuntimeError("matrix-power block score lies outside [0, 1]")
    return score


def _validated_block_context(
    sequence: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[
    np.ndarray,
    Path,
    IBlockDecomposition,
    ObservedStateMatrix,
    tuple[ExactMatrix, ...],
]:
    """Build common immutable inputs for every score in one block orbit."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    array = check_adjacency(adjacency)
    values = tuple(check_states(sequence, array))
    if len(values) <= horizon:
        raise ValueError("sequence must contain at least one history state.")
    decomposition = terminal_anchor_blocks(values, array)
    if decomposition.n_blocks < 2:
        raise ValueError("matrix-power block scores are defined here only for D>=2")
    observed = observed_state_matrix(values, array)
    powers = _exact_matrix_powers(observed, horizon)
    return array, values, decomposition, observed, powers


def matrix_power_block_scores(
    sequence: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> Iterator[BlockOrderScore]:
    """Yield Eq. (3.A5) for every one of the ``D!`` indexed orders."""
    array, values, decomposition, observed, powers = _validated_block_context(
        sequence, horizon, adjacency
    )
    history_length = len(values) - horizon
    original_counts = transition_count_matrix(values, array)
    identity_start = values[0]
    terminal_state = values[-1]
    state_to_local = observed.state_to_local

    for order in indexed_block_orders(decomposition.n_blocks):
        permuted = rebuild_sequence(decomposition, order)
        if permuted[0] != identity_start or permuted[-1] != terminal_state:
            raise RuntimeError("block permutation changed an endpoint")
        permuted_counts = transition_count_matrix(permuted, array)
        if not np.array_equal(permuted_counts, original_counts):
            raise RuntimeError("block permutation changed directed counts")
        exact_score = _exact_matrix_power_score_for_order(
            decomposition,
            order,
            history_length,
            horizon,
            powers,
            state_to_local,
        )
        yield BlockOrderScore(
            order=order,
            score=float(exact_score),
            exact_score=exact_score,
        )


def conservative_block_diagnostic(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> ConservativeBlockDiagnostic:
    """Return the exact conservative block rank and ``B0`` (Eq. 3.A6).

    The ``D<=1`` result is returned directly from the identity self-comparison;
    no matrix or matrix power is constructed in that branch.  For ``D>=2``,
    all indexed orders are evaluated, including duplicate reconstructed paths.
    """
    array, history_values, candidate_values = _validated_pair(
        history, candidate, adjacency
    )
    sequence = history_values + candidate_values
    decomposition = terminal_anchor_blocks(sequence, array)
    n_blocks = decomposition.n_blocks
    n_orders = math.factorial(n_blocks)

    if n_blocks <= 1:
        return ConservativeBlockDiagnostic(
            n_blocks=n_blocks,
            n_indexed_orders=1,
            q_conservative=1.0,
            block_feature=0.0,
            identity_score=None,
            n_at_least_identity=1,
            n_equal_identity=1,
            observed_states=(),
        )

    identity = tuple(range(n_blocks))
    order_scores = matrix_power_block_scores(sequence, len(candidate_values), array)
    first = next(order_scores, None)
    if first is None or first.order != identity:
        raise RuntimeError("full indexed enumeration omitted the identity")
    identity_score = first.score
    identity_exact_score = first.exact_score
    evaluated = 1
    n_at_least = 1
    n_equal = 1

    for row in order_scores:
        evaluated += 1
        if row.exact_score >= identity_exact_score:
            n_at_least += 1
        if row.exact_score == identity_exact_score:
            n_equal += 1

    if evaluated != n_orders:
        raise RuntimeError("indexed enumeration size differs from D factorial")
    q_conservative = n_at_least / evaluated
    block_feature = 1.0 - q_conservative
    if not 0.0 <= block_feature <= 1.0:
        raise RuntimeError("conservative block feature lies outside [0, 1]")

    return ConservativeBlockDiagnostic(
        n_blocks=n_blocks,
        n_indexed_orders=n_orders,
        q_conservative=q_conservative,
        block_feature=block_feature,
        identity_score=identity_score,
        n_at_least_identity=n_at_least,
        n_equal_identity=n_equal,
        observed_states=tuple(sorted(set(sequence))),
    )


def ochs_score(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    lam: float = 0.50,
) -> float:
    """Return the group-specific OCHS nonconformity score (Eqs. 3.6, 3.8).

    Group 1 is direct-count only and never evaluates a block diagnostic.
    Group 0 uses explicit endpoint branches, so no zero-weight infinite term
    is ever evaluated.  The fixed weight is the only method choice here.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1].")
    array, history_values, candidate_values = _validated_pair(
        history, candidate, adjacency
    )
    horizon = len(candidate_values)
    group = history_group(history_values)

    if group == GROUP_ALL_STATE_1:
        return augmented_count_score(history_values, candidate_values, array)

    if lam == 0.0:
        return conservative_block_diagnostic(
            history_values, candidate_values, array
        ).block_feature

    count = augmented_count_score(history_values, candidate_values, array)
    if lam == 1.0:
        return count / horizon

    block = conservative_block_diagnostic(
        history_values, candidate_values, array
    ).block_feature
    if math.isinf(count):
        return INF
    return lam * (count / horizon) + (1.0 - lam) * block


def reference_true_scores(
    references: Sequence[CompletedReference],
    *,
    target_id: Hashable,
    target_group: int,
    history_length: int,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    lam: float = 0.50,
) -> tuple[ReferenceTrueScore, ...]:
    """Return same-group ordinary-reference true scores (Eqs. 3.9, 3.11).

    Every score uses that reference's own realized future.  The required
    lengths enforce the fixed ``T,h`` score construction.  ``target_id`` is a
    required argument, and finding it in the ordinary reference list is an
    error.  Consequently the returned length is ``N_*`` references only.
    """
    if target_group not in (GROUP_OTHER, GROUP_ALL_STATE_1):
        raise ValueError("target_group must be 0 or 1.")
    if history_length < 1 or horizon < 1:
        raise ValueError("history_length and horizon must be positive.")
    array = check_adjacency(adjacency)
    reference_rows = tuple(references)

    seen_ids: set[Hashable] = set()
    for reference in reference_rows:
        if reference.reference_id == target_id:
            raise ValueError("target_id appears in the ordinary reference pool")
        if reference.reference_id in seen_ids:
            raise ValueError("ordinary reference identifiers must be distinct")
        seen_ids.add(reference.reference_id)

    validated: list[tuple[CompletedReference, Path, Path]] = []
    for reference in reference_rows:
        history_values = tuple(check_states(reference.history, array))
        future_values = tuple(check_states(reference.realized_future, array))
        if len(history_values) != history_length:
            raise ValueError("every reference history must have the fixed length T")
        if len(future_values) != horizon:
            raise ValueError("every reference future must have the fixed horizon h")
        validated.append((reference, history_values, future_values))

    rows: list[ReferenceTrueScore] = []
    for reference, history_values, future_values in validated:
        group = history_group(history_values)
        if group != target_group:
            continue
        diagnostic = history_graph_diagnostic(history_values, array)
        rows.append(
            ReferenceTrueScore(
                reference_id=reference.reference_id,
                history=history_values,
                realized_future=future_values,
                group=group,
                score=ochs_score(
                    history_values,
                    future_values,
                    array,
                    lam=lam,
                ),
                history_graph_compatible=diagnostic.graph_compatible,
            )
        )
    return tuple(rows)


def conservative_outer_p_value(
    reference_scores: Sequence[float],
    candidate_score: float,
) -> float:
    """Return the references-only conservative outer rank (Eq. 3.12).

    Before comparison, all scores receive the same fixed decimal
    canonicalization.  This prevents a mathematical tie from being split by
    last-bit floating-point differences while preserving a deterministic,
    exchangeability-symmetric score map.
    """
    scores = tuple(_canonical_outer_score(value) for value in reference_scores)
    candidate = _canonical_outer_score(candidate_score)
    n_at_least = sum(score >= candidate for score in scores)
    return (1 + n_at_least) / (len(scores) + 1)


def _canonical_outer_score(score: float) -> float:
    """Return the fixed computational representation used only by outer ranks."""
    value = float(score)
    if math.isnan(value):
        raise ValueError("scores must never be NaN")
    if math.isinf(value):
        return value
    return round(value, _OUTER_RANK_DECIMALS)


def conservative_outer_quantile(
    reference_scores: Sequence[float],
    alpha: float,
) -> float:
    """Return the quantile exactly equivalent to the computed strict p-rank.

    Rather than multiplying ``1-alpha`` and applying a floating-point ceiling,
    this function finds the smallest attainable comparison count whose
    computed p-value is strictly above the supplied ``alpha``.  It therefore
    agrees with :func:`conservative_outer_p_value` even when ``alpha`` itself
    lies exactly on the finite p-value grid in binary floating-point.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    scores = sorted(_canonical_outer_score(value) for value in reference_scores)
    n_reference = len(scores)
    minimum_at_least = next(
        (
            count
            for count in range(n_reference + 1)
            if (1 + count) / (n_reference + 1) > alpha
        ),
        None,
    )
    if minimum_at_least is None:
        return -INF
    if minimum_at_least == 0:
        return INF
    return scores[n_reference - minimum_at_least]


def prediction_set_via_outer_quantile(
    reference_scores: Sequence[float],
    candidate_scores: Sequence[tuple[Path, float]],
    alpha: float,
) -> tuple[Path, ...]:
    """Return Eq. (3.13) through the equivalent v8.1 quantile route."""
    threshold = conservative_outer_quantile(reference_scores, alpha)
    return tuple(
        candidate
        for candidate, score in candidate_scores
        if _canonical_outer_score(score) <= threshold
    )


def ochs_prediction_set(
    *,
    target_id: Hashable,
    target_history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    references: Sequence[CompletedReference],
    lam: float = 0.50,
) -> OCHSPredictionResult:
    """Construct ``C_alpha`` for one fixed target (Algorithm 1; Eq. 3.13).

    This interface receives no target future.  The target mark is determined
    from its history before any future score, ordinary references are checked
    against the fixed target identifier, and each target candidate is scored
    only on the target history.  Inclusion uses the required strict ``p>alpha``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1].")
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(target_history, array))
    target_group = history_group(history_values)
    target_diagnostic = history_graph_diagnostic(history_values, array)

    reference_details = reference_true_scores(
        references,
        target_id=target_id,
        target_group=target_group,
        history_length=len(history_values),
        horizon=horizon,
        adjacency=array,
        lam=lam,
    )
    scores = tuple(row.score for row in reference_details)

    candidate_rows: list[OCHSCandidateResult] = []
    for candidate in legal_candidates(history_values, horizon, array):
        score = ochs_score(history_values, candidate, array, lam=lam)
        p_value = conservative_outer_p_value(scores, score)
        candidate_rows.append(
            OCHSCandidateResult(
                candidate=candidate,
                score=score,
                p_value=p_value,
                included=p_value > alpha,
            )
        )

    return OCHSPredictionResult(
        target_id=target_id,
        target_group=target_group,
        target_history_graph_compatible=target_diagnostic.graph_compatible,
        alpha=alpha,
        lam=lam,
        n_same_group=len(reference_details),
        reference_scores=scores,
        reference_details=reference_details,
        candidates=tuple(candidate_rows),
    )


__all__ = [
    "AugmentedCounts",
    "BlockOrderScore",
    "CompletedReference",
    "ConservativeBlockDiagnostic",
    "HistoryGraphDiagnostic",
    "OCHSCandidateResult",
    "OCHSPredictionResult",
    "ObservedStateMatrix",
    "ReferenceTrueScore",
    "augmented_count_score",
    "augmented_transition_counts",
    "conservative_block_diagnostic",
    "conservative_outer_quantile",
    "conservative_outer_p_value",
    "future_is_legal",
    "history_graph_diagnostic",
    "indexed_block_orders",
    "legal_candidates",
    "matrix_power_block_scores",
    "matrix_powers",
    "observed_state_matrix",
    "ochs_prediction_set",
    "ochs_score",
    "prediction_set_via_outer_quantile",
    "reference_true_scores",
    "terminal_anchor_blocks",
]
