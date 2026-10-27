"""Exact suffix-weight backend for the OCHS v8.1 block diagnostic.

This module leaves :mod:`method3.ochs_v81` intact as the brute-force oracle.
It evaluates the same full indexed permutation measure without visiting all
``D!`` complete orders.  A right-to-left leaf fixes only the blocks needed to
determine the final ``h + 1`` states; its unvisited left-hand orders receive
the exact factorial weight ``(D-k)!``.

All probability calculations and all block-rank counts are exact.  Floating
values appear only in timing diagnostics and convenience display properties.
No randomization, permutation sampling, or partial-result fallback is used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .counts import transition_count_matrix
from .iblocks import IBlockDecomposition, decompose_i_blocks
from .states import Path, SequenceLike, check_adjacency, check_states

EXACT_SUFFIX_SCORING_VERSION = "ochs-v8.1-exact-suffix-stage2a"

ExactMatrix = tuple[tuple[Fraction, ...], ...]


class ExactSuffixResourceLimitError(RuntimeError):
    """Signal an explicitly incomplete exact calculation.

    No diagnostic or block rank is returned with this exception.  In
    particular, callers must not treat the work accumulated before the guard
    fired as an exact partial estimate.
    """

    status = "NOT_COMPLETED"

    def __init__(self, resource: str, observed: int | float, limit: int | float):
        self.resource = resource
        self.observed = observed
        self.limit = limit
        super().__init__(
            f"exact suffix computation NOT_COMPLETED: {resource} "
            f"budget exceeded ({observed!r} > {limit!r}); no partial q returned"
        )


@dataclass(frozen=True)
class ExactSuffixBudget:
    """Optional deterministic work guards for one exact calculation.

    ``None`` means that a resource is unrestricted.  These limits are purely
    computational and never alter the statistical distribution: exceeding a
    limit raises :class:`ExactSuffixResourceLimitError`.
    """

    max_tree_nodes: int | None = None
    max_grouped_leaves: int | None = None
    max_distinct_suffixes: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_tree_nodes",
            "max_grouped_leaves",
            "max_distinct_suffixes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_seconds is not None:
            value = self.max_seconds
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError("max_seconds must be a finite positive number or None")


@dataclass(frozen=True)
class WeightedSuffix:
    """One distinct final ``h+1`` state suffix and its indexed-order mass."""

    suffix: Path
    weight: int


@dataclass(frozen=True)
class ExactScoreMass:
    """One exact rational block score and its indexed-order mass."""

    score: Fraction
    weight: int


@dataclass(frozen=True)
class SuffixDepthMass:
    """Enumeration diagnostics for leaves that stop at one tree depth."""

    depth: int
    grouped_leaves: int
    indexed_terminal_assignments: int
    indexed_order_weight: int


@dataclass(frozen=True)
class ExactSuffixHistogram:
    """Complete weighted suffix distribution before matrix scoring."""

    scoring_version: str
    horizon: int
    prefix: Path
    blocks: tuple[Path, ...]
    tail: Path
    n_blocks: int
    n_indexed_orders: int
    n_content_types: int
    n_tree_nodes: int
    n_grouped_leaves: int
    n_distinct_suffixes: int
    n_suffix_aggregation_hits: int
    min_leaf_depth: int
    max_leaf_depth: int
    depth_masses: tuple[SuffixDepthMass, ...]
    suffix_weights: tuple[WeightedSuffix, ...]
    total_weight: int
    elapsed_seconds: float
    resource_status: str = "COMPLETED"

    @property
    def suffix_weight_map(self) -> Mapping[Path, int]:
        return {row.suffix: row.weight for row in self.suffix_weights}


@dataclass(frozen=True)
class ExactSuffixBlockDiagnostic:
    """Exact conservative block diagnostic computed by suffix weights."""

    scoring_version: str
    n_blocks: int
    n_indexed_orders: int
    n_content_types: int
    n_tree_nodes: int
    n_grouped_leaves: int
    n_distinct_suffixes: int
    n_suffix_aggregation_hits: int
    min_leaf_depth: int
    max_leaf_depth: int
    depth_masses: tuple[SuffixDepthMass, ...]
    suffix_weights: tuple[WeightedSuffix, ...]
    score_histogram: tuple[ExactScoreMass, ...]
    identity_suffix: Path
    identity_score: Fraction | None
    n_strictly_greater: int
    n_equal_identity: int
    n_at_least_identity: int
    q_conservative: Fraction
    block_feature: Fraction
    observed_states: Path
    score_cache_hits: int
    score_cache_misses: int
    enumeration_seconds: float
    matrix_seconds: float
    scoring_seconds: float
    total_seconds: float
    resource_status: str = "COMPLETED"

    @property
    def q_conservative_float(self) -> float:
        return float(self.q_conservative)

    @property
    def block_feature_float(self) -> float:
        return float(self.block_feature)


class _ResourceGuard:
    def __init__(self, budget: ExactSuffixBudget, started: float):
        self.budget = budget
        self.started = started
        self.tree_nodes = 0
        self.grouped_leaves = 0

    def check_time(self) -> None:
        limit = self.budget.max_seconds
        if limit is None:
            return
        observed = perf_counter() - self.started
        if observed > limit:
            raise ExactSuffixResourceLimitError("elapsed_seconds", observed, limit)

    def visit_node(self) -> None:
        self.tree_nodes += 1
        limit = self.budget.max_tree_nodes
        if limit is not None and self.tree_nodes > limit:
            raise ExactSuffixResourceLimitError("tree_nodes", self.tree_nodes, limit)
        self.check_time()

    def visit_leaf(self) -> None:
        self.grouped_leaves += 1
        limit = self.budget.max_grouped_leaves
        if limit is not None and self.grouped_leaves > limit:
            raise ExactSuffixResourceLimitError(
                "grouped_leaves", self.grouped_leaves, limit
            )
        self.check_time()

    def check_distinct_suffixes(self, observed: int) -> None:
        limit = self.budget.max_distinct_suffixes
        if limit is not None and observed > limit:
            raise ExactSuffixResourceLimitError("distinct_suffixes", observed, limit)
        self.check_time()


def _validated_sequence(
    sequence: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, Path]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    array = check_adjacency(adjacency)
    values = tuple(check_states(sequence, array))
    if len(values) <= horizon:
        raise ValueError("sequence must contain at least one history state")
    return array, values


def _suffix_histogram_from_values(
    values: Path,
    horizon: int,
    decomposition: IBlockDecomposition,
    guard: _ResourceGuard,
    started: float,
) -> ExactSuffixHistogram:
    """Enumerate grouped terminal type words and attach indexed mass."""

    suffix_length = horizon + 1
    n_blocks = decomposition.n_blocks
    n_orders = math.factorial(n_blocks)
    type_counter = Counter(decomposition.blocks)
    block_types = tuple(sorted(type_counter))
    remaining = [type_counter[block] for block in block_types]

    suffix_weights: dict[Path, int] = {}
    suffix_aggregation_hits = 0
    depth_grouped_leaves: dict[int, int] = defaultdict(int)
    depth_indexed_assignments: dict[int, int] = defaultdict(int)
    depth_indexed_weight: dict[int, int] = defaultdict(int)

    def add_leaf(
        current_suffix: Path,
        depth: int,
        assignment_count: int,
    ) -> None:
        nonlocal suffix_aggregation_hits
        guard.visit_leaf()
        if len(current_suffix) >= suffix_length:
            suffix = current_suffix[-suffix_length:]
        else:
            # All blocks have been exhausted.  I_0 is fixed and supplies the
            # still-unknown left part of the requested suffix.
            suffix = (decomposition.prefix + current_suffix)[-suffix_length:]
            if len(suffix) != suffix_length:
                raise RuntimeError("fixed prefix did not complete the h+1 suffix")

        leaf_weight = assignment_count * math.factorial(n_blocks - depth)
        if suffix in suffix_weights:
            suffix_aggregation_hits += 1
        else:
            guard.check_distinct_suffixes(len(suffix_weights) + 1)
            suffix_weights[suffix] = 0
        suffix_weights[suffix] += leaf_weight
        depth_grouped_leaves[depth] += 1
        depth_indexed_assignments[depth] += assignment_count
        depth_indexed_weight[depth] += leaf_weight

    def visit(current_suffix: Path, depth: int, assignment_count: int) -> None:
        guard.visit_node()
        if len(current_suffix) >= suffix_length or depth == n_blocks:
            add_leaf(current_suffix, depth, assignment_count)
            return

        for type_index, block in enumerate(block_types):
            multiplicity = remaining[type_index]
            if multiplicity == 0:
                continue
            remaining[type_index] -= 1
            # Before reaching a leaf, current_suffix has length at most h.
            # Once the prepended block crosses h+1, its older leading states
            # can never affect the score and may be discarded immediately.
            extended = block + current_suffix
            if len(extended) > suffix_length:
                extended = extended[-suffix_length:]
            visit(
                extended,
                depth + 1,
                assignment_count * multiplicity,
            )
            remaining[type_index] += 1

    visit(decomposition.tail, 0, 1)
    total_weight = sum(suffix_weights.values())
    if total_weight != n_orders:
        raise RuntimeError(
            "right-to-left suffix leaves do not partition the full indexed group"
        )
    if sum(depth_indexed_weight.values()) != n_orders:
        raise RuntimeError("leaf-depth weights do not sum to D factorial")

    depths = tuple(sorted(depth_grouped_leaves))
    rows = tuple(
        WeightedSuffix(suffix=suffix, weight=weight)
        for suffix, weight in sorted(suffix_weights.items())
    )
    return ExactSuffixHistogram(
        scoring_version=EXACT_SUFFIX_SCORING_VERSION,
        horizon=horizon,
        prefix=decomposition.prefix,
        blocks=decomposition.blocks,
        tail=decomposition.tail,
        n_blocks=n_blocks,
        n_indexed_orders=n_orders,
        n_content_types=len(block_types),
        n_tree_nodes=guard.tree_nodes,
        n_grouped_leaves=guard.grouped_leaves,
        n_distinct_suffixes=len(rows),
        n_suffix_aggregation_hits=suffix_aggregation_hits,
        min_leaf_depth=min(depths),
        max_leaf_depth=max(depths),
        depth_masses=tuple(
            SuffixDepthMass(
                depth=depth,
                grouped_leaves=depth_grouped_leaves[depth],
                indexed_terminal_assignments=depth_indexed_assignments[depth],
                indexed_order_weight=depth_indexed_weight[depth],
            )
            for depth in depths
        ),
        suffix_weights=rows,
        total_weight=total_weight,
        elapsed_seconds=perf_counter() - started,
    )


def exact_weighted_suffix_histogram(
    sequence: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    budget: ExactSuffixBudget | None = None,
) -> ExactSuffixHistogram:
    """Return the full indexed suffix measure without enumerating ``D!`` orders.

    Identical block contents are grouped, but their indexed multiplicity is
    retained at every tree edge.  A leaf at depth ``k`` is weighted by its
    accumulated indexed assignment count times ``(D-k)!``.
    """

    started = perf_counter()
    array, values = _validated_sequence(sequence, horizon, adjacency)
    decomposition = decompose_i_blocks(values, values[-1])
    guard = _ResourceGuard(budget or ExactSuffixBudget(), started)
    # Keep validation of adjacency in the public contract even though block
    # decomposition itself depends only on the validated state labels.
    del array
    return _suffix_histogram_from_values(values, horizon, decomposition, guard, started)


def _multiply_exact_matrices(
    left: ExactMatrix,
    right: ExactMatrix,
    guard: _ResourceGuard,
) -> ExactMatrix:
    size = len(left)
    rows: list[tuple[Fraction, ...]] = []
    for row in range(size):
        guard.check_time()
        rows.append(
            tuple(
                sum(
                    (
                        left[row][middle] * right[middle][column]
                        for middle in range(size)
                    ),
                    start=Fraction(0, 1),
                )
                for column in range(size)
            )
        )
    return tuple(rows)


def _exact_observed_context(
    values: Path,
    horizon: int,
    adjacency: np.ndarray,
    guard: _ResourceGuard,
) -> tuple[Path, tuple[ExactMatrix, ...]]:
    """Build the Stage-1 observed-state matrix and powers using Fractions."""

    states = tuple(sorted(set(values)))
    state_to_local = {state: index for index, state in enumerate(states)}
    counts = transition_count_matrix(values, adjacency)

    base_rows: list[tuple[Fraction, ...]] = []
    for source in states:
        guard.check_time()
        denominator = int(counts[source - 1].sum())
        if denominator < 1:
            raise RuntimeError(
                f"observed state {source} has no outgoing count in D>=2 branch"
            )
        base_rows.append(
            tuple(
                Fraction(int(counts[source - 1, target - 1]), denominator)
                for target in states
            )
        )
    base: ExactMatrix = tuple(base_rows)
    for row in base:
        if sum(row, start=Fraction(0, 1)) != 1:
            raise RuntimeError("exact observed-state matrix is not row-stochastic")

    powers: list[ExactMatrix] = []
    current = base
    for step in range(horizon):
        guard.check_time()
        powers.append(current)
        if step + 1 < horizon:
            current = _multiply_exact_matrices(current, base, guard)
    # Referencing the mapping here documents that matrix row order is tied to
    # the returned state tuple, not to an ambient adjacency row order.
    if len(state_to_local) != len(states):  # pragma: no cover - defensive
        raise RuntimeError("observed state labels are not unique")
    return states, tuple(powers)


def exact_suffix_score(
    suffix: SequenceLike,
    powers: Sequence[ExactMatrix],
    observed_states: Sequence[int],
) -> Fraction:
    """Evaluate Eq. (3.A5) exactly from one final ``h+1`` suffix."""

    values = tuple(int(value) for value in suffix)
    horizon = len(powers)
    if len(values) != horizon + 1:
        raise ValueError("suffix length must equal the number of powers plus one")
    state_to_local = {int(state): index for index, state in enumerate(observed_states)}
    try:
        start = state_to_local[values[0]]
        probability_sum = sum(
            (
                powers[step - 1][start][state_to_local[values[step]]]
                for step in range(1, horizon + 1)
            ),
            start=Fraction(0, 1),
        )
    except KeyError as error:
        raise ValueError("suffix contains a state absent from the matrix") from error
    score = Fraction(1, 1) - probability_sum / horizon
    if score < 0 or score > 1:
        raise RuntimeError("exact suffix score lies outside [0, 1]")
    return score


def conservative_block_diagnostic_exact_suffix(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    budget: ExactSuffixBudget | None = None,
) -> ExactSuffixBlockDiagnostic:
    """Compute the exact v8.1 conservative diagnostic by suffix weights.

    The candidate is not screened for adjacency here, matching the Stage-1
    block diagnostic.  Thus a direct off-graph call still has a well-defined
    block coordinate; the hybrid count branch decides whether the total score
    is infinite.  ``D<=1`` returns the identity self-rank without constructing
    an observed-state matrix or any matrix power.
    """

    started = perf_counter()
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(history, array))
    candidate_values = tuple(check_states(candidate, array))
    values = history_values + candidate_values
    horizon = len(candidate_values)
    decomposition = decompose_i_blocks(values, values[-1])
    guard = _ResourceGuard(budget or ExactSuffixBudget(), started)

    enumeration_started = perf_counter()
    histogram = _suffix_histogram_from_values(
        values, horizon, decomposition, guard, enumeration_started
    )
    enumeration_seconds = perf_counter() - enumeration_started
    identity_suffix = values[-(horizon + 1) :]

    if decomposition.n_blocks <= 1:
        total_seconds = perf_counter() - started
        return ExactSuffixBlockDiagnostic(
            scoring_version=EXACT_SUFFIX_SCORING_VERSION,
            n_blocks=decomposition.n_blocks,
            n_indexed_orders=1,
            n_content_types=histogram.n_content_types,
            n_tree_nodes=histogram.n_tree_nodes,
            n_grouped_leaves=histogram.n_grouped_leaves,
            n_distinct_suffixes=histogram.n_distinct_suffixes,
            n_suffix_aggregation_hits=histogram.n_suffix_aggregation_hits,
            min_leaf_depth=histogram.min_leaf_depth,
            max_leaf_depth=histogram.max_leaf_depth,
            depth_masses=histogram.depth_masses,
            suffix_weights=histogram.suffix_weights,
            score_histogram=(),
            identity_suffix=identity_suffix,
            identity_score=None,
            n_strictly_greater=0,
            n_equal_identity=1,
            n_at_least_identity=1,
            q_conservative=Fraction(1, 1),
            block_feature=Fraction(0, 1),
            observed_states=(),
            score_cache_hits=0,
            score_cache_misses=0,
            enumeration_seconds=enumeration_seconds,
            matrix_seconds=0.0,
            scoring_seconds=0.0,
            total_seconds=total_seconds,
        )

    matrix_started = perf_counter()
    observed_states, powers = _exact_observed_context(values, horizon, array, guard)
    matrix_seconds = perf_counter() - matrix_started

    scoring_started = perf_counter()
    score_cache: dict[Path, Fraction] = {}
    score_cache_hits = 0
    score_cache_misses = 0

    def score_for(suffix: Path) -> Fraction:
        nonlocal score_cache_hits, score_cache_misses
        guard.check_time()
        if suffix in score_cache:
            score_cache_hits += 1
            return score_cache[suffix]
        score_cache_misses += 1
        score = exact_suffix_score(suffix, powers, observed_states)
        score_cache[suffix] = score
        return score

    score_weights: dict[Fraction, int] = defaultdict(int)
    for row in histogram.suffix_weights:
        score_weights[score_for(row.suffix)] += row.weight
    identity_score = score_for(identity_suffix)

    n_strictly_greater = sum(
        weight for score, weight in score_weights.items() if score > identity_score
    )
    n_equal = score_weights.get(identity_score, 0)
    n_at_least = n_strictly_greater + n_equal
    if n_at_least < 1 or n_at_least > histogram.n_indexed_orders:
        raise RuntimeError("exact conservative numerator lies outside [1, D!]")
    if sum(score_weights.values()) != histogram.n_indexed_orders:
        raise RuntimeError("exact score histogram does not have total mass D!")

    q_conservative = Fraction(n_at_least, histogram.n_indexed_orders)
    block_feature = Fraction(1, 1) - q_conservative
    scoring_seconds = perf_counter() - scoring_started
    guard.check_time()
    total_seconds = perf_counter() - started

    return ExactSuffixBlockDiagnostic(
        scoring_version=EXACT_SUFFIX_SCORING_VERSION,
        n_blocks=histogram.n_blocks,
        n_indexed_orders=histogram.n_indexed_orders,
        n_content_types=histogram.n_content_types,
        n_tree_nodes=histogram.n_tree_nodes,
        n_grouped_leaves=histogram.n_grouped_leaves,
        n_distinct_suffixes=histogram.n_distinct_suffixes,
        n_suffix_aggregation_hits=histogram.n_suffix_aggregation_hits,
        min_leaf_depth=histogram.min_leaf_depth,
        max_leaf_depth=histogram.max_leaf_depth,
        depth_masses=histogram.depth_masses,
        suffix_weights=histogram.suffix_weights,
        score_histogram=tuple(
            ExactScoreMass(score=score, weight=weight)
            for score, weight in sorted(score_weights.items())
        ),
        identity_suffix=identity_suffix,
        identity_score=identity_score,
        n_strictly_greater=n_strictly_greater,
        n_equal_identity=n_equal,
        n_at_least_identity=n_at_least,
        q_conservative=q_conservative,
        block_feature=block_feature,
        observed_states=observed_states,
        score_cache_hits=score_cache_hits,
        score_cache_misses=score_cache_misses,
        enumeration_seconds=enumeration_seconds,
        matrix_seconds=matrix_seconds,
        scoring_seconds=scoring_seconds,
        total_seconds=total_seconds,
    )
