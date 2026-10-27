"""Reviewable exact OCHS v8.1 pipeline for Stage 2A.

The Stage-1 implementation in :mod:`method3.ochs_v81` remains the retained
``rounded14`` reference profile and brute-force block oracle.  This module
connects the Stage-2A raw-score numerical contract to the exact weighted-suffix
block backend.  It deliberately contains no simulation, fitted tuning,
permutation cap, or random tie breaking.

All authoritative ranks are integers or :class:`fractions.Fraction` objects.
Floating-point values in the underlying exact modules are timing diagnostics
only.  A resource-limit or uncertified comparison returns ``NOT_COMPLETED``
and never a partial prediction set labelled exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from time import perf_counter
from typing import Hashable, Sequence

import numpy as np

from .hybrid import GROUP_ALL_STATE_1, GROUP_OTHER, history_group
from .ochs_v81 import (
    CompletedReference,
    history_graph_diagnostic,
    legal_candidates,
)
from .ochs_v81_exact_numeric import (
    EXACT_SCORING_VERSION,
    ComparisonBudget,
    ComparisonRelation,
    CountWeightStatus,
    ExactBlockCoordinate,
    ExactCountWeight,
    ExactOuterDecision,
    ExactQuantileResult,
    ExactScoreComponents,
    OuterDecisionStatus,
    QuantileStatus,
    RationalInput,
    RationalParameter,
    ThresholdKind,
    compare_exact_scores,
    exact_augmented_count_weight,
    exact_outer_decision,
    exact_outer_quantile,
    parse_alpha,
    parse_lambda,
)
from .ochs_v81_exact_suffix import (
    EXACT_SUFFIX_SCORING_VERSION,
    ExactSuffixBlockDiagnostic,
    ExactSuffixBudget,
    ExactSuffixResourceLimitError,
    conservative_block_diagnostic_exact_suffix,
)
from .states import Path, SequenceLike, check_adjacency, check_states


class ExactPipelineStatus(str, Enum):
    """Completion state of an exact prediction-set request."""

    COMPLETED = "COMPLETED"
    NOT_COMPLETED = "NOT_COMPLETED"


@dataclass(frozen=True)
class ExactEpisodeKey:
    """Immutable cache identity for one fully augmented episode.

    The key includes the raw-score and suffix-backend version strings, the
    complete history and future, and every adjacency entry.  In particular, a
    block result or candidate-specific empirical matrix can never be cached by
    history alone.
    """

    scoring_version: str
    suffix_scoring_version: str
    history: Path
    future: Path
    adjacency: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ExactCacheSnapshot:
    """Cumulative explicit cache and backend-work counters."""

    count_hits: int = 0
    count_misses: int = 0
    block_hits: int = 0
    block_misses: int = 0
    matrix_contexts_built: int = 0
    suffix_score_hits: int = 0
    suffix_score_misses: int = 0
    suffix_grouped_leaves: int = 0
    suffix_distinct_states: int = 0

    def delta(self, earlier: "ExactCacheSnapshot") -> "ExactCacheSnapshot":
        """Return nonnegative work performed since ``earlier``."""

        names = self.__dataclass_fields__
        values = {name: getattr(self, name) - getattr(earlier, name) for name in names}
        if any(value < 0 for value in values.values()):
            raise ValueError("cache snapshots are not in chronological order")
        return ExactCacheSnapshot(**values)


class ExactScoreCache:
    """Explicit reusable cache for exact episode components.

    Only completed exact values are inserted.  Resource-limit exceptions are
    propagated without caching the work accumulated before failure.
    """

    def __init__(self) -> None:
        self._count: dict[ExactEpisodeKey, ExactCountWeight] = {}
        self._block: dict[ExactEpisodeKey, ExactSuffixBlockDiagnostic] = {}
        self._count_hits = 0
        self._count_misses = 0
        self._block_hits = 0
        self._block_misses = 0
        self._matrix_contexts_built = 0
        self._suffix_score_hits = 0
        self._suffix_score_misses = 0
        self._suffix_grouped_leaves = 0
        self._suffix_distinct_states = 0

    def snapshot(self) -> ExactCacheSnapshot:
        """Return immutable cumulative diagnostics."""

        return ExactCacheSnapshot(
            count_hits=self._count_hits,
            count_misses=self._count_misses,
            block_hits=self._block_hits,
            block_misses=self._block_misses,
            matrix_contexts_built=self._matrix_contexts_built,
            suffix_score_hits=self._suffix_score_hits,
            suffix_score_misses=self._suffix_score_misses,
            suffix_grouped_leaves=self._suffix_grouped_leaves,
            suffix_distinct_states=self._suffix_distinct_states,
        )

    def count_component(
        self,
        key: ExactEpisodeKey,
        *,
        history: Path,
        future: Path,
        adjacency: np.ndarray,
    ) -> tuple[ExactCountWeight, bool, float]:
        """Return an exact count component and hit/timing diagnostics."""

        if key in self._count:
            self._count_hits += 1
            return self._count[key], True, 0.0
        started = perf_counter()
        value = exact_augmented_count_weight(history, future, adjacency)
        elapsed = perf_counter() - started
        self._count[key] = value
        self._count_misses += 1
        return value, False, elapsed

    def block_component(
        self,
        key: ExactEpisodeKey,
        *,
        history: Path,
        future: Path,
        adjacency: np.ndarray,
        budget: ExactSuffixBudget | None,
    ) -> tuple[ExactSuffixBlockDiagnostic, bool, float]:
        """Return a completed suffix diagnostic and hit/timing diagnostics."""

        if key in self._block:
            self._block_hits += 1
            return self._block[key], True, 0.0
        started = perf_counter()
        value = conservative_block_diagnostic_exact_suffix(
            history,
            future,
            adjacency,
            budget=budget,
        )
        elapsed = perf_counter() - started
        self._block[key] = value
        self._block_misses += 1
        if value.n_blocks >= 2:
            self._matrix_contexts_built += 1
        self._suffix_score_hits += value.score_cache_hits
        self._suffix_score_misses += value.score_cache_misses
        self._suffix_grouped_leaves += value.n_grouped_leaves
        self._suffix_distinct_states += value.n_distinct_suffixes
        return value, False, elapsed


@dataclass(frozen=True)
class ExactPreparedScore:
    """One exact score together with auditable episode diagnostics."""

    history: Path
    future: Path
    group: int
    history_graph_compatible: bool
    components: ExactScoreComponents
    block_diagnostic: ExactSuffixBlockDiagnostic | None
    count_cache_hit: bool | None
    block_cache_hit: bool | None
    count_seconds: float
    block_seconds: float


@dataclass(frozen=True)
class ExactReferenceScore:
    """One same-group ordinary reference scored on its own realized future."""

    reference_id: Hashable
    prepared: ExactPreparedScore

    @property
    def history(self) -> Path:
        return self.prepared.history

    @property
    def realized_future(self) -> Path:
        return self.prepared.future

    @property
    def components(self) -> ExactScoreComponents:
        return self.prepared.components


@dataclass(frozen=True)
class ExactCandidateResult:
    """One candidate's exact outer rank and quantile cross-check."""

    candidate: Path
    prepared: ExactPreparedScore
    decision: ExactOuterDecision
    quantile_relation: ComparisonRelation | None

    @property
    def p_value(self) -> Fraction:
        if self.decision.p_value is None:
            raise RuntimeError("an incomplete decision has no exact p-value")
        return self.decision.p_value

    @property
    def included(self) -> bool:
        if self.decision.included is None:
            raise RuntimeError("an incomplete decision has no inclusion result")
        return self.decision.included


@dataclass(frozen=True)
class ExactPredictionResult:
    """Complete result, or an explicit failure with no partial prediction set."""

    status: ExactPipelineStatus
    reason: str | None
    target_id: Hashable
    target_group: int
    target_history: Path
    target_history_graph_compatible: bool
    horizon: int
    alpha: RationalParameter
    lam: RationalParameter
    n_same_group: int
    n_legal_candidates: int
    reference_details: tuple[ExactReferenceScore, ...]
    candidates: tuple[ExactCandidateResult, ...]
    prediction_set: tuple[Path, ...] | None
    quantile: ExactQuantileResult | None
    cache_delta: ExactCacheSnapshot
    total_seconds: float

    @property
    def exact_result_available(self) -> bool:
        return self.status is ExactPipelineStatus.COMPLETED


def _validated_episode(
    history: SequenceLike,
    future: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, Path, Path, ExactEpisodeKey]:
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(history, array))
    future_values = tuple(check_states(future, array))
    key = ExactEpisodeKey(
        scoring_version=EXACT_SCORING_VERSION,
        suffix_scoring_version=EXACT_SUFFIX_SCORING_VERSION,
        history=history_values,
        future=future_values,
        adjacency=tuple(tuple(int(value) for value in row) for row in array),
    )
    return array, history_values, future_values, key


def prepare_exact_score(
    history: SequenceLike,
    future: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    *,
    lam: RationalInput | RationalParameter = Fraction(1, 2),
    cache: ExactScoreCache | None = None,
    suffix_budget: ExactSuffixBudget | None = None,
) -> ExactPreparedScore:
    """Build only the score components active at the declared endpoint.

    Group 1 is always count-only.  For group 0, ``lambda=0`` bypasses count
    ratios and ``lambda=1`` bypasses every block/matrix calculation.  A mixed
    score builds both exact components.
    """

    lambda_parameter = parse_lambda(lam)
    array, history_values, future_values, key = _validated_episode(
        history, future, adjacency
    )
    horizon = len(future_values)
    group = history_group(history_values)
    diagnostic = history_graph_diagnostic(history_values, array)
    component_cache = cache if cache is not None else ExactScoreCache()

    need_count = group == GROUP_ALL_STATE_1 or lambda_parameter.value > 0
    need_block = group == GROUP_OTHER and lambda_parameter.value < 1

    if need_count:
        count, count_hit, count_seconds = component_cache.count_component(
            key,
            history=history_values,
            future=future_values,
            adjacency=array,
        )
    else:
        count = ExactCountWeight.not_evaluated(horizon)
        count_hit = None
        count_seconds = 0.0

    # Preserve the Stage-1 extended-real branch exactly: when the count term
    # has positive weight and is +infinity, the total score is already fixed.
    # An unused block calculation must not be allowed to fail a result whose
    # mathematical value is known.
    if count.status is CountWeightStatus.POSITIVE_INFINITY:
        need_block = False

    block_diagnostic: ExactSuffixBlockDiagnostic | None = None
    if need_block:
        block_diagnostic, block_hit, block_seconds = component_cache.block_component(
            key,
            history=history_values,
            future=future_values,
            adjacency=array,
            budget=suffix_budget,
        )
        block = ExactBlockCoordinate.from_rank_counts(
            D=block_diagnostic.n_blocks,
            n_strict_greater=block_diagnostic.n_strictly_greater,
            n_equal=block_diagnostic.n_equal_identity,
            identity_score=block_diagnostic.identity_score,
        )
    else:
        block = None
        block_hit = None
        block_seconds = 0.0

    components = ExactScoreComponents(
        group=group,
        horizon=horizon,
        count=count,
        block=block,
    )
    return ExactPreparedScore(
        history=history_values,
        future=future_values,
        group=group,
        history_graph_compatible=diagnostic.graph_compatible,
        components=components,
        block_diagnostic=block_diagnostic,
        count_cache_hit=count_hit,
        block_cache_hit=block_hit,
        count_seconds=count_seconds,
        block_seconds=block_seconds,
    )


def exact_reference_true_scores(
    references: Sequence[CompletedReference],
    *,
    target_id: Hashable,
    target_group: int,
    history_length: int,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    lam: RationalInput | RationalParameter = Fraction(1, 2),
    cache: ExactScoreCache | None = None,
    suffix_budget: ExactSuffixBudget | None = None,
) -> tuple[ExactReferenceScore, ...]:
    """Score same-group ordinary references on their own realized futures."""

    if target_group not in (GROUP_OTHER, GROUP_ALL_STATE_1):
        raise ValueError("target_group must be 0 or 1")
    if history_length < 1 or horizon < 1:
        raise ValueError("history_length and horizon must be positive")
    array = check_adjacency(adjacency)
    lambda_parameter = parse_lambda(lam)
    rows = tuple(references)
    seen_ids: set[Hashable] = set()
    validated: list[tuple[CompletedReference, Path, Path]] = []
    for reference in rows:
        if reference.reference_id == target_id:
            raise ValueError("target_id appears in the ordinary reference pool")
        if reference.reference_id in seen_ids:
            raise ValueError("ordinary reference identifiers must be distinct")
        seen_ids.add(reference.reference_id)
        history_values = tuple(check_states(reference.history, array))
        future_values = tuple(check_states(reference.realized_future, array))
        if len(history_values) != history_length:
            raise ValueError("every reference history must have the fixed length T")
        if len(future_values) != horizon:
            raise ValueError("every reference future must have the fixed horizon h")
        validated.append((reference, history_values, future_values))

    component_cache = cache if cache is not None else ExactScoreCache()
    result: list[ExactReferenceScore] = []
    for reference, history_values, future_values in validated:
        if history_group(history_values) != target_group:
            continue
        prepared = prepare_exact_score(
            history_values,
            future_values,
            array,
            lam=lambda_parameter,
            cache=component_cache,
            suffix_budget=suffix_budget,
        )
        result.append(
            ExactReferenceScore(
                reference_id=reference.reference_id,
                prepared=prepared,
            )
        )
    return tuple(result)


def _incomplete_result(
    *,
    reason: str,
    target_id: Hashable,
    target_group: int,
    target_history: Path,
    target_history_graph_compatible: bool,
    horizon: int,
    alpha: RationalParameter,
    lam: RationalParameter,
    n_same_group: int,
    n_legal_candidates: int,
    cache_delta: ExactCacheSnapshot,
    total_seconds: float,
) -> ExactPredictionResult:
    """Build a failure record that cannot expose a partial exact set."""

    return ExactPredictionResult(
        status=ExactPipelineStatus.NOT_COMPLETED,
        reason=reason,
        target_id=target_id,
        target_group=target_group,
        target_history=target_history,
        target_history_graph_compatible=target_history_graph_compatible,
        horizon=horizon,
        alpha=alpha,
        lam=lam,
        n_same_group=n_same_group,
        n_legal_candidates=n_legal_candidates,
        reference_details=(),
        candidates=(),
        prediction_set=None,
        quantile=None,
        cache_delta=cache_delta,
        total_seconds=total_seconds,
    )


def ochs_prediction_set_exact(
    *,
    target_id: Hashable,
    target_history: SequenceLike,
    horizon: int,
    alpha: RationalInput | RationalParameter,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    references: Sequence[CompletedReference],
    lam: RationalInput | RationalParameter = Fraction(1, 2),
    cache: ExactScoreCache | None = None,
    suffix_budget: ExactSuffixBudget | None = None,
    comparison_budget: ComparisonBudget = ComparisonBudget(),
) -> ExactPredictionResult:
    """Construct the raw-score exact v8.1 prediction set.

    The direct outer-rank decision and the independently derived finite-rank
    quantile decision are checked candidate by candidate.  A suffix resource
    failure or any mixed comparison that cannot be certified yields one
    explicit ``NOT_COMPLETED`` record with ``prediction_set=None``.
    """

    started = perf_counter()
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    alpha_parameter = parse_alpha(alpha)
    lambda_parameter = parse_lambda(lam)
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(target_history, array))
    target_group = history_group(history_values)
    target_diagnostic = history_graph_diagnostic(history_values, array)
    legal = legal_candidates(history_values, horizon, array)
    component_cache = cache if cache is not None else ExactScoreCache()
    before = component_cache.snapshot()

    # Materialize and validate reference metadata before component work.  This
    # makes N_* authoritative even if a later exact suffix calculation hits a
    # resource guard, and it prevents one-shot iterables from being consumed
    # differently by preflight and scoring.
    reference_rows = tuple(references)
    seen_ids: set[Hashable] = set()
    known_n_same_group = 0
    for reference in reference_rows:
        if reference.reference_id == target_id:
            raise ValueError("target_id appears in the ordinary reference pool")
        if reference.reference_id in seen_ids:
            raise ValueError("ordinary reference identifiers must be distinct")
        seen_ids.add(reference.reference_id)
        reference_history = tuple(check_states(reference.history, array))
        reference_future = tuple(check_states(reference.realized_future, array))
        if len(reference_history) != len(history_values):
            raise ValueError("every reference history must have the fixed length T")
        if len(reference_future) != horizon:
            raise ValueError("every reference future must have the fixed horizon h")
        if history_group(reference_history) == target_group:
            known_n_same_group += 1

    try:
        reference_details = exact_reference_true_scores(
            reference_rows,
            target_id=target_id,
            target_group=target_group,
            history_length=len(history_values),
            horizon=horizon,
            adjacency=array,
            lam=lambda_parameter,
            cache=component_cache,
            suffix_budget=suffix_budget,
        )
        reference_components = tuple(row.components for row in reference_details)
        quantile = exact_outer_quantile(
            reference_components,
            alpha=alpha_parameter,
            lam=lambda_parameter,
            budget=comparison_budget,
        )
        if quantile.status is not QuantileStatus.COMPLETE:
            return _incomplete_result(
                reason="UNCERTIFIED_REFERENCE_QUANTILE_COMPARISON",
                target_id=target_id,
                target_group=target_group,
                target_history=history_values,
                target_history_graph_compatible=target_diagnostic.graph_compatible,
                horizon=horizon,
                alpha=alpha_parameter,
                lam=lambda_parameter,
                n_same_group=known_n_same_group,
                n_legal_candidates=len(legal),
                cache_delta=component_cache.snapshot().delta(before),
                total_seconds=perf_counter() - started,
            )

        candidate_rows: list[ExactCandidateResult] = []
        retained: list[Path] = []
        for candidate in legal:
            prepared = prepare_exact_score(
                history_values,
                candidate,
                array,
                lam=lambda_parameter,
                cache=component_cache,
                suffix_budget=suffix_budget,
            )
            decision = exact_outer_decision(
                reference_components,
                prepared.components,
                alpha=alpha_parameter,
                lam=lambda_parameter,
                budget=comparison_budget,
            )
            if decision.status is not OuterDecisionStatus.COMPLETE:
                return _incomplete_result(
                    reason="UNCERTIFIED_CANDIDATE_OUTER_COMPARISON",
                    target_id=target_id,
                    target_group=target_group,
                    target_history=history_values,
                    target_history_graph_compatible=target_diagnostic.graph_compatible,
                    horizon=horizon,
                    alpha=alpha_parameter,
                    lam=lambda_parameter,
                    n_same_group=known_n_same_group,
                    n_legal_candidates=len(legal),
                    cache_delta=component_cache.snapshot().delta(before),
                    total_seconds=perf_counter() - started,
                )

            quantile_relation: ComparisonRelation | None = None
            if quantile.threshold_kind is ThresholdKind.EMPTY:
                quantile_included = False
            elif quantile.threshold_kind is ThresholdKind.POSITIVE_INFINITY:
                quantile_included = True
            else:
                if quantile.threshold is None:
                    raise RuntimeError("finite exact quantile omitted its threshold")
                comparison = compare_exact_scores(
                    prepared.components,
                    quantile.threshold,
                    lam=lambda_parameter,
                    budget=comparison_budget,
                )
                if not comparison.certified:
                    return _incomplete_result(
                        reason="UNCERTIFIED_CANDIDATE_QUANTILE_COMPARISON",
                        target_id=target_id,
                        target_group=target_group,
                        target_history=history_values,
                        target_history_graph_compatible=target_diagnostic.graph_compatible,
                        horizon=horizon,
                        alpha=alpha_parameter,
                        lam=lambda_parameter,
                        n_same_group=known_n_same_group,
                        n_legal_candidates=len(legal),
                        cache_delta=component_cache.snapshot().delta(before),
                        total_seconds=perf_counter() - started,
                    )
                quantile_relation = comparison.relation
                quantile_included = comparison.relation in (
                    ComparisonRelation.LESS,
                    ComparisonRelation.EQUAL,
                )

            if decision.included is not quantile_included:
                raise RuntimeError("exact p-rank and exact quantile decisions disagree")
            row = ExactCandidateResult(
                candidate=candidate,
                prepared=prepared,
                decision=decision,
                quantile_relation=quantile_relation,
            )
            candidate_rows.append(row)
            if row.included:
                retained.append(candidate)

    except ExactSuffixResourceLimitError as error:
        return _incomplete_result(
            reason=(
                f"EXACT_SUFFIX_RESOURCE_LIMIT:{error.resource}:"
                f"{error.observed!r}>{error.limit!r}"
            ),
            target_id=target_id,
            target_group=target_group,
            target_history=history_values,
            target_history_graph_compatible=target_diagnostic.graph_compatible,
            horizon=horizon,
            alpha=alpha_parameter,
            lam=lambda_parameter,
            n_same_group=known_n_same_group,
            n_legal_candidates=len(legal),
            cache_delta=component_cache.snapshot().delta(before),
            total_seconds=perf_counter() - started,
        )

    return ExactPredictionResult(
        status=ExactPipelineStatus.COMPLETED,
        reason=None,
        target_id=target_id,
        target_group=target_group,
        target_history=history_values,
        target_history_graph_compatible=target_diagnostic.graph_compatible,
        horizon=horizon,
        alpha=alpha_parameter,
        lam=lambda_parameter,
        n_same_group=len(reference_details),
        n_legal_candidates=len(legal),
        reference_details=reference_details,
        candidates=tuple(candidate_rows),
        prediction_set=tuple(retained),
        quantile=quantile,
        cache_delta=component_cache.snapshot().delta(before),
        total_seconds=perf_counter() - started,
    )


__all__ = [
    "ExactCacheSnapshot",
    "ExactCandidateResult",
    "ExactEpisodeKey",
    "ExactPipelineStatus",
    "ExactPredictionResult",
    "ExactPreparedScore",
    "ExactReferenceScore",
    "ExactScoreCache",
    "exact_reference_true_scores",
    "ochs_prediction_set_exact",
    "prepare_exact_score",
]
