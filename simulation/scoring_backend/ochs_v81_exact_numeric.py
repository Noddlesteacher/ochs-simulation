"""Exact numerical semantics for the OCHS v8.1 Stage-2A engine.

This module is deliberately separate from :mod:`method3.ochs_v81`.  The
Stage-1 module remains the reproducible ``rounded14`` comparison profile;
the primary functions below compare the unrounded mathematical scores.

The finite count coordinate is represented by its exact fitted weight

``W = product_l N_edge(H || y) / N_row(H || y)``.

No floating-point logarithm is authoritative.  Count-only order is obtained
by reversing the order of ``W``.  Mixed-score order is certified with exact
rational intervals for ``log(W_j / W_i)``.  Failure to separate an interval
from zero within the declared resource budget is returned as an explicit
``UNCERTIFIED_PRECISION_LIMIT`` status and is never converted to a tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from functools import lru_cache
import math
from typing import Sequence, TypeAlias

from .hybrid import GROUP_ALL_STATE_1, GROUP_OTHER
from .ochs_v81 import augmented_transition_counts, future_is_legal
from .states import SequenceLike, check_adjacency, check_states

EXACT_SCORING_VERSION = "ochs-v8.1-raw-exact-stage2a"
LEGACY_ROUNDED_PROFILE = "ochs-v8.1-stage1-rounded14"


class CountWeightStatus(str, Enum):
    """Extended-real state of the direct augmented-count coordinate."""

    FINITE = "FINITE"
    POSITIVE_INFINITY = "POSITIVE_INFINITY"
    NOT_EVALUATED = "NOT_EVALUATED"


class ComparisonStatus(str, Enum):
    """Whether a score ordering is exact, certified, or unresolved."""

    EXACT = "EXACT"
    CERTIFIED_INTERVAL = "CERTIFIED_INTERVAL"
    UNCERTIFIED_PRECISION_LIMIT = "UNCERTIFIED_PRECISION_LIMIT"


class ComparisonRelation(IntEnum):
    """Ordering of the left nonconformity score relative to the right."""

    LESS = -1
    EQUAL = 0
    GREATER = 1


class OuterDecisionStatus(str, Enum):
    """Completion state for an exact outer-rank decision."""

    COMPLETE = "COMPLETE"
    UNCERTIFIED_COMPARISON = "UNCERTIFIED_COMPARISON"


class QuantileStatus(str, Enum):
    """Completion state for an exact outer quantile."""

    COMPLETE = "COMPLETE"
    UNCERTIFIED_COMPARISON = "UNCERTIFIED_COMPARISON"


class ThresholdKind(str, Enum):
    """Type of threshold implied by the exact finite-rank grid."""

    EMPTY = "EMPTY"
    POSITIVE_INFINITY = "POSITIVE_INFINITY"
    FINITE = "FINITE"


RationalInput: TypeAlias = str | int | Fraction | Decimal


@dataclass(frozen=True)
class RationalParameter:
    """An exact configuration value together with its declared source token."""

    name: str
    value: Fraction
    source_text: str
    source_kind: str


def parse_rational_parameter(
    value: RationalInput | RationalParameter,
    *,
    name: str,
) -> RationalParameter:
    """Parse an exact parameter in ``[0, 1]`` without guessing from a float.

    Decimal strings (``"0.10"``), rational strings (``"1/3"``), integers,
    finite :class:`~decimal.Decimal` values, and :class:`~fractions.Fraction`
    values are accepted.  A bare binary ``float`` is rejected deliberately:
    callers must state whether they mean a decimal token or its exact binary
    value rather than having this module choose silently.
    """
    if isinstance(value, RationalParameter):
        parsed = value
        if parsed.name != name:
            parsed = RationalParameter(
                name=name,
                value=parsed.value,
                source_text=parsed.source_text,
                source_kind=parsed.source_kind,
            )
    elif isinstance(value, bool) or isinstance(value, float):
        raise TypeError(
            f"{name} must be an exact decimal string, Fraction, Decimal, or int; "
            "bare floats are not accepted"
        )
    elif isinstance(value, Fraction):
        parsed = RationalParameter(
            name=name,
            value=value,
            source_text=f"{value.numerator}/{value.denominator}",
            source_kind="fraction",
        )
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        parsed = RationalParameter(
            name=name,
            value=Fraction(value),
            source_text=str(value),
            source_kind="decimal",
        )
    elif isinstance(value, int):
        parsed = RationalParameter(
            name=name,
            value=Fraction(value, 1),
            source_text=str(value),
            source_kind="integer",
        )
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            raise ValueError(f"{name} must not be an empty string")
        try:
            fraction = Fraction(token)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                f"{name} is not an exact rational token: {value!r}"
            ) from exc
        parsed = RationalParameter(
            name=name,
            value=fraction,
            source_text=token,
            source_kind="fraction" if "/" in token else "decimal",
        )
    else:  # pragma: no cover - guarded by the public type and defensive at runtime
        raise TypeError(f"unsupported exact {name} type: {type(value).__name__}")

    if not Fraction(0, 1) <= parsed.value <= Fraction(1, 1):
        raise ValueError(f"{name} must lie in [0, 1]")
    return parsed


def parse_alpha(value: RationalInput | RationalParameter) -> RationalParameter:
    """Return the exact alpha configuration."""
    return parse_rational_parameter(value, name="alpha")


def parse_lambda(value: RationalInput | RationalParameter) -> RationalParameter:
    """Return the exact lambda configuration."""
    return parse_rational_parameter(value, name="lambda")


@dataclass(frozen=True)
class ExactCountWeight:
    """Exact fitted count weight, or an explicit extended-real state."""

    status: CountWeightStatus
    horizon: int
    weight: Fraction | None = None
    factors: tuple[Fraction, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.status is CountWeightStatus.FINITE:
            if self.weight is None or not Fraction(0, 1) < self.weight <= 1:
                raise ValueError("a finite fitted weight must lie in (0, 1]")
            if len(self.factors) != self.horizon:
                raise ValueError(
                    "a finite count record needs one factor per horizon step"
                )
            product = math.prod(self.factors, start=Fraction(1, 1))
            if product != self.weight:
                raise ValueError("count factors do not multiply to the recorded weight")
            if any(not Fraction(0, 1) < factor <= 1 for factor in self.factors):
                raise ValueError("every fitted count factor must lie in (0, 1]")
        elif self.weight is not None or self.factors:
            raise ValueError(
                "non-finite or unevaluated count records carry no fitted weight"
            )

    @classmethod
    def finite(cls, factors: Sequence[Fraction | int]) -> "ExactCountWeight":
        """Build a finite record from the ordered fitted transition factors."""
        exact_factors = tuple(Fraction(value) for value in factors)
        if not exact_factors:
            raise ValueError("at least one fitted factor is required")
        return cls(
            status=CountWeightStatus.FINITE,
            horizon=len(exact_factors),
            weight=math.prod(exact_factors, start=Fraction(1, 1)),
            factors=exact_factors,
        )

    @classmethod
    def positive_infinity(cls, horizon: int) -> "ExactCountWeight":
        """Represent the ``+infinity`` count score of an illegal future."""
        return cls(status=CountWeightStatus.POSITIVE_INFINITY, horizon=horizon)

    @classmethod
    def not_evaluated(cls, horizon: int) -> "ExactCountWeight":
        """Represent an endpoint component that was intentionally bypassed."""
        return cls(status=CountWeightStatus.NOT_EVALUATED, horizon=horizon)


def exact_augmented_count_weight(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]],
) -> ExactCountWeight:
    """Return the exact fitted weight from the unmasked counts of ``H || y``.

    A graph-illegal future retains the Stage-1 total-score convention and is
    represented by ``POSITIVE_INFINITY``.  For a legal future, every factor is
    a positive exact rational because its scored edge occurs in ``H || y``.
    """
    array = check_adjacency(adjacency)
    history_values = tuple(check_states(history, array))
    candidate_values = tuple(check_states(candidate, array))
    if not future_is_legal(history_values, candidate_values, array):
        return ExactCountWeight.positive_infinity(len(candidate_values))

    augmented = augmented_transition_counts(history_values, candidate_values, array)
    factors: list[Fraction] = []
    previous = history_values[-1]
    for state in candidate_values:
        numerator = int(augmented.counts[previous - 1, state - 1])
        denominator = int(augmented.outgoing_counts[previous - 1])
        if numerator < 1 or denominator < 1 or numerator > denominator:
            raise RuntimeError(
                "legal future produced an invalid augmented-count factor"
            )
        factors.append(Fraction(numerator, denominator))
        previous = state
    return ExactCountWeight.finite(factors)


@dataclass(frozen=True)
class ExactBlockCoordinate:
    """Exact conservative block rank with the unreduced indexed weights."""

    D: int
    total_weight: int
    n_strict_greater: int
    n_equal: int
    identity_score: Fraction | None = None

    def __post_init__(self) -> None:
        if self.D < 0:
            raise ValueError("D must be nonnegative")
        if self.total_weight != math.factorial(self.D):
            raise ValueError("total block weight must equal D factorial")
        if self.n_strict_greater < 0 or self.n_equal < 1:
            raise ValueError("the identity contributes at least one exact tie")
        if self.n_greater_equal > self.total_weight:
            raise ValueError("block rank counts exceed the full indexed mass")

    @classmethod
    def from_rank_counts(
        cls,
        *,
        D: int,
        n_strict_greater: int,
        n_equal: int,
        identity_score: Fraction | None = None,
    ) -> "ExactBlockCoordinate":
        """Construct a coordinate from exact weighted rank counts."""
        return cls(
            D=D,
            total_weight=math.factorial(D),
            n_strict_greater=n_strict_greater,
            n_equal=n_equal,
            identity_score=identity_score,
        )

    @property
    def n_greater_equal(self) -> int:
        """Full indexed weight whose score is at least the identity score."""
        return self.n_strict_greater + self.n_equal

    @property
    def q_conservative(self) -> Fraction:
        """Exact conservative block-tail rank."""
        return Fraction(self.n_greater_equal, self.total_weight)

    @property
    def block_feature(self) -> Fraction:
        """Exact block coordinate ``B = 1-q``."""
        return Fraction(self.total_weight - self.n_greater_equal, self.total_weight)


@dataclass(frozen=True)
class ExactScoreComponents:
    """Symbolic components of one raw v8.1 total score."""

    group: int
    horizon: int
    count: ExactCountWeight | None
    block: ExactBlockCoordinate | None
    scoring_version: str = EXACT_SCORING_VERSION

    def __post_init__(self) -> None:
        if self.group not in (GROUP_OTHER, GROUP_ALL_STATE_1):
            raise ValueError("group must be 0 or 1")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.count is not None and self.count.horizon != self.horizon:
            raise ValueError("count horizon differs from the score horizon")
        if self.scoring_version != EXACT_SCORING_VERSION:
            raise ValueError("exact comparisons require the Stage-2A raw-score profile")


@dataclass(frozen=True)
class ComparisonBudget:
    """Deterministic resource budget for a certified mixed comparison."""

    initial_terms: int = 8
    max_terms: int = 256

    def __post_init__(self) -> None:
        if self.initial_terms < 1 or self.max_terms < 1:
            raise ValueError("series budgets must be positive")
        if self.initial_terms > self.max_terms:
            raise ValueError("initial_terms must not exceed max_terms")


@dataclass(frozen=True)
class RationalInterval:
    """Closed interval with exact rational endpoints."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower


@dataclass(frozen=True)
class ComparisonResult:
    """A certified score relation or an explicit precision-limit result."""

    status: ComparisonStatus
    relation: ComparisonRelation | None
    reason: str
    terms_used: int = 0
    delta_interval: RationalInterval | None = None

    @property
    def certified(self) -> bool:
        return self.relation is not None


def _exact_relation(left: Fraction | int, right: Fraction | int) -> ComparisonRelation:
    if left < right:
        return ComparisonRelation.LESS
    if left > right:
        return ComparisonRelation.GREATER
    return ComparisonRelation.EQUAL


def _exact_result(relation: ComparisonRelation, reason: str) -> ComparisonResult:
    return ComparisonResult(
        status=ComparisonStatus.EXACT,
        relation=relation,
        reason=reason,
    )


def _atanh_log_interval(value: Fraction, terms: int) -> RationalInterval:
    """Enclose ``log(value)`` for ``1 <= value <= 2`` with rational bounds."""
    if not Fraction(1, 1) <= value <= Fraction(2, 1):
        raise ValueError("atanh log reduction requires a value in [1, 2]")
    if terms < 1:
        raise ValueError("terms must be positive")
    z = (value - 1) / (value + 1)
    if z == 0:
        return RationalInterval(Fraction(0, 1), Fraction(0, 1))

    z_squared = z * z
    power = z
    partial = Fraction(0, 1)
    for index in range(terms):
        partial += power / (2 * index + 1)
        power *= z_squared
    lower = 2 * partial
    # The omitted positive tail is bounded by replacing every remaining
    # denominator with its smallest value and summing a geometric series.
    remainder_upper = 2 * power / ((2 * terms + 1) * (1 - z_squared))
    return RationalInterval(lower, lower + remainder_upper)


@lru_cache(maxsize=8192)
def certified_log_fraction_interval(
    value: Fraction,
    terms: int,
) -> RationalInterval:
    """Return a proved rational enclosure of ``log(value)``.

    Range reduction is exact.  For ``value >= 1`` write
    ``value = 2**k * s`` with ``1 <= s < 2`` and enclose ``log(2)`` and
    ``log(s)`` with the positive atanh series.  Values below one are handled
    by ``log(value) = -log(1/value)`` so interval endpoints reverse safely.
    """
    if isinstance(value, float):
        raise TypeError("certified logarithms require an exact Fraction")
    value = Fraction(value)
    if value <= 0:
        raise ValueError("the logarithm argument must be positive")
    if terms < 1:
        raise ValueError("terms must be positive")
    if value == 1:
        return RationalInterval(Fraction(0, 1), Fraction(0, 1))
    if value < 1:
        reciprocal = certified_log_fraction_interval(1 / value, terms)
        return RationalInterval(-reciprocal.upper, -reciprocal.lower)

    numerator = value.numerator
    denominator = value.denominator
    k = max(0, numerator.bit_length() - denominator.bit_length())
    while numerator < denominator * (1 << k):
        k -= 1
    while numerator >= denominator * (1 << (k + 1)):
        k += 1
    scaled = value / (1 << k)
    if not Fraction(1, 1) <= scaled < Fraction(2, 1):
        raise RuntimeError("exact logarithm range reduction failed")

    log_two = _atanh_log_interval(Fraction(2, 1), terms)
    log_scaled = _atanh_log_interval(scaled, terms)
    return RationalInterval(
        k * log_two.lower + log_scaled.lower,
        k * log_two.upper + log_scaled.upper,
    )


def _require_count(score: ExactScoreComponents) -> ExactCountWeight:
    if score.count is None or score.count.status is CountWeightStatus.NOT_EVALUATED:
        raise ValueError("the active score branch requires a count component")
    return score.count


def _require_finite_weight(score: ExactScoreComponents) -> Fraction:
    count = _require_count(score)
    if count.status is not CountWeightStatus.FINITE or count.weight is None:
        raise ValueError("a finite score branch requires an exact fitted weight")
    return count.weight


def _require_block(score: ExactScoreComponents) -> ExactBlockCoordinate:
    if score.block is None:
        raise ValueError("the active score branch requires a block component")
    return score.block


def _active_count_is_infinite(
    score: ExactScoreComponents,
    lam: Fraction,
) -> bool:
    if score.group == GROUP_OTHER and lam == 0:
        return False
    return _require_count(score).status is CountWeightStatus.POSITIVE_INFINITY


def compare_exact_scores(
    left: ExactScoreComponents,
    right: ExactScoreComponents,
    *,
    lam: RationalInput | RationalParameter = Fraction(1, 2),
    budget: ComparisonBudget = ComparisonBudget(),
) -> ComparisonResult:
    """Compare two raw v8.1 nonconformity scores without float semantics.

    The result orders ``left`` relative to ``right``.  Same-group, fixed-``h``
    comparison is enforced because that is the outer calibration contract.
    """
    parameter = parse_lambda(lam)
    weight = parameter.value
    if left.scoring_version != right.scoring_version:
        raise ValueError("scores from different numerical profiles cannot be compared")
    if left.group != right.group:
        raise ValueError("outer scores must come from the same history group")
    if left.horizon != right.horizon:
        raise ValueError("outer scores must share the fixed horizon h")

    left_infinite = _active_count_is_infinite(left, weight)
    right_infinite = _active_count_is_infinite(right, weight)
    if left_infinite or right_infinite:
        if left_infinite and right_infinite:
            return _exact_result(
                ComparisonRelation.EQUAL, "both active scores are +infinity"
            )
        return _exact_result(
            ComparisonRelation.GREATER if left_infinite else ComparisonRelation.LESS,
            "extended-real ordering of a finite score and +infinity",
        )

    if left.group == GROUP_ALL_STATE_1:
        relation = _exact_relation(
            _require_finite_weight(right),
            _require_finite_weight(left),
        )
        return _exact_result(relation, "group-1 count-only reverse-W ordering")

    if weight == 0:
        relation = _exact_relation(
            _require_block(left).block_feature,
            _require_block(right).block_feature,
        )
        return _exact_result(relation, "lambda=0 exact block-only ordering")

    left_weight = _require_finite_weight(left)
    right_weight = _require_finite_weight(right)
    if weight == 1:
        relation = _exact_relation(right_weight, left_weight)
        return _exact_result(relation, "lambda=1 exact reverse-W ordering")

    left_block = _require_block(left).block_feature
    right_block = _require_block(right).block_feature
    block_delta = (1 - weight) * (left_block - right_block)
    ratio = right_weight / left_weight

    if ratio == 1:
        return _exact_result(
            _exact_relation(block_delta, 0),
            "equal fitted weights reduce the mixed comparison to exact B",
        )
    if block_delta == 0:
        return _exact_result(
            _exact_relation(ratio, 1),
            "equal block coordinates reduce the mixed comparison to log monotonicity",
        )
    if ratio > 1 and block_delta > 0:
        return _exact_result(
            ComparisonRelation.GREATER,
            "count and block differences are both positive",
        )
    if ratio < 1 and block_delta < 0:
        return _exact_result(
            ComparisonRelation.LESS,
            "count and block differences are both negative",
        )

    coefficient = weight / left.horizon
    terms = budget.initial_terms
    last_interval: RationalInterval | None = None
    while True:
        log_ratio = certified_log_fraction_interval(ratio, terms)
        delta_interval = RationalInterval(
            coefficient * log_ratio.lower + block_delta,
            coefficient * log_ratio.upper + block_delta,
        )
        last_interval = delta_interval
        if delta_interval.lower > 0:
            return ComparisonResult(
                status=ComparisonStatus.CERTIFIED_INTERVAL,
                relation=ComparisonRelation.GREATER,
                reason="certified mixed-score difference is strictly positive",
                terms_used=terms,
                delta_interval=delta_interval,
            )
        if delta_interval.upper < 0:
            return ComparisonResult(
                status=ComparisonStatus.CERTIFIED_INTERVAL,
                relation=ComparisonRelation.LESS,
                reason="certified mixed-score difference is strictly negative",
                terms_used=terms,
                delta_interval=delta_interval,
            )
        if terms >= budget.max_terms:
            break
        terms = min(2 * terms, budget.max_terms)

    return ComparisonResult(
        status=ComparisonStatus.UNCERTIFIED_PRECISION_LIMIT,
        relation=None,
        reason="mixed-score interval still contains zero at the declared term budget",
        terms_used=terms,
        delta_interval=last_interval,
    )


@dataclass(frozen=True)
class ExactOuterDecision:
    """An authoritative rational outer p-value and strict alpha decision."""

    status: OuterDecisionStatus
    alpha: RationalParameter
    denominator: int
    numerator: int | None
    n_at_least: int | None
    p_value: Fraction | None
    included: bool | None
    comparisons: tuple[ComparisonResult, ...]
    unresolved_reference_indices: tuple[int, ...] = ()


def exact_outer_decision(
    reference_scores: Sequence[ExactScoreComponents],
    candidate_score: ExactScoreComponents,
    *,
    alpha: RationalInput | RationalParameter,
    lam: RationalInput | RationalParameter = Fraction(1, 2),
    budget: ComparisonBudget = ComparisonBudget(),
) -> ExactOuterDecision:
    """Compute the exact conservative outer rank and strict decision.

    With ``p=m/M`` and ``alpha=a/b``, inclusion is decided only by the integer
    cross-product ``m*b > a*M``.  If any mixed comparison is uncertified, no
    pseudo-rank or partial p-value is returned.
    """
    alpha_parameter = parse_alpha(alpha)
    lambda_parameter = parse_lambda(lam)
    rows: list[ComparisonResult] = []
    unresolved: list[int] = []
    n_at_least = 0
    for index, reference in enumerate(reference_scores):
        result = compare_exact_scores(
            reference,
            candidate_score,
            lam=lambda_parameter,
            budget=budget,
        )
        rows.append(result)
        if not result.certified:
            unresolved.append(index)
        elif result.relation in (ComparisonRelation.EQUAL, ComparisonRelation.GREATER):
            n_at_least += 1

    denominator = len(reference_scores) + 1
    if unresolved:
        return ExactOuterDecision(
            status=OuterDecisionStatus.UNCERTIFIED_COMPARISON,
            alpha=alpha_parameter,
            denominator=denominator,
            numerator=None,
            n_at_least=None,
            p_value=None,
            included=None,
            comparisons=tuple(rows),
            unresolved_reference_indices=tuple(unresolved),
        )

    numerator = 1 + n_at_least
    p_value = Fraction(numerator, denominator)
    included = (
        numerator * alpha_parameter.value.denominator
        > alpha_parameter.value.numerator * denominator
    )
    return ExactOuterDecision(
        status=OuterDecisionStatus.COMPLETE,
        alpha=alpha_parameter,
        denominator=denominator,
        numerator=numerator,
        n_at_least=n_at_least,
        p_value=p_value,
        included=included,
        comparisons=tuple(rows),
    )


@dataclass(frozen=True)
class ExactQuantileResult:
    """Exact rank-grid index and, when requested, its score threshold."""

    status: QuantileStatus
    threshold_kind: ThresholdKind
    alpha: RationalParameter
    n_reference: int
    total_rank_size: int
    k: int
    threshold: ExactScoreComponents | None = None
    ordered_reference_indices: tuple[int, ...] = ()
    unresolved_pair: tuple[int, int] | None = None


def exact_quantile_index(
    n_reference: int,
    alpha: RationalInput | RationalParameter,
) -> ExactQuantileResult:
    """Return ``k=M-floor(M*alpha)`` using exact integer arithmetic."""
    if n_reference < 0:
        raise ValueError("n_reference must be nonnegative")
    parameter = parse_alpha(alpha)
    total = n_reference + 1
    if parameter.value == 1:
        return ExactQuantileResult(
            status=QuantileStatus.COMPLETE,
            threshold_kind=ThresholdKind.EMPTY,
            alpha=parameter,
            n_reference=n_reference,
            total_rank_size=total,
            k=0,
        )
    floor_product = total * parameter.value.numerator // parameter.value.denominator
    k = total - floor_product
    kind = ThresholdKind.POSITIVE_INFINITY if k == total else ThresholdKind.FINITE
    return ExactQuantileResult(
        status=QuantileStatus.COMPLETE,
        threshold_kind=kind,
        alpha=parameter,
        n_reference=n_reference,
        total_rank_size=total,
        k=k,
    )


def exact_outer_quantile(
    reference_scores: Sequence[ExactScoreComponents],
    *,
    alpha: RationalInput | RationalParameter,
    lam: RationalInput | RationalParameter = Fraction(1, 2),
    budget: ComparisonBudget = ComparisonBudget(),
) -> ExactQuantileResult:
    """Return the exact finite-rank threshold, or an explicit unresolved state."""
    plan = exact_quantile_index(len(reference_scores), alpha)
    if plan.threshold_kind is not ThresholdKind.FINITE:
        return plan

    lambda_parameter = parse_lambda(lam)
    ordered: list[int] = []
    for index, score in enumerate(reference_scores):
        position = len(ordered)
        while position > 0:
            previous_index = ordered[position - 1]
            result = compare_exact_scores(
                score,
                reference_scores[previous_index],
                lam=lambda_parameter,
                budget=budget,
            )
            if not result.certified:
                return ExactQuantileResult(
                    status=QuantileStatus.UNCERTIFIED_COMPARISON,
                    threshold_kind=ThresholdKind.FINITE,
                    alpha=plan.alpha,
                    n_reference=plan.n_reference,
                    total_rank_size=plan.total_rank_size,
                    k=plan.k,
                    ordered_reference_indices=tuple(ordered),
                    unresolved_pair=(index, previous_index),
                )
            if result.relation is ComparisonRelation.LESS:
                position -= 1
            else:
                break
        ordered.insert(position, index)

    threshold_index = ordered[plan.k - 1]
    return ExactQuantileResult(
        status=QuantileStatus.COMPLETE,
        threshold_kind=ThresholdKind.FINITE,
        alpha=plan.alpha,
        n_reference=plan.n_reference,
        total_rank_size=plan.total_rank_size,
        k=plan.k,
        threshold=reference_scores[threshold_index],
        ordered_reference_indices=tuple(ordered),
    )


__all__ = [
    "ComparisonBudget",
    "ComparisonRelation",
    "ComparisonResult",
    "ComparisonStatus",
    "CountWeightStatus",
    "EXACT_SCORING_VERSION",
    "ExactBlockCoordinate",
    "ExactCountWeight",
    "ExactOuterDecision",
    "ExactQuantileResult",
    "ExactScoreComponents",
    "LEGACY_ROUNDED_PROFILE",
    "OuterDecisionStatus",
    "QuantileStatus",
    "RationalInterval",
    "RationalParameter",
    "ThresholdKind",
    "certified_log_fraction_interval",
    "compare_exact_scores",
    "exact_augmented_count_weight",
    "exact_outer_decision",
    "exact_outer_quantile",
    "exact_quantile_index",
    "parse_alpha",
    "parse_lambda",
    "parse_rational_parameter",
]
