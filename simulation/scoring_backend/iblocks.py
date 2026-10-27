"""i-block decomposition, the permutation group ``S_D``, and its sampler.

Spec sections 3.2, 3.4 (sampling half) and 3.6.  Two rules in this module are
load-bearing for the finite-sample theorem and are marked ``[CRITICAL]`` at
their definitions:

* the permutation group is a group on *block indices*, never on reconstructed
  state sequences, so ties are preserved rather than collapsed (3.2);
* the evaluated set always contains the identity, and the sampler receives an
  externally seeded random stream rather than reading randomness from the
  observed state sequence (3.6).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from itertools import permutations
from typing import Sequence

import numpy as np

from .states import Path, SequenceLike, check_adjacency, check_states
from .counts import transition_count_matrix

#: Above this many block orders we never attempt a full enumeration, even when
#: ``max_permutations`` would allow it, because materialising the list costs
#: more than the sampler does.
MAX_EXACT_PERMUTATIONS = 50_000

#: ``D!`` is reported as an exact integer only up to this ``D``.
MAX_FACTORIAL_INT_D = 20


@dataclass(frozen=True)
class IBlockDecomposition:
    """The frozen/permutable split of one augmented sequence ``S = H || y``."""

    prefix: tuple[int, ...]
    """Frozen initial segment ``I_0`` before the first occurrence of ``i``.

    Empty when ``S[0] == i``.
    """

    blocks: tuple[tuple[int, ...], ...]
    """The permutable indexed blocks ``I_1, ..., I_D``."""

    tail: tuple[int, ...]
    """Frozen tail ``I_{D+1}``: the final element, which equals ``i``."""

    anchor: int
    """The anchor state ``i = y_h``."""

    @property
    def n_blocks(self) -> int:
        """``D``, the number of permutable blocks."""
        return len(self.blocks)


def decompose_i_blocks(sequence: SequenceLike, anchor: int) -> IBlockDecomposition:
    """Cut ``S`` into ``I_0``, permutable ``I_1..I_D``, and the frozen tail.

    Spec 3.2.  Segments are maximal runs that *begin* with the anchor state and
    contain no further anchor.  Because the anchor is ``y_h`` -- the final
    element of ``S`` -- the last anchor occurrence is the last position, so the
    frozen tail is the single final element.
    """
    values = [int(state) for state in sequence]
    anchor_state = int(anchor)
    positions = [index for index, state in enumerate(values) if state == anchor_state]

    if not positions:
        raise ValueError("anchor state does not appear in the sequence.")
    if positions[-1] != len(values) - 1:
        raise ValueError(
            "the anchor must be the final state of the augmented sequence; "
            "the i-block construction anchors on y_h."
        )

    prefix = tuple(values[: positions[0]])

    segments: list[tuple[int, ...]] = []
    for order, start in enumerate(positions):
        end = positions[order + 1] if order + 1 < len(positions) else len(values)
        segments.append(tuple(values[start:end]))

    return IBlockDecomposition(
        prefix=prefix,
        blocks=tuple(segments[:-1]),
        tail=segments[-1],
        anchor=anchor_state,
    )


def block_orbit_id(
    decomposition: IBlockDecomposition,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    int,
]:
    """Return a canonical identifier shared by every order in one block orbit.

    The prefix, frozen tail, anchor and multiset of indexed block contents are
    invariant under reordering.  Sorting the block tuples canonicalizes that
    multiset while retaining duplicate blocks, which remain distinct indexed
    group elements in :func:`sample_block_orders`.
    """
    return (
        decomposition.prefix,
        tuple(sorted(decomposition.blocks)),
        decomposition.tail,
        decomposition.anchor,
    )


def rebuild_sequence(
    decomposition: IBlockDecomposition,
    order: Sequence[int],
) -> list[int]:
    """Reconstruct the full permuted sequence for one block index order."""
    values = list(decomposition.prefix)
    for index in order:
        values.extend(decomposition.blocks[index])
    values.extend(decomposition.tail)
    return values


def rebuild_suffix(
    decomposition: IBlockDecomposition,
    order: Sequence[int],
    suffix_length: int,
) -> list[int]:
    """Return only the last ``suffix_length`` states of the permuted sequence.

    Both score conventions of :mod:`method3.blockdiag` read only ``X_T`` and
    ``X_{T+1..T+h}``, i.e. the final ``h + 1`` states, so long histories need
    not be rebuilt in full for every sampled order.  The result is identical to
    ``rebuild_sequence(...)[-suffix_length:]``.
    """
    if suffix_length < 1:
        raise ValueError("suffix_length must be positive.")

    pieces_from_end: list[tuple[int, ...]] = [decomposition.tail]
    pieces_from_end.extend(
        decomposition.blocks[index] for index in reversed(list(order))
    )
    pieces_from_end.append(decomposition.prefix)

    collected: list[tuple[int, ...]] = []
    remaining = suffix_length
    for piece in pieces_from_end:
        if len(piece) >= remaining:
            collected.append(piece[len(piece) - remaining :])
            remaining = 0
            break
        collected.append(piece)
        remaining -= len(piece)

    if remaining != 0:
        raise ValueError("requested suffix is longer than the sequence.")

    suffix: list[int] = []
    for piece in reversed(collected):
        suffix.extend(piece)
    return suffix


def assert_count_invariance(
    sequence: SequenceLike,
    anchor: int,
    order: Sequence[int],
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> None:
    """Assert that a legal block permutation preserves the sufficient statistic.

    Guarantee implemented: the i-block group action fixes the initial state and
    the complete transition-count matrix, which is precisely the invariance
    that makes the block orbit exchangeable under a Markov law.  Raises
    ``AssertionError`` if either is violated.
    """
    array = check_adjacency(adjacency)
    original = check_states(sequence, array)
    decomposition = decompose_i_blocks(original, anchor)
    permuted = rebuild_sequence(decomposition, order)

    assert (
        permuted[0] == original[0]
    ), f"initial state changed under permutation: {original[0]} -> {permuted[0]}"
    original_counts = transition_count_matrix(original, array)
    permuted_counts = transition_count_matrix(permuted, array)
    assert np.array_equal(
        original_counts, permuted_counts
    ), "transition-count matrix changed under a legal block permutation"


def permutation_group_summary(n_blocks: int) -> tuple[float, int | None]:
    """Return ``log(D!)`` and, when small enough to display, ``D!`` itself.

    ``|S_D| = D!`` is the mathematical group size and does not depend on any
    sampling cap.
    """
    if n_blocks < 0:
        raise ValueError("n_blocks must be nonnegative.")
    log_size = math.lgamma(n_blocks + 1)
    exact = math.factorial(n_blocks) if n_blocks <= MAX_FACTORIAL_INT_D else None
    return log_size, exact


def derive_random_seed(master_seed: int, *labels: object, salt: str = "") -> int:
    """Derive a stable child seed from an external master seed and labels.

    The labels identify a location in a random field fixed before the data are
    inspected.  They may name a method, a prespecified tuning value, or a
    canonical block-orbit identifier, but must not contain the observed block
    arrangement, candidate suffix, score, or any quantity chosen after looking
    at the scores.  This gives every caller a reproducible, order-invariant
    stream without allowing the observed representative of an orbit to choose
    a different randomization coordinate.

    SHA-256 is used instead of Python's process-salted :func:`hash`.
    """
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        raise TypeError("master_seed must be an integer.")
    payload = json.dumps(
        {
            "master_seed": int(master_seed),
            "labels": labels,
            "salt": str(salt),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def unit_rng(random_seed: int | None, salt: str = "") -> np.random.Generator:
    """Return an RNG independent of the observed history.

    ``random_seed=None`` asks NumPy for fresh entropy.  Supplying a seed gives a
    reproducible stream, with ``salt`` providing domain separation.  Callers
    that need order invariance should derive the supplied seed from a master
    seed and prespecified labels via :func:`derive_random_seed`.
    """
    if random_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(
        derive_random_seed(random_seed, salt="unit-rng|" + salt)
    )


def sample_block_orders(
    n_blocks: int,
    max_permutations: int | None,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], bool]:
    """Return the evaluated set ``V = {identity} u W`` and whether it is exhaustive.

    [CRITICAL] Spec 3.4 (sampling half).  Three properties are load-bearing and
    are each asserted below:

    1. the identity block order is always the first element of ``V``;
    2. ``W`` is drawn uniformly *without replacement* from ``S_D \\ {identity}``
       -- the "seen" set is initialised containing the identity, so no sampled
       order can duplicate it;
    3. ``|V| = n_eval = min(max_permutations, D!)`` exactly, and this is the
       denominator used by the p-value.

    The draw is completed *before* any score is evaluated, so there is no
    score-dependent stopping, filtering, or reweighting.

    Deduplication is by *index tuple* only.  Two different index orders that
    happen to rebuild the same state sequence stay two distinct group elements
    (spec 3.2, [CRITICAL]); collapsing them would discard genuine ties and make
    the p-value anti-conservative.
    """
    if n_blocks < 0:
        raise ValueError("n_blocks must be nonnegative.")
    if max_permutations is not None and max_permutations < 1:
        raise ValueError("max_permutations must be positive or None.")

    identity = tuple(range(n_blocks))
    full_size = math.factorial(n_blocks)

    if max_permutations is None or full_size <= max_permutations:
        if full_size > MAX_EXACT_PERMUTATIONS:
            raise ValueError(
                f"full enumeration would need {full_size} orders; "
                "set a smaller max_permutations."
            )
        orders = list(permutations(range(n_blocks)))
        # itertools.permutations emits the identity first for a sorted input.
        assert orders[0] == identity
        return orders, True

    n_eval = min(max_permutations, full_size)

    if full_size <= MAX_EXACT_PERMUTATIONS:
        # Small group, strict subset: enumerate once and take an exactly
        # uniform without-replacement subset of the non-identity elements.
        # Rejection sampling would also be correct here but degrades badly
        # when n_eval is a large fraction of D!.
        others = [order for order in permutations(range(n_blocks)) if order != identity]
        chosen = rng.choice(len(others), size=n_eval - 1, replace=False)
        orders = [identity] + [others[int(index)] for index in chosen]
    else:
        # Huge group, tiny sample: rejection sampling is uniform without
        # replacement and collisions are vanishingly rare.  The "seen" set is
        # initialised with the identity so no draw can duplicate it.
        orders = [identity]
        seen: set[tuple[int, ...]] = {identity}
        base = np.arange(n_blocks)
        while len(orders) < n_eval:
            order = tuple(int(value) for value in rng.permutation(base))
            if order not in seen:
                seen.add(order)
                orders.append(order)

    assert orders[0] == identity, "identity must anchor the evaluated set"
    assert len(orders) == n_eval, "denominator must equal the evaluated-set size"
    assert len(set(orders)) == n_eval, "sampled orders must be distinct"
    return orders, False


def block_order_positions(order: Sequence[int]) -> Path:
    """Return the order as a plain tuple; a small helper for tabular logging."""
    return tuple(int(value) for value in order)
