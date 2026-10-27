# The score code, without the experiment-management details

## 1. Augmented counts

For an episode (H,y), counts include the whole concatenated path, including
the history/future bridge. Only the h future transitions enter the score sum.
The count score is minus the logarithm of the product of the fitted
transition ratios.

The implementation stores that product W as an exact rational number.
Comparing count scores is therefore equivalent to comparing W in the reverse
order. This avoids treating two nearly equal floating-point logarithms as ties.

Read:

- simulation/scoring_backend/ochs_v81.py: augmented_transition_counts
- simulation/scoring_backend/ochs_v81_exact_numeric.py: exact_augmented_count_weight
- simulation/ranks.py: encode, decode, relation

## 2. The terminal-anchor blocks

The last state of H concatenated with y is the anchor. The prefix and final
anchor stay fixed; only the middle blocks are permuted. Blocks are indexed,
so two equal-content blocks still contribute separate permutations.

The block score only needs the last h+1 states of a reconstructed sequence.
The exact suffix routine works from right to left until those states are
determined. If k blocks have been specified, the unused D-k indexed blocks
have (D-k)! possible orders. Their multiplicity is counted, not sampled.

Read:

- simulation/scoring_backend/iblocks.py: decompose_i_blocks
- simulation/scoring_backend/ochs_v81_exact_suffix.py:
  _suffix_histogram_from_values, conservative_block_diagnostic_exact_suffix
- simulation/iblock.py: pair_tally

There is no random permutation approximation in the reproduction path.
The small independent exhaustive enumeration in tests/unit_checks.py is a
test oracle only, not the h=9 production algorithm.

## 3. Two distinct uses of the tallies

If gt orders have a strictly greater score and eq have an equal score:

- OCHS uses the conservative coordinate B = 1 - (gt+eq)/D!.
- The independent baseline uses q = (gt+U*eq)/D!.

For D<=1, gt=0, eq=1, total=1: B=0, but the baseline q=U, not 1.
G1 OCHS does not need B; the baseline must nevertheless compute its own block
tallies for G1. See simulation/iblock.py: exact_q and pair_tally.

## 4. Hybrid comparisons

G0 uses lambda*R/h + (1-lambda)*B; G1 uses R alone. A hybrid comparison
combines exact block fractions and certified rational bounds for a logarithm.
If a sign cannot be certified within the numerical budget, it stops. It does
not invent an epsilon tie.

Read simulation/scoring_backend/ochs_v81_exact_numeric.py: compare_exact_scores.
The usual manuscript entry is prepare_exact_score in ochs_v81_stage2a.py,
then relation and ranks in simulation/ranks.py.

## Why the backend retains historical filenames

The scoring_backend directory (previously named method3) is the existing
numerical library, preserved for traceability. Only its directory name and
import paths changed; the ten library files retain their original bytes.
Old names within historical docstrings and audit records are retained, as is
the original diagnostic salt. Its module names are not the manuscript's section numbers.
Some files also retain historical helpers (rounded-score routines, smoothed
counts, or permutation sampling). Those are **not called by this paper's
production path**. The functions identified above are the applicable path.
In particular, do not substitute conservative_outer_p_value from ochs_v81.py
for simulation/ranks.py.

Removing or rewriting that library wholesale would require a separate
algorithm-equivalence review. This package makes the scientific entry path
explicit and retains the checked exact backend; it does not claim that all
historical helper functions are part of the current method.
