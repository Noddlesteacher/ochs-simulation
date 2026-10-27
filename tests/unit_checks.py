"""Small exact identities and negative tests. No stochastic test inputs."""

import copy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from collections import Counter
from itertools import permutations
from math import factorial

from simulation import generation, iblock, ranks, ochs
from simulation.settings import A, PACKAGE
from simulation.storage import equal_rows, load, save
from simulation.fixed_prediction import template
from simulation.run_worker import run
from simulation.statistics import deterministic_tests


class ExactChecks(unittest.TestCase):
    def test_geometric_support_and_split(self):
        episode = generation.generate_episode(
            3,
            generation.Tape((1, 1, 0)),
            generation.Tape((1,)),
            generation.Tape((0,) * 8),
        )
        self.assertEqual(episode["K"], 3)
        self.assertEqual(episode["T"], 6)
        self.assertEqual(episode["H"], [1] * 6)
        self.assertEqual(episode["Y"], [1] * 3)

    def test_transition_boundaries(self):
        self.assertEqual(generation.transition(1, 989), 1)
        self.assertEqual(generation.transition(1, 990), 2)
        self.assertEqual(generation.transition(3, 721), 3)
        self.assertEqual(generation.transition(3, 722), 4)

    def test_exhausted_tape_fails(self):
        with self.assertRaises(ValueError):
            generation.positive_geometric(generation.Tape((1, 1)))

    def test_degenerate_randomized_tie(self):
        for d in (0, 1):
            tally = {"D": d, "gt": 0, "eq": 1, "total": 1}
            self.assertEqual(iblock.exact_q(tally, Fraction(1, 3)), Fraction(1, 3))

    def test_strict_threshold(self):
        membership = ochs.membership([[1, 10], [1, 1]])
        self.assertEqual(membership[0], [0, 1])
        self.assertEqual(membership[1], [0, 1])
        self.assertEqual(membership[2], [1])
        self.assertEqual(membership[20], [])

    def test_invalid_tally_fails(self):
        with self.assertRaises(ValueError):
            iblock.exact_q({"D": 2, "gt": 0, "eq": 1, "total": 1}, Fraction(1, 2))

    def test_small_block_oracle(self):
        # Separate raw-count/Fraction implementation: no production scorer here.
        # Enumerate INDEXED blocks, including repeated-content blocks. D <= 4.
        for history, future in (
            ((1, 1), (1, 1, 2)),
            ((1, 2, 4), (1, 1, 1)),
            ((2, 3, 4), (1, 2, 4)),
            ((1, 1), (1, 1, 1)),
        ):
            actual = iblock.pair_tally(history, future, A)
            expected = exhaustive_tally(history, future)
            self.assertEqual((actual["gt"], actual["eq"], actual["total"]), expected)

    def test_aggregation_identities(self):
        self.assertEqual(deterministic_tests()["status"], "PASS")


class KeyedComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"panel": "P", "method": m, "alpha": [k, 20], "value": [1, 2, 3]}
            for m in ("a", "b")
            for k in (1, 2)
        ]
        self.key = lambda row: (row["panel"], row["method"], tuple(row["alpha"]))

    def check_bad(self, changed):
        with self.assertRaises(ValueError):
            equal_rows(self.rows, changed, self.key)

    def test_reordered_rows_pass(self):
        equal_rows(self.rows, list(reversed(self.rows)), self.key)

    def test_missing_fails(self):
        self.check_bad(self.rows[:-1])

    def test_extra_fails(self):
        self.check_bad(self.rows + [self.rows[0]])

    def test_duplicate_fails(self):
        self.check_bad([self.rows[0]] * len(self.rows))

    def test_changed_number_fails(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["value"][0] = 99
        self.check_bad(changed)

    def test_changed_alpha_fails(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["alpha"] = [3, 20]
        self.check_bad(changed)

    def test_changed_method_fails(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["method"] = "c"
        self.check_bad(changed)

    def test_candidate_order_fails(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["value"].reverse()
        self.check_bad(changed)


class ImmutableStorageChecks(unittest.TestCase):
    def test_exact_large_integer_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json.gz"
            obj = {"integer": 2**1000, "fraction": [2**65 - 1, 2**65]}
            save(path, obj)
            self.assertEqual(load(path), obj)
            with self.assertRaises(ValueError):
                save(path, obj)

    def test_failed_task_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "h3/FORMAL_000000/FAILED.json", {"expected_test_failure": True})
            with self.assertRaises(ValueError):
                run("step5", 3, "FORMAL_000000", root)


def exhaustive_tally(history, future):
    """Tiny, deliberately slow independent oracle; never use for a full study."""
    z = tuple(history) + tuple(future)
    positions = [i for i, state in enumerate(z) if state == z[-1]]
    blocks = [z[left:right] for left, right in zip(positions, positions[1:])]
    d = len(blocks)
    if d <= 1:
        return 0, 1, 1
    if d > 5:
        raise ValueError("The test oracle is bounded, not a replacement scorer")
    states = sorted(set(z))
    edges = Counter(zip(z, z[1:]))
    outgoing = Counter(z[:-1])
    matrix = [[Fraction(edges[a, b], outgoing[a]) for b in states] for a in states]
    powers = [matrix]
    for _ in range(1, len(future)):
        previous = powers[-1]
        powers.append(
            [
                [
                    sum(previous[i][k] * matrix[k][j] for k in range(len(states)))
                    for j in range(len(states))
                ]
                for i in range(len(states))
            ]
        )

    def score(path):
        start = states.index(path[len(history) - 1])
        probabilities = [
            powers[ell - 1][start][states.index(path[len(history) + ell - 1])]
            for ell in range(1, len(future) + 1)
        ]
        return 1 - sum(probabilities) / len(future)

    original = score(z)
    gt = eq = 0
    for order in permutations(range(d)):
        shuffled = z[: positions[0]] + sum((blocks[i] for i in order), ()) + (z[-1],)
        value = score(shuffled)
        gt += value > original
        eq += value == original
    return gt, eq, factorial(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
