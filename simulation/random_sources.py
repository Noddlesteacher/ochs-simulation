"""Frozen addressed PCG64 streams. Reordering tasks cannot alter their data."""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import numpy as np
from .storage import canon, require
from .settings import MODEL, RUN, formal_authorization

VERSION = "OCHS_ADDRESS_FACTORY_V1"


PREFIX = b"OCHS-SOURCE-V1\x00"


def seed_words(master_hex, stage, run, accepted_address):
    require(
        stage in ("deterministic", "smoke", "formal"), "no formal source authorized"
    )
    require(isinstance(run, str) and run, "run identity")
    if stage == "formal":
        require(run == RUN, "formal run identity")
    master = bytes.fromhex(master_hex)
    require(len(master) == 32, "32-byte root required")
    envelope = [VERSION, stage, run, MODEL, accepted_address]
    address_bytes = canon(envelope).encode("utf-8")
    material = PREFIX + master + len(address_bytes).to_bytes(8, "big") + address_bytes
    raw = hashlib.sha256(material).digest()
    return (
        envelope,
        raw.hex(),
        [int.from_bytes(raw[i : i + 4], "big") for i in range(0, 32, 4)],
    )


class Source:
    def __init__(self, words):
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))
        self.integer_calls = 0
        self.raw_calls = 0

    def integer(self, low, high):
        require(type(low) is int and type(high) is int and low < high, "integer bounds")
        self.integer_calls += 1
        return int(self.rng.integers(low, high))

    def raw(self):
        self.raw_calls += 1
        return int(self.rng.bit_generator.random_raw())

    def state(self):
        return {
            "bit_generator": deepcopy(self.rng.bit_generator.state),
            "integer_calls": self.integer_calls,
            "raw_calls": self.raw_calls,
        }

    @classmethod
    def restore(cls, state):
        require(
            set(state) == {"bit_generator", "integer_calls", "raw_calls"},
            "source state fields",
        )
        require(
            state["bit_generator"]["bit_generator"] == "PCG64", "wrong bit generator"
        )
        result = cls([0])
        result.rng.bit_generator.state = deepcopy(state["bit_generator"])
        result.integer_calls = state["integer_calls"]
        result.raw_calls = state["raw_calls"]
        return result


class Factory:
    def __init__(self, master_hex, stage, run):
        if stage == "formal":
            formal_authorization(master_hex, run)
        self.master_hex = master_hex
        self.stage = stage
        self.run = run
        self.sources = {}
        self.mapping = {}

    def __call__(self, address):
        require(address not in self.sources, "duplicate source request")
        envelope, digest, words = seed_words(
            self.master_hex, self.stage, self.run, address
        )
        source = Source(words)
        self.sources[address] = source
        self.mapping[address] = {
            "envelope": envelope,
            "seed_digest_hex": digest,
            "seed_words_be32": words,
            "initial_state": source.state(),
        }
        return source

    def audit(self):
        return {
            a: {**self.mapping[a], "final_state": s.state()}
            for a, s in sorted(self.sources.items())
        }


def midpoint(word, bits=64):
    require(type(bits) is int and 1 <= bits <= 64, "grid bit count")
    require(type(word) is int and 0 <= word < 2**bits, "raw word range")
    return F(2 * word + 1, 2 ** (bits + 1))


def grid_probability(gt, eq, total, alpha, bits):
    require(eq >= 1 and total >= eq + gt and gt >= 0, "grid tally")
    m = 2**bits
    x = (alpha * total - gt) / eq
    t = m * x + F(1, 2)
    count = min(m, max(0, t.numerator // t.denominator))
    return F(m - count, m)
