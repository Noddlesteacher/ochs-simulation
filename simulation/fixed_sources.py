"""Frozen fixed-history future/U streams, shared across L and alpha."""

import hashlib
from .storage import canon, require, digest
from .settings import FIXED_RUN as RUN, FIXED_MODEL as MODEL
from .random_sources import Source, midpoint, VERSION, PREFIX
from . import generation as generator


def source(master, stage, address):
    require(stage in ("deterministic", "formal"), "stage")
    b = bytes.fromhex(master)
    require(len(b) == 32, "master length")
    envelope = [VERSION, stage, RUN, MODEL, address]
    encoded = canon(envelope).encode()
    raw = hashlib.sha256(
        PREFIX + b + len(encoded).to_bytes(8, "big") + encoded
    ).digest()
    words = [int.from_bytes(raw[i : i + 4], "big") for i in range(0, 32, 4)]
    return Source(words), dict(
        envelope=envelope, seed_digest_hex=raw.hex(), seed_words_be32=words
    )


def future_address(h, pool):
    return canon(["OCHS_FIXED_ALL1_FUTURE_V1", RUN, MODEL, h, pool, 0])


def uniform_address(h, pool, y):
    return canon(["OCHS_SINGLE_PATH_IBLOCK_TIE_V1", RUN, MODEL, h, pool, 0, list(y)])


def generate_input(h, pool, candidates, master, stage):
    # Predictor never imports this module. Numeric P belongs to frozen transition.
    address = future_address(h, pool)
    s, m = source(master, stage, address)
    initial = s.state()
    current = 1
    future = []
    innovations = []
    for _ in range(h):
        innovation = s.integer(0, 1000)
        innovations.append(innovation)
        current = generator.transition(current, innovation)
        future.append(current)
    future_manifest = dict(**m, initial_state=initial, final_state=s.state())
    words = []
    for y in candidates:
        u, _ = source(master, stage, uniform_address(h, pool, y))
        words.append(u.raw())
    return dict(
        run_id=RUN,
        model_id=MODEL,
        h=h,
        pool_id=pool,
        stage=stage,
        target_history_definition="1^L; L in {10,50,100,200}",
        future=future,
        future_innovations=innovations,
        future_source=future_manifest,
        J=words,
        U_state_scheme="Source(derived seed_words), initial state; final state after exactly one raw() call; version/schema frozen",
        candidate_order_sha256=digest(candidates),
    )
