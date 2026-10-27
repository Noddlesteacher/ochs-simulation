"""Main study: generate -> truth-free predict -> evaluate. No auto-run."""

from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import time
from .storage import save, require, equal_rows
from .settings import A, MODEL
from .random_sources import Factory, midpoint


def generate(
    h, panel, master, run, generator, baseline, failure_folder=None, stage="smoke"
):
    """Return (visible records, held-out truth, generation audit, candidate U).

    Every source is addressed by the original run, horizon, panel and purpose.
    The truth is saved for evaluation but is not passed to predict().
    """
    factory = Factory(master, stage, run)
    try:
        visible, truth, audit = generator.generate_observation(h, panel, factory)
    except BaseException:
        if failure_folder is not None:
            save(
                Path(failure_folder) / "PARTIAL_GENERATION_STATES.json.gz",
                factory.audit(),
            )
        raise
    ys = baseline.legal_candidates(visible["records"][0]["states"], h, A)
    u_factory = Factory(master, stage, run)
    entries = []
    words = []
    for y in ys:
        address = baseline.address(run, MODEL, h, panel, 0, y)
        source = u_factory(address)
        word = source.raw()
        u = midpoint(word)
        entries.append(
            {"address": address, "numerator": u.numerator, "denominator": u.denominator}
        )
        words.append(
            {
                "candidate": list(y),
                "address": address,
                "J": word,
                "U": [u.numerator, u.denominator],
            }
        )
    audit["source_manifest"] = factory.audit()
    uniforms = {
        "entries": entries,
        "words": words,
        "source_manifest": u_factory.audit(),
    }
    return visible, truth, audit, uniforms


def predict(visible, entries, study, adapter, baseline, reverse=False):
    """Truth-free controller. Does not import numerical P or read evaluation files."""
    timer = {}
    t = time.monotonic()
    ochs, cache = adapter.predict(visible, A)
    timer["ochs_seconds"] = time.monotonic() - t
    t = time.monotonic()
    b_cache = baseline.ExactScoreCache()
    ib = baseline.predict(
        visible["records"][0]["states"],
        visible["h"],
        A,
        entries,
        study=study,
        model=MODEL,
        panel=visible["panel_id"],
        target=0,
        cache=b_cache,
        reverse=reverse,
    )
    timer["baseline_seconds"] = time.monotonic() - t
    result = {"ochs": ochs, "randomized_iblock": ib}
    validate_prediction(result, visible, baseline)
    return result, timer, {"ochs": cache, "baseline": asdict(b_cache.snapshot())}


def validate_prediction(result, visible, baseline):
    require(
        set(result) == {"ochs", "randomized_iblock"}, "prediction methods container"
    )
    ochs, ib = result["ochs"], result["randomized_iblock"]
    h = visible["h"]
    ys = [
        list(y)
        for y in baseline.legal_candidates(visible["records"][0]["states"], h, A)
    ]
    require(
        len(ys) == 2 ** h and len(set(map(tuple, ys))) == 2**h, "candidate enumeration"
    )
    require(ochs["candidates"] == ib["candidates"] == ys, "candidate keys mismatch")
    require(
        len(ochs["components"]) == len(ys) and len(ochs["references"]) == ochs["N"],
        "component counts",
    )
    require(len(ib["rows"]) == len(ys), "baseline row count")
    require(
        [r["candidate"] for r in ib["rows"]] == ys,
        "missing/duplicate baseline candidate",
    )
    require(
        set(ochs["p"]) == set(ochs["members"]) == {"ochs090", "count", "ochs050"},
        "OCHS method keys",
    )
    for r in ib["rows"]:
        tally = {k: r[k] for k in ("D", "gt", "eq", "total")}
        require(F(*r["q"]) == baseline.exact_q(tally, F(*r["U"])), "baseline q formula")
        require(
            F(*r["q_cons"]) == F(r["gt"] + r["eq"], r["total"]),
            "conservative diagnostic",
        )
    pairs = {**ochs["p"], "randomized_iblock": [r["q"] for r in ib["rows"]]}
    levels = {**ochs["members"], "randomized_iblock": ib["members"]}
    for method, p in pairs.items():
        require(len(p) == len(ys) and len(levels[method]) == 21, "p/alpha count")
        for n, d in p:
            require(
                type(n) is int and type(d) is int and 0 < n <= d, "exact positive p/q"
            )
        for k in range(21):
            expected = [i for i, (n, d) in enumerate(p) if 20 * n > k * d]
            require(levels[method][k] == expected, "strict membership mismatch")


def evaluate(prediction, truth):
    """Find the true path among candidates, then record coverage and set size."""
    ys = prediction["ochs"]["candidates"]
    y = truth["future"]
    require(y in ys, "true future outside frozen support")
    index = ys.index(y)
    J = len(ys)
    levels = {
        **prediction["ochs"]["members"],
        "randomized_iblock": prediction["randomized_iblock"]["members"],
    }
    rows = []
    for method, sets in levels.items():
        for k, members in enumerate(sets):
            size = len(members)
            rows.append(
                {
                    "method": method,
                    "alpha": [k, 20],
                    "nominal": [20 - k, 20],
                    "covered": int(index in members),
                    "set_size": size,
                    "normalized_set_size": [size, J],
                    "empty": size == 0,
                    "singleton": size == 1,
                    "full": size == J,
                }
            )
    equal_rows(rows, rows, lambda r: (r["method"], r["alpha"][0]))
    require(len(rows) == 84, "84 evaluation rows required")
    return {
        "truth_index": index,
        "rows": rows,
        "Gamma": "OMITTED_OPTIONAL_NO_NEW_ORACLE_ANALYSIS",
    }
