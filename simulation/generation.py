"""Section 4.1: common shifted-geometric scoring episodes. No RNG on import."""

from dataclasses import dataclass
import hashlib
from .settings import MODEL_ID
from .storage import canon as canonical, require

P_INT = ((990, 10, 0, 0), (0, 0, 500, 500), (0, 0, 722, 278), (653, 347, 0, 0))


NAMESPACE = "OCHS_SHIFTED_EPISODE_STREAM_V1"


PURPOSES = ("length", "initial", "transition")


def integer(value, lower, upper, label):
    require(type(value) is int and lower <= value < upper, label)
    return value


def stream_address(h, panel_id, unit_id, purpose):
    integer(h, 1, 10, "h type/range")
    require(h in (3, 6, 9), "unsupported horizon")
    integer(unit_id, 0, 200, "unit ID")
    require(
        isinstance(panel_id, str) and panel_id and purpose in PURPOSES, "stream key"
    )
    # Address only: no master entropy, seed allocation or Generator constructed.
    return canonical([NAMESPACE, MODEL_ID, h, panel_id, unit_id, purpose])


def address_digest(h, panel_id, unit_id, purpose):
    return hashlib.sha256(
        stream_address(h, panel_id, unit_id, purpose).encode()
    ).hexdigest()


@dataclass
class Tape:
    """Finite deterministic QA source. Exhaustion is an error, never refill."""

    values: tuple
    position: int = 0

    def integer(self, low, high):
        require(self.position < len(self.values), "innovation tape exhausted")
        value = self.values[self.position]
        self.position += 1
        return integer(value, low, high, "innovation outside requested support")

    def state(self):
        return {"values": list(self.values), "position": self.position}

    @classmethod
    def restore(cls, state):
        require(set(state) == {"values", "position"}, "tape state fields")
        integer(state["position"], 0, len(state["values"]) + 1, "tape position")
        return cls(tuple(state["values"]), state["position"])


def positive_geometric(length_source):
    """Count trials until integer 0 appears: support 1,2,... and success 1/78.

    Keep this original sequence of integer draws. Replacing it with a library
    geometric sampler would change seeded data, even with the same distribution.
    """
    k = 1
    while length_source.integer(0, 78) != 0:
        k += 1
    return k


def transition(state, innovation):
    """Map one integer in 0,...,999 to the next state using P's cumulative row."""
    integer(state, 1, 5, "state")
    integer(innovation, 0, 1000, "transition innovation")
    cumulative = 0
    for nxt, mass in enumerate(P_INT[state - 1], 1):
        cumulative += mass
        if innovation < cumulative:
            return nxt
    raise ValueError("transition mapping exhausted")


def generate_episode(h, length_source, initial_source, transition_source):
    """Generate T=h+K history states and the next h states, then split at T."""
    require(type(h) is int and h in (3, 6, 9), "unsupported horizon")
    k = positive_geometric(length_source)
    t = h + k
    z = [initial_source.integer(1, 5)]
    for _ in range(t + h - 1):
        z.append(transition(z[-1], transition_source.integer(0, 1000)))
    return {"K": k, "T": t, "H": z[:t], "Y": z[t:], "Z": z}


def generate_observation(
    h, panel_id, source_factory, *, total_countries=200, potential_references=199
):
    """Generate 200 independent episodes; expose target H and reference H||Y.

    Returns visible data separately from target truth and the generation audit.
    The last two objects must not enter prediction.
    """
    require(
        type(total_countries) is int
        and type(potential_references) is int
        and total_countries == 200
        and potential_references == 199
        and total_countries == potential_references + 1,
        "country count must be 200/199",
    )
    require(type(h) is int and h in (3, 6, 9), "unsupported horizon")
    records, episodes, addresses, states = [], [], [], {}
    for unit in range(total_countries):
        sources = []
        for purpose in PURPOSES:
            address = stream_address(h, panel_id, unit, purpose)
            addresses.append(address)
            sources.append(source_factory(address))
        episode = generate_episode(h, *sources)
        episodes.append({"unit_id": unit, **episode})
        observed = episode["H"] if unit == 0 else episode["Z"]
        records.append(
            {
                "unit_id": unit,
                "L": len(observed),
                "start": 1 - len(observed),
                "end": 0,
                "states": list(observed),
            }
        )
        states[str(unit)] = {
            purpose: src.state() for purpose, src in zip(PURPOSES, sources)
        }
    require(len(set(addresses)) == 600, "duplicate innovation address")
    visible = {
        "model_id": MODEL_ID,
        "panel_id": panel_id,
        "h": h,
        "target_id": 0,
        "total_countries": 200,
        "potential_references": 199,
        "records": records,
    }
    # These two objects must NOT be forwarded to adapter.predict().
    truth = {
        "panel_id": panel_id,
        "future": list(episodes[0]["Y"]),
        "calendar": list(range(1, h + 1)),
    }
    audit = {"episodes": episodes, "addresses": addresses, "tape_states": states}
    return visible, truth, audit
