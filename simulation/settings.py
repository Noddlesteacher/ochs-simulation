"""Frozen study constants, not a parameter-search interface.

RUN and MODEL are part of the original random-stream addresses. Changing them
would change the data even if the master seed were unchanged.
"""

from pathlib import Path

from .storage import load, require

PACKAGE = Path(__file__).resolve().parents[1]
RUN = "OCHS_SHIFTED_STEP5_FORMAL_R01"
MODEL = MODEL_ID = "OCHS_SHIFTED_SCORING_EPISODE_V1"
FIXED_RUN = "OCHS_FIXED_ALL1_STEP6_R01"
FIXED_MODEL = "OCHS_FIXED_ALL1_CONDITIONAL_V1"
HS = (3, 6, 9)
LS = (10, 50, 100, 200)
METHODS = ("ochs090", "count", "ochs050", "randomized_iblock")
LAMBDAS = {"ochs090": "9/10", "count": "1", "ochs050": "1/2"}
A = ((1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 1, 1), (1, 1, 0, 0))
TOTAL_COUNTRIES = 200
POTENTIAL_REFERENCES = 199
REPLICATES = 5000


def frozen_seed(study):
    """Read existing public simulation entropy; never allocate a new seed."""
    require(study in ("step5", "step6"), "Unknown study")
    return load(PACKAGE / "protocol" / f"{study}_seed.json")[
        "master_entropy_hex"
    ]


def formal_authorization(master, run):
    """Publication replay guard, replacing machine-specific old launch paths.

    The source factory's bytes, hash, PCG64 calls and addresses are unchanged.
    This check does not start a run. run_simulation.py requires a separate
    explicit long-run flag and a new destination directory.
    """
    require(run == RUN, "Wrong main-study identity")
    require(master == frozen_seed("step5"), "Only the frozen seed is supported")
