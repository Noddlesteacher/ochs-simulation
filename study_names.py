"""Public experiment names and their unchanged archival identifiers.

Readers use main_study or fixed_history_study. The numerical workers keep
their original identifiers so that seeds, saved identities and checksums still
refer to the same experiments. This module does not generate any randomness.
"""

LEGACY_STUDY_IDS = {
    "main_study": "step5",
    "fixed_history_study": "step6",
}

STUDY_TITLES = {
    "main_study": "Main study: random-length histories",
    "fixed_history_study": "Fixed-history study: all-state-1 target histories",
}
