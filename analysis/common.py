"""Small CSV output helper. Existing files are never overwritten."""

import csv


def csvout(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
