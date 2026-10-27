"""Compare rebuilt figures with the actual PDFs supplied in the author's ZIP.

PDF file hashes can differ because of creation timestamps. Instead check page
dimensions and every rendered RGB pixel, at two resolutions. No tolerance is
used. A mismatch is reported, never silently accepted as a new figure style.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent
NAMES = (
    "reliability_lambda090_main.pdf",
    "step6_length_performance_nominal90.pdf",
    "step6_h6_state_composition_nominal80.pdf",
    "step6_h6_state_composition_nominal90.pdf",
)


def compare(figures, report):
    checks = []
    for name in NAMES:
        reference = ROOT / "expected/figures" / name
        rebuilt = figures / name
        with pymupdf.open(reference) as original, pymupdf.open(rebuilt) as actual:
            if len(original) != 1 or len(actual) != 1:
                raise ValueError("Expected a one-page figure: " + name)
            sizes_equal = tuple(original[0].rect) == tuple(actual[0].rect)
            row = {
                "figure": name,
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "rebuilt_sha256": hashlib.sha256(rebuilt.read_bytes()).hexdigest(),
                "page_size_points": list(actual[0].rect),
                "page_dimensions_equal": sizes_equal,
                "renders": [],
            }
            for dpi in (144, 288):
                matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
                a = original[0].get_pixmap(
                    matrix=matrix, colorspace=pymupdf.csRGB, alpha=False
                )
                b = actual[0].get_pixmap(
                    matrix=matrix, colorspace=pymupdf.csRGB, alpha=False
                )
                shape_equal = (a.width, a.height, a.n) == (b.width, b.height, b.n)
                mismatches = None
                if shape_equal:
                    left = np.frombuffer(a.samples, dtype=np.uint8).reshape(-1, 3)
                    right = np.frombuffer(b.samples, dtype=np.uint8).reshape(-1, 3)
                    mismatches = int(np.count_nonzero(np.any(left != right, axis=1)))
                row["renders"].append(
                    {
                        "dpi": dpi,
                        "dimensions_equal": shape_equal,
                        "different_pixels": mismatches,
                        "pixel_count": a.width * a.height,
                    }
                )
            row["pass"] = sizes_equal and all(
                r["different_pixels"] == 0 for r in row["renders"]
            )
            checks.append(row)
    passed = all(row["pass"] for row in checks)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "figures": checks,
        "renderer": pymupdf.VersionBind,
        "scope": "Four standalone author-supplied PDF figures; not a full manuscript LaTeX compilation",
        "comparison": "Exact RGB pixels at 144 and 288 dpi; no tolerance; PDF timestamps excluded",
    }
    with report.open("x") as output:
        output.write(json.dumps(payload, indent=2) + "\n")
    if not passed:
        raise ValueError("Figure appearance mismatch; see " + str(report))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    compare(args.figures, args.report)
    print("Four figures match the author PDFs at both render resolutions.")
