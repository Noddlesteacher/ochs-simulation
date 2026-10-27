"""Verify the files listed in the publication archive manifest. No computation."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST_SHA256.json").read_text())
    for row in manifest["files"]:
        path = root / row["path"]
        data = path.read_bytes()
        if (
            len(data) != row["bytes"]
            or hashlib.sha256(data).hexdigest() != row["sha256"]
        ):
            raise ValueError("Package file changed: " + row["path"])
    print(
        f"PASS: {len(manifest['files'])} listed package files match their SHA-256 and size."
    )


if __name__ == "__main__":
    main()
