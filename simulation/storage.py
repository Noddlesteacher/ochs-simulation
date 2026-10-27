"""Exact JSON integers, checksums, and no-clobber atomic saves."""

from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json, os, sys

sys.dont_write_bytecode = True
sys.set_int_max_str_digits(0)


def require(v, message):
    if not v:
        raise ValueError(message)


def canon(obj):
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def pack(obj):
    if type(obj) is int and abs(obj) > 2**53 - 1:
        return {"__exact_integer__": str(obj)}
    if isinstance(obj, dict):
        return {k: pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pack(v) for v in obj]
    return obj


def unpack(obj):
    if isinstance(obj, dict):
        if set(obj) == {"__exact_integer__"}:
            return int(obj["__exact_integer__"])
        return {k: unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack(v) for v in obj]
    return obj


def digest(obj):
    return hashlib.sha256(canon(pack(obj)).encode("utf-8")).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    """Atomic no-clobber commit. Interrupted .pending is retained as evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "output exists: " + str(path))
    data = (canon(pack(obj)) + "\n").encode("utf-8")
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=5, mtime=0)
    temp = path.with_name(path.name + ".pending")
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)  # Atomic exclusive publication, unlike replace().
    temp.unlink()
    require(load(path) == unpack(pack(obj)), "read-back mismatch")


def load(path):
    path = Path(path)
    data = path.read_bytes()
    return unpack(json.loads(gzip.decompress(data) if path.suffix == ".gz" else data))


def event(folder, stage, **fields):
    p = Path(folder) / "events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(canon(pack(dict(stage=stage, utc=utc(), **fields))) + "\n")
        f.flush()
        os.fsync(f.fileno())


def equal_rows(left, right, key):
    require(len(left) == len(right), "row count mismatch")

    def index(rows):
        d = {}
        for row in rows:
            k = key(row)
            require(k not in d, "duplicate key")
            d[k] = row
        return d

    a, b = index(left), index(right)
    require(a.keys() == b.keys(), "key set mismatch")
    require(a == b, "keyed values mismatch")
