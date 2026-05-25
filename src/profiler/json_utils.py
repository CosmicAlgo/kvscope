"""
json_utils.py — Robust JSON serialization for profiling artifacts
==================================================================

scipy/numpy operations inside leak detectors produce `numpy.float64`,
`numpy.int64`, and `numpy.bool_` scalars. Python's stdlib `json` encoder
refuses these with "Object of type bool is not JSON serializable".

Use `dump_json` / `to_jsonable` to sanitize and persist profiling reports
so that a research run is never lost to a serialization hiccup.

Usage:
    from src.profiler.json_utils import dump_json
    dump_json(final_report, "results/gemma4_profile.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _coerce_scalar(obj: Any) -> Any:
    """Return a JSON-native equivalent for a single scalar-ish object.

    Handles numpy scalars (np.float64, np.int64, np.bool_), numpy arrays,
    pathlib Paths, and any object exposing .item() or .tolist().
    Falls back to str(obj) for anything exotic so a save never crashes.
    """
    # numpy scalar → python scalar
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    # numpy array / tensor → list
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    # Last-resort — never crash a profiling run over a non-essential field
    return str(obj)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert `obj` into a structure the stdlib json encoder accepts."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return _coerce_scalar(obj)


def _default(obj: Any) -> Any:
    """json.dump default= hook. Coerces unknown types rather than raising."""
    return _coerce_scalar(obj)


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Sanitize `obj` and write it to `path`. Creates parent dirs.

    Returns the Path written. Never raises on numpy scalars; falls back
    to str(...) for anything truly opaque.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Double-layer defence: recursive conversion + default= fallback.
    cleaned = to_jsonable(obj)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=indent, default=_default)
    return p


def dumps_json(obj: Any, indent: int = 2) -> str:
    """String form of dump_json for logging / embedding."""
    return json.dumps(to_jsonable(obj), indent=indent, default=_default)
