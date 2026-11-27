"""Gate every ledger row through the feed validator before it is written (recipe-readiness: rows missing model, chip,
engine, precision, concurrency, isl, osl, context or launch are marked exclude_from_fit, never dropped).

The validator package is found via the MAGNI_TOOLS environment variable (a checkout that provides magni.feed.validate).
Without it the row is written unchanged and a warning is printed, so a missing validator is visible, not silent.
"""
import json
import os
import sys

_gate = None


def _load():
    global _gate
    if _gate is None:
        root = os.environ.get("MAGNI_TOOLS")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        try:
            from magni.feed.validate import gate
            _gate = gate
        except ImportError:
            print("feedgate: magni.feed.validate not importable (set MAGNI_TOOLS); rows written ungated", file=sys.stderr)
            _gate = False
    return _gate


def gate(row: dict) -> dict:
    g = _load()
    return g(row) if g else row


def append(path: str, row: dict) -> dict:
    """Gate one row and append it as a JSON line; returns what was written."""
    r = gate(row)
    with open(path, "a") as fh:
        fh.write(json.dumps(r, sort_keys=True) + "\n")
    return r


def merge_keep_fields(old: dict, new: dict) -> dict:
    """A re-measured row replaces the old one but keeps fields only the old row states (e.g. backfilled recipe fields)."""
    out = dict(old)
    out.update({k: v for k, v in new.items() if v is not None})
    return out
