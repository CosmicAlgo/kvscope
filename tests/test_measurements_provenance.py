"""Every value in core/data/measurements.json must be readable, as written,
from the result file and JSON path it cites. The Rust planner prints these
values labelled "measured", so this keeps that label honest."""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "core", "data", "measurements.json")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _entries():
    return _load(TABLE)["measurements"]


def _follow(obj, dotted):
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


@pytest.mark.parametrize("entry", _entries(), ids=lambda e: f"{e['models'][0]}:{e['path']}")
def test_value_matches_cited_source(entry):
    source = _load(os.path.join(ROOT, *entry["source"].split("/")))
    assert _follow(source, entry["path"]) == pytest.approx(entry["value"], abs=5e-5)


def test_every_entry_has_provenance_fields():
    required = {"models", "checkpoint", "quantity", "value", "gpu", "date", "engine", "source", "path"}
    for e in _entries():
        assert required <= set(e), f"missing {required - set(e)} in {e}"
