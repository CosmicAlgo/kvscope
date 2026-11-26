"""Model names hard-coded in the experiment scripts must be ones the code accepts.

A phase in run_profiling.sh once built ``KVCacheTracer(model_type='glm47flash')``
while the tracer only knew ``glm51``, so the phase crashed on start. These tests
read the shell scripts and check every such name.
"""

import os
import re

from src.profiler.kv_tracer import KVCacheTracer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")


def _read(name):
    with open(os.path.join(EXP, name), encoding="utf-8") as f:
        return f.read()


def _phases(script):
    """Phase names accepted by run_profiling.sh's final `case` statement."""
    body = script[script.rindex('case "$PHASE" in'):]
    names = set()
    for m in re.finditer(r"^  ([a-z0-9_|]+)\)", body, re.MULTILINE):
        names.update(m.group(1).split("|"))
    return names


def test_tracer_model_types_in_scripts_are_known():
    found = set()
    for name in ("run_profiling.sh", "run_longctx_sweep.sh", "run_spec_decode.sh"):
        found |= set(re.findall(r"KVCacheTracer\([^)]*model_type=['\"](\w+)['\"]", _read(name)))
    assert found, "expected at least one KVCacheTracer(model_type=...) in the scripts"
    unknown = sorted(found - KVCacheTracer.MODEL_TYPES)
    assert not unknown, f"scripts pass model_type values the tracer rejects: {unknown}"


def test_full_collection_defaults_are_real_phases():
    m = re.search(r'^MODELS="\$\{MODELS:-([^}]*)\}"', _read("run_full_collection.sh"), re.MULTILINE)
    assert m, "default MODELS line not found"
    phases = _phases(_read("run_profiling.sh"))
    unknown = sorted(set(m.group(1).split()) - phases)
    assert not unknown, f"run_full_collection.sh defaults to unknown phases: {unknown}"
