"""Compiler isolation and smoke provenance without native simulator imports."""
import copy
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robocasa_eval import configuration, evaluation, sim_worker


@pytest.fixture
def config(tmp_path):
    return configuration.build_config({"workspace": str(tmp_path / "workspace"),
                                       "envs_per_worker": 2}, environment={})[0]


def test_osmesa_compiler_default_does_not_leak_to_egl_or_parent(config, monkeypatch):
    monkeypatch.delenv("NUMBA_OPT", raising=False)
    before = dict(os.environ)
    cpu_environment = evaluation.sim_environment(config)
    gpu_environment = evaluation.sim_environment(dict(config, renderer="egl"))
    assert cpu_environment["NUMBA_OPT"] == "0"
    assert "NUMBA_OPT" not in gpu_environment
    assert dict(os.environ) == before


@pytest.mark.parametrize("renderer", ["osmesa", "egl"])
@pytest.mark.parametrize("requested", ["1", "3", "max"])
def test_explicit_compiler_setting_remains_an_operator_choice(config, monkeypatch, renderer, requested):
    monkeypatch.setenv("NUMBA_OPT", requested)
    environment = evaluation.sim_environment(dict(config, renderer=renderer))
    assert environment["NUMBA_OPT"] == requested
    assert os.environ["NUMBA_OPT"] == requested


def test_compiler_changes_do_not_reuse_previous_cache(config, monkeypatch):
    monkeypatch.delenv("NUMBA_OPT", raising=False)
    initial = evaluation.sim_environment(config)
    cache = Path(initial["NUMBA_CACHE_DIR"])
    cache.mkdir(parents=True)
    (cache / "previous-compiled-binary").write_bytes(b"previous settings")
    monkeypatch.setenv("NUMBA_OPT", "3")
    changed = evaluation.sim_environment(config)
    assert changed["NUMBA_CACHE_DIR"] != initial["NUMBA_CACHE_DIR"]
    assert not (Path(changed["NUMBA_CACHE_DIR"]) / "previous-compiled-binary").exists()
    monkeypatch.delenv("NUMBA_OPT")
    assert evaluation.sim_environment(config)["NUMBA_CACHE_DIR"] == str(cache)
    assert (cache / "previous-compiled-binary").is_file()


def test_worker_cache_namespace_ignores_existing_cache_location(tmp_path):
    environment = {"NUMBA_OPT": "0", "NUMBA_CACHE_DIR": "old-location"}
    before = evaluation.compiler_cache_path(tmp_path, environment)
    environment["NUMBA_CACHE_DIR"] = "new-location"
    assert evaluation.compiler_cache_path(tmp_path, environment) == before
    environment["NUMBA_DISABLE_JIT"] = "1"
    assert evaluation.compiler_cache_path(tmp_path, environment) != before


@pytest.fixture
def compiler_modules(monkeypatch):
    class OptLevel(int):
        def __new__(cls, value):
            result = super().__new__(cls, 3 if value == "max" else value)
            result.is_opt_max = value == "max"
            return result

    numba = ModuleType("numba")
    numba.__version__ = "0.61.2"
    numba.config = SimpleNamespace(OPT=OptLevel(3), DISABLE_JIT=0)
    llvmlite = ModuleType("llvmlite")
    llvmlite.__version__ = "0.44.0"
    monkeypatch.setitem(sys.modules, "numba", numba)
    monkeypatch.setitem(sys.modules, "llvmlite", llvmlite)
    return numba, llvmlite, OptLevel


def test_compiler_identity_records_loaded_settings_instead_of_environment(compiler_modules, monkeypatch):
    numba, _, opt_level = compiler_modules
    monkeypatch.setenv("NUMBA_OPT", "0")
    ordinary = sim_worker.compiler_identity()
    numba.config.OPT = opt_level("max")
    maximum = sim_worker.compiler_identity()
    assert ordinary["optimization_level"] == maximum["optimization_level"] == 3
    assert ordinary["is_opt_max"] is False and maximum["is_opt_max"] is True
    numba.config.DISABLE_JIT = 1
    assert sim_worker.compiler_identity()["jit_disabled"] is True


@pytest.mark.parametrize("field,value", [
    ("optimization_level", 0), ("is_opt_max", True), ("jit_disabled", True),
    ("numba", "changed-numba"), ("llvmlite", "changed-llvmlite"),
])
def test_compiler_changes_cannot_reuse_a_passed_smoke(config, compiler_modules, field, value):
    config["runtime_identity"] = {"compiler": sim_worker.compiler_identity()}
    configuration.refresh_identity(config)
    identity = config["identity"]
    receipt_path = evaluation.smoke_receipt_path(config)
    evaluation.atomic_json(receipt_path, {
        "status": "passed", "full_horizons": True, "tested_environment_count": 2,
        "tested_categories": ["atomic", "composite"],
        "protocol": configuration.gate_protocol(config),
    })
    assert evaluation.require_smoke(config)["status"] == "passed"

    changed = copy.deepcopy(config)
    changed["runtime_identity"]["compiler"][field] = value
    configuration.refresh_identity(changed)
    assert changed["identity"] != identity
    assert evaluation.smoke_receipt_path(changed) != receipt_path
    with pytest.raises(ValueError, match="full-episodes"):
        evaluation.require_smoke(changed)
