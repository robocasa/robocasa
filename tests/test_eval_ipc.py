import io
import pickle
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robocasa_eval.sim_worker import ARRAY_TAG, decode_wire, encode_wire, receive, send


def test_check_counts_selected_tasks_and_records_loaded_runtime(tmp_path, monkeypatch, capsys):
    """Exercise preflight identity collection without importing a real simulator."""
    import importlib.util
    from robocasa_eval import configuration, sim_worker, sonic_decoder
    modules = {}
    for name in ("mujoco", "onnxruntime", "robocasa", "robosuite", "gear_sonic",
                 "robosuite_models", "robosuite.environments", "robosuite.environments.base",
                 "robosuite.utils", "robosuite.utils.transform_utils",
                 "gear_sonic.utils", "gear_sonic.utils.mujoco_sim",
                 "gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge"):
        module = ModuleType(name)
        module.__file__ = str(tmp_path / name.replace(".", "/") / "__init__.py")
        module.__path__ = [str(Path(module.__file__).parent)]
        module.__version__ = "test-version"
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["robosuite.environments.base"].REGISTERED_ENVS = {"TaskA": None, "TaskB": None}
    modules["gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge"].ElasticBand = (
        lambda: SimpleNamespace(Advance=lambda state: np.zeros(6)))
    modules["mujoco"].MjModel = SimpleNamespace(from_xml_string=lambda xml: object())
    modules["mujoco"].MjData = lambda model: object()
    closed = []
    modules["mujoco"].Renderer = lambda *a, **kw: SimpleNamespace(
        update_scene=lambda data: None, render=lambda: np.zeros((32, 32, 3), np.uint8),
        close=lambda: closed.append(True))
    geometry_inputs = []

    def mat2quat(matrix):
        geometry_inputs.append(matrix.copy())
        return np.array([0., 0., 0., 1.])

    modules["robosuite.utils.transform_utils"].mat2quat = mat2quat
    compiler = {"numba": "test-numba", "llvmlite": "test-llvmlite",
                "optimization_level": 0, "is_opt_max": False, "jit_disabled": False}
    monkeypatch.setattr(sim_worker, "compiler_identity", lambda: compiler)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(sonic_decoder, "SonicDecoder", lambda *a, **kw: object())
    monkeypatch.setattr(configuration, "source_tree_identity", lambda root: {"sha256": "frozen-source"})
    sim_worker.check(dict(tasks=[{"name": "TaskA"}, {"name": "TaskB"}],
                          decoder_path=str(tmp_path / "decoder.onnx"), renderer="osmesa"))
    report = json.loads(capsys.readouterr().out)
    assert report["tasks_registered"] == 2
    assert report["runtime_identity"]["versions"]["numpy"] == np.__version__
    assert report["runtime_identity"]["versions"]["onnxruntime"] == "test-version"
    assert report["runtime_identity"]["compiler"] == compiler
    assert report["geometry_probe"] == "passed"
    assert len(geometry_inputs) == 1
    np.testing.assert_array_equal(geometry_inputs[0], np.eye(3))
    sources = report["runtime_identity"]["sources"]
    assert set(sources) == {"robocasa", "robosuite", "gear_sonic", "robosuite_models"}
    assert sources["robocasa"]["source"] == {"sha256": "frozen-source"}
    assert sources["robocasa"]["path"] == str(tmp_path / "robocasa")
    assert closed == [True]


def test_numpy_reconstruction_modules_are_not_in_wire():
    value = {"image": np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, ::2],
             "state": [np.array([1.25, -2.5], dtype=np.float32)],
             "token": (np.zeros((20, 64), dtype=np.float32), np.float32(2.5))}
    payload = pickle.dumps(encode_wire(value), protocol=5)
    assert b"numpy" not in payload
    decoded = decode_wire(pickle.loads(payload))
    np.testing.assert_array_equal(decoded["image"], value["image"])
    assert decoded["image"].flags.writeable
    assert decoded["image"].flags.c_contiguous
    np.testing.assert_array_equal(decoded["state"][0], value["state"][0])
    assert isinstance(decoded["token"], tuple)
    assert type(decoded["token"][1]) is float


@pytest.mark.parametrize("value", [
    np.empty((0, 7), dtype=np.float32),
    np.array(3, dtype=np.int64),
    np.array([1.5, -4.0], dtype=">f4"),
])
def test_framed_array_roundtrip_preserves_shape_dtype(value):
    stream = io.BytesIO()
    send(stream, value)
    stream.seek(0)
    restored = receive(stream)
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)


@pytest.mark.parametrize("value", [
    np.array([object()], dtype=object),
    np.zeros(2, dtype=[("x", "f4"), ("y", "f4")]),
    {ARRAY_TAG: "ordinary user data"},
])
def test_unsupported_array_or_reserved_marker_is_rejected(value):
    with pytest.raises(ValueError):
        encode_wire(value)


def test_array_payload_length_and_dimensions_are_validated():
    valid = encode_wire(np.ones((2, 3), dtype=np.float32))
    for change in ({"data": b"short"}, {"shape": (2, -3)}, {"shape": (True, 6)},
                   {"dtype": "O"}, {"shape": [2, 3]}):
        with pytest.raises(ValueError):
            decode_wire({**valid, **change})
