#!/usr/bin/env python3
"""Private framed IPC for a single simulator; stdout is reserved for the parent."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import struct
import sys
import traceback

import numpy as np

MAX_MESSAGE = 32 * 1024 * 1024
ARRAY_TAG = "__robocasa_sonic_ndarray_v1__"


def encode_wire(value):
    """Keep NumPy's version-specific reconstruction modules off the wire."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise ValueError("IPC does not support object or structured arrays")
        return {ARRAY_TAG: True, "dtype": value.dtype.str, "shape": value.shape,
                "data": value.tobytes(order="C")}
    if isinstance(value, np.generic):
        return encode_wire(value.item())
    if isinstance(value, dict):
        if ARRAY_TAG in value:
            raise ValueError("Reserved ndarray IPC marker in an ordinary mapping")
        return {key: encode_wire(child) for key, child in value.items()}
    if isinstance(value, list):
        return [encode_wire(child) for child in value]
    if isinstance(value, tuple):
        return tuple(encode_wire(child) for child in value)
    return value


def decode_wire(value):
    if isinstance(value, dict):
        if ARRAY_TAG in value:
            if (set(value) != {ARRAY_TAG, "dtype", "shape", "data"}
                    or value[ARRAY_TAG] is not True):
                raise ValueError("Invalid ndarray IPC envelope")
            dtype = np.dtype(value["dtype"])
            shape, data = value["shape"], value["data"]
            if (dtype.hasobject or dtype.fields is not None
                    or not isinstance(shape, tuple)
                    or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(data, bytes)
                    or math.prod(shape) * dtype.itemsize != len(data)):
                raise ValueError("Invalid ndarray IPC dtype, shape, or payload")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {key: decode_wire(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decode_wire(child) for child in value]
    if isinstance(value, tuple):
        return tuple(decode_wire(child) for child in value)
    return value


def read_exact(stream, n):
    chunks = []
    while n:
        data = stream.read(n)
        if not data:
            raise EOFError("Simulator pipe closed")
        chunks.append(data)
        n -= len(data)
    return b"".join(chunks)


def receive(stream):
    size = struct.unpack("!I", read_exact(stream, 4))[0]
    if size > MAX_MESSAGE:
        raise ValueError(f"IPC message too large: {size}")
    # Pipe is private to the child process launched by gpu_worker, never a network listener.
    return decode_wire(pickle.loads(read_exact(stream, size)))


def send(stream, value):
    payload = pickle.dumps(encode_wire(value), protocol=5)
    if len(payload) > MAX_MESSAGE:
        raise ValueError("IPC message too large")
    for part in (struct.pack("!I", len(payload)), payload):
        view = memoryview(part)
        while view:
            written = stream.write(view)
            if written is None or written <= 0:
                raise BrokenPipeError("Simulator pipe did not accept bytes")
            view = view[written:]
    stream.flush()


def compiler_identity():
    """Bind effective JIT settings, including the distinct Numba 'max' mode."""
    import llvmlite
    import numba

    return {"numba": numba.__version__, "llvmlite": llvmlite.__version__,
            "optimization_level": int(numba.config.OPT),
            "is_opt_max": bool(numba.config.OPT.is_opt_max),
            "jit_disabled": bool(numba.config.DISABLE_JIT)}


def check(config):
    import importlib
    import importlib.util
    from pathlib import Path
    from .configuration import source_tree_identity
    for name in ("numpy", "mujoco", "onnxruntime", "OpenGL", "robocasa", "robosuite"):
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"Missing simulator dependency: {name}")
    import mujoco
    import numpy as np
    import onnxruntime
    import robocasa  # noqa: F401
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand
    from robosuite.environments.base import REGISTERED_ENVS
    from .sonic_decoder import SonicDecoder
    missing = [t["name"] for t in config["tasks"] if t["name"] not in REGISTERED_ENVS]
    if missing:
        raise ValueError(f"Task registration missing: {missing}")
    decoder = SonicDecoder(config["decoder_path"], threads=1)
    del decoder
    # Native Kitchen.reset uses this before the evaluator releases the band.
    # Import the real SONIC module, including its SDK dependencies, before
    # allocating the policy. No DDS channels are created by ElasticBand.
    band_force = np.asarray(ElasticBand().Advance(
        np.array([0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])))
    if band_force.shape != (6,) or not np.isfinite(band_force).all():
        raise ValueError("SONIC ElasticBand preflight returned invalid force/torque")
    # A tiny render catches missing OSMesa/context support without launching a benchmark episode.
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=32, width=32)
    try:
        renderer.update_scene(data)
        image = renderer.render()
        assert image.shape == (32, 32, 3)
    finally:
        renderer.close()
    # Scene placement JIT-compiles this conversion after importing the render
    # stack. A tiny render alone does not exercise that native compiler path.
    from robosuite.utils.transform_utils import mat2quat
    quaternion = np.asarray(mat2quat(np.eye(3)))
    if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or not np.allclose(np.abs(quaternion), [0., 0., 0., 1.], atol=1e-6, rtol=1e-6)):
        raise ValueError("Geometry preflight returned an invalid identity unit quaternion")
    sources = {}
    for name in ("robocasa", "robosuite", "gear_sonic", "robosuite_models"):
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        root = (Path(filename).resolve().parent if filename else
                Path(next(iter(module.__path__))).resolve())
        sources[name] = {"path": str(root), "source": source_tree_identity(root)}
    runtime_identity = {
        "sources": sources,
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "mujoco": mujoco.__version__, "onnxruntime": onnxruntime.__version__},
        "compiler": compiler_identity(),
    }
    print(json.dumps(dict(tasks_registered=len(config["tasks"]), renderer=config["renderer"],
                          decoder=str(Path(config["decoder_path"])), render_probe="passed",
                          geometry_probe="passed", sonic_reset_dependency="passed",
                          runtime_identity=runtime_identity)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(json.loads(sys.stdin.read()))
        return
    wire = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)  # Native MuJoCo / ONNX libraries also print directly to fd 1.
    sys.stdout = sys.stderr
    env = None
    try:
        from .sim_backend import SonicRoboCasaEnv
        while True:
            request = receive(sys.stdin.buffer)
            try:
                command = request["command"]
                if command == "init":
                    env = SonicRoboCasaEnv(request["config"])
                    response = {"ok": True}
                elif command == "reset":
                    job = request["job"]
                    task = dict(job["task"], timeout_seconds=job["timeout_seconds"],
                                limit_sec=job["timeout_seconds"],
                                max_model_steps=job["max_model_steps"])
                    response = {"ok": True, "observation": env.reset(task, job["seed"])}
                elif command == "step":
                    response = {"ok": True, **env.advance(request["actions"])}
                elif command == "close":
                    if env is not None:
                        env.close()
                    send(wire, {"ok": True})
                    return
                else:
                    raise ValueError(f"Unknown IPC command: {command}")
            except Exception as exc:
                response = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
            send(wire, response)
    except EOFError:
        pass
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
