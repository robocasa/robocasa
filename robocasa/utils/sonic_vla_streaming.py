"""Small ZMQ helpers for SONIC VLA collection from RoboCasa.

The RoboCasa collector owns the MuJoCo clock. These helpers keep VLA-side
streaming non-blocking so camera/recording integration does not become part of
the 200 Hz control contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import threading
import time
from typing import Any

import numpy as np
import zmq


MANAGER_HEADER_SIZE = 1280
MANAGER_TOPIC = "manager_state"


def unpack_pico_message(packed_data: bytes, topic: str = MANAGER_TOPIC) -> dict[str, Any]:
    """Decode pico_manager_thread_server's topic + fixed-header binary format."""
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + MANAGER_HEADER_SIZE:
        raise ValueError(
            f"Packed data too small: {len(packed_data)} < {offset + MANAGER_HEADER_SIZE}"
        )

    header_bytes = packed_data[offset : offset + MANAGER_HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]
    header = json.loads(header_bytes.decode("utf-8"))

    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": np.bool_,
        "u8": np.uint8,
    }
    result: dict[str, Any] = {
        "version": header.get("v", 0),
        "endian": header.get("endian", "le"),
    }
    current_offset = offset + MANAGER_HEADER_SIZE
    for field in header.get("fields", []):
        dtype = np.dtype(dtype_map.get(field["dtype"], np.float32))
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * dtype.itemsize
        end = current_offset + n_bytes
        if end > len(packed_data):
            raise ValueError(f"Field '{field['name']}' extends past message end")
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset:end], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset = end
    return result


def _array_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return bool(value.flat[0])
    return bool(value)


class ManagerStateSubscriber:
    """Receive VR recording toggles from pico_manager_thread_server."""

    def __init__(self, host: str = "localhost", port: int = 5556):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, MANAGER_TOPIC)
        self._socket.setsockopt(zmq.RCVHWM, 20)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")

    def poll(self, max_messages: int = 20) -> set[str]:
        events: set[str] = set()
        for _ in range(max_messages):
            try:
                raw = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                data = unpack_pico_message(raw, topic=MANAGER_TOPIC)
            except Exception:
                continue
            if _array_bool(data, "toggle_data_collection"):
                events.add("toggle_data_collection")
            if _array_bool(data, "toggle_data_abort"):
                events.add("toggle_data_abort")
        return events

    def close(self):
        self._socket.close()
        self._ctx.term()


class VLAExporterKeyboardPublisher:
    """Forward local collector hotkeys to run_data_exporter.py's ZMQ keyboard channel."""

    def __init__(self, port: int = 5580):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 20)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{port}")
        # Avoid the PUB/SUB slow-joiner dropping the first user command.
        time.sleep(0.2)

    def send(self, key: str):
        self._socket.send_string(key, flags=zmq.NOBLOCK)

    def close(self):
        self._socket.close()
        self._ctx.term()


@dataclass
class VLACameraConfig:
    camera_name: str = "robot0_head_camera"
    output_key: str = "ego_view"
    camera_names: tuple[str, ...] | list[str] | None = None
    output_keys: tuple[str, ...] | list[str] | None = None
    width: int = 640
    height: int = 480
    hz: float = 30.0
    port: int = 5555
    flip_vertical: bool = False
    render_collision_mesh: bool = False
    render_visual_mesh: bool = True


def _prepare_image(image: np.ndarray, flip_vertical: bool) -> np.ndarray:
    if flip_vertical:
        image = image[::-1]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _resolve_names(value, fallback: str, label: str) -> tuple[str, ...]:
    if value is None:
        resolved = (fallback,)
    elif isinstance(value, str):
        resolved = (value,)
    else:
        resolved = tuple(value)
    if not resolved or any(not item for item in resolved):
        raise ValueError(f"VLA {label} must contain at least one non-empty value")
    return resolved


class RoboCasaVLACameraPublisher:
    """Publish RoboCasa frames using SONIC's camera-server schema.

    This follows the SONIC sim path: render from the collector's live MuJoCo
    sim, then hand the image to a background ZMQ publisher.
    """

    def __init__(self, config: VLACameraConfig):
        if config.hz <= 0:
            raise ValueError("VLA camera hz must be positive")
        self.config = config
        self._camera_names = _resolve_names(
            config.camera_names, config.camera_name, "camera_names"
        )
        self._output_keys = _resolve_names(
            config.output_keys, config.output_key, "output_keys"
        )
        if len(self._camera_names) != len(self._output_keys):
            raise ValueError("VLA camera_names and output_keys must have the same length")
        if len(set(self._output_keys)) != len(self._output_keys):
            raise ValueError("VLA output_keys must be unique")
        self._period = 1.0 / float(config.hz)
        self._next_publish_time = 0.0
        self._queue: queue.Queue[
            tuple[dict[str, float], dict[str, np.ndarray]]
        ] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._renderer = None
        self._renderer_model = None
        self._scene_option = None
        self._publish_count = 0
        self._drop_count = 0

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def drop_count(self) -> int:
        return self._drop_count

    def start(self, env=None):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._send_loop, name="robocasa-vla-camera", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=2.0):
            raise RuntimeError("Timed out starting RoboCasa VLA camera publisher")
        if self._startup_error is not None:
            raise RuntimeError("Failed to start RoboCasa VLA camera publisher") from self._startup_error
        if env is not None:
            self._render_images(env)

    def maybe_publish(self, env) -> bool:
        self._raise_sender_error()
        now = time.perf_counter()
        if now < self._next_publish_time:
            return False

        if self._next_publish_time == 0.0 or now - self._next_publish_time > self._period:
            self._next_publish_time = now + self._period
        else:
            self._next_publish_time += self._period

        timestamp = time.time()
        timestamps = {output_key: timestamp for output_key in self._output_keys}
        images = self._render_images(env)
        self._enqueue_images(timestamps, images)
        return True

    def _render_images(self, env) -> dict[str, np.ndarray]:
        renderer, scene_option = self._get_live_renderer(env)
        images: dict[str, np.ndarray] = {}
        for camera_name, output_key in zip(self._camera_names, self._output_keys):
            renderer.update_scene(
                env.sim.data._data, camera=camera_name, scene_option=scene_option
            )
            image = renderer.render()
            images[output_key] = _prepare_image(image, self.config.flip_vertical)
        return images

    def _raise_sender_error(self):
        if self._startup_error is not None:
            raise RuntimeError("RoboCasa VLA camera publisher failed") from self._startup_error
        if self._thread is not None and not self._thread.is_alive() and not self._stop_event.is_set():
            raise RuntimeError("RoboCasa VLA camera publisher exited unexpectedly")

    def _get_live_renderer(self, env):
        model = env.sim.model._model
        if self._renderer is None or self._renderer_model is not model:
            self._close_renderer()
            import mujoco

            self._renderer = mujoco.Renderer(model, height=self.config.height, width=self.config.width)
            self._scene_option = mujoco.MjvOption()
            self._renderer_model = model
        self._scene_option.geomgroup[0] = 1 if self.config.render_collision_mesh else 0
        self._scene_option.geomgroup[1] = 1 if self.config.render_visual_mesh else 0
        return self._renderer, self._scene_option

    def _close_renderer(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_model = None
            self._scene_option = None

    def _enqueue_images(self, timestamps: dict[str, float], images: dict[str, np.ndarray]):
        try:
            self._queue.put_nowait((timestamps, images))
        except queue.Full:
            self._increment_drop_count()
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait((timestamps, images))

    def _increment_drop_count(self):
        self._drop_count += 1

    def _send_loop(self):
        server = None
        try:
            from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

            server = SensorServer()
            server.start_server(port=self.config.port)
            self._ready_event.set()
            while not self._stop_event.is_set():
                try:
                    timestamps, images = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                message = ImageMessageSchema(
                    timestamps=timestamps,
                    images=images,
                )
                server.send_message(message.serialize())
                self._publish_count += 1
        except BaseException as exc:
            self._startup_error = exc
            self._ready_event.set()
            raise
        finally:
            if server is not None:
                server.stop_server()

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_renderer()
