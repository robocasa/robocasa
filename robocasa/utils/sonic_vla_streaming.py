"""Small ZMQ helpers for SONIC VLA collection from RoboCasa.

The RoboCasa collector owns the MuJoCo clock. These helpers keep VLA-side
streaming non-blocking so camera/recording integration does not become part of
the 200 Hz control contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
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
    width: int = 640
    height: int = 480
    hz: float = 30.0
    port: int = 5555
    flip_vertical: bool = True
    async_render: bool = True
    render_collision_mesh: bool = False
    render_visual_mesh: bool = True


def _prepare_image(image: np.ndarray, flip_vertical: bool) -> np.ndarray:
    if flip_vertical:
        image = image[::-1]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _increment_counter(counter):
    if counter is None:
        return
    with counter.get_lock():
        counter.value += 1


def _put_error(error_queue, exc: BaseException):
    try:
        error_queue.put_nowait(f"{type(exc).__name__}: {exc}")
    except queue.Full:
        pass


def _camera_process_main(
    config: VLACameraConfig,
    model_xml: str,
    snapshot_queue,
    ready_event,
    error_queue,
    publish_count,
):
    server = None
    render_sim = None
    try:
        from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer
        from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim, MjSimState

        server = SensorServer()
        server.start_server(port=config.port)
        render_sim = MjSim.from_xml_string(model_xml)
        render_context = MjRenderContextOffscreen(
            render_sim,
            device_id=-1,
            max_width=config.width,
            max_height=config.height,
        )
        render_context.vopt.geomgroup[0] = 1 if config.render_collision_mesh else 0
        render_context.vopt.geomgroup[1] = 1 if config.render_visual_mesh else 0
        ready_event.set()

        while True:
            try:
                item = snapshot_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            timestamp, sim_time, qpos, qvel = item
            if qpos.shape != render_sim.data.qpos.shape or qvel.shape != render_sim.data.qvel.shape:
                raise RuntimeError(
                    "VLA render state shape mismatch; restart collection after a model topology change"
                )

            render_sim.set_state(MjSimState(time=sim_time, qpos=qpos, qvel=qvel))
            render_sim.forward()
            image = render_sim.render(
                camera_name=config.camera_name,
                width=config.width,
                height=config.height,
                depth=False,
            )
            image = _prepare_image(image, config.flip_vertical)
            message = ImageMessageSchema(
                timestamps={config.output_key: timestamp},
                images={config.output_key: image},
            )
            server.send_message(message.serialize())
            _increment_counter(publish_count)
    except BaseException as exc:
        _put_error(error_queue, exc)
        ready_event.set()
    finally:
        if server is not None:
            server.stop_server()
        if render_sim is not None and render_sim._render_context_offscreen is not None:
            del render_sim._render_context_offscreen


class RoboCasaVLACameraPublisher:
    """Publish RoboCasa frames using SONIC's camera-server schema.

    In the normal async mode, the control loop only copies MuJoCo state. A
    render worker owns a separate offscreen sim so image rendering does not
    become part of the 200 Hz control budget.
    """

    def __init__(self, config: VLACameraConfig):
        if config.hz <= 0:
            raise ValueError("VLA camera hz must be positive")
        self.config = config
        self._period = 1.0 / float(config.hz)
        self._next_publish_time = 0.0
        self._queue: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=1)
        self._snapshot_queue = None
        self._mp_ctx = None
        self._mp_ready = None
        self._mp_errors = None
        self._mp_publish_count = None
        self._mp_drop_count = None
        self._process = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._publish_count = 0
        self._drop_count = 0

    @property
    def publish_count(self) -> int:
        if self._mp_publish_count is not None:
            return int(self._mp_publish_count.value)
        return self._publish_count

    @property
    def drop_count(self) -> int:
        if self._mp_drop_count is not None:
            return int(self._mp_drop_count.value)
        return self._drop_count

    def start(self, env=None):
        if self.config.async_render:
            if env is not None:
                self._start_camera_process(env)
            return

        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._send_loop, name="robocasa-vla-camera", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=2.0):
            raise RuntimeError("Timed out starting RoboCasa VLA camera publisher")
        if self._startup_error is not None:
            raise RuntimeError("Failed to start RoboCasa VLA camera publisher") from self._startup_error

    def maybe_publish(self, env) -> bool:
        now = time.perf_counter()
        if now < self._next_publish_time:
            return False

        if self._next_publish_time == 0.0 or now - self._next_publish_time > self._period:
            self._next_publish_time = now + self._period
        else:
            self._next_publish_time += self._period

        if self.config.async_render:
            self._start_camera_process(env)
            self._raise_camera_process_error()
            state = env.sim.get_state()
            self._enqueue_snapshot((time.time(), float(state.time), state.qpos, state.qvel))
            return True

        image = env.sim.render(
            camera_name=self.config.camera_name,
            width=self.config.width,
            height=self.config.height,
            depth=False,
        )
        self._enqueue_image(time.time(), _prepare_image(image, self.config.flip_vertical))
        return True

    def _enqueue_snapshot(self, snapshot: tuple[float, float, np.ndarray, np.ndarray]):
        if self._snapshot_queue is None:
            raise RuntimeError("RoboCasa VLA camera process is not started")
        try:
            self._snapshot_queue.put_nowait(snapshot)
        except queue.Full:
            self._increment_drop_count()
            try:
                self._snapshot_queue.get_nowait()
            except queue.Empty:
                pass
            self._snapshot_queue.put_nowait(snapshot)

    def _enqueue_image(self, timestamp: float, image: np.ndarray):
        try:
            self._queue.put_nowait((timestamp, image))
        except queue.Full:
            self._increment_drop_count()
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait((timestamp, image))

    def _increment_drop_count(self):
        if self._mp_drop_count is not None:
            _increment_counter(self._mp_drop_count)
            return
        self._drop_count += 1

    def _start_camera_process(self, env):
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            self._raise_camera_process_error()
            raise RuntimeError("RoboCasa VLA camera process exited unexpectedly")

        # Use the compiled sim XML, not env.model.get_xml(). RoboCasa applies
        # camera overrides in edit_model_xml(), so env.model.get_xml() can be
        # stale and render a different camera than playback / env.sim.render().
        model_xml = env.sim.model.get_xml()
        self._mp_ctx = mp.get_context("fork")
        self._snapshot_queue = self._mp_ctx.Queue(maxsize=1)
        self._mp_ready = self._mp_ctx.Event()
        self._mp_errors = self._mp_ctx.Queue(maxsize=1)
        self._mp_publish_count = self._mp_ctx.Value("i", 0)
        self._mp_drop_count = self._mp_ctx.Value("i", 0)
        self._process = self._mp_ctx.Process(
            target=_camera_process_main,
            args=(
                self.config,
                model_xml,
                self._snapshot_queue,
                self._mp_ready,
                self._mp_errors,
                self._mp_publish_count,
            ),
            daemon=True,
        )
        self._process.start()
        if not self._mp_ready.wait(timeout=8.0):
            self.close()
            raise RuntimeError("Timed out starting RoboCasa VLA camera process")
        self._raise_camera_process_error()

    def _raise_camera_process_error(self):
        if self._mp_errors is None:
            return
        try:
            error = self._mp_errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"RoboCasa VLA camera process failed: {error}")

    def _send_loop(self):
        server = None
        try:
            from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

            server = SensorServer()
            server.start_server(port=self.config.port)
            self._ready_event.set()
            while not self._stop_event.is_set():
                try:
                    timestamp, image = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                message = ImageMessageSchema(
                    timestamps={self.config.output_key: timestamp},
                    images={self.config.output_key: image},
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
        if self._process is not None:
            if self._snapshot_queue is not None:
                try:
                    self._snapshot_queue.put_nowait(None)
                except queue.Full:
                    try:
                        self._snapshot_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._snapshot_queue.put_nowait(None)
                    except queue.Full:
                        pass
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None
            if self._snapshot_queue is not None:
                self._snapshot_queue.close()
                self._snapshot_queue = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
