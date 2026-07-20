import json
import socket
import time

import msgpack
import numpy as np
import pytest
import zmq

from robocasa.utils.sonic_vla_streaming import (
    MANAGER_HEADER_SIZE,
    ManagerStateSubscriber,
    RoboCasaVLACameraPublisher,
    VLACameraConfig,
    VLAExporterKeyboardPublisher,
)


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return port


def _pack_manager_state(toggle=False, abort=False, stream_mode=1):
    fields = [
        {"name": "stream_mode", "dtype": "i32", "shape": [1]},
        {"name": "toggle_data_collection", "dtype": "bool", "shape": [1]},
        {"name": "toggle_data_abort", "dtype": "bool", "shape": [1]},
    ]
    header = json.dumps({"v": 3, "endian": "le", "count": 1, "fields": fields}).encode("utf-8")
    header = header.ljust(MANAGER_HEADER_SIZE, b"\x00")
    payload = b"".join(
        [
            np.array([stream_mode], dtype=np.int32).tobytes(),
            np.array([toggle], dtype=bool).tobytes(),
            np.array([abort], dtype=bool).tobytes(),
        ]
    )
    return b"manager_state" + header + payload


def test_manager_state_subscriber_receives_mock_vr_toggles():
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    sub = ManagerStateSubscriber(host="127.0.0.1", port=port)

    try:
        time.sleep(0.2)
        deadline = time.time() + 2.0
        events = set()
        while time.time() < deadline and not {"toggle_data_collection", "toggle_data_abort"} <= events:
            pub.send(_pack_manager_state(toggle=True, abort=False))
            pub.send(_pack_manager_state(toggle=False, abort=True))
            time.sleep(0.02)
            events |= sub.poll()

        assert "toggle_data_collection" in events
        assert "toggle_data_abort" in events
    finally:
        sub.close()
        pub.close()
        ctx.term()


def test_vla_keyboard_publisher_sends_exporter_keys():
    port = _free_port()
    pub = VLAExporterKeyboardPublisher(port=port)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://127.0.0.1:{port}")

    try:
        time.sleep(0.2)
        pub.send("c")
        assert sub.poll(1000)
        assert sub.recv_string() == "c"
    finally:
        sub.close()
        ctx.term()
        pub.close()


def test_vla_camera_config_defaults_to_sonic_ego_view():
    config = VLACameraConfig()

    assert config.camera_name == "robot0_head_camera"
    assert config.output_key == "ego_view"
    assert config.flip_vertical is False
    assert config.render_collision_mesh is False
    assert config.render_visual_mesh is True


def test_robocasa_vla_camera_publisher_rate_limits_fake_sim(monkeypatch):
    sensor_server = pytest.importorskip("gear_sonic.camera.sensor_server")

    class FakeRenderer:
        def __init__(self):
            self.render_count = 0

        def update_scene(self, data, camera, scene_option=None):
            assert camera == "robot0_head_camera"

        def render(self):
            self.render_count += 1
            width, height = 32, 24
            rows = (np.arange(height, dtype=np.uint8) * 10)[:, None, None]
            cols = np.zeros((1, width, 3), dtype=np.uint8)
            return rows + cols

    class FakeEnv:
        def __init__(self):
            self.sim = type(
                "FakeSim",
                (),
                {"data": type("FakeData", (), {"_data": object()})()},
            )()

    port = _free_port()
    publisher = RoboCasaVLACameraPublisher(
        VLACameraConfig(width=32, height=24, hz=30.0, port=port)
    )
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://127.0.0.1:{port}")
    env = FakeEnv()
    fake_renderer = FakeRenderer()
    monkeypatch.setattr(publisher, "_get_live_renderer", lambda env: (fake_renderer, None))

    try:
        publisher.start()
        time.sleep(0.2)
        assert publisher.maybe_publish(env)
        for _ in range(10):
            assert not publisher.maybe_publish(env)
        assert fake_renderer.render_count == 1

        assert sub.poll(1000)
        message = msgpack.unpackb(sub.recv(), raw=False)
        assert "ego_view" in message["images"]
        assert "ego_view" in message["timestamps"]
        image = sensor_server.ImageMessageSchema.deserialize(message).images["ego_view"]
        assert image[0].mean() < 50
        assert image[-1].mean() > 180

        time.sleep(0.06)
        assert publisher.maybe_publish(env)
        assert fake_renderer.render_count == 2
    finally:
        publisher.close()
        sub.close()
        ctx.term()


def test_robocasa_vla_camera_publisher_warms_cameras_during_start(monkeypatch):
    class FakeRenderer:
        def __init__(self):
            self.cameras = []
            self.render_count = 0

        def update_scene(self, data, camera, scene_option=None):
            self.cameras.append(camera)

        def render(self):
            self.render_count += 1
            return np.zeros((12, 16, 3), dtype=np.uint8)

    class FakeEnv:
        def __init__(self):
            self.sim = type(
                "FakeSim",
                (),
                {"data": type("FakeData", (), {"_data": object()})()},
            )()

    camera_names = (
        "robot0_head_camera",
        "robot0_left_wrist_camera",
        "robot0_right_wrist_camera",
    )
    publisher = RoboCasaVLACameraPublisher(
        VLACameraConfig(
            camera_names=camera_names,
            output_keys=("ego_view", "left_wrist", "right_wrist"),
            width=16,
            height=12,
            hz=30.0,
        )
    )
    env = FakeEnv()
    fake_renderer = FakeRenderer()

    def fake_send_loop():
        publisher._ready_event.set()
        publisher._stop_event.wait(timeout=1.0)

    monkeypatch.setattr(publisher, "_send_loop", fake_send_loop)
    monkeypatch.setattr(
        publisher,
        "_get_live_renderer",
        lambda env: (fake_renderer, None),
    )

    try:
        publisher.start(env)

        assert fake_renderer.cameras == list(camera_names)
        assert fake_renderer.render_count == len(camera_names)
    finally:
        publisher.close()


def test_robocasa_vla_camera_publisher_sends_multiple_camera_keys(monkeypatch):
    class FakeRenderer:
        def __init__(self):
            self.cameras = []

        def update_scene(self, data, camera, scene_option=None):
            self.cameras.append(camera)

        def render(self):
            value = len(self.cameras)
            return np.full((12, 16, 3), value, dtype=np.uint8)

    class FakeEnv:
        def __init__(self):
            self.sim = type(
                "FakeSim",
                (),
                {"data": type("FakeData", (), {"_data": object()})()},
            )()

    camera_names = (
        "robot0_head_camera",
        "robot0_left_wrist_camera",
        "robot0_right_wrist_camera",
    )
    output_keys = ("ego_view", "left_wrist", "right_wrist")
    publisher = RoboCasaVLACameraPublisher(
        VLACameraConfig(
            camera_names=camera_names,
            output_keys=output_keys,
            width=16,
            height=12,
            hz=30.0,
        )
    )
    env = FakeEnv()
    fake_renderer = FakeRenderer()
    enqueued = []
    monkeypatch.setattr(publisher, "_get_live_renderer", lambda env: (fake_renderer, None))
    monkeypatch.setattr(
        publisher,
        "_enqueue_images",
        lambda timestamps, images: enqueued.append((timestamps, images)),
    )

    assert publisher.maybe_publish(env)

    timestamps, images = enqueued[0]
    assert set(images) == set(output_keys)
    assert set(timestamps) == set(output_keys)
    assert fake_renderer.cameras == list(camera_names)
    assert images["ego_view"].mean() == 1
    assert images["left_wrist"].mean() == 2
    assert images["right_wrist"].mean() == 3
