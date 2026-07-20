import sys
from types import SimpleNamespace

from robocasa.scripts import collect_sonic_demos as collector


def _parse_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["collect_sonic_demos.py", *args])
    return collector.get_args()


def test_render_camera_defaults_to_frontview(monkeypatch):
    args = _parse_args(monkeypatch)

    assert args.render_camera == "robot0_frontview"


def test_render_camera_accepts_robot_head_camera(monkeypatch):
    args = _parse_args(monkeypatch, "--render-camera", "robot0_head_camera")

    assert args.render_camera == "robot0_head_camera"


def test_make_env_forwards_render_camera(monkeypatch):
    captured = {}
    fake_env = object()

    def fake_make(environment, **kwargs):
        captured["environment"] = environment
        captured.update(kwargs)
        return fake_env

    monkeypatch.setattr(collector.robosuite, "make", fake_make)
    args = SimpleNamespace(
        environment="Kitchen",
        robot="SonicG1",
        layout=1,
        style=None,
        control_freq=200,
        render_camera="robot0_head_camera",
    )

    env, _ = collector.make_env(args, cfg={})

    assert env is fake_env
    assert captured["render_camera"] == "robot0_head_camera"
    assert captured["initialization_noise"] is None
