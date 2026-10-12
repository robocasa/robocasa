from types import SimpleNamespace

import pytest

from robocasa.environments.kitchen.atomic.kitchen_pick_place import (
    PickPlaceCounterToStove,
)
from robocasa.utils import env_utils as EnvUtils


def _make_task(robot_name):
    robot_model_cls = type(robot_name, (), {})
    task = object.__new__(PickPlaceCounterToStove)
    task.robots = [SimpleNamespace(robot_model=robot_model_cls())]
    task.obj_groups = "food"
    task.exclude_obj_groups = None
    task.stove = object()
    task.counter = object()
    task._ep_meta = {}
    task.model = SimpleNamespace(merge_objects=lambda _: None)
    return task


def _food_placement(task):
    return next(
        cfg["placement"] for cfg in task._get_obj_cfgs() if cfg["name"] == "obj"
    )


def test_sonic_g1_restricts_only_generated_plate_to_counter_front():
    task = _make_task("SonicG1")
    food_placement = _food_placement(task)

    assert food_placement["size"] == (0.30, 0.30)
    assert "side" not in food_placement
    plate_placement = food_placement["try_to_place_in_kwargs"]["placement"]
    assert plate_placement["fixture"] is task.counter
    assert plate_placement["sample_region_kwargs"]["ref"] is task.stove
    assert plate_placement["size"] == (0.30, 0.30)
    assert plate_placement["pos"] == ("ref", -1.0)
    assert plate_placement["side"] == "front"
    assert "rotation" not in plate_placement


@pytest.mark.parametrize("robot_name", ["PandaOmron", "G1", "SonicG1Fixed"])
def test_other_robots_keep_original_plate_sampling(robot_name):
    placement = _food_placement(_make_task(robot_name))

    assert placement["size"] == (0.30, 0.30)
    assert placement["pos"] == ("ref", -1.0)
    assert "try_to_place_in_kwargs" not in placement


def test_sonic_plate_override_reaches_generated_container(monkeypatch):
    task = _make_task("SonicG1")

    def fake_create_obj(_env, cfg):
        groups = ["in_container"] if cfg["name"] == "obj" else []
        return SimpleNamespace(name=cfg["name"]), {
            "groups_containing_sampled_obj": groups
        }

    monkeypatch.setattr(EnvUtils, "create_obj", fake_create_obj)
    task._create_objects()

    plate_cfg = next(c for c in task.object_cfgs if c["name"] == "obj_container")
    food_cfg = next(c for c in task.object_cfgs if c["name"] == "obj")
    assert plate_cfg["placement"]["side"] == "front"
    assert "rotation" not in plate_cfg["placement"]
    assert food_cfg["placement"]["sample_args"]["reference"] == "obj_container"
