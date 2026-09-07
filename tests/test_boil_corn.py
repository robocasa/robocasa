from types import SimpleNamespace

import pytest

from robocasa.environments.kitchen.composite.boiling import boil_corn as task_module
from robocasa.environments.kitchen.composite.boiling.boil_corn import BoilCorn


def _bare_task():
    task = object.__new__(BoilCorn)
    task.knob = "front_left"
    task.stove = SimpleNamespace()
    return task


def _install_success_state(monkeypatch, task, state, calls):
    def check_saucepan_location(env, obj_name, threshold):
        calls["saucepan"] = (obj_name, threshold)
        return state.burner_location

    def check_receptacle(env, obj_name, receptacle_name, th=None):
        calls[obj_name] = (receptacle_name, th)
        return state.in_receptacle[obj_name]

    task.stove.check_obj_location_on_stove = check_saucepan_location
    monkeypatch.setattr(task_module.OU, "check_obj_in_receptacle", check_receptacle)
    monkeypatch.setattr(
        task_module.OU,
        "gripper_obj_far",
        lambda env, obj_name: state.gripper_far,
    )


def test_boil_corn_uses_relaxed_local_thresholds(monkeypatch):
    task = _bare_task()
    calls = {}
    state = SimpleNamespace(
        burner_location=task.knob,
        in_receptacle={
            "corn1": True,
            "corn2": True,
            "saucepan_auxiliary": True,
        },
        gripper_far=True,
    )
    _install_success_state(monkeypatch, task, state, calls)

    assert task._check_success()
    assert calls["saucepan"] == (
        "saucepan",
        BoilCorn.SAUCEPAN_BURNER_THRESHOLD,
    )
    assert BoilCorn.SAUCEPAN_BURNER_THRESHOLD == 0.15
    assert calls["saucepan_auxiliary"] == (
        "saucepan",
        BoilCorn.LID_ON_SAUCEPAN_THRESHOLD,
    )
    assert BoilCorn.LID_ON_SAUCEPAN_THRESHOLD == 0.05
    assert calls["corn1"] == ("saucepan", None)
    assert calls["corn2"] == ("saucepan", None)


@pytest.mark.parametrize(
    ("failed_condition", "value"),
    [
        ("burner_location", None),
        ("burner_location", "front_right"),
        ("corn1", False),
        ("corn2", False),
        ("lid", False),
        ("gripper_far", False),
    ],
)
def test_boil_corn_still_requires_every_semantic_condition(
    monkeypatch, failed_condition, value
):
    task = _bare_task()
    calls = {}
    state = SimpleNamespace(
        burner_location=task.knob,
        in_receptacle={
            "corn1": True,
            "corn2": True,
            "saucepan_auxiliary": True,
        },
        gripper_far=True,
    )
    if failed_condition == "burner_location":
        state.burner_location = value
    elif failed_condition == "lid":
        state.in_receptacle["saucepan_auxiliary"] = value
    elif failed_condition in ("corn1", "corn2"):
        state.in_receptacle[failed_condition] = value
    else:
        state.gripper_far = value
    _install_success_state(monkeypatch, task, state, calls)

    assert not task._check_success()
