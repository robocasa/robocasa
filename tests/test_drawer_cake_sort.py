from types import SimpleNamespace

from robocasa.environments.kitchen.composite.tidying_cabinets_and_drawers import (
    drawer_cake_sort as task_module,
)
from robocasa.environments.kitchen.composite.tidying_cabinets_and_drawers.drawer_cake_sort import (
    DrawerCakeSort,
)


def _bare_task():
    task = object.__new__(DrawerCakeSort)
    task.counter = object()
    task.drawer = SimpleNamespace(is_closed=lambda env: True)
    return task


def test_drawer_cake_sort_uses_two_cakes_without_lateral_offsets():
    task = _bare_task()

    configs = task._get_obj_cfgs()

    assert [config["name"] for config in configs] == ["cake1", "cake2"]
    assert [config["obj_groups"] for config in configs] == ["cake", "cake"]
    assert all("offset" not in config["placement"] for config in configs)
    assert all(
        config["placement"]["pos"] == ("ref", -1.0) for config in configs
    )
    assert [config["placement"]["size"][1] for config in configs] == [0.15, 0.15]


def test_drawer_cake_sort_instruction_describes_goal_without_placement_details(
    monkeypatch,
):
    task = _bare_task()
    monkeypatch.setattr(
        task_module.ManipulateDrawer,
        "get_ep_meta",
        lambda self: {"lang": "Open the left drawer."},
    )

    assert task.get_ep_meta()["lang"] == (
        "Open the drawer, place both cakes inside it, then close the drawer."
    )


def test_drawer_cake_sort_success_requires_cakes_stored_released_and_drawer_closed(
    monkeypatch,
):
    task = _bare_task()
    state = SimpleNamespace(
        inside={"cake1": True, "cake2": True},
        counter_contact={"cake1": False, "cake2": False},
        gripper_far={"cake1": True, "cake2": True},
        drawer_closed=True,
    )
    task.drawer.is_closed = lambda env: state.drawer_closed
    monkeypatch.setattr(
        task_module.OU,
        "obj_inside_of",
        lambda env, name, drawer: state.inside[name],
    )
    monkeypatch.setattr(
        task_module.OU,
        "check_obj_any_counter_contact",
        lambda env, name: state.counter_contact[name],
    )
    monkeypatch.setattr(
        task_module.OU,
        "gripper_obj_far",
        lambda env, obj_name: state.gripper_far[obj_name],
    )

    assert task._check_success()

    state.counter_contact["cake2"] = True
    assert not task._check_success()

    state.counter_contact["cake2"] = False
    state.gripper_far["cake1"] = False
    assert not task._check_success()

    state.gripper_far["cake1"] = True
    state.drawer_closed = False
    assert not task._check_success()
