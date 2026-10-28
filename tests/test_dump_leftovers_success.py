"""DumpLeftovers success semantics without constructing a simulator."""

from itertools import product

import pytest

from robocasa.environments.kitchen.composite.washing_dishes import dump_leftovers


def _task_with_state(monkeypatch, *, in_bowl, bowl_in_sink, gripper_far):
    task = object.__new__(dump_leftovers.DumpLeftovers)
    task.sink = object()
    receptacle_contents = {
        ("leftover1", "bowl"): in_bowl[0],
        ("leftover2", "bowl"): in_bowl[1],
    }
    fixture_contents = {("bowl", task.sink): bowl_in_sink}
    hand_clearance = {"bowl": gripper_far}

    def in_receptacle(env, obj_name, receptacle_name):
        assert env is task
        return receptacle_contents[(obj_name, receptacle_name)]

    def inside_fixture(env, obj_name, fixture):
        assert env is task
        return fixture_contents[(obj_name, fixture)]

    def far_from_hand(env, obj_name):
        assert env is task
        return hand_clearance[obj_name]

    monkeypatch.setattr(dump_leftovers.OU, "check_obj_in_receptacle", in_receptacle)
    monkeypatch.setattr(dump_leftovers.OU, "obj_inside_of", inside_fixture)
    monkeypatch.setattr(dump_leftovers.OU, "gripper_obj_far", far_from_hand)
    return task


def _case_id(state):
    first, second, sink, hand = state
    return "_".join(
        (
            "leftover1-in" if first else "leftover1-out",
            "leftover2-in" if second else "leftover2-out",
            "bowl-in-sink" if sink else "bowl-outside-sink",
            "hand-far" if hand else "hand-near",
        )
    )


STATES = list(product((False, True), repeat=4))


@pytest.mark.parametrize("state", STATES, ids=[_case_id(state) for state in STATES])
def test_success_requires_both_leftovers_removed_and_bowl_released_in_sink(
    monkeypatch, state,
):
    first_in_bowl, second_in_bowl, bowl_in_sink, gripper_far = state
    task = _task_with_state(
        monkeypatch,
        in_bowl=(first_in_bowl, second_in_bowl),
        bowl_in_sink=bowl_in_sink,
        gripper_far=gripper_far,
    )

    # Of the 16 physical-condition combinations, only the cleared, deposited,
    # and released bowl is a successful completion.
    expected = state == (False, False, True, True)
    assert bool(task._check_success()) is expected
