import numpy as np

from robocasa.environments.kitchen.kitchen import (
    Kitchen,
    _refresh_controller_goals,
)


class _RecordingController:
    def __init__(self, events, name):
        self._events = events
        self._name = name

    def update(self, force=False):
        self._events.append((self._name, "update", force))

    def reset_goal(self):
        self._events.append((self._name, "reset_goal"))


class _RecordingCompositeController:
    def __init__(self, events):
        self._events = events
        self.part_controllers = {
            "arm": _RecordingController(events, "arm"),
            "gripper": _RecordingController(events, "gripper"),
        }

    def update_state(self):
        self._events.append(("composite", "update_state"))

    def reset(self):
        self._events.append(("composite", "reset"))
        for controller in self.part_controllers.values():
            controller.reset_goal()


class _RecordingRobot:
    def __init__(self, composite_controller):
        self.composite_controller = composite_controller


class _RecordingSim:
    def __init__(self):
        self.step1_calls = 0
        self.step2_calls = 0

    def step1(self):
        self.step1_calls += 1

    def step2(self):
        self.step2_calls += 1


def test_refresh_controller_goals_updates_state_before_reset():
    events = []
    composite_controller = _RecordingCompositeController(events)

    _refresh_controller_goals([_RecordingRobot(composite_controller)])

    assert events == [
        ("composite", "update_state"),
        ("arm", "update", True),
        ("gripper", "update", True),
        ("composite", "reset"),
        ("arm", "reset_goal"),
        ("gripper", "reset_goal"),
    ]


def test_settle_simulation_does_not_submit_policy_actions():
    env = Kitchen.__new__(Kitchen)
    env.robots = [_RecordingRobot(None)]
    env.robots[0].action_limits = (np.zeros(4), np.ones(4))
    env.control_timestep = 0.1
    env.model_timestep = 0.02
    env.sim = _RecordingSim()
    policy_steps = []
    actions = []

    env._pre_action = lambda action, policy_step: (
        actions.append(np.array(action)),
        policy_steps.append(policy_step),
    )

    env._settle_simulation()

    assert env.sim.step1_calls == 50
    assert env.sim.step2_calls == 50
    assert len(actions) == 50
    assert all(not policy_step for policy_step in policy_steps)
    assert all(np.array_equal(action, np.zeros(4)) for action in actions)
