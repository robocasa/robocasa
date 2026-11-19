import copy
import unittest

import mujoco
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
from termcolor import colored

import robocasa
from robocasa.models.fixtures import Fixture
from robocasa.utils import env_utils as EnvUtils
from robocasa.utils import object_utils as OU

# placements are re-sampled on every reset, so a reconstruction has to be retried
# a few times before it lands on one that differs from the recorded scene
MAX_RESAMPLE = 8
MAX_ATTEMPTS = 6
# a plain rollout is the same on every reset, so a couple of layouts is enough to
# cover the cabinets and counters the fixtures are drawn from
RESETS = 4


def make_env(task):
    return robosuite.make(
        task,
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(
            controller=None, robot="PandaOmron"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        obj_instance_split="pretrain",
        layout_ids=-2,
        style_ids=-2,
    )


def restore(env, ep_meta, model_xml, states, search=None):
    """
    Rebuild a recorded scene, as `reset_to` in
    `robocasa/scripts/dataset_scripts/playback_dataset_hdf5.py` does.

    `reset_from_xml_string` is a soft reset -- it skips `_load_model` -- so the python
    fixtures still belong to the scene the preceding `reset` sampled.  `search` keeps
    re-sampling until it returns True, i.e. until those fixtures have drifted far
    enough from the recording to expose the disagreement.
    """
    env.set_ep_meta(copy.deepcopy(ep_meta))
    env.reset()
    for _ in range(MAX_RESAMPLE):
        if search is None or search(env):
            break
        env.reset()

    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(states)
    env.sim.forward()
    env.update_state()


def is_free_jointed(env, fixture):
    """
    Whether the model can move this fixture on its own.

    A body held by a free joint is one that was dropped into the scene rather than
    placed in it -- a blender lid -- and it settles under gravity before anything
    reads it back, so its simulated pose genuinely stops matching the pose the
    model was built with.
    """
    body_id = env.sim.model.body_name2id(fixture.root_body)
    for jnt_id in range(env.sim.model.njnt):
        if env.sim.model.jnt_bodyid[jnt_id] != body_id:
            continue
        if env.sim.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            return True
    return False


def base_pose(env):
    body_id = env.sim.model.body_name2id("mobilebase0_base")
    pos = np.array(env.sim.data.body_xpos[body_id])
    yaw = T.mat2euler(np.array(env.sim.data.body_xmat[body_id]).reshape((3, 3)))[2]
    return pos, yaw


def set_base_yaw(env, yaw):
    joint = env.sim.model.joint_name2id("mobilebase0_joint_mobile_yaw")
    addr = env.sim.model.jnt_qposadr[joint]
    env.sim.data.qpos[addr] += yaw - base_pose(env)[1]
    env.sim.forward()


def park(env, goal, goal_ori):
    """
    Put the mobile base at `goal` facing `goal_ori[2]`.

    `EnvUtils.set_robot_to_position` solves the two prismatic mobile joints in the
    frame of `init_robot_base_ori_anchor`, and those axes rotate with the base, so it
    is only exact while the base still has that yaw.  The joint values are re-solved
    here against the yaw we actually want.
    """
    set_base_yaw(env, goal_ori[2])
    addrs = [
        env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(name)]
        for name in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
        )
    ]

    for _ in range(2):
        start = base_pose(env)[0][:2]
        jac = np.zeros((2, 2))
        eps = 1e-4
        for i, addr in enumerate(addrs):
            qpos = env.sim.data.qpos[addr]
            env.sim.data.qpos[addr] = qpos + eps
            env.sim.forward()
            jac[:, i] = (base_pose(env)[0][:2] - start) / eps
            env.sim.data.qpos[addr] = qpos
            env.sim.forward()

        delta = np.linalg.solve(jac, goal[:2] - start)
        for i, addr in enumerate(addrs):
            env.sim.data.qpos[addr] += delta[i]
        env.sim.forward()


class TestStateRestore(unittest.TestCase):
    def test_fixture_poses_agree_with_sim_after_restore(self):
        """
        A fixture's python pose has to describe the scene the simulator is running.

        `get_ep_meta` pins the layout and the style but not the placements, so
        re-sampling the same episode moves the fixtures while the python objects stay
        behind.  Rebuilding the recording on top of that has to re-point them.
        """
        env = make_env("CloseBlenderLid")
        try:
            for _ in range(MAX_ATTEMPTS):
                env.unset_ep_meta()
                env.reset()

                # while python and simulator still describe the same scene, the
                # simulator-read pose has to agree with the python one. Without this
                # the rest of the test proves nothing: "sync makes them equal" would
                # hold just as well if the two quantities were different things.
                #
                # every fixture the model places rigidly matches to machine precision.
                # the ones that do not are bodies a free joint can move: a blender lid
                # is dropped into the scene and settles under gravity, so by the time
                # anything reads it back it has moved as far as 22 cm from where the
                # model put it. that is the simulator having a different scene, not the
                # two quantities being different things, so those are skipped.
                for name, fxtr in env.fixtures.items():
                    if not isinstance(fxtr, Fixture):
                        continue
                    try:
                        pos, yaw = OU.get_fixture_pose_from_sim(env, fxtr)
                    except ValueError:
                        continue
                    if is_free_jointed(env, fxtr):
                        continue
                    np.testing.assert_allclose(
                        pos,
                        np.array(fxtr.pos),
                        atol=1e-2,
                        err_msg=f"{name}: the simulator pose is not the fixture pose",
                    )
                    self.assertAlmostEqual(fxtr.rot, yaw, places=3)

                ep_meta = env.get_ep_meta()
                model_xml = env.sim.model.get_xml()
                states = env.sim.get_state().flatten()
                recorded = {
                    name: np.array(f.pos)
                    for name, f in env.fixtures.items()
                    if isinstance(f, Fixture)
                }

                drifted = {}

                def search(e):
                    for name, pos in recorded.items():
                        fxtr = e.fixtures.get(name)
                        if fxtr is None:
                            continue
                        if np.linalg.norm(np.array(fxtr.pos)[:2] - pos[:2]) > 0.05:
                            drifted[name] = fxtr
                    return bool(drifted)

                restore(env, ep_meta, model_xml, states, search=search)
                if not drifted:
                    continue

                # the python fixtures are stale -- this is the condition under test
                name, fxtr = drifted.popitem()
                self.assertGreater(
                    np.linalg.norm(
                        np.array(fxtr.pos)
                        - env.sim.data.body_xpos[env.sim.model.body_name2id(fxtr.root_body)]
                    ),
                    0.05,
                    f"{name} was expected to still describe the discarded placement",
                )

                # ...and reading through the simulator resolves it
                pos, yaw = OU.get_fixture_pose_from_sim(env, fxtr)
                np.testing.assert_allclose(
                    pos, env.sim.data.body_xpos[env.sim.model.body_name2id(fxtr.root_body)]
                )
                OU.sync_fixture_pose_from_sim(env, fxtr)
                np.testing.assert_allclose(np.array(fxtr.pos), pos, atol=1e-7)
                self.assertAlmostEqual(fxtr.rot, yaw, places=6)
                return

            self.skipTest("no reconstruction drifted far enough to test the sync")
        finally:
            env.close()

    def test_sync_is_a_noop_when_the_poses_already_agree(self):
        """
        Syncing a scene that is already consistent must leave it alone.

        `HousingCabinet` and `Counter` override `set_pos` to re-place their
        interior object, so writing a pose back through it moves a fixture that
        was already in the right place -- and drags along anything derived from
        that fixture, including the goal `NavigateKitchen` navigates to.  A plain
        rollout never desyncs python from the simulator, so every call here has
        to leave every pose, and the goal, exactly where it found them.
        """
        env = make_env("NavigateKitchen")
        try:
            for attempt in range(RESETS):
                print(colored(f"NavigateKitchen rollout {attempt}...", "green"))
                env.unset_ep_meta()
                env.reset()

                before = {
                    name: np.array(f.pos)
                    for name, f in env.fixtures.items()
                    if isinstance(f, Fixture)
                }
                goal = np.array(
                    EnvUtils.compute_robot_base_placement_pose(env, env.target_fixture)[0]
                )

                OU.sync_fixture_poses_from_sim(env)

                for name, pos in before.items():
                    np.testing.assert_allclose(
                        np.array(env.fixtures[name].pos),
                        pos,
                        atol=1e-9,
                        err_msg=f"{name} was moved by a sync that had nothing to do",
                    )
                np.testing.assert_allclose(
                    np.array(
                        EnvUtils.compute_robot_base_placement_pose(env, env.target_fixture)[0]
                    ),
                    goal,
                    atol=1e-9,
                    err_msg="the navigation goal moved by a sync that had nothing to do",
                )
        finally:
            env.close()

    def test_close_blender_lid_survives_state_restore(self):
        """
        CloseBlenderLid: a state that the check accepts has to keep passing after it
        has been recorded and replayed, whatever the surrounding scene was resampled to.
        """
        env = make_env("CloseBlenderLid")
        try:
            for attempt in range(MAX_ATTEMPTS):
                print(colored(f"CloseBlenderLid reconstruction {attempt}...", "green"))
                env.unset_ep_meta()
                env.reset()
                blender = next(
                    f for f in env.fixtures.values() if type(f).__name__ == "Blender"
                )
                lid = blender.blender_lid

                # successful final state: the lid resting on the blender
                closed = blender.pos + np.dot(
                    T.euler2mat([0, 0, blender.rot]), blender.anchor_offset
                )
                addr = env.sim.model.jnt_qposadr[
                    env.sim.model.joint_name2id(lid.joints[0])
                ]
                env.sim.data.qpos[addr : addr + 7] = list(closed) + [1, 0, 0, 0]
                env.sim.forward()
                env.update_state()
                if not env._check_success():
                    continue

                ep_meta = env.get_ep_meta()
                model_xml = env.sim.model.get_xml()
                states = env.sim.get_state().flatten()
                recorded_pos = np.array(blender.pos)
                recorded_root = blender.root_body
                shifted = {}

                def search(e):
                    fxtr = next(
                        f
                        for f in e.fixtures.values()
                        if type(f).__name__ == "Blender"
                    )
                    distance = np.linalg.norm(np.array(fxtr.pos) - recorded_pos)
                    if fxtr.root_body == recorded_root and distance > 0.25:
                        shifted["distance"] = distance
                        return True
                    return False

                restore(env, ep_meta, model_xml, states, search=search)
                if not shifted:
                    continue

                print(
                    f"  blender re-sampled {shifted['distance'] * 100:.1f} cm away; "
                    f"replaying the recorded success -> {env._check_success()}"
                )
                self.assertTrue(
                    env._check_success(),
                    "a recorded successful state failed its own success check after "
                    "being replayed into the same recorded scene",
                )
                return

            self.skipTest("no reconstruction drifted far enough to test the check")
        finally:
            env.close()

    def test_navigate_kitchen_survives_state_restore(self):
        """
        NavigateKitchen: the navigation target is derived from the target fixture, so
        it has to follow the fixture the restored scene actually shows.
        """
        env = make_env("NavigateKitchen")
        try:
            for attempt in range(MAX_ATTEMPTS):
                print(colored(f"NavigateKitchen reconstruction {attempt}...", "green"))

                # Whether an episode can show the bug at all is up to its layout. The
                # goal is derived from a placement, and some placements never move --
                # an island sink is fixed to the room -- so the goal comes out
                # identical on every resample and there is nothing to desync. Probe
                # that first. A layout that cannot show the bug is not a failed
                # attempt, so draw another one rather than spending one of the few
                # attempts below on it -- otherwise the test skips on a run of bad
                # layout draws instead of testing anything.
                for _ in range(MAX_RESAMPLE):
                    env.unset_ep_meta()
                    env.reset()

                    ep_meta = env.get_ep_meta()
                    fixture_name = env.target_fixture.name
                    sampled_goal = np.array(env.target_pos)

                    env.set_ep_meta(copy.deepcopy(ep_meta))
                    env.reset()
                    if env.target_fixture.name != fixture_name:
                        continue
                    if (
                        np.linalg.norm(np.array(env.target_pos)[:2] - sampled_goal[:2])
                        >= 0.25
                    ):
                        break
                else:
                    continue

                target = env.target_fixture
                goal = np.array(env.target_pos)
                goal_ori = np.array(env.target_ori)

                park(env, goal, goal_ori)
                env.update_state()
                if not env._check_success():
                    continue

                ep_meta = env.get_ep_meta()
                model_xml = env.sim.model.get_xml()
                states = env.sim.get_state().flatten()
                shifted = {}

                def search(e):
                    distance = np.linalg.norm(np.array(e.target_pos)[:2] - goal[:2])
                    if e.target_fixture.name == target.name and distance > 0.25:
                        shifted["distance"] = distance
                        return True
                    return False

                restore(env, ep_meta, model_xml, states, search=search)
                if not shifted:
                    continue

                print(
                    f"  goal re-sampled {shifted['distance'] * 100:.1f} cm away; "
                    f"replaying the recorded success -> {env._check_success()}"
                )
                self.assertTrue(
                    env._check_success(),
                    "a recorded successful state failed its own success check after "
                    "being replayed into the same recorded scene",
                )
                return

            self.skipTest("no reconstruction drifted far enough to test the check")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
