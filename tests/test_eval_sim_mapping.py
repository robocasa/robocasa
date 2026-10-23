"""Pure-array contract tests; no MuJoCo, ONNX, model, or GPU required."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import time

import numpy as np

from robocasa_eval.sim_backend import (
    HAND_MOTOR_TO_DATASET, hand_dataset_to_motor, hand_motor_to_dataset,
    model_step_limit, projected_gravity_from_wxyz, state_from_motors,
    SonicRoboCasaEnv, camera_configuration, CAMERAS,
)


class SimMappingTest(unittest.TestCase):
    def test_policy_cameras_are_independent_of_video_recording(self):
        cameras, height, width = camera_configuration({"record_multiview_video": True})
        self.assertEqual(cameras, {"ego_view": CAMERAS["ego_view"]})
        self.assertEqual((height, width), (480, 640))
        self.assertEqual(camera_configuration({"camera_names": list(CAMERAS)})[0], CAMERAS)
        for bad in ([], "ego_view", ["missing"], ["ego_view", "ego_view"]):
            with self.assertRaises(ValueError):
                camera_configuration({"camera_names": bad})
        for bad in (0, -2, True, 480.5, "480"):
            with self.assertRaises(ValueError):
                camera_configuration({"camera_height": bad})

    def test_hand_names_and_inverse(self):
        motor_names = ["thumb0", "thumb1", "thumb2", "index0", "index1", "middle0", "middle1"]
        self.assertEqual(
            np.array(motor_names)[HAND_MOTOR_TO_DATASET].tolist(),
            ["index0", "index1", "middle0", "middle1", "thumb0", "thumb1", "thumb2"],
        )
        motor = np.arange(7, dtype=float)
        np.testing.assert_array_equal(hand_dataset_to_motor(hand_motor_to_dataset(motor)), motor)

    def test_state_exact_live_exporter_ranges(self):
        result = state_from_motors(np.arange(29), np.arange(100, 107),
                                   np.arange(200, 207), [0, 0, -1])
        self.assertEqual(sum(len(v) for v in result.values()), 46)
        np.testing.assert_array_equal(result["left_arm"], np.arange(15, 22))
        np.testing.assert_array_equal(result["right_arm"], np.arange(22, 29))
        np.testing.assert_array_equal(result["left_hand"], [103, 104, 105, 106, 100, 101, 102])
        self.assertTrue(all(v.dtype == np.float32 for v in result.values()))

    def test_gravity_orientation_and_normalization(self):
        np.testing.assert_allclose(projected_gravity_from_wxyz([2, 0, 0, 0]), [0, 0, -1])
        # +90 degree pitch: world down becomes +body-x.
        q = np.sqrt(0.5)
        np.testing.assert_allclose(projected_gravity_from_wxyz([q, 0, q, 0]),
                                   [1, 0, 0], atol=1e-7)
        # Pure yaw cannot change gravity.
        np.testing.assert_allclose(projected_gravity_from_wxyz([q, 0, 0, q]), [0, 0, -1])
        with self.assertRaises(ValueError):
            projected_gravity_from_wxyz([0, 0, 0, 0])

    def test_timeout_is_simulation_time(self):
        self.assertEqual(model_step_limit({"category": "atomic"}), 2000)
        self.assertEqual(model_step_limit({"category": "composite"}), 9000)
        self.assertEqual(model_step_limit({"category": "atomic", "limit_sec": .02}), 1)
        self.assertEqual(model_step_limit({
            "category": "atomic", "timeout_seconds": .1, "max_model_steps": 5
        }), 5)
        with self.assertRaises(ValueError):
            model_step_limit({"category": "atomic", "timeout_seconds": 40,
                              "max_model_steps": 9000})
        with self.assertRaises(ValueError):
            model_step_limit({"category": "atomic", "timeout_seconds": 40, "limit_sec": 180})
        for bad in (0, -1, .001, float("nan")):
            with self.assertRaises(ValueError):
                model_step_limit({"category": "atomic", "limit_sec": bad})

    def test_bad_or_nonfinite_vectors_rejected(self):
        with self.assertRaises(ValueError):
            hand_dataset_to_motor(np.zeros(6))
        with self.assertRaises(ValueError):
            state_from_motors(np.full(29, np.nan), np.zeros(7), np.zeros(7), [0, 0, -1])

    def test_lowstate_uses_fresh_freejoint_values(self):
        backend = SonicRoboCasaEnv.__new__(SonicRoboCasaEnv)
        qpos, qvel = np.zeros(50), np.zeros(49)
        a = np.sqrt(.5)
        qpos[43:50] = [1., 2., .79, a, 0., a, 0.]
        qvel[46:49] = [.1, -.2, .3]
        backend.env = SimpleNamespace(sim=SimpleNamespace(
            data=SimpleNamespace(_data=SimpleNamespace(qpos=qpos, qvel=qvel))
        ))
        backend.maps = SimpleNamespace(
            qpos_adr=np.arange(29), qvel_adr=np.arange(29),
            _lh=(np.arange(29, 36),), _rh=(np.arange(36, 43),),
            free_qadr=43, free_vadr=43,
        )
        state = backend._motor_state()
        np.testing.assert_allclose(state[4], [.1, -.2, .3])
        np.testing.assert_allclose(state[5], [1., 0., 0.], atol=1e-7)

    def test_action_slots_preserved_and_budget_enforced(self):
        # Fake physics tests the adapter boundary without loading a sim or GPU.
        backend = SonicRoboCasaEnv.__new__(SonicRoboCasaEnv)
        qpos = np.zeros(50)
        qpos[45] = .79
        data = SimpleNamespace(time=0.0, qpos=qpos)
        observed_actions = []

        def step(action):
            observed_actions.append(action.copy())
            data.time += .005
            data.qpos[45] -= .01
            return None, 0.0, False, {}

        backend.env = SimpleNamespace(
            sim=SimpleNamespace(data=SimpleNamespace(_data=data, qpos=data.qpos))
        )
        backend.wall = SimpleNamespace(step=step)
        backend.maps = SimpleNamespace(_lh=(np.arange(29, 36),),
                                       _rh=(np.arange(36, 43),), free_qadr=43)
        backend.decoder = SimpleNamespace(step=lambda *args: np.full(29, .1))
        backend._motor_state = lambda: (np.zeros(29), np.zeros(29),
                                        np.zeros(7), np.zeros(7), np.zeros(3),
                                        np.array([0., 0., -1.]))
        backend._observation = lambda: {"fake": True}
        backend._finished = backend._success = False
        backend._model_steps = backend._physics_steps = 0
        backend._limit = 2
        backend._sim_start = 0
        backend._pelvis = 0
        backend._initial_root_height = backend._min_root_height = backend._final_root_height = .79
        backend._wall_start = time.monotonic()
        backend._metadata = {}
        backend._reason = "running"
        hand = np.array([.01, .02, .03, -.04, -.05, -.06, -.07])
        result = backend.advance({
            "motion_token": np.zeros((3, 64)),
            "left_hand_joints": np.tile(hand, (3, 1)),
            "right_hand_joints": np.zeros((3, 7)),
        })
        self.assertEqual(len(observed_actions), 8)
        np.testing.assert_allclose(observed_actions[0][29:36], hand, atol=1e-7)
        self.assertEqual(result["sim_steps"], 2)
        self.assertEqual(result["sim_seconds"], .04)
        self.assertTrue(result["finished"])
        self.assertFalse(result["success"])
        self.assertFalse(result["metadata"]["first_physics_step_success"])
        self.assertEqual(result["reason"], "simulation_time_limit")
        self.assertIsNone(result["observation"])
        self.assertEqual(result["metadata"]["initial_root_height"], .79)
        self.assertAlmostEqual(result["metadata"]["final_root_height"], .71)
        self.assertAlmostEqual(result["metadata"]["min_root_height"], .71)


if __name__ == "__main__":
    unittest.main()
