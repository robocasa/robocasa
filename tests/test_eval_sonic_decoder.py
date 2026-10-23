"""Adapter tests; no simulator, accelerator, model download, or file output."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

from robocasa_eval.sonic_decoder import (
    BODY_ACTION_SCALE, BODY_KD, BODY_KP, DEFAULT_BODY_Q,
    HAND_KD, HAND_KP, MOTOR_JOINT_NAMES, MOTOR_TO_POLICY, POLICY_TO_MOTOR,
    SonicDecoder, clamp_hand_target,
)


class FakeSession:
    def __init__(self):
        self.observations = []
        self.action = np.zeros((1, 29), np.float32)

    def get_inputs(self):
        return [SimpleNamespace(name="obs_dict", shape=[1, 994], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="action", shape=[1, 29], type="tensor(float)")]

    def run(self, outputs, feed):
        assert outputs == ["action"]
        self.observations.append(feed["obs_dict"].copy())
        return [self.action.copy()]


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.decoder = SonicDecoder("unused.onnx", _session=self.session)

    def step(self, **kwargs):
        args = {
            "token64": np.zeros(64),
            "body_q29_motor_order": DEFAULT_BODY_Q.copy(),
            "body_dq29_motor_order": np.zeros(29),
            "angular_velocity_body3": np.zeros(3),
            "projected_gravity3": np.array([0, 0, -1]),
        }
        args.update(kwargs)
        return self.decoder.step(**args)

    def test_reference_first_tick_padding_and_obs_layout(self):
        qdelta = np.arange(29) / 100
        token = np.arange(64) / 64
        angular = np.array([0.1, 0.2, 0.3])
        self.step(token64=token, body_q29_motor_order=DEFAULT_BODY_Q + qdelta,
                  body_dq29_motor_order=np.arange(29) / 10,
                  angular_velocity_body3=angular)
        obs = self.session.observations[-1]
        self.assertEqual(obs.shape, (1, 994))
        self.assertEqual(obs.dtype, np.float32)
        row = obs[0]
        np.testing.assert_allclose(row[:64], token)
        np.testing.assert_allclose(row[64:91], 0)
        np.testing.assert_allclose(row[91:94], angular)
        np.testing.assert_allclose(row[94:355], 0)
        np.testing.assert_allclose(row[355:384], qdelta[MOTOR_TO_POLICY])
        np.testing.assert_allclose(row[384:645], 0)
        np.testing.assert_allclose(row[645:674], (np.arange(29) / 10)[MOTOR_TO_POLICY])
        np.testing.assert_allclose(row[674:964], 0)
        np.testing.assert_array_equal(row[964:991].reshape(9, 3),
                                      np.tile([0, 0, 1], (9, 1)))
        np.testing.assert_array_equal(row[991:994], [0, 0, -1])

    def test_history_is_oldest_first_and_previous_action_is_unscaled(self):
        for tick in range(12):
            self.session.action[:] = 100 + tick + np.arange(29)
            self.step(body_q29_motor_order=DEFAULT_BODY_Q + tick,
                      angular_velocity_body3=np.full(3, tick))
        row = self.session.observations[-1][0]
        np.testing.assert_array_equal(
            row[64:94].reshape(10, 3), np.repeat(np.arange(2, 12)[:, None], 3, axis=1)
        )
        np.testing.assert_allclose(
            row[94:384].reshape(10, 29),
            np.repeat(np.arange(2, 12)[:, None], 29, axis=1),
        )
        np.testing.assert_array_equal(
            row[674:964].reshape(10, 29),
            100 + np.arange(1, 11)[:, None] + np.arange(29),
        )

    def test_raw_action_is_reordered_scaled_and_offset_once(self):
        raw = np.linspace(-2, 2, 29, dtype=np.float32)
        self.session.action[0] = raw
        target = self.step()
        np.testing.assert_allclose(target,
                                   DEFAULT_BODY_Q + raw[POLICY_TO_MOTOR] * BODY_ACTION_SCALE)
        np.testing.assert_array_equal(MOTOR_TO_POLICY[POLICY_TO_MOTOR], np.arange(29))
        # Action 1 on policy index 3 is left hip roll (motor index 1).
        self.session.action[:] = 0
        self.session.action[0, 3] = 1
        target = self.step()
        delta = target - DEFAULT_BODY_Q
        self.assertAlmostEqual(delta[1], BODY_ACTION_SCALE[1])
        np.testing.assert_array_equal(np.delete(delta, 1), np.zeros(28))

    def test_reset_restores_padding_and_zero_previous_actions(self):
        self.session.action[:] = 2
        self.step()
        self.step()
        self.decoder.reset()
        self.step()
        row = self.session.observations[-1][0]
        np.testing.assert_array_equal(row[674:964], 0)
        np.testing.assert_array_equal(row[964:991].reshape(9, 3),
                                      np.tile([0, 0, 1], (9, 1)))

    def test_bad_shapes_and_nonfinite_values_fail_before_inference(self):
        for key, bad in (("token64", np.zeros((1, 64))),
                         ("body_q29_motor_order", np.zeros(28)),
                         ("body_dq29_motor_order", np.full(29, np.nan)),
                         ("angular_velocity_body3", [0, np.inf, 0]),
                         ("projected_gravity3", [0, 0]),
                         ("token64", np.full(64, 1e300))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.step(**{key: bad})
        self.assertEqual(len(self.session.observations), 0)

    def test_bad_output_does_not_advance_history(self):
        self.session.action[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.step(angular_velocity_body3=[1, 2, 3])
        self.session.action[:] = 0
        self.step()
        np.testing.assert_array_equal(self.session.observations[-1][0, 64:94], 0)

    def test_rejects_incompatible_network_signature(self):
        self.session.get_inputs = lambda: [
            SimpleNamespace(name="obs_dict", shape=[1, 436], type="tensor(float)")
        ]
        with self.assertRaises(ValueError):
            SonicDecoder("unused.onnx", _session=self.session)

    def test_reference_gains_and_motor_names(self):
        self.assertEqual(len(MOTOR_JOINT_NAMES), 29)
        self.assertEqual(len(set(MOTOR_JOINT_NAMES)), 29)
        self.assertEqual(MOTOR_JOINT_NAMES[15], "left_shoulder_pitch_joint")
        self.assertEqual(MOTOR_JOINT_NAMES[28], "right_wrist_yaw_joint")
        self.assertAlmostEqual(BODY_KP[0], 99.09842777473803)
        self.assertAlmostEqual(BODY_KD[0], 6.308801853824895)
        self.assertAlmostEqual(BODY_KP[4] / BODY_KP[15], 2)
        self.assertAlmostEqual(BODY_KD[4] / BODY_KD[15], 2)
        self.assertAlmostEqual(BODY_ACTION_SCALE[4], BODY_ACTION_SCALE[15])
        self.assertEqual((HAND_KP, HAND_KD), (1.5, 0.1))

    def test_hand_directional_limits_then_measured_delta_clamp(self):
        result = clamp_hand_target(np.full(7, -10), np.zeros(7), is_left=True)
        # Reference clipToMaxOpen leaves a zero-limit direction alone;
        # the measured-delta clamp still limits every commanded displacement.
        np.testing.assert_allclose(result, -0.25)
        current = np.array([0.8, 0.8, 1.5, 0, 0, 0, 0])
        result = clamp_hand_target(np.full(7, 10), current, is_left=True,
                                   max_close_ratio=0.5)
        np.testing.assert_allclose(result[:3], [0.55, 0.55, 1.25])
        np.testing.assert_allclose(result[3:], 0.25)


if __name__ == "__main__":
    unittest.main()
