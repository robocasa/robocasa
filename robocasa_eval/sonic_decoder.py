"""CPU SONIC token decoder matching the deployed G1 control loop.

Reference files under GR00T-WholeBodyControl/gear_sonic_deploy:
  policy/release/observation_config_sonic_release.yaml
  src/g1/g1_deploy_onnx_ref/include/{policy_parameters,dex3_hands,math_utils}.hpp
  src/g1/g1_deploy_onnx_ref/src/{g1_deploy_onnx_ref,state_logger}.cpp

Call ``step`` at 50 Hz of simulated time, with motor-order joint states and
body-frame angular velocity / projected gravity. Hold its position target for
four 200 Hz physics steps and apply BODY_KP / BODY_KD in the simulator.
The network consumes raw motion tokens; no additional normalization or action
clipping is applied by the reference deployment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _constant(values, dtype=np.float64):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


MOTOR_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Index a motor-order array with MOTOR_TO_POLICY to obtain IsaacLab order.
MOTOR_TO_POLICY = _constant(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    np.int64,
)
# Index a policy-order array with POLICY_TO_MOTOR to obtain motor order.
POLICY_TO_MOTOR = _constant(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    np.int64,
)
DEFAULT_BODY_Q = _constant(
    [-0.312, 0, 0, 0.669, -0.363, 0,
     -0.312, 0, 0, 0.669, -0.363, 0,
     0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0,
     0.2, -0.2, 0, 0.6, 0, 0, 0]
)

_MOTOR_TYPES = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
)
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520,
             "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT_FOR_SCALE = {"5020": 25.0, "7520_14": 88.0,
                     "7520_22": 139.0, "4010": 5.0}
_OMEGA = 10 * 2.0 * 3.1415926535
_STIFFNESS = np.array([_ARMATURE[m] * _OMEGA**2 for m in _MOTOR_TYPES])
_DAMPING = np.array([4.0 * _ARMATURE[m] * _OMEGA for m in _MOTOR_TYPES])
_GAIN_MULTIPLIER = np.ones(29)
_GAIN_MULTIPLIER[[4, 5, 10, 11, 13, 14]] = 2.0
BODY_KP = _constant(_STIFFNESS * _GAIN_MULTIPLIER)
BODY_KD = _constant(_DAMPING * _GAIN_MULTIPLIER)
# Scale uses the base stiffness, including at the double-gain joints.
BODY_ACTION_SCALE = _constant(
    0.25 * np.array([_EFFORT_FOR_SCALE[m] for m in _MOTOR_TYPES]) / _STIFFNESS
)

HAND_KP = 1.5
HAND_KD = 0.1
HAND_MAX_DELTA_Q = 0.25
HAND_MAX_LEFT = _constant([1.05, 1.05, 1.75, 0, 0, 0, 0])
HAND_MIN_LEFT = _constant([-1.05, -0.724, 0, -1.57, -1.75, -1.57, -1.75])
HAND_MAX_RIGHT = _constant([1.05, 0.742, 0, 1.57, 1.75, 1.57, 1.75])
HAND_MIN_RIGHT = _constant([-1.05, -1.05, -1.75, 0, 0, 0, 0])

CONTROL_HZ = 50
HISTORY_FRAMES = 10
INPUT_DIM = 994


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def clamp_hand_target(target, current_q, *, is_left, max_close_ratio=1.0):
    """Replicate Dex3Hands::writeOnce, using thumb-first SDK joint order.

    This follows the reference's *directional* close-limit logic, followed by
    a +/-0.25 rad command offset from the measured joint position. It is not a
    velocity limit or a trajectory interpolator. Call at the command cadence.
    """
    desired = _vector(target, 7, "hand target").copy()
    measured = _vector(current_q, 7, "hand current_q")
    if not np.isfinite(max_close_ratio):
        raise ValueError("max_close_ratio must be finite")
    ratio = np.clip(float(max_close_ratio), 0.2, 1.0)
    maximum = HAND_MAX_LEFT if is_left else HAND_MAX_RIGHT
    minimum = HAND_MIN_LEFT if is_left else HAND_MIN_RIGHT
    positive = (desired > 0) & (maximum > 0)
    negative = (desired < 0) & (minimum < 0)
    desired[positive] = np.minimum(desired[positive], ratio * maximum[positive])
    desired[negative] = np.maximum(desired[negative], ratio * minimum[negative])
    return measured + np.clip(desired - measured, -HAND_MAX_DELTA_Q, HAND_MAX_DELTA_Q)


class SonicDecoder:
    """One independent 50 Hz decoder and history buffer per simulated robot.

    The optional private ``_session`` argument permits adapter tests without
    loading a model. Normal construction always selects CPUExecutionProvider.
    """

    def __init__(self, model_path, threads=1, *, _session=None):
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.model_path = str(Path(model_path).expanduser())
        if _session is None:
            if not Path(self.model_path).is_file():
                raise FileNotFoundError(f"SONIC decoder is missing: {self.model_path}")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Many independent simulator processes share host CPU resources.
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self.session = ort.InferenceSession(
                self.model_path, sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        else:
            self.session = _session
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "obs_dict"
                or list(inputs[0].shape) != [1, INPUT_DIM]
                or inputs[0].type != "tensor(float)"):
            raise ValueError("Expected SONIC input obs_dict: float32[1,994]")
        if (len(outputs) != 1 or outputs[0].name != "action"
                or list(outputs[0].shape) != [1, 29]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("Expected SONIC output action: float32[1,29]")
        self.reset()

    def reset(self):
        """Clear history exactly as the C++ StateLogger missing-frame padding."""
        self._angular = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
        self._positions = np.zeros((HISTORY_FRAMES, 29), dtype=np.float64)
        self._velocities = np.zeros((HISTORY_FRAMES, 29), dtype=np.float64)
        self._actions = np.zeros((HISTORY_FRAMES, 29), dtype=np.float64)
        self._gravity = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
        # Reference makeZeroEntry_ uses quaternion (0,0,0,0). Its quaternion
        # rotation formula maps world gravity -Z to +Z for these padded frames.
        # Preserve that behavior instead of silently substituting identity.
        self._gravity[:, 2] = 1.0
        self._last_action = np.zeros(29, dtype=np.float64)

    def step(self, token64, body_q29_motor_order, body_dq29_motor_order,
             angular_velocity_body3, projected_gravity3):
        token = _vector(token64, 64, "token64")
        q = _vector(body_q29_motor_order, 29, "body_q29_motor_order")
        dq = _vector(body_dq29_motor_order, 29, "body_dq29_motor_order")
        angular = _vector(angular_velocity_body3, 3, "angular_velocity_body3")
        gravity = _vector(projected_gravity3, 3, "projected_gravity3")

        def append(history, current):
            return np.concatenate((history[1:], current[None, :]), axis=0)

        # The state at time t is logged before inference; last_action is a(t-1).
        histories = (
            append(self._angular, angular),
            append(self._positions, (q - DEFAULT_BODY_Q)[MOTOR_TO_POLICY]),
            append(self._velocities, dq[MOTOR_TO_POLICY]),
            append(self._actions, self._last_action),
            append(self._gravity, gravity),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            observation = np.concatenate(
                [token] + [item.reshape(-1) for item in histories]
            ).astype(np.float32)[None, :]
        if not np.isfinite(observation).all():
            raise ValueError("SONIC observation exceeds the float32 range")
        results = self.session.run(["action"], {"obs_dict": observation})
        if len(results) != 1 or np.asarray(results[0]).shape != (1, 29):
            raise ValueError("SONIC returned an action with an invalid shape")
        action = _vector(np.asarray(results[0])[0], 29, "SONIC action").copy()
        target = DEFAULT_BODY_Q + action[POLICY_TO_MOTOR] * BODY_ACTION_SCALE
        if not np.isfinite(target).all():
            raise ValueError("SONIC target contains NaN or infinity")
        (self._angular, self._positions, self._velocities,
         self._actions, self._gravity) = histories
        self._last_action = action
        return target
