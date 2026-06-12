"""Shared Unitree G1 43DoF DDS utilities used by the SONIC RoboCasa bridge."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SIM_TORSO_IDS = [12, 13, 14]
SIM_TORSO_JOINTS = [
    "robot0_waist_yaw_joint",
    "robot0_waist_roll_joint",
    "robot0_waist_pitch_joint",
]
SIM_IMU_BODY_CANDIDATES = [
    "robot0_pelvis",
    "robot0_base",
    "robot0_torso_link",
]
SIM_LEFT_ARM_IDS = [15, 16, 17, 18, 19, 20, 21]
SIM_RIGHT_ARM_IDS = [22, 23, 24, 25, 26, 27, 28]
SIM_LEFT_ARM_JOINTS = [
    "robot0_left_shoulder_pitch_joint",
    "robot0_left_shoulder_roll_joint",
    "robot0_left_shoulder_yaw_joint",
    "robot0_left_elbow_joint",
    "robot0_left_wrist_roll_joint",
    "robot0_left_wrist_pitch_joint",
    "robot0_left_wrist_yaw_joint",
]
SIM_RIGHT_ARM_JOINTS = [
    "robot0_right_shoulder_pitch_joint",
    "robot0_right_shoulder_roll_joint",
    "robot0_right_shoulder_yaw_joint",
    "robot0_right_elbow_joint",
    "robot0_right_wrist_roll_joint",
    "robot0_right_wrist_pitch_joint",
    "robot0_right_wrist_yaw_joint",
]
SIM_LEFT_HAND_JOINTS = [
    "robot0_left_hand_thumb_0_joint",
    "robot0_left_hand_thumb_1_joint",
    "robot0_left_hand_thumb_2_joint",
    "robot0_left_hand_middle_0_joint",
    "robot0_left_hand_middle_1_joint",
    "robot0_left_hand_index_0_joint",
    "robot0_left_hand_index_1_joint",
]
SIM_RIGHT_HAND_JOINTS = [
    "robot0_right_hand_thumb_0_joint",
    "robot0_right_hand_thumb_1_joint",
    "robot0_right_hand_thumb_2_joint",
    "robot0_right_hand_middle_0_joint",
    "robot0_right_hand_middle_1_joint",
    "robot0_right_hand_index_0_joint",
    "robot0_right_hand_index_1_joint",
]
SIM_LEG_JOINTS = [
    "robot0_left_hip_pitch_joint",
    "robot0_left_hip_roll_joint",
    "robot0_left_hip_yaw_joint",
    "robot0_left_knee_joint",
    "robot0_left_ankle_pitch_joint",
    "robot0_left_ankle_roll_joint",
    "robot0_right_hip_pitch_joint",
    "robot0_right_hip_roll_joint",
    "robot0_right_hip_yaw_joint",
    "robot0_right_knee_joint",
    "robot0_right_ankle_pitch_joint",
    "robot0_right_ankle_roll_joint",
]


@dataclass
class DdsCommandState:
    lowcmd: Any | None = None
    left_hand_cmd: Any | None = None
    right_hand_cmd: Any | None = None
    lowcmd_time: float = 0.0
    lowcmd_topic: str | None = None
    left_hand_time: float = 0.0
    right_hand_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class UnitreeDdsReceiver:
    """Read the Unitree DDS topics consumed by the SONIC bridge."""

    def __init__(self, args):
        try:
            from unitree_sdk2py.core.channel import (  # noqa: PLC0415
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: PLC0415
                HandCmd_,
                LowCmd_,
            )
        except ImportError as exc:
            raise ImportError(
                "unitree_sdk2py is required for Unitree DDS communication."
            ) from exc

        ChannelFactoryInitialize(args.domain_id, networkInterface=args.network_interface)

        self.state = DdsCommandState()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self.lowcmd_sub = ChannelSubscriber(args.cmd_topic, LowCmd_)
        self.lowcmd_sub.Init()
        self._threads.append(
            threading.Thread(
                target=self._read_loop,
                args=(self.lowcmd_sub, "lowcmd", "lowcmd_time", "lowcmd_topic", args.cmd_topic),
                daemon=True,
            )
        )

        self.extra_lowcmd_sub = None
        if args.extra_cmd_topic and args.extra_cmd_topic != args.cmd_topic:
            self.extra_lowcmd_sub = ChannelSubscriber(args.extra_cmd_topic, LowCmd_)
            self.extra_lowcmd_sub.Init()
            self._threads.append(
                threading.Thread(
                    target=self._read_loop,
                    args=(
                        self.extra_lowcmd_sub,
                        "lowcmd",
                        "lowcmd_time",
                        "lowcmd_topic",
                        args.extra_cmd_topic,
                    ),
                    daemon=True,
                )
            )

        self.left_hand_sub = None
        self.right_hand_sub = None
        if args.subscribe_hands:
            self.left_hand_sub = ChannelSubscriber(args.left_hand_topic, HandCmd_)
            self.right_hand_sub = ChannelSubscriber(args.right_hand_topic, HandCmd_)
            self.left_hand_sub.Init()
            self.right_hand_sub.Init()
            self._threads.extend(
                [
                    threading.Thread(
                        target=self._read_loop,
                        args=(self.left_hand_sub, "left_hand_cmd", "left_hand_time"),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._read_loop,
                        args=(self.right_hand_sub, "right_hand_cmd", "right_hand_time"),
                        daemon=True,
                    ),
                ]
            )

        for thread in self._threads:
            thread.start()

    def _read_loop(
        self,
        subscriber: Any,
        attr_name: str,
        time_name: str,
        source_attr_name: str | None = None,
        source_value: str | None = None,
    ) -> None:
        while not self._stop.is_set():
            msg = subscriber.Read()
            if msg is not None:
                with self.state.lock:
                    setattr(self.state, attr_name, msg)
                    setattr(self.state, time_name, time.monotonic())
                    if source_attr_name is not None:
                        setattr(self.state, source_attr_name, source_value)
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop.set()


class UnitreeStatePublisher:
    """Static helpers for filling Unitree state messages from MuJoCo state."""

    @staticmethod
    def _set_imu_state(msg, env) -> None:
        body_name = None
        body_names = getattr(env.sim.model, "body_names", ())
        for candidate in SIM_IMU_BODY_CANDIDATES:
            if candidate in body_names:
                body_name = candidate
                break
        if body_name is None:
            msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
            msg.imu_state.gyroscope = [0.0, 0.0, 0.0]
            msg.imu_state.rpy = [0.0, 0.0, 0.0]
            msg.imu_state.accelerometer = [0.0, 0.0, 0.0]
            return

        quat = np.array(env.sim.data.get_body_xquat(body_name), dtype=float)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            quat = quat / norm

        rot = np.array(env.sim.data.get_body_xmat(body_name), dtype=float).reshape(3, 3)
        try:
            ang_vel_world = np.array(env.sim.data.get_body_xvelr(body_name), dtype=float)
        except (AttributeError, ValueError):
            ang_vel_world = np.zeros(3, dtype=float)
        ang_vel_body = rot.T @ ang_vel_world

        msg.imu_state.quaternion = quat.tolist()
        msg.imu_state.gyroscope = ang_vel_body.tolist()
        msg.imu_state.rpy = _quat_wxyz_to_rpy(quat).tolist()
        msg.imu_state.accelerometer = [0.0, 0.0, 0.0]

    @staticmethod
    def _set_motor_state(msg, motor_id: int, env, joint_name: str) -> None:
        if joint_name not in env.sim.model.joint_names:
            return
        msg.motor_state[motor_id].q = float(env.sim.data.get_joint_qpos(joint_name))
        msg.motor_state[motor_id].dq = float(env.sim.data.get_joint_qvel(joint_name))

    @staticmethod
    def _read_hand_state(env, joint_name: str, fallback_q: float = 0.0) -> tuple[float, float]:
        if joint_name not in env.sim.model.joint_names:
            return fallback_q, 0.0

        q = float(env.sim.data.get_joint_qpos(joint_name))
        if fallback_q and abs(q) < fallback_q:
            q = fallback_q
        dq = float(env.sim.data.get_joint_qvel(joint_name))
        return q, dq

    @classmethod
    def _set_hand_states(
        cls,
        msg,
        env,
        joint_names: list[str],
        side: str,
        hand_mapping_profile: str,
        fallback_q: float = 0.0,
    ) -> None:
        del side, hand_mapping_profile
        for motor_id, joint_name in enumerate(joint_names):
            q, dq = cls._read_hand_state(env, joint_name, fallback_q=fallback_q)
            msg.motor_state[motor_id].q = float(q)
            msg.motor_state[motor_id].dq = float(dq)


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(pitch_arg, -1.0, 1.0)))
    yaw = _yaw_from_quat_wxyz(quat)
    return np.array([roll, pitch, yaw], dtype=float)
