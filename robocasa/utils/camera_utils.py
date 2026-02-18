"""
Collection of constants for cameras / robots / etc
in kitchen environments
"""

from scipy.spatial.transform import Rotation
import numpy as np
from copy import deepcopy
import collections

# default free cameras for different kitchen layouts
LAYOUT_CAMS = {
    1: dict(
        lookat=[2.26593463, -1.00037131, 1.38769295],
        distance=3.0505089839567323,
        azimuth=90.71563812375285,
        elevation=-12.63948837207208,
    ),
    2: dict(
        lookat=[2.66147999, -1.00162429, 1.2425155],
        distance=3.7958766287746255,
        azimuth=89.75784013699234,
        elevation=-15.177406642875091,
    ),
    3: dict(
        lookat=[3.02344359, -1.48874618, 1.2412914],
        distance=3.6684844368165512,
        azimuth=51.67880851867874,
        elevation=-13.302619131542388,
    ),
    # 4: dict(
    #     lookat=[11.44842548, -11.47664723, 11.24115989],
    #     distance=43.923271794728187,
    #     azimuth=227.12928449329333,
    #     elevation=-16.495686334624907,
    # ),
    5: dict(
        lookat=[1.6, -1.0, 1.0],
        distance=5,
        azimuth=89.70301806083651,
        elevation=-18.02177994296577,
    ),
}

DEFAULT_LAYOUT_CAM = {
    "lookat": [2.25, -1, 1.05312667],
    "distance": 5,
    "azimuth": 89.70301806083651,
    "elevation": -18.02177994296577,
}


def euler_to_calc_quat(x, y, z, x2=-15):
    from scipy.spatial.transform import Rotation as R

    roll = R.from_euler("x", x, degrees=True)
    pitch = R.from_euler("y", y, degrees=True)
    yaw = R.from_euler("z", z, degrees=True)
    pitch2 = R.from_euler("x", x2, degrees=True)
    final_rot = roll * pitch * pitch2 * yaw
    return final_rot.as_quat()[[3, 0, 1, 2]].tolist()


CAM_CONFIGS = dict(
    DEFAULT=dict(
        robot0_agentview_center=dict(
            pos=[-0.6, 0.0, 1.15],
            quat=[
                0.636945903301239,
                0.3325185477733612,
                -0.3199238181114197,
                -0.6175596117973328,
            ],
            parent_body="mobilebase0_support",
        ),
        robot0_agentview_left=dict(
            pos=[-0.5, 0.35, 1.05],
            quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_agentview_right=dict(
            pos=[-0.5, -0.35, 1.05],
            quat=[
                0.6775091886520386,
                0.3767866790294647,
                -0.2993525564670563,
                -0.55623859167099,
            ],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_frontview=dict(
            pos=[-0.50, 0, 0.95],
            quat=[
                0.6088936924934387,
                0.3814677894115448,
                -0.3673907518386841,
                -0.5905545353889465,
            ],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_eye_in_hand=dict(
            pos=[0.05, 0, 0],
            quat=[0, 0.707107, 0.707107, 0],
            parent_body="robot0_right_hand",
        ),
    ),
    ### Add robot specific configs here ####
    PandaMobile=dict(),
    GR1FixedLowerBody=dict(),
)

COTRAIN_CAM_CONFIGS = dict(
    DEFAULT=dict(
        robot0_agentview_center=dict(
            pos=[-0.6, 0.0, 1.15],
            quat=[
                0.636945903301239,
                0.3325185477733612,
                -0.3199238181114197,
                -0.6175596117973328,
            ],
            parent_body="mobilebase0_support",
        ),
        robot0_agentview_left=dict(
            pos=[-0.35, 0.4, 0.85],
            # quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
            # euler=[-0.959931, -1.309, 3.1415 + 0.6],
            quat=euler_to_calc_quat(x=90, y=-110, z=0, x2=-28),
            camera_attribs=dict(fovy="65"),
            parent_body="mobilebase0_support",
        ),
        robot0_agentview_right=dict(
            pos=[-0.35, -0.35, 0.85],
            # quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
            # euler=[-0.959931, -1.309, 3.1415 + 0.6],
            quat=euler_to_calc_quat(x=90, y=-65, z=0, x2=-28),
            camera_attribs=dict(fovy="65"),
            parent_body="mobilebase0_support",
        ),
        robot0_frontview=dict(
            pos=[-0.50, 0, 0.95],
            quat=[
                0.6088936924934387,
                0.3814677894115448,
                -0.3673907518386841,
                -0.5905545353889465,
            ],
            camera_attribs=dict(fovy="60"),
            parent_body="mobilebase0_support",
        ),
        robot0_eye_in_hand=dict(
            pos=[-0.029, 0, 0.05],
            camera_attribs=dict(
                focalpixel="606 606", resolution="640 480", sensorsize="5.5 3"
            ),
            # euler=[-3.1415927, -0.7853981 / 1.5, 1.5707963],
            quat=[-0.116144, -0.6975031, 0.6975031, 0.1161439],
            parent_body="robot0_right_hand",
        ),
    ),
)


def deep_update(d, u):
    """
    Copied from https://stackoverflow.com/a/3233356
    """

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_robot_cam_configs(robot, use_cotraining_cameras=False):
    if use_cotraining_cameras:
        default_cotraining_configs = deepcopy(COTRAIN_CAM_CONFIGS["DEFAULT"])
        return default_cotraining_configs
    default_configs = deepcopy(CAM_CONFIGS["DEFAULT"])
    robot_specific_configs = deepcopy(CAM_CONFIGS.get(robot, {}))
    return deep_update(default_configs, robot_specific_configs)


def set_cameras(env):
    """
    Adds new kitchen-relevant cameras to the environment. Will randomize cameras if specified.
    """
    env._cam_configs = get_robot_cam_configs(
        env.robots[0].name, use_cotraining_cameras=env.use_cotraining_cameras
    )
    if env.randomize_cameras:
        randomize_cameras(env)

    for (cam_name, cam_cfg) in env._cam_configs.items():
        if cam_cfg.get("parent_body", None) is not None:
            continue

        env.mujoco_arena.set_camera(
            camera_name=cam_name,
            pos=cam_cfg["pos"],
            quat=cam_cfg["quat"],
            camera_attribs=cam_cfg.get("camera_attribs", None),
        )


def randomize_cameras(env):
    """
    Randomizes the position and rotation of the wrist and agentview cameras.
    Note: This function is called only if randomize_cameras is set to True.
    """
    for camera in env._cam_configs:
        if "agentview" in camera:
            # use less noise for third person cotraining cameras
            pos_scale = 0.025 if env.use_cotraining_cameras else 0.1
            rot_scale = 0.75 if env.use_cotraining_cameras else 6
            pos_noise = env.rng.normal(loc=0, scale=pos_scale, size=(1, 3))[0]
            euler_noise = env.rng.normal(loc=0, scale=rot_scale, size=(1, 3))[0]
        elif "eye_in_hand" in camera:
            # bias and clip wristview position so as to not penetrate robot
            pos_noise = np.random.normal(
                loc=[-0.008, 0, 0], scale=0.00625, size=(1, 3)
            )[0]
            pos_noise[0] = min(0, pos_noise[0])
            pos_noise[2] = min(0.00125, pos_noise[2] / 2.5)
            # # clip z direction if the camera is close to the eef bc the camera will penetrate robot
            # if pos_noise[0] < 0.02:
            #     pos_noise[2] = max(0, pos_noise[2])
            euler_noise = np.random.normal(loc=0, scale=0.75, size=(1, 3))[0]
        else:
            # skip randomization for cameras not implemented
            continue

        old_pos = env._cam_configs[camera]["pos"]
        new_pos = [pos + n for pos, n in zip(old_pos, pos_noise)]
        env._cam_configs[camera]["pos"] = list(new_pos)

        old_euler = Rotation.from_quat(env._cam_configs[camera]["quat"]).as_euler(
            "xyz", degrees=True
        )
        new_euler = [eul + n for eul, n in zip(old_euler, euler_noise)]
        new_quat = Rotation.from_euler("xyz", new_euler, degrees=True).as_quat()
        env._cam_configs[camera]["quat"] = list(new_quat)
