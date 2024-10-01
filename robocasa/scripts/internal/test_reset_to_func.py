import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import cv2

env_meta = {
    "env_name": "PnPCabToCounter",
    "env_version": "1.4.1",
    "type": 1,
    "env_kwargs": {
        "robots": "PandaOmron",
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_heights": 128,
        "camera_widths": 128,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "env_name": "PnPCabToCounter",
        "seed": 43,
    },
    "env_lang": None,
}

env_kwargs = dict(
    env_meta=env_meta,
    env_name="PnPCabToCounter",
    render=False,
    render_offscreen=True,
    use_image_obs=True,
    seed=42,
)

dummy_spec = dict(
    obs=dict(
        low_dim=["robot0_eef_pos"],
        rgb=[],
    ),
)
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

env = EnvUtils.create_env_from_metadata(**env_kwargs)

obs = env.reset()
print(env._ep_lang_str)
state_dict = env.get_state()

obs = env.reset()
print(env._ep_lang_str)

obs = env.reset_to(state_dict)
print(env._ep_lang_str)

obs = env.reset()
print(env._ep_lang_str)

# cv2.imshow("image0", obs["robot0_agentview_left_image"][::-1,:,::-1])
# cv2.waitKey(1)
