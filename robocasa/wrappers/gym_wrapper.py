import cv2
import gymnasium as gym
import numpy as np
import robocasa  # we need this to register environments  # noqa: F401
import robosuite
import sys

from gymnasium import spaces
from gymnasium.envs.registration import register
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.environments.base import REGISTERED_ENVS


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)
from robocasa.utils.env_utils import create_env


class PandaOmronKeyConverter:
    @classmethod
    def get_camera_config(cls):
        mapped_names = [
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ]
        camera_names = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        camera_widths, camera_heights = 256, 256
        return mapped_names, camera_names, camera_widths, camera_heights

    @classmethod
    def map_obs(cls, input_obs):
        output_obs = type(input_obs)()
        output_obs = {
            "hand.gripper_qpos": input_obs["robot0_gripper_qpos"],
            "body.base_position": input_obs["robot0_base_pos"],
            "body.base_rotation": input_obs["robot0_base_quat"],
            "body.end_effector_position_relative": input_obs["robot0_base_to_eef_pos"],
            "body.end_effector_rotation_relative": input_obs["robot0_base_to_eef_quat"],
        }
        return output_obs

    @classmethod
    def deduce_observation_space(cls, env):
        obs = (
            env.viewer._get_observations(force_update=True)
            if env.viewer_get_obs
            else env._get_observations(force_update=True)
        )
        # obs.update(gather_robot_observations(env))
        obs = cls.map_obs(obs)
        observation_space = spaces.Dict()

        for k, v in obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                observation_space["state." + k[5:]] = spaces.Box(
                    low=-1, high=1, shape=(len(v),), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown key: {k}")

        return observation_space

    @classmethod
    def deduce_action_space(cls, env):
        # action = cls.map_action(reconstruct_latest_actions(env))
        action = {
            "hand.gripper_close": np.zeros(1),
            "body.end_effector_position": np.zeros(3),
            "body.end_effector_rotation": np.zeros(3),
            "body.base_motion": np.zeros(4),
            "body.control_mode": np.zeros(1),
        }
        action_space = spaces.Dict()
        for k, v in action.items():
            if isinstance(v, np.int64):
                action_space["action." + k[5:]] = spaces.Discrete(2)
            elif isinstance(v, np.ndarray):
                action_space["action." + k[5:]] = spaces.Box(
                    low=-1, high=1, shape=(len(v),), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown type: {type(v)}")
        return action_space

    @classmethod
    def map_obs_in_eval(cls, input_obs):
        output_obs = {}
        mapped_obs = cls.map_obs(input_obs)
        for k, v in mapped_obs.items():
            assert k.startswith("hand.") or k.startswith("body.")
            output_obs["state." + k[5:]] = v
        return output_obs

    @classmethod
    def convert_to_float64(cls, input):
        for k, v in input.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float32:
                input[k] = v.astype(np.float64)
        return input

    @classmethod
    def unmap_action(cls, input_action):
        output_action = type(input_action)()
        output_action = {
            "robot0_right_gripper": (
                -1.0 if input_action["action.gripper_close"] < 0.5 else 1.0
            ),
            "robot0_right": np.concatenate(
                (
                    input_action["action.end_effector_position"],
                    input_action["action.end_effector_rotation"],
                ),
                axis=-1,
            ),
            "robot0_base": input_action["action.base_motion"][..., 0:3],
            "robot0_torso": input_action["action.base_motion"][..., 3:4],
            "robot0_base_mode": (
                -1.0 if input_action["action.control_mode"] < 0.5 else 1.0
            ),
        }
        return output_action


class RoboCasaGymEnv(gym.Env):
    def __init__(
        self,
        env_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        split="test",
        **kwargs,  # Accept additional kwargs
    ):
        self.key_converter = PandaOmronKeyConverter
        (
            _,
            camera_names,
            default_camera_widths,
            default_camera_heights,
        ) = self.key_converter.get_camera_config()

        if camera_widths is None:
            camera_widths = default_camera_widths
        if camera_heights is None:
            camera_heights = default_camera_heights

        self.env_name = env_name
        print(f"Creating {env_name} with split={split}")
        self.env = create_env(
            env_name=env_name,
            render_onscreen=False,
            # seed=0, # set seed=None to run unseeded
            split=split,
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            **kwargs,
        )
        self.env.reset()

        # TODO: the following info should be output by grootrobocasa
        self.camera_names = camera_names
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.enable_render = enable_render
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None

        self._create_obs_and_action_space()

    def _create_obs_and_action_space(self):
        # setup spaces
        action_space = spaces.Dict()
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            for part_name, controller in cc.part_controllers.items():
                min_value, max_value = -1, 1
                start_idx, end_idx = cc._action_split_indexes[part_name]
                shape = [end_idx - start_idx]
                this_space = spaces.Box(
                    low=min_value, high=max_value, shape=shape, dtype=np.float32
                )
                action_space[f"{pf}{part_name}"] = this_space
            if isinstance(cc, HybridMobileBase):
                this_space = spaces.Discrete(2)
                action_space[f"{pf}base_mode"] = this_space

            action_space = spaces.Dict(action_space)
            self.action_space = action_space

        obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        # obs.update(gather_robot_observations(self.env))
        observation_space = spaces.Dict()
        for obs_name, obs_value in obs.items():
            shape = list(obs_value.shape)
            if obs_name.endswith("_image"):
                continue
            # min_value, max_value = -1, 1
            min_value, max_value = -1000, 1000
            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[obs_name] = this_space

        for camera_name in self.camera_names:
            shape = [self.camera_heights, self.camera_widths, 3]
            this_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            observation_space[f"{camera_name}_image"] = this_space

        observation_space["language"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        self.observation_space = observation_space

        # now remap observation and action space
        self.observation_space = self.key_converter.deduce_observation_space(self.env)
        mapped_names, _, _, _ = self.key_converter.get_camera_config()
        for mapped_name in mapped_names:
            self.observation_space[mapped_name] = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_heights, self.camera_widths, 3),
                dtype=np.uint8,
            )

        self.observation_space["annotation.human.task_description"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )
        self.action_space = self.key_converter.deduce_action_space(self.env)

    def get_basic_observation(self, raw_obs):
        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        # Return black image if rendering is disabled
        if not self.enable_render:
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights, self.camera_widths, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def get_observation(self, raw_obs):
        basic_obs = self.get_basic_observation(raw_obs)
        obs = {}
        temp_obs = self.key_converter.map_obs(basic_obs)
        for k, v in temp_obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                obs["state." + k[5:]] = v
            else:
                raise ValueError(f"Unknown key: {k}")
        mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
        for mapped_name, camera_name in zip(mapped_names, camera_names):
            obs[mapped_name] = basic_obs[camera_name + "_image"]
            # self.process_img(
            #     basic_obs[camera_name + "_image"]
            # )

        obs["annotation.human.task_description"] = basic_obs["language"]

        return obs

    # def process_img(self, img):
    #     h, w, _ = img.shape
    #     if h != w:
    #         dim = max(h, w)
    #         y_offset = (dim - h) // 2
    #         x_offset = (dim - w) // 2
    #         img = np.pad(img, ((y_offset, y_offset), (x_offset, x_offset), (0, 0)))
    #         h, w = dim, dim
    #     if (h, w) != (self.camera_heights, self.camera_widths):
    #         img = cv2.resize(img, (self.camera_heights, self.camera_widths), cv2.INTER_AREA)
    #     return np.copy(img)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)

        raw_obs = self.env.reset()
        # return obs
        obs = self.get_observation(raw_obs)

        info = {}
        info["success"] = False

        return obs, info

    def step(self, action_dict):
        action_dict = self.key_converter.unmap_action(action_dict)

        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act
            if isinstance(cc, HybridMobileBase):
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        assert len(action_dict) == 0, f"Unprocessed actions: {action_dict}"
        env_action = np.concatenate(env_action)

        raw_obs, reward, done, info = self.env.step(env_action)
        # sparse reward
        is_success = self.env._check_success()
        reward = 1.0 if is_success else 0.0

        obs = self.get_observation(raw_obs)

        truncated = False

        info["success"] = reward > 0

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        # If this wrapper doesn't have the attribute, forward to inner env
        return getattr(self.env, name)


def create_robocasa_gym_env_class(env):
    class_name = f"{env}"
    id_name = f"robocasa/{class_name}"

    env_class_type = type(
        class_name,
        (RoboCasaGymEnv,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=env,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["robocasa.wrappers.gym_wrapper"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,  # Unique ID for the environment
        entry_point=f"robocasa.wrappers.gym_wrapper:{class_name}",  # Path to your environment class
    )


for ENV in REGISTERED_ENVS:
    create_robocasa_gym_env_class(ENV)
