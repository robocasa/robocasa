"""Compatibility helpers for older RoboCasa callers.

Model evaluation is implemented in :mod:`robocasa_eval`. The helpers below
retain the legacy raw-RoboSuite interface; random rollouts are a demo, not a
benchmark policy. Heavy dependencies are imported only inside these helpers.
"""

from copy import deepcopy
from pathlib import Path
import sys


def create_eval_env(
    env_name,
    # robosuite-related configs
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
):
    import robosuite
    from robosuite.controllers import (
        load_composite_controller_config,
        load_part_controller_config,
    )

    robot_name = robots if isinstance(robots, str) else robots[0]
    controller_configs = load_composite_controller_config(robot=robot_name)
    part_controller = load_part_controller_config(default_controller=controllers)
    for name, old_part in controller_configs["body_parts"].items():
        if name in {"left", "right"}:
            new_part = deepcopy(part_controller)
            if "gripper" in old_part:
                new_part["gripper"] = deepcopy(old_part["gripper"])
            controller_configs["body_parts"][name] = new_part

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    """Keep the legacy raw-RoboSuite random-action demo API for existing callers."""
    import imageio
    import numpy as np
    from tqdm import tqdm
    from termcolor import colored

    video_writer = None
    if video_path is not None:
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=512, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


def main():
    """Route the former evaluation script to the shared model evaluator."""
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robocasa_eval.cli import main as evaluate

    return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
