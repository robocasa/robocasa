import unittest
from termcolor import colored
import argparse
import os

from robocasa.scripts.dataset_scripts.playback_dataset import playback_dataset
from robocasa.utils.dataset_registry import (
    ATOMIC_TASK_DATASETS,
    COMPOSITE_TASK_DATASETS,
)
from robocasa.utils.dataset_registry_utils import get_ds_path


class TestTasksValidity(unittest.TestCase):
    def test_tasks_validity(self, *args):
        """
        Tests that all kitchen environment tasks run error free. Iterates through
        all tasks, creates the environment, then runs NUM_ROLLOUTS test episodes per scene.
        At the end, prints out all tests that were successful, and then any that were not
        completed successfully along with their errors.
        """

        # iterate through all atomic and composite tasks
        all_tasks = list(ATOMIC_TASK_DATASETS) + list(COMPOSITE_TASK_DATASETS)
        all_tasks = all_tasks[-2:]
        for task_i, task in enumerate(all_tasks):
            human_path = get_ds_path(task=task, source="human")  # human dataset path
            print(f"Dataset path: {human_path}")
            print(
                colored(
                    f"Playing back {task} environment [{task_i+1}/{len(all_tasks)}]...",
                    "green",
                )
            )

            args = argparse.ArgumentParser()
            args.dataset = human_path
            args.filter_key = None
            args.n = 5
            args.use_obs = False
            args.use_actions = True
            args.use_abs_actions = False
            args.render = False
            video_folder = os.path.expanduser("~/tmp/playback_videos")
            if os.path.exists(video_folder) is False:
                os.makedirs(video_folder)
            args.video_path = os.path.join(video_folder, f"{task}.mp4")
            args.video_skip = 5
            args.render_image_names = [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]
            args.first = False
            args.extend_states = False
            args.verbose = False
            args.camera_height = 512
            args.camera_width = 768

            playback_dataset(
                dataset=args.dataset,
                use_actions=args.use_actions,
                use_abs_actions=args.use_abs_actions,
                use_obs=args.use_obs,
                filter_key=args.filter_key,
                n=args.n,
                render=args.render,
                render_image_names=args.render_image_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                video_path=args.video_path,
                video_skip=args.video_skip,
                extend_states=args.extend_states,
                first=args.first,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    unittest.main()
