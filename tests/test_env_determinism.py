import unittest
from unittest import mock

import robocasa
import robosuite
from robosuite import load_controller_config
from termcolor import colored

DEFAULT_SEED = 3

class TestEnvDeterminism(unittest.TestCase):

    @mock.patch("random.choice")
    @mock.patch("random.choices")
    @mock.patch("random.randint")
    @mock.patch("random.shuffle")
    @mock.patch("numpy.random.randint")
    @mock.patch("numpy.random.normal")
    def test_no_random(self, 
                       mock_random_choice, 
                       mock_random_choices, 
                       mock_random_randint,
                       mock_random_shuffle, 
                       mock_numpy_random_randint, 
                       mock_numpy_random_normal):
        
        start_with = "BowlAndCup"
        start = False
        for env in sorted(robosuite.ALL_ENVIRONMENTS):
            print(colored(f"Testing {env} environment...", "green"))

            if env == start_with:
                start = True

            if not start:
                continue

            config = {
                "env_name": env,
                "robots": "PandaMobile",
                "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            }

            env = robosuite.make(
                **config,
                has_renderer=True,
                has_offscreen_renderer=False,
                ignore_done=True,
                use_camera_obs=False,
                control_freq=20,
                seed=DEFAULT_SEED,
                randomize_cameras=False
            )
            env.reset()

            mock_random_choice.assert_not_called()
            mock_random_choices.assert_not_called()
            mock_random_randint.assert_not_called()
            mock_random_shuffle.assert_not_called()
            mock_numpy_random_randint.assert_not_called()
            mock_numpy_random_normal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
