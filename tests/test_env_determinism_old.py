import unittest
import robocasa
import robosuite
from robosuite import load_controller_config
from termcolor import colored

DEFAULT_SEED = 3

class TestRandomKitchen(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        super().setUpClass()
        config = {
            "env_name": "Kitchen",
            "robots": "PandaMobile",
            "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        }

        self.env = robosuite.make(
            **config,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
            seed=DEFAULT_SEED
        )
        self.env.reset()

    def test_random_layout_style(self) -> None:
        # envs = sorted(robosuite.ALL_ENVIRONMENTS)
        # print(envs)
        self.assertEqual(self.env.layout_id, 6)
        self.assertEqual(self.env.style_id, 6)

    def test_base_position(self) -> None:
        self.assertEqual(self.env.fixture_name, "stool_3_island_group")
        pos = self.env.robots[0].robot_model._elements["root_body"].get("pos")
        pos = pos.split()
        self.assertEqual(len(pos), 3)
        self.assertAlmostEqual(float(pos[0]), 2.5500857073604597)
        self.assertAlmostEqual(float(pos[1]), -1.7999397791371388)
        self.assertAlmostEqual(float(pos[2]), 0.0)

class TestAllEnvironments(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        super().setUpClass()
        self.random_attrs = {
            "ArrangeBreadBasket": ["cab"]
        }
        
    def test_all_environments(self) -> None:
        for env in sorted(robosuite.ALL_ENVIRONMENTS):
            print(colored(f"Testing {env} environment...", "green"))
            
            if env == "AfterwashSorting":
                continue
            
            config = {
                "env_name": env,
                "robots": "PandaMobile",
                "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            }

            env = robosuite.make(
                **config,
                has_renderer=False,
                has_offscreen_renderer=False,
                ignore_done=True,
                use_camera_obs=False,
                control_freq=20,
                seed=DEFAULT_SEED
            )
            env.reset()

            for attr in self.random_attrs[env]:
                if attr == "cab":
                    door_state = env.cab.get_door_state(env)

            env.close()


if __name__ == "__main__":
    unittest.main()
