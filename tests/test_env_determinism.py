import unittest
from unittest import mock

import numpy as np
from lxml import etree as ET

import mujoco
import robocasa
import robosuite
from robosuite import load_controller_config
from termcolor import colored

DEFAULT_SEED = 3

class TestEnvDeterminism(unittest.TestCase):

    skip_envs = set(["AfterwashSorting", "BowlAndCup", "ClearingCleaningReceptacles", 
                     "DrinkwareConsolidation", "HumanoidTransport", "PnP", "SetBowlsForSoup", 
                     "SetupJuicing", "WineServingPrep"])
    
    def create_env(self, config):
        env = robosuite.make(**config)
        env.reset()
        return env

    @mock.patch("random.choice")
    @mock.patch("random.choices")
    @mock.patch("random.randint")
    @mock.patch("random.shuffle")
    @mock.patch("numpy.random.randint")
    @mock.patch("numpy.random.normal")
    @mock.patch("numpy.random.uniform")
    def test_env_determinism(self, *args):
        """
        Tests environment determinism for all Kichen environments excluding those in
        skip_envs (defined above). We test for similarity in scene layout, style, all
        objects and fixtures in the scene including their position and orientation,
        and randomized cameras. 
        """
        
        def compare_scene_appearance(env_1, env_2):
            """
            Compares the appearance of two environments based on their layout
            and their style.
            """
            self.assertEqual(env_1.layout_id, env_2.layout_id)
            self.assertEqual(env_1.style_id, env_2.style_id)

        def compare_objects_in_scene(env_1, env_2):
            """
            Compares all added objects in the scene and their positions and
            quaternions.
            """
            env_1_objects = env_1.object_placements
            env_2_objects = env_2.object_placements

            self.assertEqual(env_1_objects.keys(), env_2_objects.keys())

            # Checks the position and rotation of all the geoms in the scene
            for name in env_1_objects.keys():
                pos_1, quat_1 = env_1_objects[name][:2]
                pos_2, quat_2 = env_2_objects[name][:2]
                np.testing.assert_allclose(pos_1, pos_2, atol=1e-7)
                np.testing.assert_allclose(quat_1, quat_2, atol=1e-7)

        def compare_fixtures_in_scene(env_1, env_2):
            """
            Compares all added fixtures in the scene and their positions and
            quaternions.
            """
            env_1_fixtures = env_1.fxtr_placements
            env_2_fixtures = env_2.fxtr_placements

            self.assertEqual(env_1_fixtures.keys(), env_2_fixtures.keys())

            for name in env_1_fixtures.keys():
                pos_1, quat_1 = env_1_fixtures[name][:2]
                pos_2, quat_2 = env_2_fixtures[name][:2]
                np.testing.assert_allclose(pos_1, pos_2, atol=1e-7)
                np.testing.assert_allclose(quat_1, quat_2, atol=1e-7)

        envs = sorted(robocasa.ALL_KITCHEN_ENVIRONMENTS)

        for i, env in enumerate(envs):
            if env in self.skip_envs or env.startswith("MG_"):
                continue

            print(colored(f"Testing {env} environment...", "green"))

            config = {
                "env_name": env,
                "robots": "PandaMobile",
                "controller_configs": load_controller_config(default_controller="OSC_POSE"),
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "ignore_done": True,
                "use_camera_obs": False,
                "control_freq": 20,
                "seed": DEFAULT_SEED,
                "randomize_cameras": False
            }

            env_1 = self.create_env(config)
            env_2 = self.create_env(config)

            compare_scene_appearance(env_1, env_2)
            compare_objects_in_scene(env_1, env_2)
            compare_fixtures_in_scene(env_1, env_2)

            env_1.close()
            env_2.close()

            for mock in args:
                mock.assert_not_called()

    def test_random_generative_textures(self):
        """
        Tests env determinism when using generative textures to ensure random generation 
        of textures results in the same generated texture replacement file.
        """

        config = {
            "env_name": "PnPCounterToCab",
            "robots": "PandaMobile",
            "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "seed": DEFAULT_SEED,
            "randomize_cameras": False,
            "generative_textures": "100p"
        }

        env_1 = self.create_env(config)
        env_2 = self.create_env(config)

        texture_names = ["cab_tex",
                         "counter_tex",
                         "wall_tex",
                         "floor_tex"]
        
        for texture_name in texture_names:
            self.assertEqual(env_1._curr_gen_fixtures[texture_name], env_2._curr_gen_fixtures[texture_name])

    def test_randomized_cameras(self):
        """
        Tests env determinism when using randomized cameras. Ensures that the position
        and orientation of all respective cameras are the same when using the same seed.
        """

        config = {
            "env_name": "PnPCounterToCab",
            "robots": "PandaMobile",
            "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "seed": DEFAULT_SEED,
            "randomize_cameras": False,
        }

        env_1 = self.create_env(config)
        env_2 = self.create_env(config)
        
        self.assertListEqual(list(env_1._cam_configs.keys()), list(env_2._cam_configs.keys()))
        for camera_name in env_1._cam_configs.keys():
            self.assertEqual(env_1._cam_configs[camera_name]["pos"], env_2._cam_configs[camera_name]["pos"])
            self.assertEqual(env_1._cam_configs[camera_name]["quat"], env_2._cam_configs[camera_name]["quat"])

if __name__ == "__main__":
    unittest.main()
