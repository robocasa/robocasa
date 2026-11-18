import hashlib
import unittest
from unittest import mock

import numpy as np
from lxml import etree as ET

import mujoco
import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config
from termcolor import colored
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.environments.kitchen.composite.clearing_table.drinkware_consolidation import (
    DrinkwareConsolidation,
)
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

DEFAULT_SEED = 3


class TestEnvDeterminism(unittest.TestCase):
    def test_gym_reset_forwards_explicit_episode_seed(self):
        env = object.__new__(RoboCasaGymEnv)
        env.env = mock.Mock()
        env.env.reset.return_value = {"raw": 1}
        env.get_observation = mock.Mock(return_value={"mapped": 1})
        self.assertEqual(env.reset(options={"episode_seed": 123})[0], {"mapped": 1})
        env.env.reset.assert_called_once_with(episode_seed=123)

    def test_gym_reset_rejects_conflicting_seeds(self):
        env = object.__new__(RoboCasaGymEnv)
        env.env = mock.Mock()
        with self.assertRaises(ValueError):
            env.reset(seed=8, options={"episode_seed": 7})
        env.env.reset.assert_not_called()

    def test_overridden_reset_forwards_episode_seed(self):
        env = object.__new__(DrinkwareConsolidation)
        env.cab = mock.Mock()
        with mock.patch.object(Kitchen, "reset", return_value={"obs": 1}) as reset:
            self.assertEqual(env.reset(episode_seed=123), {"obs": 1})
        reset.assert_called_once_with(episode_seed=123)
        env.cab.open_door.assert_called_once_with(env=env)

    def test_episode_seed_is_independent_of_previous_rollout(self):
        """The same episode seed reproduces the scene after different rollouts."""
        config = {
            "env_name": "ArrangeDrinkware",
            "robots": "PandaOmron",
            "controller_configs": load_composite_controller_config(
                controller=None, robot="PandaOmron"
            ),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "seed": 7,
            "randomize_cameras": True,
        }
        env_a = robosuite.make(**config)
        env_b = robosuite.make(**config)
        try:
            env_a.reset(episode_seed=101)
            env_b.reset(episode_seed=101)
            for _ in range(10):
                env_b.step(np.zeros(env_b.action_dim))

            env_a.reset(episode_seed=102)
            env_b.reset(episode_seed=102)
            self.assertEqual(
                (env_a.layout_id, env_a.style_id),
                (env_b.layout_id, env_b.style_id),
            )
            self.assertEqual(env_a._cam_configs, env_b._cam_configs)
            self.assertEqual(
                hashlib.sha256(env_a.model.get_xml().encode()).hexdigest(),
                hashlib.sha256(env_b.model.get_xml().encode()).hexdigest(),
            )
            self.assertEqual(
                env_a.object_placements.keys(), env_b.object_placements.keys()
            )
            self.assertEqual(env_a.fxtr_placements.keys(), env_b.fxtr_placements.keys())
            for placements_a, placements_b in (
                (env_a.object_placements, env_b.object_placements),
                (env_a.fxtr_placements, env_b.fxtr_placements),
            ):
                for name in placements_a:
                    np.testing.assert_array_equal(
                        placements_a[name][0], placements_b[name][0]
                    )
                    np.testing.assert_array_equal(
                        placements_a[name][1], placements_b[name][1]
                    )
            np.testing.assert_array_equal(env_a.sim.data.qpos, env_b.sim.data.qpos)
            np.testing.assert_array_equal(env_a.sim.data.qvel, env_b.sim.data.qvel)
        finally:
            env_a.close()
            env_b.close()

    skip_envs = set(
        [
            "AfterwashSorting",
            "BowlAndCup",
            "ClearingCleaningReceptacles",
            "DrinkwareConsolidation",
            "PnP",
            "SetBowlsForSoup",
            "WineServingPrep",
        ]
    )

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
                "robots": "PandaOmron",
                "controller_configs": load_composite_controller_config(
                    controller=None, robot="PandaOmron"
                ),
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
            "env_name": "PickPlaceCounterToCabinet",
            "robots": "PandaOmron",
            "controller_configs": load_composite_controller_config(
                controller=None, robot="PandaOmron"
            ),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "seed": DEFAULT_SEED,
            "randomize_cameras": False,
            "generative_textures": "100p",
        }

        env_1 = self.create_env(config)
        env_2 = self.create_env(config)

        texture_names = ["cab_tex", "counter_tex", "wall_tex", "floor_tex"]

        for texture_name in texture_names:
            self.assertEqual(
                env_1._curr_gen_fixtures[texture_name],
                env_2._curr_gen_fixtures[texture_name],
            )

    def test_randomized_cameras(self):
        """
        Tests env determinism when using randomized cameras. Ensures that the position
        and orientation of all respective cameras are the same when using the same seed.
        """

        config = {
            "env_name": "PickPlaceCounterToCabinet",
            "robots": "PandaOmron",
            "controller_configs": load_composite_controller_config(
                controller=None, robot="PandaOmron"
            ),
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

        self.assertListEqual(
            list(env_1._cam_configs.keys()), list(env_2._cam_configs.keys())
        )
        for camera_name in env_1._cam_configs.keys():
            self.assertEqual(
                env_1._cam_configs[camera_name]["pos"],
                env_2._cam_configs[camera_name]["pos"],
            )
            self.assertEqual(
                env_1._cam_configs[camera_name]["quat"],
                env_2._cam_configs[camera_name]["quat"],
            )


if __name__ == "__main__":
    unittest.main()
