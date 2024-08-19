from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa
from robocasa.models.arenas.layout_builder import create_fixtures


# base class for kitchens
class KitchenArena(Arena):
    """
    Kitchen arena class holding all of the fixtures

    Args:
        yaml_path (str): path to the file containing the fixture layout

        style (int): style of the kitchen to load

        rng (np.random.Generator): random number generator used for initializing
            fixture state in the KitchenArena
    """

    def __init__(self, yaml_path, style=0, rng=None):
        super().__init__(
            xml_path_completion(
                "arenas/empty_kitchen_arena.xml", root=robocasa.models.assets_root
            )
        )
        self.fixtures = create_fixtures(
            yaml_path,
            style=style,
            rng=rng,
        )

    def get_fixture_cfgs(self):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for (name, fxtr) in self.fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs
