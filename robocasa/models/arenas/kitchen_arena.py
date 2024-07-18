from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robocasa.models.arenas.layout_builder import create_fixtures

import robocasa


# base class for kitchens
class KitchenArena(Arena):
    def __init__(self, json_path, style=0, rng=None):
        super().__init__(xml_path_completion("arenas/empty_kitchen_arena.xml", root=robocasa.models.assets_root))
        self.fixtures = create_fixtures(
            json_path, style=style, rng=rng,
        )

    def get_fixture_cfgs(self):
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
