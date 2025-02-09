from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa
import yaml
from robocasa.models.scenes.scene_builder import (
    create_fixtures,
    get_layout_path,
    get_style_path,
)


# base class for kitchens
class KitchenArena(Arena):
    """
    Kitchen arena class holding all of the fixtures

    Args:
        layout_id (int or LayoutType): layout of the kitchen to load

        style_id (int or StyleType): style of the kitchen to load

        rng (np.random.Generator): random number generator used for initializing
            fixture state in the KitchenArena
    """

    def __init__(self, layout_id, style_id, rng=None):
        super().__init__(
            xml_path_completion(
                "arenas/empty_kitchen_arena.xml", root=robocasa.models.assets_root
            )
        )

        # load layout config
        if isinstance(layout_id, dict):
            layout_config = layout_id
        else:
            layout_path = get_layout_path(layout_id=layout_id)
            with open(layout_path, "r") as f:
                layout_config = yaml.safe_load(f)

        # load style config
        if isinstance(style_id, dict):
            style_config = style_id
        else:
            style_path = get_style_path(style_id=style_id)
            with open(style_path, "r") as f:
                style_config = yaml.safe_load(f)

        self.fixtures = create_fixtures(
            layout_config=layout_config,
            style_config=style_config,
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
