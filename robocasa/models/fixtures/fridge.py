import numpy as np
from robocasa.models.fixtures import Fixture


class Fridge(Fixture):
    """
    Fridge fixture class
    """

    def __init__(
        self, xml="fixtures/fridges/Refrigerator033", name="fridge", *args, **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    def get_reset_regions(self, env, reg_type="fridge"):
        assert reg_type in ["fridge", "freezer"]
        reset_region_names = [
            reg_name for reg_name in self.RESET_REGION_NAMES if reg_type in reg_name
        ]
        reset_regions = {}
        for reg_name in reset_region_names:
            reg_dict = self._regions.get(reg_name, None)
            if reg_dict is None:
                continue
            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]
            pz = reg_dict["pz"]
            height = pz[2] - pz[0]
            if height < 0.20:
                continue
            reset_regions[reg_name] = {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        return reset_regions

    @property
    def nat_lang(self):
        return "fridge"


class FridgeFrenchDoor(Fridge):
    RESET_REGION_NAMES = [
        "fridge_left_shelf0",
        "fridge_left_shelf1",
        "fridge_left_shelf2",
        "fridge_right_shelf0",
        "fridge_right_shelf1",
        "fridge_right_shelf2",
    ]


class FridgeSideBySide(Fridge):
    RESET_REGION_NAMES = [
        "freezer_shelf0",
        "freezer_shelf1",
        "freezer_shelf2",
        "fridge_shelf0",
        "fridge_shelf1",
        "fridge_shelf2",
    ]


class FridgeBottomFreezer(Fridge):
    RESET_REGION_NAMES = [
        "fridge_shelf0",
        "fridge_shelf1",
        "fridge_shelf2",
    ]
