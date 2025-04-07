import numpy as np
from robocasa.models.fixtures import Fixture
from robosuite.utils.mjcf_utils import (
    string_to_array,
    find_elements,
)


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
        self._fridge_door_joint_info = dict()
        self._freezer_door_joint_info = dict()
        joint_elems = find_elements(
            root=self.worldbody, tags="joint", return_first=False
        )
        for elem in joint_elems:
            elem_name = elem.get("name")
            if elem_name is None:
                continue
            # strip out name prefix
            stripped_name = elem_name[len(self.name) + 1 :]
            if "door" in stripped_name:
                if "fridge" in stripped_name:
                    self._fridge_door_joint_info[elem_name] = dict(
                        range=string_to_array(elem.get("range")),
                    )
                elif "freezer" in stripped_name:
                    self._freezer_door_joint_info[elem_name] = dict(
                        range=string_to_array(elem.get("range")),
                    )

    def get_reset_regions(self, env, reg_type="fridge"):
        assert reg_type in ["fridge", "freezer"]
        reset_region_names = [
            reg_name
            for reg_name in self.get_reset_region_names()
            if reg_type in reg_name
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

    def set_door_state(self, min, max, env, door_type="fridge"):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is
        Args:
            min (float): minimum percentage of how open the door is
            max (float): maximum percentage of how open the door is
            env (MujocoEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        rng = None or env.rng

        assert door_type in ["fridge", "freezer"]
        joint_info = (
            self._fridge_door_joint_info
            if door_type == "fridge"
            else self._freezer_door_joint_info
        )

        for joint_name, info in joint_info.items():
            joint_min, joint_max = info["range"]
            desired_min = joint_min + (joint_max - joint_min) * min
            desired_max = joint_min + (joint_max - joint_min) * max
            env.sim.data.set_joint_qpos(
                joint_name,
                rng.uniform(desired_min, desired_max),
            )

    @property
    def nat_lang(self):
        return "fridge"


class FridgeFrenchDoor(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator033", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "fridge_left_shelf0",
            "fridge_left_shelf1",
            "fridge_left_shelf2",
            "fridge_right_shelf0",
            "fridge_right_shelf1",
            "fridge_right_shelf2",
            "freezer_top",
            "freezer_bottom",
        )


class FridgeSideBySide(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator031", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "freezer_shelf0",
            "freezer_shelf1",
            "freezer_shelf2",
            "fridge_shelf0",
            "fridge_shelf1",
            "fridge_shelf2",
        )


class FridgeBottomFreezer(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator060", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "fridge_shelf0",
            "fridge_shelf1",
            "fridge_shelf2",
            "freezer_top",
            "freezer_bottom",
        )
