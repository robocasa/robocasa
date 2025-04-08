import numpy as np
from robocasa.models.fixtures import Fixture
import robocasa.utils.object_utils as OU
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
        self._fridge_door_joint_names = []
        self._freezer_door_joint_names = []
        for joint_name in self._joint_infos:
            stripped_name = joint_name[len(self.name) + 1 :]
            if "door" in stripped_name and "fridge" in stripped_name:
                self._fridge_door_joint_names.append(joint_name)
            elif "door" in stripped_name and "freezer" in stripped_name:
                self._freezer_door_joint_names.append(joint_name)

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

    def is_open(self, env, entity="fridge", th=0.90):
        """
        checks whether the fixture is open
        """
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names

        return super().is_open(env, joint_names, th=th)

    def is_closed(self, env, entity="fridge", th=0.005):
        """
        checks whether the fixture is closed
        """
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names

        return super().is_closed(env, joint_names, th=th)

    def open_door(self, env, min=0.90, max=1.0, entity="fridge"):
        """
        helper function to open the door. calls set_door_state function
        """
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names
        self.set_joint_state(env=env, min=min, max=max, joint_names=joint_names)

    def close_door(self, env, min=0.0, max=0.0, entity="fridge"):
        """
        helper function to close the door. calls set_door_state function
        """
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names
        self.set_joint_state(env=env, min=min, max=max, joint_names=joint_names)

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
            "freezer0",
            "freezer1",
            "freezer2",
        )


class FridgeSideBySide(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator031", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "fridge_shelf0",
            "fridge_shelf1",
            "fridge_shelf2",
            "freezer_shelf0",
            "freezer_shelf1",
            "freezer_shelf2",
        )


class FridgeBottomFreezer(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator060", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "fridge_shelf0",
            "fridge_shelf1",
            "fridge_shelf2",
            "freezer0",
            "freezer1",
            "freezer2",
        )
