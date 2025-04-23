import re
import numpy as np
from robocasa.models.fixtures import Fixture
import robocasa.utils.object_utils as OU
from robosuite.utils.mjcf_utils import (
    string_to_array,
    find_elements,
)


class Fridge(Fixture):
    """
    Fridge fixture class that:
      - Extracts door joint info,
      - Provides door state setting,
      - Implements reset region retrieval,
      - And optionally extracts rack geoms.
    """

    def __init__(
        self, xml="fixtures/fridges/Refrigerator064", name="fridge", *args, **kwargs
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

        self._fridge_reg_names = [
            reg_name for reg_name in self._regions.keys() if "fridge" in reg_name
        ]
        self._freezer_reg_names = [
            reg_name for reg_name in self._regions.keys() if "freezer" in reg_name
        ]

    def _get_rack_geoms(self):
        rack_geoms = {}

        rack_pattern = re.compile(r"fridge_(left|right|main)_group_reg_fridge_shelf\d+")
        for elem in self.root.iter():
            if elem.tag == "geom":
                geom_name = elem.get("name", "")
                if rack_pattern.search(geom_name):
                    rack_geoms[geom_name] = elem
        return rack_geoms

    def get_reset_regions(self, env, reg_type="fridge", z_range=(0.50, 1.50)):
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
            height = pz[2] - p0[2]
            if height < 0.20:
                # region is too small, skip
                continue

            if z_range is not None:
                reg_abs_z = self.pos[2] + p0[2]
                if reg_abs_z < z_range[0] or reg_abs_z > z_range[1]:
                    # region hard to reach, skip
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

    def get_reset_region_names(self):
        return self._fridge_reg_names + self._freezer_reg_names

    @property
    def nat_lang(self):
        return "fridge"


class FridgeFrenchDoor(Fridge):
    """
    A variant of Fridge for French Door style.
    Implements a custom get_reset_regions method that groups shelves based on side.
    """

    def __init__(self, xml="fixtures/fridges/Refrigerator064", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "fridge_shelf0" "fridge_shelf1",
            "freezer0_l",
            "freezer0_r",
        )

    def get_reset_regions(self, env, reg_type="fridge", rack_index=None):
        """
        Returns a dictionary of reset regions for a fridge/freezer

        Args:
            env: the environment.
            rack_index (int, optional): A 0-indexed value selecting a specific shelf for "fridge".
                                        For freezer regions, valid indices are 0 or 1.
                                        If rack_index is -1, choose the highest rack (largest shelf number).
                                        If rack_index is -2, choose the rack below the highest.
            reg_type (str): Either "fridge" or "freezer".

        Returns:
            dict: Mapping of region name to {'offset': (x, y, z), 'size': (sx, sy)}.
        """
        if reg_type == "freezer":
            reset_regions = {}
            for reg_name in self.get_reset_region_names():
                if "freezer" in reg_name:
                    reg_dict = self._regions.get(reg_name, None)
                    if reg_dict is None:
                        continue

                    p0 = reg_dict["p0"]
                    px = reg_dict["px"]
                    py = reg_dict["py"]
                    region_offset = (
                        np.mean((p0[0], px[0])),
                        np.mean((p0[1], py[1])),
                        p0[2],
                    )
                    region_size = (px[0] - p0[0], py[1] - p0[1])
                    reset_regions[reg_name] = {
                        "offset": region_offset,
                        "size": region_size,
                    }
            if rack_index is not None:
                if rack_index == 0:
                    result = {"freezer_top": reset_regions.get("freezer_top", None)}
                elif rack_index == 1:
                    result = {
                        "freezer_bottom": reset_regions.get("freezer_bottom", None)
                    }
                else:
                    raise ValueError(
                        f"rack_index {rack_index} out of range for freezer regions. Valid indices are 0 or 1."
                    )
                if None in result.values():
                    raise ValueError(
                        f"Requested freezer region for rack_index {rack_index} not found in self._regions."
                    )
                return result
            else:
                available_keys = list(reset_regions.keys())
                if not available_keys:
                    raise ValueError("No freezer regions found in self._regions.")
                chosen_key = env.rng.choice(available_keys)
                return {chosen_key: reset_regions[chosen_key]}

        else:
            all_shelves = list(self._get_rack_geoms().items())
            shelves = {}
            for geom_name, geom_elem in all_shelves:
                match = re.search(r"reg_fridge_shelf(\d+)", geom_name)
                if match:
                    num = int(match.group(1))
                    shelves[num] = (geom_name, geom_elem)

            if not shelves:
                raise ValueError("No shelf regions found for the fridge.")

            if rack_index is not None:
                if rack_index == -1:
                    shelf_number = max(shelves.keys())
                elif rack_index == -2:
                    sorted_keys = sorted(shelves.keys())
                    if len(sorted_keys) < 2:
                        raise ValueError(
                            "Not enough shelves to select the second highest (rack_index -2)."
                        )
                    shelf_number = sorted_keys[-2]
                else:
                    shelf_number = rack_index
                if shelf_number not in shelves:
                    valid_indices = sorted(shelves.keys())
                    raise ValueError(
                        f"rack_index {rack_index} out of range for fridge regions. Available rack indices: {valid_indices}"
                    )
                chosen = (
                    shelf_number,
                    shelves[shelf_number][0],
                    shelves[shelf_number][1],
                )
            else:
                valid_keys = list(shelves.keys())
                chosen_number = env.rng.choice(valid_keys)
                chosen = (
                    chosen_number,
                    shelves[chosen_number][0],
                    shelves[chosen_number][1],
                )

            shelf_number, geom_name, geom_elem = chosen

            pos_str = geom_elem.get("pos", None)
            size_str = geom_elem.get("size", None)
            if pos_str is None or size_str is None:
                raise ValueError(
                    f"Selected shelf geom '{geom_name}' is missing pos or size attributes."
                )
            shelf_pos = np.fromstring(pos_str, sep=" ")
            shelf_size = np.fromstring(size_str, sep=" ")

            z_top = shelf_pos[2] - shelf_size[2]
            region_offset = (shelf_pos[0], shelf_pos[1], z_top)
            region_size = (shelf_size[0] * 2, shelf_size[1] * 2)
            region_key = f"shelf{shelf_number}"

            return {region_key: {"offset": region_offset, "size": region_size}}


class FridgeSideBySide(Fridge):
    def __init__(
        self, xml="fixtures/fridges/Refrigerator031/model.xml", *args, **kwargs
    ):
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

    def get_reset_regions(self, env, reg_type="fridge", rack_index=None):
        """
        Returns a dictionary of reset regions for a side-by-side fridge.

        Args:
            env: the environment.
            rack_index (int, optional): A 0-indexed value selecting a specific shelf.
                                        If rack_index is -1, the highest-numbered region is chosen.
                                        If rack_index is -2, the region below the highest is chosen.
            reg_type (str): Either "fridge" or "freezer". Determines which regions to return.

        Returns:
            dict: Mapping of region name to { 'offset': (x,y,z), 'size': (sx, sy) }.
        """
        region_names = [
            name for name in self.get_reset_region_names() if reg_type in name
        ]
        reset_regions = {}

        for name in region_names:
            reg_dict = self._regions.get(name, None)
            if reg_dict is None:
                continue
            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]

            region_offset = (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2])
            region_size = (px[0] - p0[0], py[1] - p0[1])
            reset_regions[name] = {"offset": region_offset, "size": region_size}

        if rack_index is not None:
            if rack_index == -1:

                def extract_number(region_name):
                    m = re.search(r"(\d+)", region_name)
                    if m:
                        return int(m.group(1))
                    else:
                        return -1

                sorted_region_names = sorted(region_names, key=extract_number)
                selected_name = sorted_region_names[-1]
                if selected_name in reset_regions:
                    return {selected_name: reset_regions[selected_name]}
                else:
                    raise ValueError(
                        f"Selected region '{selected_name}' not found in reset_regions."
                    )
            elif rack_index == -2:

                def extract_number(region_name):
                    m = re.search(r"(\d+)", region_name)
                    if m:
                        return int(m.group(1))
                    else:
                        return -1

                sorted_region_names = sorted(region_names, key=extract_number)
                if len(sorted_region_names) < 2:
                    raise ValueError(
                        "Not enough regions to select the second highest (rack_index=-2)."
                    )
                selected_name = sorted_region_names[-2]
                if selected_name in reset_regions:
                    return {selected_name: reset_regions[selected_name]}
                else:
                    raise ValueError(
                        f"Selected region '{selected_name}' not found in reset_regions."
                    )
            else:
                idx = rack_index
                if idx < len(region_names):
                    selected_name = region_names[idx]
                    if selected_name in reset_regions:
                        return {selected_name: reset_regions[selected_name]}
                else:
                    sorted_names = sorted(
                        region_names, key=lambda r: int(re.search(r"\d+", r).group())
                    )
                    raise ValueError(
                        f"rack_index {rack_index} out of range for {reg_type} regions. Available indices: {list(range(1, len(sorted_names)+1))}"
                    )
        return reset_regions


class FridgeBottomFreezer(Fridge):
    def __init__(
        self, xml="fixtures/fridges/Refrigerator060/model.xml", *args, **kwargs
    ):
        super().__init__(xml=xml, *args, **kwargs)

    def get_reset_region_names(self):
        return (
            "fridge_shelf0",
            "fridge_shelf1",
            "freezer0",
            "freezer1",
        )

    def get_reset_regions(self, env, reg_type="fridge", rack_index=None):
        """
        Returns reset regions for a bottom-freezer fridge

        Args:
            env: The environment (not used here, except for possible debug or context).
            rack_index (int, optional): 0-indexed shelf/region to return from 0 to N (lowest to highest).
                If rack_index is -1, the highest rack is returned.
                If rack_index is -2, the region below the highest is chosen.
            reg_type (str): "fridge" or "freezer".

        Returns:
            dict: A dictionary mapping region names to:
                { "offset": (x, y, z), "size": (size_x, size_y) }
        """

        valid_names = [
            name for name in self.get_reset_region_names() if reg_type in name
        ]

        all_reset_regions = {}
        for name in valid_names:
            reg_dict = self._regions.get(name, None)
            if reg_dict is None:
                continue

            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]

            offset = (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2])
            size = (px[0] - p0[0], py[1] - p0[1])
            all_reset_regions[name] = {"offset": offset, "size": size}

        if rack_index is not None:
            sorted_names = sorted(valid_names)
            if rack_index == -1:
                chosen_name = sorted_names[-1]
                if chosen_name in all_reset_regions:
                    return {chosen_name: all_reset_regions[chosen_name]}
                else:
                    raise ValueError(
                        f"Rack region '{chosen_name}' not found in self._regions."
                    )
            elif rack_index == -2:
                if len(sorted_names) < 2:
                    raise ValueError(
                        "Not enough regions to select the second highest (rack_index=-2)."
                    )
                chosen_name = sorted_names[-2]
                if chosen_name in all_reset_regions:
                    return {chosen_name: all_reset_regions[chosen_name]}
                else:
                    raise ValueError(
                        f"Rack region '{chosen_name}' not found in self._regions."
                    )
            else:
                idx = rack_index
                if 0 <= idx < len(sorted_names):
                    chosen_name = sorted_names[idx]
                    if chosen_name in all_reset_regions:
                        return {chosen_name: all_reset_regions[chosen_name]}
                else:
                    raise ValueError(
                        f"rack_index {rack_index} out of range for {reg_type} regions. Available indices: {list(range(len(sorted_names)))}"
                    )

        return all_reset_regions