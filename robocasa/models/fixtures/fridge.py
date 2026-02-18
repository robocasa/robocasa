import numpy as np
from robocasa.models.fixtures import Fixture
import robosuite.utils.transform_utils as T
from robocasa.utils.object_utils import obj_inside_of, get_fixture_to_point_rel_offset
from robosuite.utils.mjcf_utils import string_to_array as s2a


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
        self._drawer_joint_names = []

        for joint_name in self._joint_infos:
            stripped_name = joint_name[len(self.name) + 1 :]
            if "door" in stripped_name and "fridge" in stripped_name:
                self._fridge_door_joint_names.append(joint_name)
            elif "door" in stripped_name and "freezer" in stripped_name:
                self._freezer_door_joint_names.append(joint_name)
            elif "drawer" in stripped_name:
                self._drawer_joint_names.append(joint_name)

        self._fridge_reg_names = [
            reg_name for reg_name in self._regions.keys() if "fridge" in reg_name
        ]
        self._freezer_reg_names = [
            reg_name for reg_name in self._regions.keys() if "freezer" in reg_name
        ]

    def get_reset_regions(
        self,
        env,
        compartment="fridge",
        z_range=(0.50, 1.50),
        rack_index=None,
        reg_type=("shelf"),
    ):
        """
        Returns reset regions inside the fridge or freezer as a dictionary of offsets and sizes.

        Args:
            env (Kitchen): the environment the fridge belongs to
            compartment (str): "fridge" or "freezer" — which compartment to query
            z_range (tuple): optional Z bounds to filter usable regions by height
            rack_index (int or None): if set, selects a specific shelf/drawer by index
                (0 = lowest, -1 = highest, -2 = second highest)
            reg_type (tuple or str): can be combination of shelf or drawer. specifies
                whether to use shelves or drawers, or both

        Returns:
            dict: mapping from region name to { "offset": (x, y, z), "size": (sx, sy) }
        """
        assert compartment in ["fridge", "freezer"]

        if isinstance(reg_type, str):
            reg_type = tuple([reg_type])
            for t in reg_type:
                assert t in ["shelf", "drawer"]
        region_names = []
        for reg_name in self.get_reset_region_names():
            if compartment not in reg_name:
                continue

            if "drawer" in reg_name:
                # it is a drawer
                if "drawer" not in reg_type:
                    continue
            else:
                # it is a shelf
                if "shelf" not in reg_type:
                    continue

            region_names.append(reg_name)

        reset_regions = {}

        for name in region_names:
            reg = self._regions.get(name)
            if reg is None:
                continue
            p0, px, py, pz = reg["p0"], reg["px"], reg["py"], reg["pz"]
            height = pz[2] - p0[2]
            if height < 0.20 and "drawer" not in reg_type:
                continue

            # bypass z-range check if rack_index is specified
            bypass_z_range = rack_index is not None
            if not bypass_z_range:
                reg_abs_z = self.pos[2] + p0[2]
                if reg_abs_z < z_range[0] or reg_abs_z > z_range[1]:
                    continue

            offset = (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2])
            size = (px[0] - p0[0], py[1] - p0[1])
            reset_regions[name] = {"offset": offset, "size": size}

        # sort by Z height (top shelf/drawer first)
        sorted_regions = sorted(
            reset_regions.items(),
            key=lambda item: self.pos[2] + self._regions[item[0]]["p0"][2],
            reverse=False,
        )

        if rack_index is not None:
            if rack_index == -1:
                return dict([sorted_regions[-1]]) if sorted_regions else {}
            elif rack_index == -2:
                if len(sorted_regions) > 1:
                    return dict([sorted_regions[-2]])
                else:
                    return dict([sorted_regions[-1]]) if sorted_regions else {}
            elif 0 <= rack_index < len(sorted_regions):
                return dict([sorted_regions[rack_index]])
            else:
                raise ValueError(
                    f"rack_index {rack_index} out of range for {compartment} regions. "
                    f"Available indices: {list(range(len(sorted_regions)))}"
                )

        return dict(sorted_regions)

    def _get_drawer_joints(self, compartment="fridge", drawer_rack_index=None):
        """
        Returns the list of drawer joint names for a given compartment. If drawer_rack_index is provided,
        selects a single joint based on reverse-alphabetical sorting (highest first).

        Args:
            compartment (str): "fridge" or "freezer"
            drawer_rack_index (int or None):
                - None: return all matching drawer joints
                - -1: highest drawer
                - -2: second highest drawer (falls back to highest if only one)
                - N >= 0: Nth drawer in reverse-sorted order

        Returns:
            list[str]: selected joint names (can be empty if none found)
        """
        compartment_drawer_joints = []
        for joint_name in self._drawer_joint_names:
            stripped_name = joint_name[len(self.name) + 1 :]
            if compartment in stripped_name:
                compartment_drawer_joints.append(joint_name)

        if not compartment_drawer_joints:
            return []

        if drawer_rack_index is None:
            return compartment_drawer_joints

        sorted_joints = sorted(compartment_drawer_joints, reverse=True)

        if drawer_rack_index == -1:
            return [sorted_joints[0]] if sorted_joints else []
        elif drawer_rack_index == -2:
            if len(sorted_joints) > 1:
                return [sorted_joints[1]]
            return [sorted_joints[0]] if sorted_joints else []
        elif 0 <= drawer_rack_index < len(sorted_joints):
            return [sorted_joints[drawer_rack_index]]
        else:
            raise ValueError(
                f"drawer_rack_index {drawer_rack_index} out of range for {compartment} drawer joints. "
                f"Available indices: {list(range(len(sorted_joints)))}"
            )

    def is_open(
        self,
        env,
        compartment="fridge",
        th=0.90,
        reg_type="door",
        drawer_rack_index=None,
    ):
        """
        checks whether the fixture is open

        Args:
            env (Kitchen): the environment the fridge belongs to
            compartment (str): "fridge" or "freezer" — which compartment to check
            th (float): threshold for determining if open (default: 0.90)
            reg_type (str): "door" or "drawer" — which type of opening to check
            drawer_rack_index (int or None): if reg_type is "drawer", checks specific drawer rack by index
                (0 = lowest, -1 = highest, -2 = second highest). If None, checks all drawer racks.
                Ignored when reg_type is "door".
        Returns:
            bool: True if the fixture is open, False otherwise
        """
        if reg_type == "door":
            joint_names = None
            if compartment == "fridge":
                joint_names = self._fridge_door_joint_names
            elif compartment == "freezer":
                joint_names = self._freezer_door_joint_names
            return super().is_open(env, joint_names, th=th)
        elif reg_type == "drawer":
            drawer_joints = self._get_drawer_joints(
                compartment=compartment, drawer_rack_index=drawer_rack_index
            )
            if not drawer_joints:
                return False
            return super().is_open(env, joint_names=drawer_joints, th=th)
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def is_closed(
        self,
        env,
        compartment="fridge",
        th=0.005,
        reg_type="door",
        drawer_rack_index=None,
    ):
        """
        checks whether the fixture is closed

        Args:
            env (Kitchen): the environment the fridge belongs to
            compartment (str): "fridge" or "freezer" — which compartment to check
            th (float): threshold for determining if closed (default: 0.005)
            reg_type (str): "door" or "drawer" — which type of opening to check
            drawer_rack_index (int or None): if reg_type is "drawer", checks specific drawer rack by index
                (0 = lowest, -1 = highest, -2 = second highest). If None, checks all drawer racks.
                Ignored when reg_type is "door".

        Returns:
            bool: True if the fixture is closed, False otherwise
        """
        if reg_type == "door":
            joint_names = None
            if compartment == "fridge":
                joint_names = self._fridge_door_joint_names
            elif compartment == "freezer":
                joint_names = self._freezer_door_joint_names
            return super().is_closed(env, joint_names, th=th)
        elif reg_type == "drawer":
            drawer_joints = self._get_drawer_joints(
                compartment=compartment, drawer_rack_index=drawer_rack_index
            )
            if not drawer_joints:
                return True
            return super().is_closed(env, joint_names=drawer_joints, th=th)
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def open_door(
        self,
        env,
        min=0.90,
        max=1.0,
        compartment="fridge",
        reg_type="door",
        drawer_rack_index=None,
    ):
        """
        helper function to open the door or drawer. calls set_door_state function

        Args:
            env (Kitchen): the environment the fridge belongs to
            min (float): minimum joint position (default: 0.90)
            max (float): maximum joint position (default: 1.0)
            compartment (str): "fridge" or "freezer" — which compartment to operate on
            reg_type (str): "door" or "drawer" — which type of opening to operate on
            drawer_rack_index (int or None): if reg_type is "drawer", selects specific drawer rack by index
                (0 = lowest, -1 = highest, -2 = second highest). If None, opens all drawer racks.
        """
        if reg_type == "door":
            joint_names = None
            if compartment == "fridge":
                joint_names = self._fridge_door_joint_names
            elif compartment == "freezer":
                joint_names = self._freezer_door_joint_names
            self.set_joint_state(env=env, min=min, max=max, joint_names=joint_names)
        elif reg_type == "drawer":
            drawer_joints = self._get_drawer_joints(
                compartment=compartment, drawer_rack_index=drawer_rack_index
            )
            if drawer_joints:
                self.set_joint_state(
                    env=env, min=min, max=max, joint_names=drawer_joints
                )
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def close_door(
        self,
        env,
        min=0.0,
        max=0.0,
        compartment="fridge",
        reg_type="door",
        drawer_rack_index=None,
    ):
        """
        helper function to close the door or drawer. calls set_door_state function

        Args:
            env (Kitchen): the environment the fridge belongs to
            min (float): minimum joint position (default: 0.0)
            max (float): maximum joint position (default: 0.0)
            compartment (str): "fridge" or "freezer" — which compartment to operate on
            reg_type (str): "door" or "drawer" — which type of opening to operate on
            drawer_rack_index (int or None): if reg_type is "drawer", selects specific drawer rack by index
                (0 = lowest, -1 = highest, -2 = second highest). If None, closes all drawer racks.
        """
        if reg_type == "door":
            joint_names = None
            if compartment == "fridge":
                joint_names = self._fridge_door_joint_names
            elif compartment == "freezer":
                joint_names = self._freezer_door_joint_names
            self.set_joint_state(env=env, min=min, max=max, joint_names=joint_names)
        elif reg_type == "drawer":
            drawer_joints = self._get_drawer_joints(
                compartment=compartment, drawer_rack_index=drawer_rack_index
            )
            if drawer_joints:
                self.set_joint_state(
                    env=env, min=min, max=max, joint_names=drawer_joints
                )
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def get_reset_region_names(self):
        return self._fridge_reg_names + self._freezer_reg_names

    def check_rack_contact(
        self,
        env,
        object_name,
        rack_index=None,
        compartment="fridge",
        reg_type=("shelf"),
    ):
        """
        Check if an object is in contact with a specific shelf or drawer in the fridge.

        Args:
            env (Kitchen): the environment the fridge belongs to
            object_name (str): name of the object to check
            rack_index (int or None): if set, selects a specific shelf/drawer by index
                (0 = lowest, -1 = highest, -2 = second highest)
            compartment (str): "fridge" or "freezer" — which compartment to query
            reg_type (tuple or str): can be combination of shelf or drawer. specifies
                whether to use shelves or drawers, or both

        Returns:
            bool: True if the object is in contact with the specified shelf/drawer, False otherwise
        """
        if env.sim is None or env.sim.data is None:
            return False

        if object_name not in env.obj_body_id:
            return False

        region_names = [
            name
            for name in self.get_reset_regions(
                env, compartment=compartment, rack_index=rack_index, reg_type=reg_type
            )
        ]

        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[object_name]])
        obj_z = obj_pos[2]

        # filter shelf regions based on the bottom to 90% of the total height
        filtered_region_names = []
        for region_name in region_names:
            reg = self._regions[region_name]
            p0, pz = reg["p0"], reg["pz"]

            region_min_z = self.pos[2] + p0[2]
            region_max_z = self.pos[2] + pz[2]

            is_drawer = "drawer" in region_name
            if is_drawer:
                restricted_max_z = region_max_z
            else:
                restricted_max_z = region_min_z + 0.9 * (region_max_z - region_min_z)
            if region_min_z <= obj_z <= restricted_max_z:
                filtered_region_names.append(region_name)

        if not filtered_region_names:
            return False

        orig_get_int = self.get_int_sites

        def get_int_sites_filtered(relative=False):
            sites = orig_get_int(relative=relative)
            return {rn: sites[rn] for rn in filtered_region_names}

        self.get_int_sites = get_int_sites_filtered
        inside = obj_inside_of(env, object_name, self.name, partial_check=False)

        self.get_int_sites = orig_get_int
        return inside

    def update_state(self, env):
        """
        Updates the interior bounding boxes of the fridge drawers to be matched with
        how open the drawers are. This is needed when determining if an object
        is inside the drawer or when placing an object inside an open drawer.

        Args:
            env (MujocoEnv): environment
        """
        for compartment in ["fridge", "freezer"]:
            drawer_regions = []
            for reg_name in self._regions.keys():
                if "drawer" in reg_name and compartment in reg_name:
                    drawer_regions.append(reg_name)
            for reg_name in drawer_regions:
                if reg_name in self._regions:
                    geom_name = f"{self.naming_prefix}reg_{reg_name}"
                    pos = get_fixture_to_point_rel_offset(
                        self, env.sim.data.get_geom_xpos(geom_name)
                    )
                    hs = s2a(self._regions[reg_name]["elem"].get("size"))
                    self.update_regions(
                        {reg_name: {"pos": pos, "halfsize": hs}}, update_elem=False
                    )

    @property
    def nat_lang(self):
        return "fridge"


class FridgeFrenchDoor(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator064", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)


class FridgeSideBySide(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator031", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)


class FridgeBottomFreezer(Fridge):
    def __init__(self, xml="fixtures/fridges/Refrigerator060", *args, **kwargs):
        super().__init__(xml=xml, *args, **kwargs)
