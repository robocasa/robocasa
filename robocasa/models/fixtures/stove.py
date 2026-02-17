from copy import deepcopy
import numpy as np
from robocasa.environments.kitchen.kitchen import *
from robocasa.models.fixtures import Fixture

STOVE_LOCATIONS = [
    "rear_left",
    "rear_center",
    "rear_right",
    "front_left",
    "front_center",
    "front_right",
    "center",
    "left",
    "right",
]


class Stove(Fixture):
    """
    Stove fixture class. The stove has knob joints that can be turned on and off to simulate burner flames

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    STOVE_LOW_MIN = 0.35
    STOVE_HIGH_MIN = np.deg2rad(80)

    def __init__(
        self,
        xml="fixtures/stoves/basic_sleek_induc",
        name="stove",
        stove_type="standard",
        *args,
        **kwargs
    ):

        assert stove_type in ["standard", "wide"]
        self.stove_type = stove_type
        self._knob_joints = None
        self._burner_sites = None

        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    def get_reset_regions(self, env, locs=None):
        """
        Returns dictionary of reset regions, usually used when initializing a receptacle on the stove.
        The regions are the sites where the burner flames are located.

        Args:
            env (MujocoEnv): environment

            locs (list): list of locations to get reset regions for. If None, uses all locations

        Returns:
            dict: dictionary of reset regions
        """
        regions = dict()

        if locs is None:
            locs = STOVE_LOCATIONS
        for location in locs:
            site = self.worldbody.find(
                "./body/body/site[@name='{}burner_{}_place_site']".format(
                    self.naming_prefix, location
                )
            )
            if site is None:
                site = self.worldbody.find(
                    "./body/body/site[@name='{}burner_on_{}']".format(
                        self.naming_prefix, location
                    )
                )
            if site is None:
                continue
            burner_pos = [float(x) for x in site.get("pos").split()]
            regions[location] = {
                "offset": burner_pos,
                "size": [0.10, 0.10],
            }

        return regions

    def update_state(self, env):
        """
        Updates the burner flames of the stove based on the knob joint positions

        Args:
            env (MujocoEnv): environment
        """
        for location in STOVE_LOCATIONS:
            site = self.burner_sites[location]
            if site is None:
                continue
            site_id = env.sim.model.site_name2id(
                "{}burner_on_{}".format(self.naming_prefix, location)
            )

            joint = self.knob_joints[location]
            if joint is None:
                env.sim.model.site_rgba[site_id][3] = 0.0
                continue
            joint_id = env.sim.model.get_joint_qpos_addr(
                "{}knob_{}_joint".format(self.naming_prefix, location)
            )

            joint_qpos = deepcopy(env.sim.data.qpos[joint_id])
            joint_qpos = joint_qpos % (2 * np.pi)
            if joint_qpos < 0:
                joint_qpos += 2 * np.pi

            if 0.35 <= np.abs(joint_qpos) <= 2 * np.pi - 0.35:
                env.sim.model.site_rgba[site_id][3] = 0.5
            else:
                env.sim.model.site_rgba[site_id][3] = 0.0

    def set_knob_state(self, env, rng, knob, mode="on"):
        """
        Sets the state of the knob joint based on the mode parameter

        Args:
            env (MujocoEnv): environment

            rng (np.random.RandomState): random number generator

            knob (str): location of the knob

            mode (str): "on" or "off"
        """
        assert mode in ["on", "off", "high", "low"]
        _, joint_max = self._joint_infos[
            "{}knob_{}_joint".format(self.naming_prefix, knob)
        ]["range"]
        if mode == "off":
            joint_val = 0.0
        elif mode == "low":
            joint_val = rng.uniform(0.35, self.STOVE_HIGH_MIN - 0.00001)
        elif mode == "high":
            joint_val = rng.uniform(self.STOVE_HIGH_MIN, joint_max)
        else:
            joint_val = rng.uniform(0.50, joint_max)

        env.sim.data.set_joint_qpos(
            "{}knob_{}_joint".format(self.naming_prefix, knob), joint_val
        )

    def get_knobs_state(self, env):
        """
        Gets the angle of which knob joints are turned

        Args:
            env (MujocoEnv): environment

        Returns:
            dict: maps location of knob to the angle of the knob joint
        """
        knobs_state = {}
        for location in STOVE_LOCATIONS:
            joint = self.knob_joints[location]
            if joint is None:
                continue
            site = self.burner_sites[location]
            if site is None:
                continue

            joint_id = env.sim.model.get_joint_qpos_addr(
                "{}knob_{}_joint".format(self.naming_prefix, location)
            )

            joint_qpos = deepcopy(env.sim.data.qpos[joint_id])

            # normalize between 0 and 2pi
            joint_qpos = joint_qpos % (2 * np.pi)
            if joint_qpos < 0:
                joint_qpos += 2 * np.pi

            knobs_state[location] = joint_qpos
        return knobs_state

    def is_burner_on(self, env, burner_loc, th=0.35):
        """
        checks if a specified burner is on or off
        """
        knobs_state = self.get_knobs_state(env=env)
        knob_value = knobs_state[burner_loc]
        knob_on = th <= np.abs(knob_value) <= 2 * np.pi - 0.35
        return knob_on

    def check_obj_location_on_stove(self, env, obj_name, threshold=0.08):
        """
        Check if the object is on the stove and close to a burner and the knob is on.
        Returns the location of the burner if the object is on the stove, close to a burner, and the burner is on.
        None otherwise.
        """

        knobs_state = self.get_knobs_state(env=env)

        obj = env.objects[obj_name]
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])[0:2]

        obj_on_stove = OU.check_obj_fixture_contact(env, obj_name, self)

        if obj_on_stove:
            for location, site in self.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(env.sim.data.get_site_xpos(site.get("name")))[
                        0:2
                    ]
                    dist = np.linalg.norm(burner_pos - obj_pos)

                    obj_on_site = dist < threshold
                    if location in knobs_state:
                        knob_angle = np.abs(knobs_state[location])
                        knob_on = 0.35 <= knob_angle <= (2 * np.pi - 0.35)
                    else:
                        knob_on = False

                    if obj_on_site and knob_on:
                        return location

        return None

    def get_obj_location_on_stove(self, env, obj_name, threshold=0.08):
        """
        Check if the object (e.g., pan) is correctly placed on the stove and aligned with a burner.
        Args:
            obj_name (str): The name of the object to check.
            threshold (float): Maximum allowed distance from the burner to consider it aligned.
        Returns:
            str or None: The closest burner if the object is within the threshold, else None.
        """

        obj = env.objects[obj_name]
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])[0:2]

        obj_on_stove = OU.check_obj_fixture_contact(env, obj_name, self)
        if not obj_on_stove:
            return None

        closest_burner = None
        closest_distance = float("inf")

        for burner_name, site in self.burner_sites.items():
            if site is None:
                continue

            burner_pos = np.array(env.sim.data.get_site_xpos(site.get("name")))[0:2]
            distance = np.linalg.norm(burner_pos - obj_pos)

            if distance < closest_distance:
                closest_burner = burner_name
                closest_distance = distance

        return closest_burner

    def obj_fully_inside_receptacle(self, env, obj_name, receptacle_id, tol=0.01):
        """
        Checks if an object is fully inside a receptacle (mostly for pans/pots),
        with a tolerance margin applied only to the height (z-axis).

        Args:
            env (MujocoEnv): The simulation environment.
            obj_name (str): Name of the object.
            receptacle_id (str): Identifier of the pot/pan.
            tol (float): Tolerance margin to relax the check on the z-axis (default: 0.02).

        Returns:
            bool: True if the object is considered fully inside the receptacle, False otherwise.
        """
        obj = env.objects[obj_name]
        receptacle = env.objects[receptacle_id]

        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
        obj_quat = T.convert_quat(
            env.sim.data.body_xquat[env.obj_body_id[obj_name]], to="xyzw"
        )

        receptacle_pos = np.array(
            env.sim.data.body_xpos[env.obj_body_id[receptacle_id]]
        )
        receptacle_quat = T.convert_quat(
            env.sim.data.body_xquat[env.obj_body_id[receptacle_id]], to="xyzw"
        )

        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
        recep_points = receptacle.get_bbox_points(
            trans=receptacle_pos, rot=receptacle_quat
        )

        recep_min = np.min(recep_points, axis=0)
        recep_max = np.max(recep_points, axis=0)
        rx_min, ry_min, rz_min = recep_min
        rx_max, ry_max, rz_max = recep_max

        rz_min_relaxed = rz_min - tol
        rz_max_relaxed = rz_max + tol

        fully_inside = True
        for point in obj_points:
            x, y, z = point
            if not (
                rx_min <= x <= rx_max
                and ry_min <= y <= ry_max
                and rz_min_relaxed <= z <= rz_max_relaxed
            ):
                fully_inside = False
                break

        return fully_inside

    @property
    def knob_joints(self):
        """
        Returns the knob joints of the stove
        """
        if self._knob_joints is None:
            self._knob_joints = {}
            for location in STOVE_LOCATIONS:
                joint = self.worldbody.find(
                    "./body/body/body/joint[@name='{}knob_{}_joint']".format(
                        self.naming_prefix, location
                    )
                )
                self._knob_joints[location] = joint

        return self._knob_joints

    @property
    def burner_sites(self):
        """
        Returns the burner sites of the stove
        """
        if self._burner_sites is None:
            self._burner_sites = {}
            for location in STOVE_LOCATIONS:
                site = self.worldbody.find(
                    "./body/body/site[@name='{}burner_on_{}']".format(
                        self.naming_prefix, location
                    )
                )
                self._burner_sites[location] = site

        return self._burner_sites

    @property
    def nat_lang(self):
        return "stove"


class Stovetop(Stove):
    """
    Stovetop fixture class. The stovetop has knob joints that can be turned on and off to simulate burner flames

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    def __init__(
        self, xml="fixtures/stoves/sleek_silver_top_gas", name="stove", *args, **kwargs
    ):
        super().__init__(xml=xml, name=name, *args, **kwargs)
