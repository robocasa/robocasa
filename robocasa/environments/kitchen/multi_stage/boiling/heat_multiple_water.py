from robocasa.environments.kitchen.kitchen import *


class HeatMultipleWater(Kitchen):
    """
    Heat Multiple Water: composite task for Boiling activity.

    Simulates the process of heating water in a pot and a kettle.

    Steps:
        Take the kettle from the cabinet and place it on a stove burner.
        Take the pot from the counter and place it on another stove burner.
        Turn both burners on.

    Args:
        init_robot_base_pos (str): Specifies a fixture to initialize the robot
            in front of. Default is "stove".
    """

    def __init__(self, init_robot_base_pos="stove", *args, **kwargs):
        super().__init__(init_robot_base_pos=init_robot_base_pos, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.ref_cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_TOP, ref=self.stove)
        )
        self.ref_counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.ref_cab, size=(0.2, 0.2))
        )

        self.init_robot_base_pos = self.ref_cab

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("pot"),
                graspable=True,
                heatable=True,
                placement=dict(
                    fixture=self.ref_counter,
                    sample_region_kwargs=dict(
                        ref=self.ref_cab,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", 0),
                    # ensure_object_boundary_in_range=False because the pots handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                placement=dict(
                    fixture=self.ref_cab,
                    size=(0.50, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the kettle from the cabinet and place it on a stove burner. "
            "Then pick the pot from the counter and place it on another stove burner. "
            "Finally, turn both burners on."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.ref_cab.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        valid_knobs = self.stove.get_knobs_state(env=self).keys()

        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):
        """
        Check if the object is placed on any of the stove burners

        Args:
            obj_name (str): object name

            threshold (float): distance threshold from object center to stove burner site center

        Returns:
            str: location of the stove burner site if the object is placed on it, None otherwise.
        """

        knobs_state = self.stove.get_knobs_state(env=self)
        obj = self.objects[obj_name]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])[0:2]
        obj_on_stove = OU.check_obj_fixture_contact(self, obj_name, self.stove)
        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(
                        self.sim.data.get_site_xpos(site.get("name"))
                    )[0:2]
                    dist = np.linalg.norm(burner_pos - obj_pos)

                    obj_on_site = dist < threshold
                    knob_on = (
                        (0.35 <= np.abs(knobs_state[location]) <= 2 * np.pi - 0.35)
                        if location in knobs_state
                        else False
                    )

                    if obj_on_site and knob_on:
                        return location

        return None

    def _check_success(self):

        pan_loc = self._check_obj_location_on_stove("obj", threshold=0.15)
        kettle_loc = self._check_obj_location_on_stove("obj2")

        # both objects placed on different parts of the stove
        successful_stove_placement = (
            (pan_loc is not None)
            and (kettle_loc is not None)
            and (pan_loc != kettle_loc)
        )

        return (
            successful_stove_placement
            and OU.gripper_obj_far(self)
            and OU.gripper_obj_far(self, "obj2")
        )
