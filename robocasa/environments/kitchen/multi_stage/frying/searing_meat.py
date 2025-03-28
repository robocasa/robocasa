from robocasa.environments.kitchen.kitchen import *


class SearingMeat(Kitchen):
    """
    Searing Meat: composite task for Frying activity.

    Simulates the task of searing meat.

    Steps:
        Place the pan on the specified burner on the stove,
        then place the meat on the pan and turn the burner on.

    Args:
        knob_id (str): The id of the knob who's burner the pan will be placed on.
            If "random", a random knob is chosen.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.DOOR_TOP_HINGE_DOUBLE, ref=self.stove)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        ep_meta["lang"] = (
            f"Grab the pan from the cabinet and place it on the {self.knob.replace('_', ' ')} burner on the stove. "
            f"Then place the {meat_name} on the stove and turn the burner on."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob

        self.stove.set_knob_state(mode="off", knob=self.knob, env=self, rng=self.rng)
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.cab,
                    pos=(0.0, -1.0),
                    size=(0.8, 0.5),
                    rotation=(np.pi / 2 - np.pi / 16, np.pi / 2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):
        """
        Check if the object is on the stove and close to a burner and the knob is on.
        Returns the location of the burner if the object is on the stove, close to a burner, and the burner is on.
        None otherwise.
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
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="meat")
        pan_loc = self._check_obj_location_on_stove("pan", threshold=0.15) == self.knob
        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "pan", th=0.07)
        return gripper_obj_far and pan_loc and meat_in_pan
