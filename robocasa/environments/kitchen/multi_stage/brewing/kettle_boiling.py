from robocasa.environments.kitchen.kitchen import *


class KettleBoiling(Kitchen):
    """
    Kettle Boiling: composite task for Brewing activity.

    Simulates the task of boiling water in a kettle.

    Steps:
        Take the kettle from the counter and place it on a stove burner.
        Turn the burner on.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_pos = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.2, 0.2))
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                heatable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="stove_distr",
                obj_groups=("pan", "pot"),
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    # apply rotations so the handle doesnt stick too much
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the kettle from the counter and place it on a stove burner. Then, turn the burner on"
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=self, rng=self.rng)

    def _check_success(self):
        """
        Check if the kettle is placed on the stove burner and the burner is turned on.
        """
        knobs_state = self.stove.get_knobs_state(env=self)
        kettle = self.objects["obj"]
        kettle_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[kettle.name]])[
            0:2
        ]
        obj_on_stove = OU.check_obj_fixture_contact(self, "obj", self.stove)
        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(
                        self.sim.data.get_site_xpos(site.get("name"))
                    )[0:2]
                    dist = np.linalg.norm(burner_pos - kettle_pos)

                    kettle_on_site = dist < 0.15
                    knob_on = (
                        (0.35 <= np.abs(knobs_state[location]) <= 2 * np.pi - 0.35)
                        if location in knobs_state
                        else False
                    )

                    if kettle_on_site and knob_on and OU.gripper_obj_far(self):
                        return True

        return False
