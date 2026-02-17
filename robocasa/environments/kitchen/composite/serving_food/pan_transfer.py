from robocasa.environments.kitchen.kitchen import *


class PanTransfer(Kitchen):
    """
    Pan Transfer: composite task for Serving Food activity.

    Simulates the task of transferring vegetables from a pan to a plate.

    Steps:
        Pick up the pan and dump the vegetables in it onto the plate.
        Then, return the pan to the stove.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the pan and dump the vegetables in it onto the plate. "
                "Then return the pan to the stove."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self._robot_touched_food = False

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="pan",
                ),
            )
        )
        # cfgs.append(dict(
        #     name="vegetable2",
        #     obj_groups="vegetable",
        #     placement=dict(
        #         size=(0.01, 0.01),
        #         ensure_object_boundary_in_range=False,
        #         sample_args=dict(
        #             reference="vegetable_container"
        #         )
        #     ),
        # ))

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                exclude_obj_groups=["plate", "pan", "vegetable"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.20),
                    pos=(0.5, 0.5),
                ),
            )
        )
        return cfgs

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):
        """
        Check if the object is on the stove and close to a burner
        Returns the location of the burner if the object is on the stove, and close to a burner.
        None otherwise.
        """

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

                    if obj_on_site:
                        return location

        return None

    def _post_action(self, action):
        ret = super()._post_action(action)
        contact = self.check_contact(
            self.robots[0].gripper["right"], self.objects["vegetable"]
        )
        self._robot_touched_food = self._robot_touched_food or contact
        return ret

    def _check_success(self):
        vegetable_on_plate = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        pan_on_stove = (
            self._check_obj_location_on_stove("vegetable_container") is not None
        )
        gripper_obj_far = OU.gripper_obj_far(self, "vegetable_container")
        return (
            vegetable_on_plate
            and pan_on_stove
            and gripper_obj_far
            and (not self._robot_touched_food)
        )
