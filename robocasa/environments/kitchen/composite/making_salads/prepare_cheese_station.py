from robocasa.environments.kitchen.kitchen import *


class PrepareCheeseStation(Kitchen):
    """
    PrepareCheeseStation: composite task for Making Salads activity.

    Simulates the task of setting up cheese to be grated for a salad.

    Steps:
        Pick the cheese from the fridge and pick the cheese grater from the
        cabinet and place them next to the salad bowl.
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet, size=(0.6, 0.6))
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the cheese from the fridge and the cheese grater from the cabinet and place them next to the salad bowl."
        return ep_meta

    def _reset_internal(self):
        self.fridge.open_door(self)
        self.cabinet.open_door(self)
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cheese",
                obj_groups=("cheese"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(z_range=(1.0, 1.5)),
                    size=(0.3, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="lettuce",
                obj_groups="lettuce",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cabinet, top_size=(0.4, 0.4)),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )

        cfgs.append(
            dict(
                name="grater",
                obj_groups="cheese_grater",
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.5, 0.2),
                    pos=(0, -1),
                ),
                object_scale=0.75,
            )
        )

        cfgs.append(
            dict(
                name="distr1",
                fridgable=True,
                exclude_obj_groups=["cheese"],
                placement=dict(
                    fixture=self.fridge,
                    size=(0.6, 0.2),
                    pos=(0, 1),
                    sample_region_kwargs=dict(),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr2",
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.6, 0.2),
                    pos=(0, 1),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        distance_thresh = 0.25

        bowl_pos_xy = self.sim.data.body_xpos[self.obj_body_id["lettuce_container"]][
            0:2
        ]
        bowl_radius = np.linalg.norm(
            self.objects["lettuce_container"].horizontal_radius
        )

        grater_pos_xy = self.sim.data.body_xpos[self.obj_body_id["grater"]][0:2]

        cheese_pos_xy = self.sim.data.body_xpos[self.obj_body_id["cheese"]][0:2]

        cheese_on_counter = OU.check_obj_fixture_contact(self, "cheese", self.counter)
        grater_on_counter = OU.check_obj_fixture_contact(self, "grater", self.counter)
        cheese_far = OU.gripper_obj_far(self, "cheese")
        grater_far = OU.gripper_obj_far(self, "grater")

        objs_on_counter = cheese_on_counter and grater_on_counter
        objs_gripper_far = cheese_far and grater_far
        objs_close = (
            np.linalg.norm(bowl_pos_xy - grater_pos_xy) - bowl_radius
        ) < distance_thresh
        objs_close = (
            objs_close
            and (np.linalg.norm(bowl_pos_xy - cheese_pos_xy) - bowl_radius)
            < distance_thresh
        )
        return objs_on_counter and objs_gripper_far and objs_close
