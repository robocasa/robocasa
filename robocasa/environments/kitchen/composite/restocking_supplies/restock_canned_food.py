from robocasa.environments.kitchen.kitchen import *


class RestockCannedFood(Kitchen):
    """
    Restock Canned Food: composite task for Restocking Supplies activity.
    Simulates the task of restocking canned food in a cabinet.
    Steps:
        1. Take the new canned food from the counter
        2. Place the new canned food in the cabinet
        3. Take the old canned food from the cabinet
        4. Place the old canned food on the counter
    """

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Restock the canned food cabinet by taking the new fresh canned food from the counter and placing it in the cabinet. Then, take the old canned food from the cabinet and place it on the counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):

        cfgs = []
        cfgs.append(
            dict(
                name="canned_food_new",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.20),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                exclude_obj_groups="canned_food",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="canned_food_new",
                    size=(1.0, 0.30),
                    pos=(0, 0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="canned_food_old",
                obj_groups="canned_food",
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):

        new_canned_food_in_cab = OU.obj_inside_of(self, "canned_food_new", self.cab)
        old_canned_food_on_counter = OU.check_obj_any_counter_contact(
            self, "canned_food_old"
        )
        gripper_obj_far = OU.gripper_obj_far(
            self, obj_name="canned_food_old"
        ) and OU.gripper_obj_far(self, obj_name="canned_food_new")
        return new_canned_food_in_cab and old_canned_food_on_counter and gripper_obj_far
