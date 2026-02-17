from robocasa.environments.kitchen.kitchen import *


class PJSandwichPrep(Kitchen):
    """
    PJ Sandwich Preparation: composite task for Toasting Bread activity.

    Simulates the task of preparing toasted bread for a peanut butter and jelly sandwich.

    Steps:
        1. Place the toasted bread in a bowl
        2. Carry the bowl to the dining counter and place it next to the peanut butter and jelly jars
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("objaverse", "lightwheel", "aigen"), *args, **kwargs
    ):
        obj_registries = list(obj_registries or [])
        # make sure to use aigen objects to access the butter
        if "aigen" not in obj_registries:
            obj_registries.append("aigen")
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.toaster = self.register_fixture_ref(
            "toaster", dict(id=FixtureType.TOASTER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the two toasted bread slices and place them in a bowl next to the toaster. "
            "Then carry the bowl to the dining counter and place it next to the peanut butter bottle and jelly jar "
            "to make a peanut butter and jelly sandwich."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bread_slice_1",
                obj_groups=("sandwich_bread"),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    sample_region_kwargs=dict(
                        side="left",
                    ),
                    rotation=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread_slice_2",
                obj_groups=("sandwich_bread"),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    sample_region_kwargs=dict(
                        side="right",
                    ),
                    rotation=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster,
                    ),
                    size=(0.8, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="peanut_butter",
                obj_groups="peanut_butter",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.35, 0.3),
                    pos=(0, "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="jelly",
                obj_groups="jam",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.7, 0.3),
                    pos=(0, "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        bread_in_bowl = []
        for i in range(1, 3):
            if OU.check_obj_in_receptacle(self, f"bread_slice_{i}", "bowl"):
                bread_in_bowl.append(i)

        if not bread_in_bowl:
            bread1_in_bowl = False
            bread2_in_bowl = False
        else:
            bread1_in_bowl = 1 in bread_in_bowl
            bread2_in_bowl = 2 in bread_in_bowl

            if not bread1_in_bowl:
                for j in bread_in_bowl:
                    if self.check_contact(
                        self.objects["bread_slice_1"], self.objects[f"bread_slice_{j}"]
                    ):
                        bread1_in_bowl = True
                        break

            if not bread2_in_bowl:
                for j in bread_in_bowl:
                    if self.check_contact(
                        self.objects["bread_slice_2"], self.objects[f"bread_slice_{j}"]
                    ):
                        bread2_in_bowl = True
                        break

        bowl_on_dining_counter = OU.check_obj_fixture_contact(
            self, "bowl", self.dining_counter
        )

        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]]
        pb_pos = self.sim.data.body_xpos[self.obj_body_id["peanut_butter"]]
        jelly_pos = self.sim.data.body_xpos[self.obj_body_id["jelly"]]

        dist_to_pb = np.linalg.norm(bowl_pos[:2] - pb_pos[:2])
        dist_to_jelly = np.linalg.norm(bowl_pos[:2] - jelly_pos[:2])

        bowl_positioned_correctly = dist_to_pb < 0.4 and dist_to_jelly < 0.4
        gripper_far_from_bowl = OU.gripper_obj_far(self, "bowl")

        return (
            bread1_in_bowl
            and bread2_in_bowl
            and bowl_on_dining_counter
            and bowl_positioned_correctly
            and gripper_far_from_bowl
        )
