from robocasa.environments.kitchen.kitchen import *


class SetupButterPlate(Kitchen):
    """
    Place Butter on the Table: composite task for Setting The Table activity.

    Simulates the task of retrieving butter and a butter knife and placing it on a plate on the dining table.

    Steps:
        1. Open the refrigerator and pick up the butter.
        2. Place the butter on the plate on the dining table.
        3. Place the butter knife near the plate on the dining table.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("objaverse", "lightwheel", "aigen"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Retrieve the butter from the fridge and place it on a plate on the dining table. "
            "Then retrieve the butter knife by the fridge and place it near the plate."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="butter",
                obj_groups="butter_stick",
                object_scale=1.15,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.20, 0.20),
                    pos=(0, -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                exclude_obj_groups="butter_stick",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.35, 0.35),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.15,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="butter_knife",
                obj_groups="knife",
                object_scale=[1.1, 1.1, 1.75],
                info={"mjcf_path": "aigen_objs/knife/knife_8/model.xml"},
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.30, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        butter_on_plate = OU.check_obj_in_receptacle(self, "butter", "plate")

        knife_pos = self.sim.data.body_xpos[self.obj_body_id["butter_knife"]]
        plate_pos = self.sim.data.body_xpos[self.obj_body_id["plate"]]

        xy_dist = np.linalg.norm(knife_pos[:2] - plate_pos[:2])
        knife_near_plate = xy_dist <= 0.3

        knife_on_counter = OU.check_obj_any_counter_contact(self, "butter_knife")

        gripper_obj_far_butter = OU.gripper_obj_far(self, obj_name="butter")
        gripper_obj_far_knife = OU.gripper_obj_far(self, obj_name="butter_knife")

        return (
            butter_on_plate
            and knife_near_plate
            and gripper_obj_far_butter
            and gripper_obj_far_knife
            and knife_on_counter
        )
