from robocasa.environments.kitchen.kitchen import *


class ScalePortioning(Kitchen):
    """
    Scale Portioning: composite task for Portioning Meals activity.

    Simulates the task of weighing meat on a digital scale and then placing it on a plate.
    The task involves taking meat from the fridge, placing it on a digital scale,
    waiting for 50 timesteps, then moving it to a plate on the dining counter.

    Steps:
        1. Take meat from the fridge
        2. Place meat on the digital scale
        3. Wait 50 timesteps
        4. Move meat from scale to plate on dining counter
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )

        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        ep_meta["lang"] = (
            f"Take the {meat_name} from the fridge and place it on the digital scale on the counter by the fridge. "
            "Wait a few seconds for a reading, then move it to the plate on the dining counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.scale_wait_counter = 0
        self.scale_wait_complete = False
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
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
                name="digital_scale",
                obj_groups="digital_scale",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                    ),
                    size=(0.4, 0.5),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                fridgable=True,
                exclude_obj_groups=("meat"),
                placement=dict(
                    fixture=self.fridge,
                    size=(0.6, 0.3),
                    pos=(0, 1),
                ),
            )
        )

        return cfgs

    def update_state(self):
        super().update_state()
        meat_on_scale = OU.check_obj_in_receptacle(self, "meat", "digital_scale")
        gripper_far_from_meat_on_scale = OU.gripper_obj_far(self, "meat", th=0.1)
        if (
            meat_on_scale
            and not self.scale_wait_complete
            and gripper_far_from_meat_on_scale
        ):
            self.scale_wait_counter += 1
            if self.scale_wait_counter >= 50:
                self.scale_wait_complete = True

    def _check_success(self):
        meat_on_plate = OU.check_obj_in_receptacle(self, "meat", "plate")
        gripper_far = OU.gripper_obj_far(self, "meat")
        return meat_on_plate and gripper_far and self.scale_wait_complete
