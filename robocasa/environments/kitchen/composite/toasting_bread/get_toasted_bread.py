from robocasa.environments.kitchen.kitchen import *


class GetToastedBread(Kitchen):
    """
    Get Toasted Bread: composite task for Toasting Bread activity.

    Steps:
        1. Pull the toaster lever down and wait for the toaster to finish toasting the bread.
        2. Once the lever pops up, pull the bread out.
        3. Place the bread on the plate on the dining counter.
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
        self.toaster = self.register_fixture_ref(
            "toaster", dict(id=FixtureType.TOASTER)
        )
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Start the toaster. "
                "Once the lever pops up, take the bread to the plate on the dining counter."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.toaster_on = False

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("sandwich_bread"),
                object_scale=[1, 1, 1.15],
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
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
                    size=(0.50, 0.30),
                    pos=(0, "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        toast_slot = 0
        toast_in_slot = False
        for slot_pair in range(len(self.toaster.get_state(self).keys())):
            if self.toaster.check_slot_contact(self, "obj", slot_pair=slot_pair):
                toast_slot = slot_pair
                toast_in_slot = True
                break

        if (
            toast_in_slot
            and self.toaster.get_state(self, slot_pair=toast_slot)["turned_on"]
        ):
            self.toaster_on = True

        return (
            self.toaster_on
            and OU.check_obj_in_receptacle(self, "obj", "plate")
            and OU.gripper_obj_far(self, "obj")
        )
