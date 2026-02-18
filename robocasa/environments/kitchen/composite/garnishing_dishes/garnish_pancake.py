from robocasa.environments.kitchen.kitchen import *


class GarnishPancake(Kitchen):
    """
    Garnish Pancake: composite task for Garnishing Dishes activity.

    Simulates the task of garnishing a pancake with a strawberry topping.

    Steps:
        Take a strawberry from the fridge and place it on a pancake
        on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        kwargs["obj_registries"] = ("aigen", "objaverse", "lightwheel")
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Take the strawberry from the fridge and place it on top of the pancake, located on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="strawberry",
                obj_groups="strawberry",
                object_scale=0.9,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.15, 0.15),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(z_range=(1.0, 1.5)),
                ),
            )
        )
        # Distractor object in fridge
        cfgs.append(
            dict(
                name="distr_fridge",
                obj_groups="food",
                placement=dict(
                    fixture=self.fridge,
                    size=(0.30, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="pancake",
                obj_groups="pancake",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        strawberry_on_pancake = OU.check_obj_in_receptacle(
            self, "strawberry", "pancake"
        )
        pancake_on_plate = OU.check_obj_in_receptacle(
            self, "pancake", "pancake_container"
        )
        plate_on_table = OU.check_obj_fixture_contact(
            self, "pancake_container", self.dining_counter
        )
        gripper_far_strawberry = OU.gripper_obj_far(self, "strawberry")
        return (
            strawberry_on_pancake
            and pancake_on_plate
            and plate_on_table
            and gripper_far_strawberry
        )
