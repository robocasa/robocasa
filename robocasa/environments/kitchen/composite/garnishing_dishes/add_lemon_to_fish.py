from robocasa.environments.kitchen.kitchen import *


class AddLemonToFish(Kitchen):
    """
    Add Lemon to Fish: composite task for Garnishing Dishes activity.

    Simulates the task of garnishing a fish dish with a lemon wedge.

    Steps:
        Take a lemon wedge from the fridge and place it on a plate with fish
        on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
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
        lemon_name = self.get_obj_lang("lemon_wedge")
        fish_name = self.get_obj_lang("fish")
        ep_meta[
            "lang"
        ] = f"Take the {lemon_name} from the fridge and place it on the plate with {fish_name} on the dining counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="lemon_wedge",
                obj_groups="lemon_wedge",
                graspable=True,
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
                name="fish",
                obj_groups="fish",
                graspable=False,
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
        lemon_on_plate = OU.check_obj_in_receptacle(
            self, "lemon_wedge", "fish_container"
        )
        fish_on_plate = OU.check_obj_in_receptacle(self, "fish", "fish_container")
        plate_on_table = OU.check_obj_fixture_contact(
            self, "fish_container", self.dining_counter
        )
        gripper_far = OU.gripper_obj_far(self, "lemon_wedge")
        return lemon_on_plate and fish_on_plate and plate_on_table and gripper_far
