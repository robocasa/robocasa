from robocasa.environments.kitchen.kitchen import *


class RetrieveIceTray(Kitchen):
    """
    Retrieve Ice Tray: composite task for Adding Ice to Beverages activity.

    Simulates retrieving the ice cube tray from the freezer and placing it on the dining counter, where there is a glass cup.

    Steps:
        1. Open the freezer (fridge compartment).
        2. Pick up the ice cube tray from the freezer.
        3. Place it on the dining counter, where there is a glass cup.
    """

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + Kitchen.FREEZER_EXCLUDED_LAYOUTS
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Open the freezer door, pick up the ice cube tray from the freezer, "
            f"and place it on the dining counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ice_cube_tray",
                obj_groups="ice_cube_tray",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=-1,
                    ),
                    size=(0.25, 0.3),
                    pos=(0, -1.0),
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                fridgable=True,
                freezable=True,
                exclude_obj_groups="ice_cube_tray",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=-2,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups="glass_cup",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.15, 0.15),
                    pos=("ref", "ref"),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        on_counter = OU.check_obj_fixture_contact(
            self, "ice_cube_tray", self.dining_counter
        )
        gripper_far = OU.gripper_obj_far(self, "ice_cube_tray")
        return on_counter and gripper_far
