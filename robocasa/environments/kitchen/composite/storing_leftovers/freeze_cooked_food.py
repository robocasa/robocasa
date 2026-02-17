from robocasa.environments.kitchen.kitchen import *


class FreezeCookedFood(Kitchen):
    """
    Freeze Cooked Food: composite task for Storing Leftovers activity.

    Simulates the process of storing cooked food into the freezer for long-term storage.
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cooked_food = self.get_obj_lang("cooked_food")

        ep_meta["lang"] = (
            f"Store the {cooked_food} in the freezer for long-term storage. "
            f"Close the freezer door after placing the cooked food."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cooked_food",
                obj_groups="cooked_food",
                init_robot_here=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        rack_index=0,
                    ),
                    size=(1.0, 0.3),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                    ),
                    size=(1.0, 0.2),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cooked_food_in_freezer = self.fridge.check_rack_contact(
            self, "cooked_food", compartment="freezer"
        )

        freezer_door_closed = self.fridge.is_closed(
            env=self,
            compartment="freezer",
        )

        return cooked_food_in_freezer and freezer_door_closed
