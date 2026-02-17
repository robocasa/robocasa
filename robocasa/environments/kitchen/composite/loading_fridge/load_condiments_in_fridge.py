from robocasa.environments.kitchen.kitchen import *


class LoadCondimentsInFridge(Kitchen):
    """
    Load Condiments In Fridge: composite task for Loading Fridge.

    Simulates the task of organizing condiments in the fridge by placing them
    on the top shelf for easy access. There are 2 distractor
    items already in the fridge that may need to be moved if they're on the top shelf.

    Steps:
        Place 2 condiments from the counter next to the fridge onto the top shelf
        of the fridge. If any distractor items are on the top shelf, move them
        to other shelves to make space for the condiments.
    """

    # certain fridge-side-by-side fridges only have 1 shelf
    EXCLUDE_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )
        self.target_rack_index = -1

        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        condiment1_lang = self.get_obj_lang("condiment1")
        condiment2_lang = self.get_obj_lang("condiment2")

        ep_meta["lang"] = (
            f"Place the {condiment1_lang} and {condiment2_lang} from the counter "
            f"to the top shelf of the fridge. "
            f"If the existing items in the fridge are on the top shelf, "
            f"move them to other shelves."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="condiment1",
                obj_groups="condiment",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.fridge,
                        full_depth_region=True,
                    ),
                    size=(0.40, 0.30),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups="condiment",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="condiment1",
                    size=(0.40, 0.30),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor1",
                exclude_obj_groups="condiment",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(-0.3, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor2",
                exclude_obj_groups="condiment",
                fridgable=True,
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.2),
                    pos=(0.3, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        condiment1_on_top_rack = self.fridge.check_rack_contact(
            self, "condiment1", compartment="fridge", rack_index=self.target_rack_index
        )

        condiment2_on_top_rack = self.fridge.check_rack_contact(
            self, "condiment2", compartment="fridge", rack_index=self.target_rack_index
        )

        distractor1_not_on_top = not self.fridge.check_rack_contact(
            self, "distractor1", compartment="fridge", rack_index=self.target_rack_index
        )

        distractor2_not_on_top = not self.fridge.check_rack_contact(
            self, "distractor2", compartment="fridge", rack_index=self.target_rack_index
        )

        distractor1_on_any_rack = self.fridge.check_rack_contact(
            self, "distractor1", compartment="fridge"
        )

        distractor2_on_any_rack = self.fridge.check_rack_contact(
            self, "distractor2", compartment="fridge"
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj)
            for obj in ["condiment1", "condiment2", "distractor1", "distractor2"]
        )

        return (
            condiment1_on_top_rack
            and condiment2_on_top_rack
            and distractor1_not_on_top
            and distractor2_not_on_top
            and distractor1_on_any_rack
            and distractor2_on_any_rack
            and gripper_far
        )
