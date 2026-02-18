from robocasa.environments.kitchen.kitchen import *


class PlaceVeggiesInDrawer(Kitchen):
    """
    Place Veggies In Drawer: composite task for Loading Fridge.

    Simulates the task of opening the fridge drawer and placing vegetables inside.

    Steps:
        Open the fridge drawer and place the vegetables inside.
    """

    # no drawer in side-by-side fridge in these styles
    EXCLUDE_STYLES = [23, 24, 25, 27, 28, 37, 38, 40, 44, 47, 56]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )

        self.init_robot_base_ref = self.counter

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, min=1.0, max=1.0)
        self.fridge.open_door(
            self, min=0.5, max=0.8, reg_type="drawer", drawer_rack_index=-1
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg1 = self.get_obj_lang("vegetable1")
        veg2 = self.get_obj_lang("vegetable2")

        if veg1 == veg2:
            veg_text = f"{veg1}s"
        else:
            veg_text = f"{veg1} and {veg2}"

        ep_meta[
            "lang"
        ] = f"Place the {veg_text} inside the fridge drawer. Close the drawer after placing the vegetables."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                init_robot_here=True,
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(0.5, 0.2),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="vegetable1",
                    size=(0.5, 0.2),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_item",
                exclude_obj_groups=("vegetable"),
                fridgable=False,
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 1.0),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.5, 0.5),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veg_in_drawer = all(
            self.fridge.check_rack_contact(
                self, veg, compartment="fridge", reg_type=("drawer"), rack_index=-1
            )
            for veg in ["vegetable1", "vegetable2"]
        )
        is_drawer_closed = self.fridge.is_closed(
            self, compartment="fridge", reg_type="drawer", drawer_rack_index=-1, th=0.02
        )

        gripper_far = all(
            OU.gripper_obj_far(self, obj) for obj in ["vegetable1", "vegetable2"]
        )

        return veg_in_drawer and gripper_far and is_drawer_closed
