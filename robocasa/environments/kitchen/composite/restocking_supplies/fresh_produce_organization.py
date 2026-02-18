from robocasa.environments.kitchen.kitchen import *


class FreshProduceOrganization(Kitchen):
    """
    Fresh Produce Organization: composite task for Restocking Supplies activity.
    Simulates the task of organizing fresh produce in the fridge.
    Steps:
        1. Place the fresh produce in the correct fridge rack based on whether it's a fruit or vegetable.
        2. Close the fridge door.
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

        if "refs" in self._ep_meta:
            self.fruit_rack_index = self._ep_meta["refs"]["fruit_rack_index"]
            self.veg_rack_index = self._ep_meta["refs"]["veg_rack_index"]
            self.restock_obj_cat = self._ep_meta["refs"]["restock_obj_cat"]
        else:
            self.fruit_rack_index = -1 if self.rng.random() < 0.5 else -2
            self.veg_rack_index = -2 if self.fruit_rack_index == -1 else -1
            self.restock_obj_cat = self.rng.choice(["fruit", "vegetable"])

        self.init_robot_base_ref = self.counter

    def _setup_scene(self):
        self.fridge.open_door(env=self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_lang = self.get_obj_lang("food")

        ep_meta[
            "lang"
        ] = f"Place the {food_lang} in the correct fridge rack based on whether it is a fruit or vegetable. Finally, close the fridge door."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["fruit_rack_index"] = self.fruit_rack_index
        ep_meta["refs"]["veg_rack_index"] = self.veg_rack_index
        ep_meta["refs"]["restock_obj_cat"] = self.restock_obj_cat
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="food",
                obj_groups=self.restock_obj_cat,
                init_robot_here=True,
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(1.0, 0.40),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable_fridge",
                obj_groups="vegetable",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        rack_index=self.veg_rack_index,
                    ),
                    size=(0.40, 0.20),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit_fridge",
                obj_groups="fruit",
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="fridge",
                        rack_index=self.fruit_rack_index,
                    ),
                    size=(0.40, 0.20),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        veg_on_correct_shelf = self.fridge.check_rack_contact(
            self,
            "vegetable_fridge",
            rack_index=self.veg_rack_index,
            compartment="fridge",
        )
        fruit_on_correct_shelf = self.fridge.check_rack_contact(
            self, "fruit_fridge", rack_index=self.fruit_rack_index, compartment="fridge"
        )
        food_rack_index = (
            self.fruit_rack_index
            if self.restock_obj_cat == "fruit"
            else self.veg_rack_index
        )
        food_on_correct_shelf = self.fridge.check_rack_contact(
            self, "food", rack_index=food_rack_index, compartment="fridge"
        )
        gripper_far = all(
            OU.gripper_obj_far(self, obj)
            for obj in ["vegetable_fridge", "fruit_fridge", "food"]
        )
        fridge_door_closed = self.fridge.is_closed(self)

        return (
            veg_on_correct_shelf
            and fruit_on_correct_shelf
            and food_on_correct_shelf
            and gripper_far
            and fridge_door_closed
        )
