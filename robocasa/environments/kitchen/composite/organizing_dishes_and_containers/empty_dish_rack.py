from robocasa.environments.kitchen.kitchen import *


class EmptyDishRack(Kitchen):
    """
    Empty Dish Drying Rack: composite task for Organizing Dishes and Containers.

    Simulates the task of removing a drinking container from the dish rack and placing it in the cabinet.

    Steps:
        Pick up the mug from the dish rack and place it in the cabinet.
        Close the cabinet door(s).
    """

    # excluded models due to placement issues
    EXCLUDE_STYLES = [
        12,
        14,
        15,
        16,
        18,
        19,
        20,
        22,
        23,
        25,
        27,
        29,
        31,
        32,
        34,
        36,
        38,
        39,
        43,
        44,
        45,
        48,
        49,
        50,
        54,
        55,
        56,
        57,
        58,
        60,
    ]

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["dish_rack"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dish_rack = self.register_fixture_ref(
            "dish_rack", dict(id=FixtureType.DISH_RACK)
        )
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_WITH_DOOR, ref=self.dish_rack)
        )
        self.init_robot_base_ref = self.dish_rack

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the mug from the dish rack and place it inside the open cabinet where all mugs go. Then close the cabinet door(s)."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        for i in range(2):
            cfgs.append(
                dict(
                    name=f"cabinet_mug{i+1}",
                    obj_groups="mug",
                    placement=dict(
                        fixture=self.cabinet,
                        size=(0.40, 0.20),
                        pos=(-0.9 + (i * 0.9), -1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="mug",
                obj_groups="mug",
                object_scale=0.95,
                placement=dict(
                    fixture=self.dish_rack,
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        mug_inside = OU.obj_inside_of(self, "mug", self.cabinet)
        gripper_far = OU.gripper_obj_far(self, obj_name="mug")
        cabinet_closed = self.cabinet.is_closed(env=self)

        return mug_inside and gripper_far and cabinet_closed
