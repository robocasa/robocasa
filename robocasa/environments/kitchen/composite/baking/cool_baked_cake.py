from robocasa.environments.kitchen.kitchen import *


class CoolBakedCake(Kitchen):
    """
    Cool Baked Cake: composite task for Baking Cookies and Cakes activity.
    Simulates the task of cooling a baked cake.
    Steps:
        Take the baked cake out of the oven and place it on the plate for cooling.
        Then close the oven door.
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.oven)
        )
        self.init_robot_base_ref = self.oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Take the baked cake out of the oven and place it on the plate for cooling. Then close the oven door."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cake",
                obj_groups=("cake"),
                graspable=True,
                placement=dict(
                    fixture=self.oven,
                    size=(1.0, 0.45),
                    pos=(0, -1.0),
                    try_to_place_in="oven_tray",
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.oven,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj_in_recep = OU.check_obj_in_receptacle(self, "cake", "container")
        recep_on_counter = self.check_contact(self.objects["container"], self.counter)
        oven_closed = self.oven.is_closed(self)

        return (
            oven_closed
            and obj_in_recep
            and recep_on_counter
            and OU.gripper_obj_far(self, "cake")
        )
