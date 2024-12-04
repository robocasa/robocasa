from robocasa.environments.kitchen.kitchen import *


class DessertAssembly(Kitchen):
    """
    Dessert Assembly: composite task for Clearing Table activity.

    Simulates the task of assembling desserts.

    Steps:
        Pick the container with the dessert on it and place in on the tray.
        Pick the cupcake and place it on the tray.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(1.0, 0.4))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        dessert1 = self.get_obj_lang("dessert1")
        container = self.get_obj_lang("dessert1_container")

        ep_meta["lang"] = (
            f"Pick up the {container} with {dessert1} and place them on the tray. "
            "Pick up the cupcake and place it on the tray."
        )

        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="tray",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(1, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert1",
                obj_groups=["donut", "cake"],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0, -1),
                    try_to_place_in="bowl",
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert2",
                obj_groups="cupcake",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0, -1),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        sweets_on_tray = (
            OU.check_obj_in_receptacle(self, "dessert1", "dessert1_container")
            and OU.check_obj_in_receptacle(self, "dessert2", "receptacle")
            and OU.check_obj_in_receptacle(self, "dessert1_container", "receptacle")
        )

        return sweets_on_tray and OU.gripper_obj_far(self, "receptacle")
