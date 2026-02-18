from robocasa.environments.kitchen.kitchen import *


class PackSnack(Kitchen):
    """
    Pack Snack: composite task for Packing Lunches activity.

    Simulates the task of taking a snack (boxed_food) from a cabinet and placing it
    next to a packed lunch (tupperware) on the counter to take on the go.

    Steps:
        1. Take a snack item from the cabinet.
        2. Place it next to the tupperware containing the packed lunch on the counter.
        3. Ensure the snack is close enough to the tupperware for easy access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, full_depth_region=True)
        )
        self.init_robot_base_ref = self.cabinet

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Take the snack from the cabinet and place it next to the tupperware "
            f"containing the packed lunch on the counter to take on the go."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[1.75, 1.75, 1.25],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(full_depth_region=True),
                    size=(0.5, 0.4),
                    pos=(1.0, -1.0),
                    rotation=(0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1",
                obj_groups=("cooked_food", "meat"),
                exclude_obj_groups=("kebab_skewer", "pizza", "shrimp"),
                placement=dict(
                    object="tupperware",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food2",
                obj_groups=("vegetable"),
                placement=dict(
                    object="tupperware",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="snack",
                obj_groups=("boxed_food", "chips"),
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.5, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cabinet",
                obj_groups=("receptacle", "stackable", "tool"),
                exclude_obj_groups=("tupperware"),
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        snack_on_counter = OU.check_obj_any_counter_contact(self, "snack")

        obj1_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects["snack"].name]
        ][:2]
        obj2_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects["tupperware"].name]
        ][:2]
        snack_tupperware_close = np.linalg.norm(obj1_pos - obj2_pos) <= 0.3

        gripper_far = OU.gripper_obj_far(self, "snack")
        return snack_on_counter and snack_tupperware_close and gripper_far
