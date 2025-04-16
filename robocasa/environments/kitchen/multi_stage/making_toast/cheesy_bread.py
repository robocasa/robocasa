from robocasa.environments.kitchen.kitchen import *


class CheesyBread(Kitchen):
    """
    Cheesy Bread: composite task for Making Toast activity.

    Simulates the task of making cheesy bread.

    Steps:
        Start with a slice of bread already on a plate and a wedge of cheese on the
        counter. Pick up the wedge of cheese and place it on the slice of bread to
        prepare a simple cheese on bread dish.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6))
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread_flat",
                object_scale=1.5,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.7),
                    pos=(0, -1.0),
                    try_to_place_in="cutting_board",
                ),
            )
        )
        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
                init_robot_here=True,
                placement=dict(
                    ref_obj="bread_container",
                    fixture=self.counter,
                    size=(1.0, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        # Distractor on the counter
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(fixture=self.counter, size=(1.0, 0.20), pos=(0, 1.0)),
            )
        )
        return cfgs

    def _check_success(self):
        # Bread is still on the cutting board, and cheese is on top
        return (
            OU.check_obj_in_receptacle(self, "bread", "bread_container")
            and OU.gripper_obj_far(self, obj_name="cheese")
            and self.check_contact(self.objects["cheese"], self.objects["bread"])
        )
